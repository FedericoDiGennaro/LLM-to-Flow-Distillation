"""Shared request transport for the peptide teacher roles."""

from __future__ import annotations

from typing import Any, Mapping, Sequence
from pydantic import BaseModel
from adm.preferences.artifacts import PreferenceArtifactStore
from adm.preferences.clients import (
    StructuredLLMClient,
    StructuredRequest,
    value_as_json,
)


class ProviderCallBudgetExhausted(RuntimeError):
    """Raised before a genuinely new provider dispatch would exceed its ceiling."""


class StructuredSemanticGoalAgent:
    """Shared structured requests, durable replay, and provider-call accounting."""

    # Callers may defer coarse capacity checks to this agent.  It accounts for
    # exact request hashes and charges only requests that would really dispatch.
    request_aware_provider_budget = True

    def __init__(
        self,
        *,
        client: StructuredLLMClient,
        artifact_store: PreferenceArtifactStore,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
        max_provider_calls: int,
        initial_provider_calls: int = 0,
        judge_mode: str = "dual_goal_v1",
        judge_chunk_size: int | None = None,
        use_generic_sequence_evidence: bool = False,
        provider_order_seed: int = 0,
    ) -> None:
        if not model.strip():
            raise ValueError("model must be non-empty")
        if reasoning_effort not in {"none", "low", "medium", "high"}:
            raise ValueError("reasoning_effort must be none, low, medium, or high")
        if max_output_tokens <= 0 or max_provider_calls <= 0:
            raise ValueError("token and provider-call budgets must be positive")
        if (
            isinstance(initial_provider_calls, bool)
            or not isinstance(initial_provider_calls, int)
            or not 0 <= initial_provider_calls <= max_provider_calls
        ):
            raise ValueError(
                "initial_provider_calls must be between zero and the call ceiling"
            )
        if judge_mode not in {
            "dual_goal_v1",
            "current_only_tools_v2",
            "directional_prompt_annealing_v3",
        }:
            raise ValueError("unsupported semantic judge mode")
        if judge_mode in {
            "current_only_tools_v2",
            "directional_prompt_annealing_v3",
        } and not (use_generic_sequence_evidence):
            raise ValueError("the tools-v2 judge requires generic sequence evidence")
        if judge_chunk_size is not None and (
            isinstance(judge_chunk_size, bool)
            or not isinstance(judge_chunk_size, int)
            or judge_chunk_size <= 0
        ):
            raise ValueError("judge_chunk_size must be a positive integer")
        if isinstance(provider_order_seed, bool) or not isinstance(
            provider_order_seed, int
        ):
            raise ValueError("provider_order_seed must be an integer")
        self.client = client
        self.artifact_store = artifact_store
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens
        self.max_provider_calls = max_provider_calls
        self.initial_provider_calls = initial_provider_calls
        self.judge_mode = judge_mode
        self.judge_chunk_size = judge_chunk_size
        self.use_generic_sequence_evidence = use_generic_sequence_evidence
        self.provider_order_seed = provider_order_seed
        self.logical_calls = 0
        # Count durable attempts from prior invocations so a resumed run cannot
        # silently reset the experiment-wide external-call ceiling.
        self.provider_calls = initial_provider_calls

    def _requires_provider_dispatch(
        self,
        request: StructuredRequest,
        response_model: type[BaseModel],
    ) -> bool:
        """Return whether resolving this cache miss would make a new API attempt."""
        checker = getattr(self.client, "requires_provider_dispatch", None)
        if checker is None:
            return True
        required = checker(request, response_model)
        if not isinstance(required, bool):
            raise TypeError("requires_provider_dispatch must return a boolean")
        return required

    def prospective_provider_calls(
        self,
        requests: Sequence[tuple[StructuredRequest, type[BaseModel]]],
    ) -> int:
        """Count unique request hashes that still require a new provider attempt."""
        required = 0
        seen: set[str] = set()
        for request, response_model in requests:
            if request.request_hash in seen:
                continue
            seen.add(request.request_hash)
            if self.artifact_store.load_response(request.request_hash) is not None:
                continue
            if self._requires_provider_dispatch(request, response_model):
                required += 1
        return required

    def ensure_provider_capacity(
        self,
        requests: Sequence[tuple[StructuredRequest, type[BaseModel]]],
    ) -> int:
        """Fail before dispatch unless all prospective cache misses fit the ceiling."""
        required = self.prospective_provider_calls(requests)
        if self.provider_calls + required > self.max_provider_calls:
            raise ProviderCallBudgetExhausted(
                "semantic-agent provider-call budget exhausted"
            )
        return required

    def _request(
        self,
        *,
        schema_id: str,
        system_prompt: str,
        user_payload: Mapping[str, Any],
        logical_context: Mapping[str, Any],
        reasoning_effort_override: str | None = None,
    ) -> StructuredRequest:
        effort = (
            self.reasoning_effort
            if reasoning_effort_override is None
            else reasoning_effort_override
        )
        if effort not in {"none", "low", "medium", "high"}:
            raise ValueError("reasoning effort override is invalid")
        return StructuredRequest.from_payload(
            model=self.model,
            schema_id=schema_id,
            system_prompt=system_prompt,
            user_payload=user_payload,
            logical_context=logical_context,
            max_output_tokens=self.max_output_tokens,
            reasoning_effort=effort,
        )

    def _execute(
        self,
        request: StructuredRequest,
        response_model: type[BaseModel],
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        self.logical_calls += 1
        self.artifact_store.save_request(request)
        cached = self.artifact_store.load_response(request.request_hash)
        if cached is None:
            dispatch_required = self._requires_provider_dispatch(
                request, response_model
            )
            if dispatch_required:
                if self.provider_calls >= self.max_provider_calls:
                    raise ProviderCallBudgetExhausted(
                        "semantic-agent provider-call budget exhausted"
                    )
                # Charge before entering the transport: its durable start marker
                # means an exception can still represent a consumed attempt.
                self.provider_calls += 1
            response = self.client.parse(request, response_model)
            self.artifact_store.save_response(request, response)
            parsed = response.parsed
            usage = response.usage
            raw = response.raw_response
            cache_hit = False
        else:
            parsed = response_model.model_validate(cached.parsed)
            usage = cached.usage
            raw = cached.raw_response
            cache_hit = True
        values = value_as_json(parsed)
        if not isinstance(values, Mapping):
            raise TypeError("structured response must be a mapping")
        raw_mapping = raw if isinstance(raw, Mapping) else {}
        model_version_observer = getattr(
            self.client, "observe_model_version_metadata", None
        )
        if callable(model_version_observer):
            model_version_observer(raw_mapping)
        metadata = {
            "request_hash": request.request_hash,
            "usage": value_as_json(usage),
            "model_version": (
                str(raw_mapping["model_version"])
                if raw_mapping.get("model_version") is not None
                else None
            ),
            "thought_summary": (
                str(raw_mapping["thought_summary"])
                if raw_mapping.get("thought_summary") is not None
                else None
            ),
        }
        # Cache/replay is an invocation detail and is intentionally absent from
        # scientific records, keeping reconstructed outputs byte-stable.
        del cache_hit
        return values, metadata
