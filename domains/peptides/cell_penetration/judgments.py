"""No-menu semantic curriculum discovery for the PepDFM cancer pilot.

Version 4 deliberately separates the high-volume candidate judge from the
low-volume curriculum controller.  The judge sees only the current goal and
raw one-letter sequences.  The controller receives categorical judgments and
must derive any intermediate construct from its own knowledge; no descriptor
cards or menu of proxy objectives is supplied.

The module is intentionally independent of the v1--v3 harness so historical
requests and cached provider responses remain reproducible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import hashlib
import json
import math
import random
import re
from typing import Any, Callable, Literal, Mapping, Protocol, Sequence

from pydantic import BaseModel, ConfigDict, Field

from adm.preferences.artifacts import PreferenceArtifactStore
from adm.preferences.clients import StructuredLLMClient, StructuredRequest
from domains.peptides.sequence_evidence import (
    candidate_evidence_payload,
    developability_evidence_tool_definition,
    evidence_tool_definition,
    generic_sequence_evidence,
)
from domains.peptides.agent import StructuredSemanticGoalAgent
from domains.peptides.types import PeptideCandidate, SemanticLabel






class SupportLevel(str, Enum):
    NONE = "none"
    WEAK_SEQUENCE_INFERENCE = "weak_sequence_inference"
    STRONG_SEQUENCE_INFERENCE = "strong_sequence_inference"
    RECOGNIZED_SEQUENCE_SPECIFIC_KNOWLEDGE = (
        "recognized_sequence_specific_knowledge"
    )






@dataclass(frozen=True)
class V4SampleJudgment:
    candidate_id: str
    model_label: SemanticLabel
    label: SemanticLabel
    goal_specific_support: str
    goal_application_support: SupportLevel
    safety_support: SupportLevel
    generic_proxy_only: bool
    known_nonspecific_toxicity_concern: bool
    negative_due_only_to_missing_evidence: bool
    contraindications: str
    critical_unknowns: str
    rationale: str
    override_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        for value in (
            self.goal_specific_support,
            self.contraindications,
            self.critical_unknowns,
            self.rationale,
        ):
            if not value.strip():
                raise ValueError("judgment rationale fields must be non-empty")
        if self.override_reason is not None and not self.override_reason.strip():
            raise ValueError("override_reason must be non-empty when present")


@dataclass(frozen=True)
class V4JudgeResult:
    goal: str
    exact_final_goal: bool
    judgments: tuple[V4SampleJudgment, ...]
    request_hashes: tuple[str, ...]
    usage: tuple[Any, ...]
    model_versions: tuple[str, ...]
    thought_summaries: tuple[str, ...]
    coverage_recovery_request_hashes: tuple[str, ...] = ()
    coverage_redispatch_request_hashes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.goal.strip() or not self.judgments:
            raise ValueError("a v4 judge result requires a goal and judgments")
        ids = [row.candidate_id for row in self.judgments]
        if len(ids) != len(set(ids)):
            raise ValueError("v4 judge result contains duplicate candidates")

    def counts(self) -> dict[str, int | float]:
        positive = sum(row.label == SemanticLabel.POSITIVE for row in self.judgments)
        negative = sum(row.label == SemanticLabel.NEGATIVE for row in self.judgments)
        abstain = len(self.judgments) - positive - negative
        return {
            "sample_count": len(self.judgments),
            "positive_count": positive,
            "negative_count": negative,
            "abstain_count": abstain,
            "positive_fraction_all": positive / len(self.judgments),
            "abstain_fraction": abstain / len(self.judgments),
        }


class _V4JudgmentRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(min_length=1)
    goal_specific_support: str = Field(min_length=1)
    goal_application_support: Literal[
        "none",
        "weak_sequence_inference",
        "strong_sequence_inference",
        "recognized_sequence_specific_knowledge",
    ]
    safety_support: Literal[
        "none",
        "weak_sequence_inference",
        "strong_sequence_inference",
        "recognized_sequence_specific_knowledge",
    ]
    generic_proxy_only: bool
    known_nonspecific_toxicity_concern: bool
    negative_due_only_to_missing_evidence: bool
    contraindications: str = Field(min_length=1)
    critical_unknowns: str = Field(min_length=1)
    model_label: Literal["positive", "negative", "abstain"]
    rationale: str = Field(min_length=1)


class _V4JudgmentResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    results: list[_V4JudgmentRow]


_FENCED_JSON = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    flags=re.IGNORECASE | re.DOTALL,
)
_LABEL_ENFORCEMENT_FIELDS = (
    "model_label",
    "goal_application_support",
    "safety_support",
    "generic_proxy_only",
    "known_nonspecific_toxicity_concern",
    "negative_due_only_to_missing_evidence",
)


def _recover_complete_v4_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    provider_ids: Sequence[str],
    thought_summary: str | None,
    response_schema: type[BaseModel] = _V4JudgmentResponse,
    label_enforcement_fields: Sequence[str] = _LABEL_ENFORCEMENT_FIELDS,
) -> tuple[list[dict[str, Any]], bool]:
    """Recover a truncated final array from an exact, consistent reasoning JSON.

    Gemini can return ``STOP`` while omitting a suffix of a structured array
    even when its visible reasoning summary contains the complete JSON object.
    Recovery is deliberately fail-closed: the reasoning object must validate
    against the same response schema, cover every expected item exactly once,
    and agree field-for-field with every row present in the final answer.
    """

    final_rows = [dict(row) for row in rows]
    expected = set(provider_ids)
    observed = [str(row.get("item_id", "")) for row in final_rows]
    if len(observed) == len(set(observed)) and set(observed) == expected:
        return final_rows, False
    if (
        thought_summary is None
        or len(observed) != len(set(observed))
        or not set(observed).issubset(expected)
    ):
        raise ValueError("v4 judgments must cover every item exactly once")

    final_by_id = {str(row["item_id"]): row for row in final_rows}
    valid: dict[str, list[dict[str, Any]]] = {}
    for match in _FENCED_JSON.finditer(thought_summary):
        try:
            parsed = response_schema.model_validate_json(match.group(1))
        except (ValueError, json.JSONDecodeError):
            continue
        candidate_rows = [
            row.model_dump(mode="json") for row in parsed.results
        ]
        candidate_ids = [str(row["item_id"]) for row in candidate_rows]
        if len(candidate_ids) != len(set(candidate_ids)) or set(candidate_ids) != expected:
            continue
        candidate_by_id = {
            str(row["item_id"]): row for row in candidate_rows
        }
        if any(
            any(
                candidate_by_id[item_id][field] != final_row[field]
                for field in label_enforcement_fields
            )
            for item_id, final_row in final_by_id.items()
        ):
            continue
        # Preserve the provider's visible final rows verbatim and use the
        # reasoning object only for the missing suffix.  Rationale wording may
        # differ between deliberation and final rendering, but every field that
        # determines the categorical/guarded label must agree on the overlap.
        merged = [
            final_by_id.get(provider_id, candidate_by_id[provider_id])
            for provider_id in provider_ids
        ]
        canonical = json.dumps(merged, sort_keys=True, separators=(",", ":"))
        valid[canonical] = merged
    if len(valid) != 1:
        raise ValueError("v4 judgments must cover every item exactly once")
    return next(iter(valid.values())), True


def _validated_partial_v4_rows(
    *,
    rows: Sequence[Mapping[str, Any]],
    provider_ids: Sequence[str],
    row_schema: type[BaseModel] = _V4JudgmentRow,
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    """Validate a provider-final proper subset before missing-ID redispatch.

    This helper intentionally accepts only a unique, schema-valid subset of the
    exact provider IDs.  Duplicate or unexpected IDs are ambiguous and must not
    trigger another paid request.
    """

    expected_order = tuple(str(item) for item in provider_ids)
    if len(expected_order) != len(set(expected_order)):
        raise ValueError("v4 provider IDs must be unique")
    validated = [
        row_schema.model_validate(dict(row)).model_dump(mode="json")
        for row in rows
    ]
    observed = [str(row["item_id"]) for row in validated]
    if len(observed) != len(set(observed)):
        raise ValueError("v4 partial judgment contains duplicate item IDs")
    expected = set(expected_order)
    if not set(observed).issubset(expected):
        raise ValueError("v4 partial judgment contains an unexpected item ID")
    missing = tuple(item for item in expected_order if item not in set(observed))
    if not missing:
        raise ValueError("v4 partial judgment recovery found no missing item")
    return validated, missing


def _validated_singleton_redispatch_row(
    *,
    rows: Sequence[Mapping[str, Any]],
    provider_id: str,
    row_schema: type[BaseModel] = _V4JudgmentRow,
) -> dict[str, Any]:
    """Require one exact schema-valid provider-final row for a redispatch.

    Unlike an original multi-item response, a paid singleton repair is never
    recoverable from visible reasoning.  Its provider-final structured output
    must itself contain the one requested ID exactly once.
    """

    validated = [
        row_schema.model_validate(dict(row)).model_dump(mode="json")
        for row in rows
    ]
    if len(validated) != 1 or str(validated[0]["item_id"]) != provider_id:
        raise ValueError(
            "v4 missing-item redispatch did not return its exact item"
        )
    return validated[0]



def _enforce_label(
    row: Mapping[str, Any], *, exact_final_goal: bool
) -> tuple[SemanticLabel, str | None]:
    model_label = SemanticLabel(str(row["model_label"]))
    if bool(row["negative_due_only_to_missing_evidence"]) and (
        model_label == SemanticLabel.NEGATIVE
    ):
        return SemanticLabel.ABSTAIN, "missing evidence cannot justify negative"
    if not exact_final_goal or model_label != SemanticLabel.POSITIVE:
        return model_label, None
    if bool(row["known_nonspecific_toxicity_concern"]):
        return SemanticLabel.ABSTAIN, (
            "known nonspecific-toxicity concern precludes final positive"
        )
    if bool(row["generic_proxy_only"]):
        return SemanticLabel.ABSTAIN, "generic proxy alone cannot justify final positive"
    if SupportLevel(str(row["goal_application_support"])) == SupportLevel.NONE:
        return SemanticLabel.ABSTAIN, "final positive lacks activity-specific support"
    if SupportLevel(
        str(row["safety_support"])
    ) == SupportLevel.NONE:
        return SemanticLabel.ABSTAIN, "final positive lacks mammalian-safety support"
    return model_label, None


@dataclass(frozen=True)
class _V4JudgmentPlan:
    chunk: tuple[PeptideCandidate, ...]
    provider_ids: tuple[str, ...]
    request: StructuredRequest


class V4CandidateJudge(StructuredSemanticGoalAgent):
    """Sequence-only categorical judge with deterministic final-goal guards."""

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
        chunk_size: int = 25,
        provider_order_seed: int = 0,
        system_prompt: str,
        use_generic_evidence_cards: bool = False,
        experiment_context: Mapping[str, Any] | None = None,
        show_final_goal_context_for_intermediate: bool = False,
        developability_evidence_tool: Mapping[str, Any] | None = None,
        require_model_version: bool = False,
        maximum_coverage_redispatches_per_chunk: int = 0,
        response_schema: type[BaseModel] = _V4JudgmentResponse,
        response_row_schema: type[BaseModel] = _V4JudgmentRow,
        response_row_adapter: Callable[[Mapping[str, Any]], Mapping[str, Any]] = dict,
        response_label_enforcement_fields: Sequence[str] = (
            _LABEL_ENFORCEMENT_FIELDS
        ),
        request_schema_id: str | None = None,
    ) -> None:
        super().__init__(
            client=client,
            artifact_store=artifact_store,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            max_provider_calls=max_provider_calls,
            initial_provider_calls=initial_provider_calls,
            judge_mode="dual_goal_v1",
            judge_chunk_size=chunk_size,
            use_generic_sequence_evidence=False,
            provider_order_seed=provider_order_seed,
        )
        self.system_prompt = system_prompt
        self.use_generic_evidence_cards = use_generic_evidence_cards
        self.experiment_context = dict(experiment_context or {})
        self.developability_evidence_tool = (
            developability_evidence_tool_definition()
            if developability_evidence_tool is None
            else dict(developability_evidence_tool)
        )
        if not isinstance(require_model_version, bool):
            raise TypeError("require_model_version must be boolean")
        self.require_model_version = require_model_version
        self.response_schema = response_schema
        self.response_row_schema = response_row_schema
        self.response_row_adapter = response_row_adapter
        self.response_label_enforcement_fields = tuple(
            str(value) for value in response_label_enforcement_fields
        )
        self.request_schema_id = (
            None if request_schema_id is None else request_schema_id.strip()
        )
        if request_schema_id is not None and not self.request_schema_id:
            raise ValueError("request_schema_id must be non-empty when supplied")
        if (
            isinstance(maximum_coverage_redispatches_per_chunk, bool)
            or not isinstance(maximum_coverage_redispatches_per_chunk, int)
            or maximum_coverage_redispatches_per_chunk < 0
            or maximum_coverage_redispatches_per_chunk > chunk_size
        ):
            raise ValueError(
                "maximum_coverage_redispatches_per_chunk must lie between zero "
                "and chunk_size"
            )
        self.maximum_coverage_redispatches_per_chunk = (
            maximum_coverage_redispatches_per_chunk
        )
        if not isinstance(show_final_goal_context_for_intermediate, bool):
            raise TypeError(
                "show_final_goal_context_for_intermediate must be boolean"
            )
        self.show_final_goal_context_for_intermediate = (
            show_final_goal_context_for_intermediate
        )

    def _judgment_plans(
        self,
        goal: str,
        samples: Sequence[PeptideCandidate],
        *,
        final_goal: str,
        arm: str,
        iteration: int,
        repeat_index: int,
        proxy_specification: Mapping[str, Any] | None,
    ) -> tuple[_V4JudgmentPlan, ...]:
        exact_final = goal.strip() == final_goal.strip()

        def order_key(sample: PeptideCandidate) -> tuple[str, str]:
            material = (
                f"{self.provider_order_seed}:{iteration}:{repeat_index}:"
                f"{sample.sequence}"
            )
            return hashlib.sha256(material.encode()).hexdigest(), sample.sequence

        ordered = tuple(sorted(samples, key=order_key))
        has_developability_evidence = any(
            sample.developability_evidence for sample in ordered
        )
        if has_developability_evidence and not all(
            sample.developability_evidence for sample in ordered
        ):
            raise ValueError(
                "candidate batch mixes present and absent developability evidence"
            )
        chunk_size = self.judge_chunk_size or len(ordered)
        plans: list[_V4JudgmentPlan] = []
        for chunk_index, start in enumerate(range(0, len(ordered), chunk_size)):
            chunk = ordered[start : start + chunk_size]
            provider_ids = tuple(
                f"item-{index:03d}" for index in range(1, len(chunk) + 1)
            )
            request = self._request(
                schema_id=(
                    self.request_schema_id
                    or (
                        "peptide-semantic-current-goal-context-tools-v5"
                        if self.use_generic_evidence_cards
                        else "peptide-semantic-current-goal-sequence-only-v4"
                    )
                ),
                system_prompt=self.system_prompt,
                user_payload={
                    "current_goal": goal,
                    "exact_final_goal": exact_final,
                    **(
                        {"immutable_final_goal_context": final_goal}
                        if self.show_final_goal_context_for_intermediate
                        else {}
                    ),
                    "representation_note": (
                        "Each string is a canonical amino-acid sequence written "
                        "N-to-C and represents one computationally generated "
                        "linear peptide."
                        if self.use_generic_evidence_cards
                        else "Unmodified canonical one-letter peptide sequences only."
                    ),
                    **(
                        {
                            "experiment_context": self.experiment_context,
                            "generic_sequence_evidence_tool": (
                                evidence_tool_definition()
                            ),
                            **(
                                {
                                    "developability_evidence_tool": (
                                        self.developability_evidence_tool
                                    )
                                }
                                if has_developability_evidence
                                else {}
                            ),
                            "operative_proxy_specification": (
                                dict(proxy_specification)
                                if proxy_specification is not None
                                else None
                            ),
                        }
                        if self.use_generic_evidence_cards
                        else {}
                    ),
                    "candidates": [
                        {
                            "item_id": provider_id,
                            "media_type": "chemical/x-amino-acid-sequence",
                            "sequence": sample.sequence,
                            **(
                                candidate_evidence_payload(
                                    sample.sequence,
                                    sample.developability_evidence,
                                )
                                if self.use_generic_evidence_cards
                                else {}
                            ),
                        }
                        for provider_id, sample in zip(provider_ids, chunk)
                    ],
                },
                logical_context={
                    "arm": arm,
                    "iteration": iteration,
                    "repeat_index": repeat_index,
                    "chunk_index": chunk_index,
                    "candidate_ids_hidden_from_provider": [
                        sample.candidate_id for sample in chunk
                    ],
                    **(
                        {"immutable_final_goal_visible_as_context": final_goal}
                        if self.show_final_goal_context_for_intermediate
                        else {"final_goal_hidden_unless_current_is_exact": final_goal}
                    ),
                },
            )
            plans.append(_V4JudgmentPlan(chunk, provider_ids, request))
        return tuple(plans)

    def _coverage_redispatch_plans(
        self,
        *,
        original: _V4JudgmentPlan,
        missing_provider_ids: Sequence[str],
    ) -> tuple[_V4JudgmentPlan, ...]:
        """Create content-addressed singleton requests for missing final rows."""

        missing = tuple(str(item) for item in missing_provider_ids)
        if not missing or len(missing) > self.maximum_coverage_redispatches_per_chunk:
            raise ValueError("v4 missing-item redispatch exceeds its bounded ceiling")
        sample_by_id = dict(zip(original.provider_ids, original.chunk))
        payload = original.request.user_payload
        candidates = payload.get("candidates")
        if not isinstance(candidates, list):
            raise TypeError("v4 original request omitted candidate payloads")
        payload_by_id: dict[str, Mapping[str, Any]] = {}
        for row in candidates:
            if not isinstance(row, Mapping):
                raise TypeError("v4 original candidate payload is malformed")
            item_id = str(row.get("item_id", ""))
            if not item_id or item_id in payload_by_id:
                raise ValueError("v4 original candidate payload IDs are invalid")
            payload_by_id[item_id] = row
        if set(payload_by_id) != set(original.provider_ids):
            raise ValueError("v4 original request/provider-ID binding differs")

        plans: list[_V4JudgmentPlan] = []
        for provider_id in missing:
            if provider_id not in sample_by_id:
                raise ValueError("v4 missing item is absent from the original plan")
            recovery_payload = dict(payload)
            recovery_payload["candidates"] = [dict(payload_by_id[provider_id])]
            logical_context = dict(original.request.logical_context)
            logical_context.update(
                {
                    "coverage_redispatch": "missing-singleton-v1",
                    "coverage_redispatch_of_request_hash": (
                        original.request.request_hash
                    ),
                    "coverage_redispatch_provider_item_id": provider_id,
                    "candidate_ids_hidden_from_provider": [
                        sample_by_id[provider_id].candidate_id
                    ],
                }
            )
            request = StructuredRequest.from_payload(
                model=original.request.model,
                schema_id=original.request.schema_id,
                system_prompt=original.request.system_prompt,
                user_payload=recovery_payload,
                logical_context=logical_context,
                max_output_tokens=original.request.max_output_tokens,
                temperature=original.request.temperature,
                reasoning_effort=original.request.reasoning_effort,
            )
            plans.append(
                _V4JudgmentPlan(
                    chunk=(sample_by_id[provider_id],),
                    provider_ids=(provider_id,),
                    request=request,
                )
            )
        return tuple(plans)

    def judge_current_goal(
        self,
        goal: str,
        samples: Sequence[PeptideCandidate],
        *,
        final_goal: str,
        arm: str,
        iteration: int,
        repeat_index: int = 0,
        proxy_specification: Mapping[str, Any] | None = None,
    ) -> V4JudgeResult:
        if not samples:
            raise ValueError("candidate judgment requires at least one sample")
        exact_final = goal.strip() == final_goal.strip()
        plans = self._judgment_plans(
            goal,
            samples,
            final_goal=final_goal,
            arm=arm,
            iteration=iteration,
            repeat_index=repeat_index,
            proxy_specification=proxy_specification,
        )
        self.ensure_provider_capacity(
            tuple((plan.request, self.response_schema) for plan in plans)
        )
        by_candidate: dict[str, V4SampleJudgment] = {}
        request_hashes: list[str] = []
        usages: list[Any] = []
        versions: list[str] = []
        thoughts: list[str] = []
        coverage_recoveries: list[str] = []
        coverage_redispatches: list[str] = []

        def record_metadata(metadata: Mapping[str, Any]) -> None:
            request_hashes.append(str(metadata["request_hash"]))
            usages.append(metadata["usage"])
            if metadata["model_version"] is None:
                if self.require_model_version:
                    raise ValueError("Gemini judge chunk omitted modelVersion")
            else:
                versions.append(str(metadata["model_version"]))
            if metadata["thought_summary"] is not None:
                thoughts.append(str(metadata["thought_summary"]))

        for plan in plans:
            chunk = plan.chunk
            provider_ids = plan.provider_ids
            request = plan.request
            values, metadata = self._execute(request, self.response_schema)
            rows = values.get("results")
            if not isinstance(rows, list):
                raise TypeError("v4 judgment results must be a list")
            if not all(isinstance(row, Mapping) for row in rows):
                raise TypeError("v4 judgment rows must be objects")
            record_metadata(metadata)
            try:
                rows, recovered = _recover_complete_v4_rows(
                    rows=rows,
                    provider_ids=provider_ids,
                    thought_summary=(
                        None
                        if metadata["thought_summary"] is None
                        else str(metadata["thought_summary"])
                    ),
                    response_schema=self.response_schema,
                    label_enforcement_fields=(
                        self.response_label_enforcement_fields
                    ),
                )
            except ValueError:
                if self.maximum_coverage_redispatches_per_chunk == 0:
                    raise
                partial_rows, missing = _validated_partial_v4_rows(
                    rows=rows,
                    provider_ids=provider_ids,
                    row_schema=self.response_row_schema,
                )
                recovery_plans = self._coverage_redispatch_plans(
                    original=plan,
                    missing_provider_ids=missing,
                )
                self.ensure_provider_capacity(
                    tuple(
                        (recovery.request, self.response_schema)
                        for recovery in recovery_plans
                    )
                )
                merged = {
                    str(row["item_id"]): dict(row) for row in partial_rows
                }
                for recovery in recovery_plans:
                    recovery_values, recovery_metadata = self._execute(
                        recovery.request, self.response_schema
                    )
                    recovery_rows = recovery_values.get("results")
                    if not isinstance(recovery_rows, list) or not all(
                        isinstance(row, Mapping) for row in recovery_rows
                    ):
                        raise TypeError(
                            "v4 missing-item redispatch results must be object rows"
                        )
                    record_metadata(recovery_metadata)
                    recovery_id = recovery.provider_ids[0]
                    recovery_row = _validated_singleton_redispatch_row(
                        rows=recovery_rows,
                        provider_id=recovery_id,
                        row_schema=self.response_row_schema,
                    )
                    if recovery_id in merged:
                        raise ValueError(
                            "v4 missing-item redispatch did not return its exact item"
                        )
                    merged[recovery_id] = recovery_row
                    coverage_redispatches.append(
                        str(recovery_metadata["request_hash"])
                    )
                rows = [merged[provider_id] for provider_id in provider_ids]
                recovered = False
            rows = [
                _V4JudgmentRow.model_validate(
                    dict(self.response_row_adapter(row))
                ).model_dump(mode="json")
                for row in rows
            ]
            if recovered:
                coverage_recoveries.append(str(metadata["request_hash"]))
            by_provider = {str(row["item_id"]): row for row in rows}
            for provider_id, sample in zip(provider_ids, chunk):
                row = by_provider[provider_id]
                enforced, override = _enforce_label(
                    row, exact_final_goal=exact_final
                )
                judgment = V4SampleJudgment(
                    candidate_id=sample.candidate_id,
                    model_label=SemanticLabel(str(row["model_label"])),
                    label=enforced,
                    goal_specific_support=str(row["goal_specific_support"]),
                    goal_application_support=SupportLevel(
                        str(row["goal_application_support"])
                    ),
                    safety_support=SupportLevel(
                        str(row["safety_support"])
                    ),
                    generic_proxy_only=bool(row["generic_proxy_only"]),
                    known_nonspecific_toxicity_concern=bool(
                        row["known_nonspecific_toxicity_concern"]
                    ),
                    negative_due_only_to_missing_evidence=bool(
                        row["negative_due_only_to_missing_evidence"]
                    ),
                    contraindications=str(row["contraindications"]),
                    critical_unknowns=str(row["critical_unknowns"]),
                    rationale=str(row["rationale"]),
                    override_reason=override,
                )
                if judgment.candidate_id in by_candidate:
                    raise ValueError("v4 chunked judge returned a duplicate candidate")
                by_candidate[judgment.candidate_id] = judgment
        expected_ids = [sample.candidate_id for sample in samples]
        if set(by_candidate) != set(expected_ids):
            raise ValueError("v4 judge did not cover the complete sample")
        return V4JudgeResult(
            goal=goal,
            exact_final_goal=exact_final,
            judgments=tuple(by_candidate[item_id] for item_id in expected_ids),
            request_hashes=tuple(request_hashes),
            usage=tuple(usages),
            model_versions=tuple(dict.fromkeys(versions)),
            thought_summaries=tuple(thoughts),
            coverage_recovery_request_hashes=tuple(coverage_recoveries),
            coverage_redispatch_request_hashes=tuple(coverage_redispatches),
        )
