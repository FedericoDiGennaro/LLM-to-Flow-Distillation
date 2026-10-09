"""Native Vertex Express Gemini transport for structured preference requests.

The transport deliberately performs no automatic retries.  Before sending a
request it writes an immutable attempt marker; after the HTTP response arrives
it durably stores the unparsed provider response.  A restarted process may
reparse a stored response, but it will never resend an ambiguous attempt.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping

from pydantic import ValidationError
import requests

from adm.preferences.allocation_window import require_allocation_window
from adm.preferences.clients import StructuredRequest, StructuredResponse


DEFAULT_BASE_URL = "https://aiplatform.googleapis.com/v1/publishers/google/models"
THINKING_BUDGETS = {"none": 0, "low": 1024, "medium": 4096}
GEMINI_3_THINKING_LEVELS = {
    "low": "LOW",
    "medium": "MEDIUM",
    "high": "HIGH",
}


@dataclass(frozen=True)
class GeminiUsageBudget:
    """Durable pre-dispatch ceilings for one experiment-wide Gemini run."""

    maximum_prompt_tokens: int
    maximum_output_tokens: int
    maximum_estimated_usd: float
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float
    prompt_token_overhead_per_request: int = 1024

    def __post_init__(self) -> None:
        integer_fields = (
            "maximum_prompt_tokens",
            "maximum_output_tokens",
            "prompt_token_overhead_per_request",
        )
        for field in integer_fields:
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        numeric_fields = (
            "maximum_estimated_usd",
            "input_usd_per_million_tokens",
            "output_usd_per_million_tokens",
        )
        for field in numeric_fields:
            value = getattr(self, field)
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{field} must be finite")
            if value <= 0:
                raise ValueError(f"{field} must be positive")

    def estimated_usd(self, *, prompt_tokens: int, output_tokens: int) -> float:
        return (
            prompt_tokens * self.input_usd_per_million_tokens
            + output_tokens * self.output_usd_per_million_tokens
        ) / 1_000_000


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _write_immutable_json(path: Path, value: Any) -> None:
    serialized = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(f"immutable Gemini artifact differs: {path}")
        return
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        try:
            os.link(temporary_path, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != serialized:
                raise FileExistsError(
                    f"immutable Gemini artifact differs: {path}"
                )
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validate_api_key(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise EnvironmentError("GEMINI_API_KEY must be a nonempty string")
    if not value.isascii() or value != value.strip():
        raise EnvironmentError("Gemini credential must be trimmed ASCII")
    if not value.isprintable() or any(character in value for character in "'\"`"):
        raise EnvironmentError("Gemini credential contains a control or quote character")
    return value


def probe_model_access(
    api_key: str,
    *,
    model: str,
    base_url: str = DEFAULT_BASE_URL,
    timeout: float = 60.0,
    session: Any | None = None,
) -> dict[str, Any]:
    """Verify Express credentials/model access without generating content."""
    key = validate_api_key(api_key)
    if not model.strip():
        raise ValueError("probe model must be non-empty")
    if timeout <= 0 or not math.isfinite(timeout):
        raise ValueError("probe timeout must be positive")
    endpoint = f"{base_url.rstrip('/')}/{model}:countTokens"
    response = (session or requests.Session()).post(
        endpoint,
        headers={
            "Content-Type": "application/json; charset=utf-8",
            "X-Goog-Api-Key": key,
        },
        json={
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "credential and model access preflight"}],
                }
            ]
        },
        timeout=timeout,
    )
    response.raise_for_status()
    value = response.json()
    total_tokens = value.get("totalTokens") if isinstance(value, Mapping) else None
    if isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens <= 0:
        raise ValueError("Gemini countTokens probe returned invalid metadata")
    return {
        "schema_version": 1,
        "method": "publishers.models.countTokens",
        "endpoint": endpoint,
        "model": model,
        "http_status": int(getattr(response, "status_code", 200)),
        "total_tokens": total_tokens,
        "generation_requested": False,
        "credential_value_recorded": False,
    }


def _schema_for_vertex(value: Any) -> Any:
    """Remove annotation-only JSON-schema keys unsupported by Vertex."""
    if isinstance(value, list):
        return [_schema_for_vertex(item) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"title", "default", "examples"}:
            continue
        if key == "const":
            result["enum"] = [item]
            continue
        result[key] = _schema_for_vertex(item)
    return result


def response_schema(response_model: type[Any]) -> dict[str, Any]:
    if not hasattr(response_model, "model_json_schema"):
        raise TypeError("Gemini response model must be a Pydantic v2 model")
    schema = _schema_for_vertex(response_model.model_json_schema())
    if not isinstance(schema, dict) or schema.get("type") != "object":
        raise ValueError("Gemini response schema must describe one object")
    return schema


def _gemini_3_request(request: StructuredRequest) -> bool:
    return request.model.startswith("gemini-3")


def _thinking_configuration(
    request: StructuredRequest,
) -> tuple[dict[str, Any], bool]:
    """Return the model-family-specific thinking payload and its semantics."""
    effort = request.reasoning_effort or "none"
    if _gemini_3_request(request):
        if effort == "none":
            raise ValueError(
                "Gemini 3 has no zero-thinking mode; label the condition with "
                "an explicit supported reasoning effort"
            )
        level = GEMINI_3_THINKING_LEVELS.get(effort)
        if level is None:
            raise ValueError(f"unsupported Gemini 3 reasoning effort: {effort}")
        return {"thinkingLevel": level, "includeThoughts": True}, True
    if effort not in THINKING_BUDGETS:
        raise ValueError(f"unsupported Gemini reasoning effort: {effort}")
    thinking_budget = THINKING_BUDGETS[effort]
    return {
        "thinkingBudget": thinking_budget,
        "includeThoughts": thinking_budget > 0,
    }, thinking_budget > 0


def request_body(
    request: StructuredRequest,
    response_model: type[Any],
    *,
    temperature: float,
) -> dict[str, Any]:
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("Gemini temperature must be finite and nonnegative")
    thinking_config, _ = _thinking_configuration(request)
    generation_config: dict[str, Any] = {
        "candidateCount": 1,
        "maxOutputTokens": request.max_output_tokens,
        "responseMimeType": "application/json",
        "responseJsonSchema": response_schema(response_model),
        "thinkingConfig": thinking_config,
    }
    if not _gemini_3_request(request):
        generation_config["temperature"] = temperature
        generation_config["topP"] = 1.0
    return {
        "systemInstruction": {"parts": [{"text": request.system_prompt}]},
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": json.dumps(
                            request.user_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                            sort_keys=True,
                        )
                    }
                ],
            }
        ],
        "generationConfig": generation_config,
    }


def _thought_and_final(parts: Any) -> tuple[str | None, str]:
    if not isinstance(parts, list) or not parts:
        raise ValueError("Gemini response has no candidate parts")
    thoughts: list[str] = []
    final: list[str] = []
    for part in parts:
        if not isinstance(part, Mapping):
            raise ValueError("Gemini response part is not an object")
        text = part.get("text")
        if text is None or not str(text).strip():
            continue
        if part.get("thought") is True:
            thoughts.append(str(text).strip())
        else:
            final.append(str(text).strip())
    if len(final) != 1:
        raise ValueError("Gemini response must have exactly one visible final part")
    return "\n\n".join(thoughts) or None, final[0]


def normalize_usage(value: Any, *, reasoning: bool) -> dict[str, int]:
    if not isinstance(value, Mapping):
        raise ValueError("Gemini response has no usage metadata")
    provider_names = {
        "prompt_tokens": "promptTokenCount",
        "completion_tokens": "candidatesTokenCount",
        "thought_tokens": "thoughtsTokenCount",
        "total_tokens": "totalTokenCount",
    }
    normalized: dict[str, int] = {}
    for canonical, provider_name in provider_names.items():
        default = 0 if canonical == "thought_tokens" else None
        raw = value.get(provider_name, default)
        if isinstance(raw, bool) or not isinstance(raw, int) or raw < 0:
            raise ValueError(f"invalid Gemini usage field: {provider_name}")
        normalized[canonical] = raw
    normalized["output_tokens"] = (
        normalized["completion_tokens"] + normalized["thought_tokens"]
    )
    expected_total = normalized["prompt_tokens"] + normalized["output_tokens"]
    if normalized["total_tokens"] != expected_total:
        raise ValueError("Gemini total tokens differ from prompt + completion + thoughts")
    if not reasoning and normalized["thought_tokens"] != 0:
        raise ValueError("no-reasoning response reported thought tokens")
    return normalized


def parse_provider_response(
    raw: Mapping[str, Any],
    response_model: type[Any],
    *,
    reasoning: bool,
) -> dict[str, Any]:
    """Parse one raw provider response through the production validation path."""
    candidates = raw.get("candidates")
    if (
        not isinstance(candidates, list)
        or len(candidates) != 1
        or candidates[0].get("finishReason") != "STOP"
    ):
        reason = candidates[0].get("finishReason") if candidates else None
        raise ValueError(f"Gemini candidate did not finish with STOP: {reason}")
    content = candidates[0].get("content")
    if not isinstance(content, Mapping):
        raise ValueError("Gemini candidate content is missing")
    thought_summary, final_text = _thought_and_final(content.get("parts"))
    usage = normalize_usage(raw.get("usageMetadata"), reasoning=reasoning)
    if not reasoning and thought_summary is not None:
        raise ValueError("no-reasoning response returned thought text")
    try:
        parsed = response_model.model_validate_json(final_text)
    except (ValidationError, json.JSONDecodeError) as error:
        raise ValueError("Gemini structured output failed validation") from error
    version = str(raw.get("modelVersion", "")).strip()
    if not version:
        raise ValueError("Gemini response omitted modelVersion")
    return {
        "parsed": parsed,
        "usage": usage,
        "model_version": version,
        "thought_summary": thought_summary,
        "thought_summary_available": thought_summary is not None,
        "final_text": final_text,
        "response_id": str(raw.get("responseId")) if raw.get("responseId") else None,
    }


class GeminiVertexClient:
    """One-attempt structured client for Gemini on Vertex Express."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300.0,
        temperature: float = 0.0,
        session: Any | None = None,
        api_key_provider: Callable[[], str] | None = None,
        attempt_root: Path | None = None,
        requests_per_minute: float = 8.0,
        usage_budget: GeminiUsageBudget | None = None,
        count_tokens_on_budget_rejection: bool = False,
    ) -> None:
        if timeout <= 0 or not math.isfinite(timeout):
            raise ValueError("Gemini timeout must be positive")
        if requests_per_minute <= 0 or not math.isfinite(requests_per_minute):
            raise ValueError("requests_per_minute must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.temperature = temperature
        self.session = session or requests.Session()
        self.api_key_provider = api_key_provider or self._environment_key
        self.attempt_root = attempt_root
        self.requests_per_minute = requests_per_minute
        if usage_budget is not None and attempt_root is None:
            raise ValueError("a usage budget requires a durable attempt_root")
        self.usage_budget = usage_budget
        self.count_tokens_on_budget_rejection = count_tokens_on_budget_rejection
        self._last_dispatch_monotonic: float | None = None
        self._observed_model_version: str | None = None

    def observe_model_version(self, version: object) -> str:
        """Bind one resolved provider revision across fresh and cached calls."""

        normalized = str(version or "").strip()
        if not normalized:
            raise ValueError("Gemini response omitted modelVersion")
        if self._observed_model_version is None:
            self._observed_model_version = normalized
        elif self._observed_model_version != normalized:
            raise ValueError("Gemini modelVersion changed within one client run")
        return normalized

    def observe_model_version_metadata(self, raw: Mapping[str, Any]) -> str:
        """Validate the normalized wrapper against the raw provider field."""

        wrapper = self.observe_model_version(raw.get("model_version"))
        provider = raw.get("provider_response")
        if not isinstance(provider, Mapping):
            raise ValueError("Gemini cached response omitted raw provider response")
        provider_version = str(provider.get("modelVersion", "")).strip()
        if provider_version != wrapper:
            raise ValueError("Gemini wrapper/provider modelVersion mismatch")
        return wrapper

    @staticmethod
    def _usage_counts(raw: Mapping[str, Any]) -> tuple[int, int]:
        usage = raw.get("usageMetadata")
        if not isinstance(usage, Mapping):
            raise ValueError("completed Gemini response has no usage metadata")
        prompt = usage.get("promptTokenCount")
        total = usage.get("totalTokenCount")
        for name, value in (("promptTokenCount", prompt), ("totalTokenCount", total)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"invalid completed Gemini usage field: {name}")
        if total < prompt:
            raise ValueError("Gemini totalTokenCount is smaller than promptTokenCount")
        # Some Vertex responses omit candidatesTokenCount when thinking consumes
        # the full generation budget.  total-prompt remains the authoritative
        # billed output count and includes both visible and thinking tokens.
        return prompt, total - prompt

    def _completed_usage(self) -> tuple[int, int]:
        if self.attempt_root is None:
            return 0, 0
        prompt_total = 0
        output_total = 0
        for path in sorted(self.attempt_root.glob("*/provider_response.json")):
            envelope = _read_json(path)
            status = envelope.get("http_status")
            if isinstance(status, bool) or not isinstance(status, int):
                raise ValueError(f"Gemini attempt has invalid status: {path}")
            # Google documents that non-2xx responses are not token billed.
            if not 200 <= status < 300:
                continue
            raw = envelope.get("provider_response")
            if not isinstance(raw, Mapping):
                raise ValueError(f"completed Gemini attempt has no response: {path}")
            prompt, output = self._usage_counts(raw)
            prompt_total += prompt
            output_total += output
        # An interrupted HTTP request may have been billed despite having no
        # local response. Carry its full reservation across process restarts.
        if self.usage_budget is not None:
            for path in self.attempt_root.glob("*/attempt_started.json"):
                if (path.parent / "provider_response.json").exists():
                    continue
                started = _read_json(path)
                projection = started.get("budget_projection") or {}
                body = started["provider_request_body"]
                prompt_total += int(projection.get(
                    "next_prompt_token_upper_bound",
                    len(_canonical_json(body).encode("utf-8"))
                    + self.usage_budget.prompt_token_overhead_per_request,
                ))
                output_total += int(body["generationConfig"]["maxOutputTokens"])
        return prompt_total, output_total

    def _budget_projection(
        self,
        *,
        body: Mapping[str, Any],
        request: StructuredRequest,
    ) -> dict[str, Any] | None:
        if self.usage_budget is None:
            return None
        completed_prompt, completed_output = self._completed_usage()
        # Every model-visible token consumes at least one UTF-8 byte. The full
        # serialized request contains all prompt/schema bytes; the additional
        # fixed allowance conservatively covers provider framing/special tokens.
        prompt_reservation = len(_canonical_json(body).encode("utf-8")) + (
            self.usage_budget.prompt_token_overhead_per_request
        )
        projected_prompt = completed_prompt + prompt_reservation
        projected_output = completed_output + request.max_output_tokens
        projected_usd = self.usage_budget.estimated_usd(
            prompt_tokens=projected_prompt,
            output_tokens=projected_output,
        )
        reservation_method = "serialized_bytes_upper_bound"
        if self.count_tokens_on_budget_rejection and (
            projected_prompt > self.usage_budget.maximum_prompt_tokens
            or projected_usd > self.usage_budget.maximum_estimated_usd
        ):
            # Count the identical prompt without generating a response. Keep
            # a conservative byte allowance for all non-content fields (schema
            # included), since their token accounting can differ by endpoint.
            count_body = {
                name: body[name]
                for name in ("contents", "systemInstruction")
                if name in body
            }
            other_fields = {
                name: value for name, value in body.items() if name not in count_body
            }
            endpoint = f"{self.base_url}/{request.model}:countTokens"
            count_hash = hashlib.sha256(_canonical_json(body).encode()).hexdigest()
            count_path = self.attempt_root / "token_counts" / f"{count_hash}.json"
            if count_path.is_file():
                receipt = _read_json(count_path)
                if receipt.get("endpoint") != endpoint:
                    raise ValueError("token-count receipt endpoint differs")
                count = receipt["total_tokens"]
            else:
                response = self.session.post(
                    endpoint,
                    headers={
                        "Content-Type": "application/json; charset=utf-8",
                        "X-Goog-Api-Key": validate_api_key(self.api_key_provider()),
                    },
                    json=count_body,
                    timeout=min(self.timeout, 60.0),
                )
                response.raise_for_status()
                count = response.json().get("totalTokens")
                if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                    raise ValueError("invalid Gemini countTokens result")
                _write_immutable_json(count_path, {
                    "endpoint": endpoint,
                    "total_tokens": count,
                    "provider_request_body_sha256": count_hash,
                    "generation_requested": False,
                })
            prompt_reservation = (
                count + len(_canonical_json(other_fields).encode("utf-8"))
                + self.usage_budget.prompt_token_overhead_per_request
            )
            projected_prompt = completed_prompt + prompt_reservation
            projected_usd = self.usage_budget.estimated_usd(
                prompt_tokens=projected_prompt, output_tokens=projected_output
            )
            reservation_method = "countTokens_plus_schema_bytes_and_overhead"
        if projected_prompt > self.usage_budget.maximum_prompt_tokens:
            raise PermissionError("Gemini prompt-token budget projection exceeded")
        if projected_output > self.usage_budget.maximum_output_tokens:
            raise PermissionError("Gemini output-token budget projection exceeded")
        if projected_usd > self.usage_budget.maximum_estimated_usd:
            raise PermissionError("Gemini USD budget projection exceeded")
        return {
            "prompt_reservation_method": reservation_method,
            "completed_prompt_tokens": completed_prompt,
            "completed_output_tokens": completed_output,
            "next_prompt_token_upper_bound": prompt_reservation,
            "next_maximum_output_tokens": request.max_output_tokens,
            "projected_prompt_tokens": projected_prompt,
            "projected_output_tokens": projected_output,
            "projected_estimated_usd": projected_usd,
        }

    def _validate_completed_budget(self) -> None:
        if self.usage_budget is None:
            return
        prompt_tokens, output_tokens = self._completed_usage()
        estimated_usd = self.usage_budget.estimated_usd(
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )
        if prompt_tokens > self.usage_budget.maximum_prompt_tokens:
            raise PermissionError("completed Gemini prompt-token budget exceeded")
        if output_tokens > self.usage_budget.maximum_output_tokens:
            raise PermissionError("completed Gemini output-token budget exceeded")
        if estimated_usd > self.usage_budget.maximum_estimated_usd:
            raise PermissionError("completed Gemini USD budget exceeded")

    @staticmethod
    def _environment_key() -> str:
        present = [
            name
            for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY")
            if os.environ.get(name)
        ]
        if present != ["GEMINI_API_KEY"]:
            raise EnvironmentError(
                "set exactly GEMINI_API_KEY and unset GOOGLE_API_KEY"
            )
        return validate_api_key(os.environ["GEMINI_API_KEY"])

    def _dispatch_or_replay(
        self,
        *,
        request: StructuredRequest,
        endpoint: str,
        body: Mapping[str, Any],
        key: str,
    ) -> tuple[dict[str, Any], bool]:
        if self.attempt_root is None:
            require_allocation_window(
                required_seconds=self.timeout + 600.0,
                operation="generateContent",
                request_hash=request.request_hash,
            )
            response = self.session.post(
                endpoint,
                headers={
                    "Content-Type": "application/json; charset=utf-8",
                    "X-Goog-Api-Key": key,
                },
                json=body,
                timeout=self.timeout,
            )
            response.raise_for_status()
            raw = response.json()
            if not isinstance(raw, dict):
                raise ValueError("Gemini response is not a JSON object")
            return raw, False

        request_directory = self.attempt_root / request.request_hash
        start_path = request_directory / "attempt_started.json"
        response_path = request_directory / "provider_response.json"
        body_hash = hashlib.sha256(_canonical_json(body).encode()).hexdigest()

        if response_path.exists():
            envelope = _read_json(response_path)
            if envelope.get("schema_version") != 1:
                raise ValueError(f"unsupported Gemini attempt schema: {response_path}")
            if envelope.get("request_hash") != request.request_hash:
                raise ValueError(f"Gemini attempt request hash mismatch: {response_path}")
            if envelope.get("provider_request_body_sha256") != body_hash:
                raise ValueError(f"Gemini attempt body hash mismatch: {response_path}")
            status = envelope.get("http_status")
            if isinstance(status, bool) or not isinstance(status, int):
                raise ValueError(f"Gemini attempt has invalid status: {response_path}")
            if not 200 <= status < 300:
                raise RuntimeError(
                    f"preserved Gemini attempt returned HTTP {status}; refusing resend"
                )
            raw = envelope.get("provider_response")
            if not isinstance(raw, dict):
                raise ValueError(f"preserved Gemini response is invalid: {response_path}")
            self._validate_completed_budget()
            return copy.deepcopy(raw), True

        if start_path.exists():
            raise RuntimeError(
                "Gemini attempt has a durable start marker but no raw response; "
                "provider completion is ambiguous and this request will not be resent"
            )

        budget_projection = self._budget_projection(body=body, request=request)

        minimum_interval = 60.0 / self.requests_per_minute
        throttle_wait = 0.0
        if self._last_dispatch_monotonic is not None:
            throttle_wait = minimum_interval - (
                time.monotonic() - self._last_dispatch_monotonic
            )
        throttle_wait = max(0.0, throttle_wait)
        require_allocation_window(
            required_seconds=self.timeout + throttle_wait + 600.0,
            operation="generateContent",
            request_hash=request.request_hash,
        )
        if throttle_wait > 0:
            time.sleep(throttle_wait)
            require_allocation_window(
                required_seconds=self.timeout + 600.0,
                operation="generateContent",
                request_hash=request.request_hash,
            )
        self._last_dispatch_monotonic = time.monotonic()
        _write_immutable_json(
            start_path,
            {
                "schema_version": 1,
                "request_hash": request.request_hash,
                "endpoint": endpoint,
                "provider_request_body_sha256": body_hash,
                "provider_request_body": body,
                "budget_projection": budget_projection,
                "automatic_retries": 0,
                "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        response = self.session.post(
            endpoint,
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "X-Goog-Api-Key": key,
            },
            json=body,
            timeout=self.timeout,
        )
        status = int(getattr(response, "status_code", 200))
        try:
            provider_response = response.json()
        except (ValueError, json.JSONDecodeError):
            provider_response = None
        _write_immutable_json(
            response_path,
            {
                "schema_version": 1,
                "request_hash": request.request_hash,
                "provider_request_body_sha256": body_hash,
                "http_status": status,
                "provider_response": provider_response,
                "response_text": (
                    None
                    if provider_response is not None
                    else str(getattr(response, "text", ""))
                ),
                "received_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
        )
        response.raise_for_status()
        if not isinstance(provider_response, dict):
            raise ValueError("Gemini response is not a JSON object")
        self._validate_completed_budget()
        return provider_response, False

    def requires_provider_dispatch(
        self,
        request: StructuredRequest,
        response_model: type[Any],
    ) -> bool:
        """Return false when this request is already durable and cannot be resent."""
        del response_model
        if self.attempt_root is None:
            return True
        request_directory = self.attempt_root / request.request_hash
        return not (
            (request_directory / "provider_response.json").exists()
            or (request_directory / "attempt_started.json").exists()
        )

    def parse(
        self,
        request: StructuredRequest,
        response_model: type[Any],
    ) -> StructuredResponse[Any]:
        key = validate_api_key(self.api_key_provider())
        body = request_body(request, response_model, temperature=self.temperature)
        endpoint = f"{self.base_url}/{request.model}:generateContent"
        raw, replayed = self._dispatch_or_replay(
            request=request,
            endpoint=endpoint,
            body=body,
            key=key,
        )
        _, reasoning = _thinking_configuration(request)
        response_sha256 = hashlib.sha256(_canonical_json(raw).encode()).hexdigest()
        request_directory = (
            None
            if self.attempt_root is None
            else self.attempt_root / request.request_hash
        )
        try:
            normalized = parse_provider_response(
                raw,
                response_model,
                reasoning=reasoning,
            )
        except Exception as error:
            if request_directory is not None:
                if (request_directory / "parsed_response.json").exists():
                    raise RuntimeError(
                        "Gemini response was previously parse-validated but no longer parses"
                    ) from error
                _write_immutable_json(
                    request_directory / "parse_failure.json",
                    {
                        "schema_version": 1,
                        "request_hash": request.request_hash,
                        "provider_response_sha256": response_sha256,
                        "exception_type": type(error).__name__,
                        "exception_message": str(error)[:1000],
                    },
                )
            raise
        version = self.observe_model_version(normalized["model_version"])
        if request_directory is not None:
            if (request_directory / "parse_failure.json").exists():
                raise RuntimeError(
                    "Gemini response has a durable parse-failure marker"
                )
            _write_immutable_json(
                request_directory / "parsed_response.json",
                {
                    "schema_version": 1,
                    "request_hash": request.request_hash,
                    "provider_response_sha256": response_sha256,
                    "model_version": version,
                    "response_id": normalized["response_id"],
                    "usage": normalized["usage"],
                },
            )
        return StructuredResponse(
            request_hash=request.request_hash,
            parsed=normalized["parsed"],
            response_id=normalized["response_id"],
            usage=normalized["usage"],
            raw_response={
                "provider_request_body": body,
                "provider_response": copy.deepcopy(raw),
                "model_version": version,
                "thought_summary": normalized["thought_summary"],
                "thought_summary_available": normalized[
                    "thought_summary_available"
                ],
                "final_text": normalized["final_text"],
                "automatic_retries": 0,
                "provider_response_replayed": replayed,
            },
        )
