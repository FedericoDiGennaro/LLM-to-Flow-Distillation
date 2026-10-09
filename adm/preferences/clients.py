"""Provider-independent contracts for structured LLM requests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Generic, Mapping, Protocol, TypeVar

ParsedResponse = TypeVar("ParsedResponse")


class StructuredResponseParseError(RuntimeError):
    """Raised when a provider response cannot be decoded as the requested schema."""


def canonical_json(value: Any) -> str:
    """Serialize JSON deterministically and reject non-portable numeric values."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def value_as_json(value: Any) -> Any:
    """Convert SDK/Pydantic values to JSON-compatible Python values."""
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


@dataclass(frozen=True)
class StructuredRequest:
    """Immutable logical request used for hashing, execution, and replay."""

    model: str
    schema_id: str
    system_prompt: str
    user_payload_json: str
    max_output_tokens: int
    logical_context_json: str = "{}"
    temperature: float | None = None
    reasoning_effort: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("model", "schema_id", "system_prompt"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must be non-empty")
        try:
            payload = json.loads(self.user_payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("user_payload_json must contain valid JSON") from error
        if not isinstance(payload, (dict, list)):
            raise ValueError("user payload must be a JSON object or array")
        canonical_payload = canonical_json(payload)
        object.__setattr__(self, "user_payload_json", canonical_payload)
        try:
            logical_context = json.loads(self.logical_context_json)
        except json.JSONDecodeError as error:
            raise ValueError("logical_context_json must contain valid JSON") from error
        if not isinstance(logical_context, dict):
            raise ValueError("logical context must be a JSON object")
        object.__setattr__(
            self,
            "logical_context_json",
            canonical_json(logical_context),
        )
        if self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")
        if self.temperature is not None:
            if not math.isfinite(self.temperature) or self.temperature < 0:
                raise ValueError("temperature must be finite and non-negative")
        if self.reasoning_effort is not None and not self.reasoning_effort.strip():
            raise ValueError("reasoning_effort must be non-empty when supplied")
        if self.temperature is not None and self.reasoning_effort is not None:
            raise ValueError("temperature and reasoning_effort are mutually exclusive")

    @classmethod
    def from_payload(
        cls,
        *,
        model: str,
        schema_id: str,
        system_prompt: str,
        user_payload: Any,
        max_output_tokens: int,
        logical_context: Mapping[str, Any] | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> "StructuredRequest":
        return cls(
            model=model,
            schema_id=schema_id,
            system_prompt=system_prompt,
            user_payload_json=canonical_json(user_payload),
            max_output_tokens=max_output_tokens,
            logical_context_json=canonical_json(logical_context or {}),
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

    @property
    def user_payload(self) -> dict[str, Any] | list[Any]:
        return json.loads(self.user_payload_json)

    @property
    def logical_context(self) -> dict[str, Any]:
        """Return experiment identity that is recorded but never sent."""
        return json.loads(self.logical_context_json)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "schema_id": self.schema_id,
            "system_prompt": self.system_prompt,
            "user_payload": self.user_payload,
            "logical_context": self.logical_context,
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }

    @property
    def request_hash(self) -> str:
        return hashlib.sha256(canonical_json(self.as_dict()).encode()).hexdigest()


@dataclass(frozen=True)
class StructuredResponse(Generic[ParsedResponse]):
    """Normalized result of one external structured-output call."""

    request_hash: str
    parsed: ParsedResponse
    response_id: str | None
    usage: Any
    raw_response: Any

    def __post_init__(self) -> None:
        if len(self.request_hash) != 64:
            raise ValueError("request_hash must be a SHA-256 hexadecimal digest")
        try:
            int(self.request_hash, 16)
        except ValueError as error:
            raise ValueError("request_hash must be hexadecimal") from error


class StructuredLLMClient(Protocol):
    """Execute one logical request against a structured-output provider."""

    def parse(
        self,
        request: StructuredRequest,
        response_model: type[ParsedResponse],
    ) -> StructuredResponse[ParsedResponse]:
        """Execute and normalize one logical request."""
