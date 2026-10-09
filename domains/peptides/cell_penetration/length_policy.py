"""One-shot, pre-p0 selection of a campaign-wide PepDFM length policy.

Peptide length is an external PepDFM sampling control, not a model output and
not evidence that a candidate satisfies a biological goal.  Selecting length
again after each generator update would therefore confound a claimed
``p_k -> p_(k+1)`` shift with a change in the sampling distribution.  This
module asks one structured LLM call to choose a scientifically useful length
mixture before p0, then writes a content-bound artifact that freezes that
mixture for the entire campaign.

No labelled peptide dataset or reference peptide is supplied.  The selection
uses only the immutable final objective and the model's pretrained scientific
knowledge.
"""

from __future__ import annotations

from domains.peptides.prompts import CELL_PENETRATION as DOMAIN_PROMPTS

import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.peptides.lengths import (
    MAXIMUM_PEPTIDE_LENGTH,
    MAXIMUM_SAMPLES_PER_LENGTH,
    MINIMUM_PEPTIDE_LENGTH,
    MINIMUM_SAMPLES_PER_LENGTH,
    POLICY_SAMPLE_COUNT,
    AdaptiveLengthPolicy,
)

from adm.preferences.clients import (
    StructuredLLMClient,
    StructuredRequest,
    canonical_json,
    value_as_json,
)
from domains.peptides.cell_penetration.goals import FrozenGoalLengthPolicy


PRE_P0_LENGTH_POLICY_SCHEMA_ID = "pepdfm-pre-p0-campaign-length-policy-v13"
PRE_P0_LENGTH_POLICY_ARTIFACT_VERSION = 1


PRE_P0_LENGTH_POLICY_SYSTEM_PROMPT = DOMAIN_PROMPTS['length_policy']


class PreP0LengthPolicyResponse(BaseModel):
    """Strict response to the one and only pre-p0 policy-selection call."""

    model_config = ConfigDict(extra="forbid")

    length_policy: FrozenGoalLengthPolicy
    selected_before_p0: Literal[True]
    frozen_for_entire_campaign: Literal[True]
    external_dataset_or_reference_context_used: Literal[False]
    target_specific_activity_predictor_used: Literal[False]
    novel_sequences_are_expected: Literal[True]

    @model_validator(mode="after")
    def _require_two_to_four_lengths(self) -> "PreP0LengthPolicyResponse":
        count = len(self.length_policy.allocations)
        if not 2 <= count <= 4:
            raise ValueError("pre-p0 policy must select between two and four lengths")
        return self


class PreP0LengthPolicyModelMetadata(BaseModel):
    """Provider metadata retained without archiving the full raw response."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_model: str = Field(min_length=1)
    requested_reasoning_effort: str = Field(min_length=1)
    max_output_tokens: int = Field(gt=0)
    response_id: str | None
    observed_model_version: str | None
    usage: Any
    thought_summary_available: bool


class PreP0LengthPolicyArtifact(BaseModel):
    """Immutable campaign input produced by the pre-p0 selector."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_schema_version: Literal[1]
    protocol: Literal["pre-p0-campaign-length-policy-v13"]
    final_goal: str = Field(min_length=1)
    final_goal_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    pepdfm_supported_length_minimum: Literal[6]
    pepdfm_supported_length_maximum: Literal[49]
    sample_count: Literal[100]
    selected_length_count_minimum: Literal[2]
    selected_length_count_maximum: Literal[4]
    minimum_samples_per_selected_length: Literal[10]
    maximum_samples_per_selected_length: Literal[60]
    policy_selected_before_p0: Literal[True]
    policy_frozen_for_entire_campaign: Literal[True]
    parent_child_comparisons_require_this_exact_policy: Literal[True]
    length_is_not_evidence_of_goal_satisfaction: Literal[True]
    external_dataset_or_reference_context_visible: Literal[False]
    target_specific_activity_predictor_visible: Literal[False]
    novel_generated_sequences_are_expected: Literal[True]
    length_policy: FrozenGoalLengthPolicy
    model_metadata: PreP0LengthPolicyModelMetadata
    artifact_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_content(self) -> "PreP0LengthPolicyArtifact":
        if not 2 <= len(self.length_policy.allocations) <= 4:
            raise ValueError("campaign length policy must contain two to four lengths")
        normalized = _normalize_policy(self.length_policy)
        if normalized.model_dump(mode="json") != self.length_policy.model_dump(
            mode="json"
        ):
            raise ValueError("campaign length-policy allocations must be canonical")
        content = self.model_dump(mode="json", exclude={"artifact_content_sha256"})
        if _sha256_json(content) != self.artifact_content_sha256:
            raise ValueError("pre-p0 length-policy artifact content hash differs")
        return self


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _normalize_json_value(value: Any) -> Any:
    """Return a detached JSON value and reject non-portable provider metadata."""

    return json.loads(canonical_json(value_as_json(value)))


def _normalize_policy(policy: FrozenGoalLengthPolicy) -> FrozenGoalLengthPolicy:
    """Validate with the shared utility and sort allocations by exact length."""

    allocation = {item.length: item.count for item in policy.allocations}
    if len(allocation) != len(policy.allocations):
        raise ValueError("pre-p0 policy contains duplicate lengths")
    normalized = AdaptiveLengthPolicy(allocation)
    if not 2 <= len(normalized) <= 4:
        raise ValueError("pre-p0 policy must select between two and four lengths")
    return FrozenGoalLengthPolicy.model_validate(
        {
            "allocations": [
                {"length": length, "count": count}
                for length, count in normalized.entries
            ],
            "rationale": policy.rationale.strip(),
            "total_samples": POLICY_SAMPLE_COUNT,
            "frozen_until_goal_changes": True,
            "length_is_not_evidence_of_goal_satisfaction": True,
        }
    )


def build_pre_p0_length_policy_request(
    *,
    final_goal: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    logical_context: Mapping[str, Any] | None = None,
) -> StructuredRequest:
    """Build the deterministic request used for selection and replay identity."""

    normalized_goal = final_goal.strip()
    if not normalized_goal:
        raise ValueError("final_goal must be non-empty")
    return StructuredRequest.from_payload(
        model=model,
        schema_id=PRE_P0_LENGTH_POLICY_SCHEMA_ID,
        system_prompt=PRE_P0_LENGTH_POLICY_SYSTEM_PROMPT,
        user_payload={
            "immutable_final_goal": normalized_goal,
            "generator": {
                "name": "PepDFM",
                "length_is_chosen_before_sequence_sampling": True,
                "supported_exact_length_minimum": MINIMUM_PEPTIDE_LENGTH,
                "supported_exact_length_maximum": MAXIMUM_PEPTIDE_LENGTH,
            },
            "campaign_length_policy_contract": {
                "selected_exact_lengths_minimum": 2,
                "selected_exact_lengths_maximum": 4,
                "total_samples": POLICY_SAMPLE_COUNT,
                "minimum_count_per_selected_length": MINIMUM_SAMPLES_PER_LENGTH,
                "maximum_count_per_selected_length": MAXIMUM_SAMPLES_PER_LENGTH,
                "selected_once_before_p0": True,
                "frozen_for_entire_campaign_including_goal_changes": True,
                "parent_child_comparisons_use_identical_length_policy": True,
                "length_is_not_goal_evidence": True,
            },
            "external_evidence_contract": {
                "labelled_peptide_dataset_visible": False,
                "reference_peptides_visible": False,
                "database_lookup_visible": False,
                "internet_visible": False,
                "target_specific_activity_predictor_visible": False,
                "use_pretrained_domain_knowledge": True,
            },
            "no_generated_candidates_are_visible": True,
            "return_exactly_one_policy": True,
        },
        logical_context=dict(logical_context or {}),
        max_output_tokens=max_output_tokens,
        reasoning_effort=reasoning_effort,
    )


def _write_immutable_json(path: Path, value: Any) -> None:
    """Atomically create ``path``; allow only a byte-identical replay."""

    text = json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"immutable artifact differs: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_pre_p0_length_policy_artifact(path: Path) -> PreP0LengthPolicyArtifact:
    """Load and integrity-check a previously frozen campaign policy."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid pre-p0 length-policy artifact: {path}") from error
    return PreP0LengthPolicyArtifact.model_validate(value)


def _model_version(raw_response: Any) -> str | None:
    if not isinstance(raw_response, Mapping):
        return None
    for key in ("model_version", "modelVersion"):
        value = raw_response.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    provider = raw_response.get("provider_response")
    if isinstance(provider, Mapping):
        value = provider.get("modelVersion")
        if value is not None and str(value).strip():
            return str(value).strip()
    return None


def _thought_summary_available(raw_response: Any) -> bool:
    if not isinstance(raw_response, Mapping):
        return False
    value = raw_response.get("thought_summary")
    return isinstance(value, str) and bool(value.strip())


def _artifact_content(
    *,
    final_goal: str,
    request: StructuredRequest,
    response: Any,
    policy: FrozenGoalLengthPolicy,
) -> dict[str, Any]:
    metadata = PreP0LengthPolicyModelMetadata(
        requested_model=request.model,
        requested_reasoning_effort=request.reasoning_effort or "none",
        max_output_tokens=request.max_output_tokens,
        response_id=response.response_id,
        observed_model_version=_model_version(response.raw_response),
        usage=_normalize_json_value(response.usage),
        thought_summary_available=_thought_summary_available(response.raw_response),
    )
    return {
        "artifact_schema_version": PRE_P0_LENGTH_POLICY_ARTIFACT_VERSION,
        "protocol": "pre-p0-campaign-length-policy-v13",
        "final_goal": final_goal,
        "final_goal_sha256": hashlib.sha256(final_goal.encode("utf-8")).hexdigest(),
        "request_hash": request.request_hash,
        "pepdfm_supported_length_minimum": MINIMUM_PEPTIDE_LENGTH,
        "pepdfm_supported_length_maximum": MAXIMUM_PEPTIDE_LENGTH,
        "sample_count": POLICY_SAMPLE_COUNT,
        "selected_length_count_minimum": 2,
        "selected_length_count_maximum": 4,
        "minimum_samples_per_selected_length": MINIMUM_SAMPLES_PER_LENGTH,
        "maximum_samples_per_selected_length": MAXIMUM_SAMPLES_PER_LENGTH,
        "policy_selected_before_p0": True,
        "policy_frozen_for_entire_campaign": True,
        "parent_child_comparisons_require_this_exact_policy": True,
        "length_is_not_evidence_of_goal_satisfaction": True,
        "external_dataset_or_reference_context_visible": False,
        "target_specific_activity_predictor_visible": False,
        "novel_generated_sequences_are_expected": True,
        "length_policy": policy.model_dump(mode="json"),
        "model_metadata": metadata.model_dump(mode="json"),
    }


def select_pre_p0_length_policy(
    *,
    client: StructuredLLMClient,
    final_goal: str,
    model: str,
    reasoning_effort: str,
    max_output_tokens: int,
    artifact_path: Path,
    logical_context: Mapping[str, Any] | None = None,
) -> PreP0LengthPolicyArtifact:
    """Select or replay one campaign-wide policy with at most one client call.

    A valid existing artifact is replayed without consulting ``client``.  A new
    selection invokes ``client.parse`` exactly once; there is no proposal loop
    or correction call in this component.
    """

    normalized_goal = final_goal.strip()
    request = build_pre_p0_length_policy_request(
        final_goal=normalized_goal,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        logical_context=logical_context,
    )

    if artifact_path.is_file():
        existing = load_pre_p0_length_policy_artifact(artifact_path)
        expected_bindings = {
            "final_goal": normalized_goal,
            "final_goal_sha256": hashlib.sha256(
                normalized_goal.encode("utf-8")
            ).hexdigest(),
            "request_hash": request.request_hash,
        }
        observed = existing.model_dump(mode="json")
        mismatches = {
            key: {"expected": value, "observed": observed.get(key)}
            for key, value in expected_bindings.items()
            if observed.get(key) != value
        }
        if mismatches:
            raise FileExistsError(
                "immutable pre-p0 length-policy artifact has different bindings: "
                f"{mismatches}"
            )
        return existing

    # Deliberately exactly one structured provider invocation.
    response_schema = PreP0LengthPolicyResponse
    response = client.parse(request, response_schema)
    if response.request_hash != request.request_hash:
        raise ValueError("length-policy response request hash differs")
    parsed = response_schema.model_validate(response.parsed)
    policy = _normalize_policy(parsed.length_policy)

    content = _artifact_content(
        final_goal=normalized_goal,
        request=request,
        response=response,
        policy=policy,
    )
    artifact = PreP0LengthPolicyArtifact.model_validate(
        {
            **content,
            "artifact_content_sha256": _sha256_json(content),
        }
    )
    _write_immutable_json(artifact_path, artifact.model_dump(mode="json"))
    return artifact


__all__ = [
    "PRE_P0_LENGTH_POLICY_ARTIFACT_VERSION",
    "PRE_P0_LENGTH_POLICY_SCHEMA_ID",
    "PRE_P0_LENGTH_POLICY_SYSTEM_PROMPT",
    "PreP0LengthPolicyArtifact",
    "PreP0LengthPolicyModelMetadata",
    "PreP0LengthPolicyResponse",
    "build_pre_p0_length_policy_request",
    "load_pre_p0_length_policy_artifact",
    "select_pre_p0_length_policy",
]
