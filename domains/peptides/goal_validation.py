"""Schema validation for qualitative intermediate goals."""

from __future__ import annotations


import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_ITEM_ID_PATTERN = r"^item-\d{3}$"
_NUMERIC_COMPARATOR = re.compile(r"(?:>=|<=|==|>|<)\s*[+-]?(?:\d+(?:\.\d*)?|\.\d+)")
_VERBAL_NUMERIC_CUTOFF = re.compile(
    r"(?:at\s+least|at\s+most|no\s+more\s+than|no\s+less\s+than|"
    r"fewer\s+than|more\s+than|minimum(?:\s+of)?|maximum(?:\s+of)?)"
    r"\s+[+-]?(?:\d+(?:\.\d*)?|\.\d+)",
    re.IGNORECASE,
)
_DESCRIPTOR_VALUE_CUTOFF = re.compile(
    r"\b(?:charge|hydrophobic(?:ity|\s+fraction)?|f\s*sp3|molecular\s+weight|"
    r"rotatable\s+bonds?|solubility|half[- ]life|non[- ]hemolysis|"
    r"non[- ]fouling)\b.{0,32}?(?:=|of|is)\s*[+-]?\d+(?:\.\d+)?\b",
    re.IGNORECASE,
)
_QUOTA_LANGUAGE = re.compile(
    r"(?:\bquota\b|\bclass\s+balance\b|\bbalanced\s+classes\b|"
    r"\bpositive\s+(?:rate|fraction|percentage)\b|"
    r"\bnegative\s+(?:rate|fraction|percentage)\b|"
    r"\b\d+(?:\.\d+)?\s*%\b|\b50\s*/\s*50\b)",
    re.IGNORECASE,
)


def operative_text_violations(*texts: str) -> tuple[str, ...]:
    """Detect explicit quota or numeric-cutoff rules in operative goal text."""

    failures: list[str] = []
    for text in texts:
        if _NUMERIC_COMPARATOR.search(text) or _VERBAL_NUMERIC_CUTOFF.search(text):
            failures.append("operative goal text contains a numeric cutoff")
        if _DESCRIPTOR_VALUE_CUTOFF.search(text):
            failures.append("operative goal text fixes a numeric descriptor value")
        if _QUOTA_LANGUAGE.search(text):
            failures.append("operative goal text targets a class quota")
    return tuple(dict.fromkeys(failures))


class VisibleDiscoveryCohortComparison(BaseModel):
    """Qualitative observations over controller-visible discovery data only."""

    model_config = ConfigDict(extra="forbid")

    prior_generator_observation: str = Field(min_length=1)
    current_generator_observation: str = Field(min_length=1)
    prior_representative_item_ids: list[str] = Field(min_length=1, max_length=8)
    current_representative_item_ids: list[str] = Field(min_length=1, max_length=8)
    comparison_is_qualitative_hypothesis_only: Literal[True]
    observed_gate_enrichment_claimed: Literal[False]

    @model_validator(mode="after")
    def _validate_item_ids(self) -> "VisibleDiscoveryCohortComparison":
        for item_id in (
            *self.prior_representative_item_ids,
            *self.current_representative_item_ids,
        ):
            if re.fullmatch(_ITEM_ID_PATTERN, item_id) is None:
                raise ValueError("discovery-cohort references must use item-NNN IDs")
        return self


class _MovingFrontierNonFinalResponse(BaseModel):
    """Fields shared by the one-proposal broadening/tightening responses."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["broaden", "tighten"]
    frontier_criterion: str = Field(min_length=1, max_length=1400)
    bridge_to_final: str = Field(min_length=1)
    limitations: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    direction_preserved: Literal[True]
    proxy_is_temporary: Literal[True]
    visible_evidence_cues_are_neither_necessary_nor_sufficient_biological_markers: (
        Literal[True]
    )
    positive_evidence_definition: str = Field(min_length=1)
    negative_evidence_definition: str = Field(min_length=1)
    positive_witness_item_id: str = Field(pattern=_ITEM_ID_PATTERN)
    negative_witness_item_id: str = Field(pattern=_ITEM_ID_PATTERN)
    proposal_count: Literal[1]
    class_quota_targeted: Literal[False]
    numeric_descriptor_cutoff_used: Literal[False]
    hidden_gate_or_confirmation_data_used: Literal[False]

    @model_validator(mode="after")
    def _validate_operative_contract(self) -> "_MovingFrontierNonFinalResponse":
        if self.positive_witness_item_id == self.negative_witness_item_id:
            raise ValueError("positive and negative witnesses must be distinct")
        violations = operative_text_violations(
            self.frontier_criterion,
            self.positive_evidence_definition,
            self.negative_evidence_definition,
        )
        if violations:
            raise ValueError("; ".join(violations))
        return self


class MovingFrontierBroadeningResponse(_MovingFrontierNonFinalResponse):
    """One initial easier goal when the exact final goal is label-sparse."""

    action: Literal["broaden"]
    removed_or_deferred_requirements: list[str] = Field(min_length=1, max_length=6)
    easier_positive_decision_explanation: str = Field(min_length=1)
    expertise_beyond_evidence_cards: str = Field(min_length=1)
    multiple_plausible_sequence_routes_admitted: Literal[True]
    source_positive_set_plausibly_subset_of_proposed_positive_set: Literal[True]


class MovingFrontierTighteningResponse(_MovingFrontierNonFinalResponse):
    """One cross-cohort tightening to be frozen and independently tested."""

    action: Literal["tighten"]
    discovery_cohort_comparison: VisibleDiscoveryCohortComparison
    newly_unlocked_hypothesis: str = Field(min_length=1)
    restored_final_goal_requirement: str = Field(min_length=1)
    restored_final_goal_requirement_count: Literal[1]
    one_semantic_restriction_added: Literal[True]
    same_proxy_threshold_only: Literal[False]
    proposed_positive_set_plausibly_subset_of_source_positive_set: Literal[True]


class MovingFrontierReachedFinalResponse(BaseModel):
    """Transport schema; the host still verifies the final text verbatim."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["reached_final"]
    frontier_criterion: str = Field(min_length=1, max_length=1400)
    rationale: str = Field(min_length=1)
    proposal_count: Literal[1]
    hidden_gate_or_confirmation_data_used: Literal[False]


def response_schema_for_action(
    action: Literal["broaden", "tighten", "reached_final"],
) -> type[BaseModel]:
    """Return the strict provider-response schema for a host-selected action."""

    schemas: dict[str, type[BaseModel]] = {
        "broaden": MovingFrontierBroadeningResponse,
        "tighten": MovingFrontierTighteningResponse,
        "reached_final": MovingFrontierReachedFinalResponse,
    }
    try:
        return schemas[action]
    except KeyError as error:  # pragma: no cover - protects untyped callers.
        raise ValueError(f"unsupported moving-frontier action: {action}") from error
