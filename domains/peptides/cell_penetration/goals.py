"""V13 prompts and schemas for a generator-reachable semantic goal ladder.

This module is intentionally separate from the frozen V12 protocol.  V13 makes
the alternating generator/goal mechanism explicit, permits bounded correction
of an unsuitable proposed goal on one 100-candidate working cohort, supplies a
strict no-external-dataset evidence boundary, and uses a campaign-wide
sampling-length policy selected once before p0.
"""

from __future__ import annotations

from domains.peptides.prompts import CELL_PENETRATION as DOMAIN_PROMPTS

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from domains.peptides.cell_penetration.goal_validation import operative_text_violations
from domains.peptides.cell_penetration.task import TASK


GEOMETRIC_V13_JUDGE_SYSTEM_PROMPT = DOMAIN_PROMPTS['LLM-Judge']


GEOMETRIC_V13_CONTROLLER_PROMPT = DOMAIN_PROMPTS['SetGoal_proposer']


GEOMETRIC_V13_CORRECTION_ADDENDUM = DOMAIN_PROMPTS['goal_controller']


_ITEM_ID_PATTERN = r"^item-\d{3}$"
_CANONICAL_PEPTIDE_PATTERN = r"^[ACDEFGHIKLMNPQRSTVWY]{6,49}$"


class LengthCount(BaseModel):
    """One exact peptide length and its sample count."""

    model_config = ConfigDict(extra="forbid")

    length: int = Field(ge=6, le=49)
    count: int = Field(ge=10, le=60)


class FrozenGoalLengthPolicy(BaseModel):
    """A bounded 100-sample policy; V13 freezes it for the whole campaign.

    ``frozen_until_goal_changes`` is retained in the provider schema for
    backwards compatibility.  The pre-p0 selector and campaign artifact impose
    the stronger invariant that the same policy survives goal changes too.
    """

    model_config = ConfigDict(extra="forbid")

    allocations: list[LengthCount] = Field(min_length=1, max_length=4)
    rationale: str = Field(min_length=1)
    total_samples: Literal[100]
    frozen_until_goal_changes: Literal[True]
    length_is_not_evidence_of_goal_satisfaction: Literal[True]

    @model_validator(mode="after")
    def _validate_allocations(self) -> "FrozenGoalLengthPolicy":
        lengths = [item.length for item in self.allocations]
        if len(lengths) != len(set(lengths)):
            raise ValueError("length-policy lengths must be distinct")
        if sum(item.count for item in self.allocations) != 100:
            raise ValueError("length-policy counts must sum to 100")
        return self


class ObservedCategoricalCounts(BaseModel):
    """Stable categorical outcomes on exactly one 100-candidate cohort."""

    model_config = ConfigDict(extra="forbid")

    sample_count: Literal[100]
    positive_count: int = Field(ge=0, le=100)
    negative_count: int = Field(ge=0, le=100)
    abstain_count: int = Field(ge=0, le=100)
    unstable_count: int = Field(ge=0, le=100)

    @model_validator(mode="after")
    def _validate_partition(self) -> "ObservedCategoricalCounts":
        total = (
            self.positive_count
            + self.negative_count
            + self.abstain_count
            + self.unstable_count
        )
        if total != self.sample_count:
            raise ValueError("categorical counts must partition all 100 candidates")
        return self


class RepresentativeProposalOutcome(BaseModel):
    """One provider-safe example from the failed proposal's same cohort."""

    model_config = ConfigDict(extra="forbid")

    item_id: str = Field(pattern=_ITEM_ID_PATTERN)
    sequence: str = Field(pattern=_CANONICAL_PEPTIDE_PATTERN)
    stable_outcome: Literal["positive", "negative", "abstain", "unstable"]
    pass_1_label: Literal["positive", "negative", "abstain"]
    pass_2_label: Literal["positive", "negative", "abstain"]
    pass_1_rationale: str = Field(min_length=1, max_length=1200)
    pass_2_rationale: str = Field(min_length=1, max_length=1200)
    pass_1_critical_unknowns: str = Field(min_length=1, max_length=1200)
    pass_2_critical_unknowns: str = Field(min_length=1, max_length=1200)


HostRevisionDirection = Literal[
    "broaden_materially",
    "narrow_slightly",
    "simplify_criterion",
    "repair_semantic_direction",
]


class GoalCorrectionRequest(BaseModel):
    """Host feedback for one of at most three corrections after proposal one."""

    model_config = ConfigDict(extra="forbid")

    proposal_attempt: int = Field(ge=2, le=4)
    correction_number: int = Field(ge=1, le=3)
    prior_frontier_criterion: str = Field(min_length=1)
    observed_counts: ObservedCategoricalCounts
    representative_proposed_goal_outcomes: list[
        RepresentativeProposalOutcome
    ] = Field(min_length=1, max_length=6)
    failed_conditions: list[str] = Field(min_length=1, max_length=12)
    host_revision_direction: HostRevisionDirection
    semantic_feedback: str = Field(min_length=1)
    outcomes_are_from_same_working_cohort: Literal[True]
    exact_class_quota_is_not_the_target: Literal[True]

    @model_validator(mode="after")
    def _validate_attempt_number(self) -> "GoalCorrectionRequest":
        if self.proposal_attempt != self.correction_number + 1:
            raise ValueError("proposal attempt must equal correction number plus one")
        return self


class _GeometricNonFinalResponse(BaseModel):
    """Fields shared by V13 broadening and tightening responses."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["broaden", "tighten"]
    frontier_criterion: str = Field(min_length=1, max_length=1400)
    bridge_to_final: str = Field(min_length=1)
    limitations: str = Field(min_length=1)
    rationale: str = Field(min_length=1)
    positive_evidence_definition: str = Field(min_length=1)
    negative_evidence_definition: str = Field(min_length=1)
    positive_witness_item_id: str = Field(pattern=_ITEM_ID_PATTERN)
    negative_witness_item_id: str = Field(pattern=_ITEM_ID_PATTERN)
    length_policy: FrozenGoalLengthPolicy
    proposal_attempt_acknowledged: int = Field(ge=1, le=4)
    proposals_in_this_response: Literal[1]
    direction_preserved: Literal[True]
    proxy_is_temporary: Literal[True]
    external_dataset_or_reference_evidence_used: Literal[False]
    target_specific_activity_predictor_used: Literal[False]
    novel_sequences_can_be_positive: Literal[True]
    class_quota_targeted: Literal[False]
    numeric_descriptor_cutoff_used: Literal[False]
    biological_success_claimed: Literal[False]

    @model_validator(mode="after")
    def _validate_operative_contract(self) -> "_GeometricNonFinalResponse":
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


class GeometricBroadeningResponse(_GeometricNonFinalResponse):
    """A deliberately broad and generator-reachable first rung."""

    action: Literal["broaden"]
    removed_or_deferred_requirements: list[str] = Field(min_length=1, max_length=8)
    weakest_meaningful_first_rung_selected: Literal[True]
    most_final_requirements_postponed: Literal[True]
    multiple_plausible_mechanistic_routes_admitted: Literal[True]
    no_single_mechanism_claimed_universally_necessary: Literal[True]
    source_positive_set_plausibly_subset_of_proposed_positive_set: Literal[True]


class GeometricTighteningResponse(_GeometricNonFinalResponse):
    """One small goal increment newly reachable after generator movement."""

    action: Literal["tighten"]
    restored_or_refined_final_requirement: str = Field(min_length=1)
    smallest_meaningful_semantic_increment: Literal[True]
    newly_reachable_on_current_distribution: Literal[True]
    remains_discriminative_on_current_distribution: Literal[True]
    why_more_difficult_on_goal_entry_distribution: str = Field(min_length=1)
    why_reachable_on_current_distribution: str = Field(min_length=1)
    proposed_positive_set_plausibly_subset_of_source_positive_set: Literal[True]


class GeometricReachedFinalResponse(BaseModel):
    """Exact-final transport response with a frozen length policy."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["reached_final"]
    frontier_criterion: str = Field(min_length=1, max_length=1400)
    rationale: str = Field(min_length=1)
    length_policy: FrozenGoalLengthPolicy
    proposal_attempt_acknowledged: int = Field(ge=1, le=4)
    proposals_in_this_response: Literal[1]
    external_dataset_or_reference_evidence_used: Literal[False]
    target_specific_activity_predictor_used: Literal[False]
    novel_sequences_can_be_positive: Literal[True]
    biological_success_claimed: Literal[False]


def response_schema_for_action(
    action: Literal["broaden", "tighten", "reached_final"],
) -> type[BaseModel]:
    schemas: dict[str, type[BaseModel]] = {
        "broaden": GeometricBroadeningResponse,
        "tighten": GeometricTighteningResponse,
        "reached_final": GeometricReachedFinalResponse,
    }
    try:
        return schemas[action]
    except KeyError as error:  # pragma: no cover - protects untyped callers.
        raise ValueError(f"unsupported geometric V13 action: {action}") from error


__all__ = [
    "FrozenGoalLengthPolicy",
    "GEOMETRIC_V13_CONTROLLER_PROMPT",
    "GEOMETRIC_V13_CORRECTION_ADDENDUM",
    "GEOMETRIC_V13_JUDGE_SYSTEM_PROMPT",
    "GeometricBroadeningResponse",
    "GeometricReachedFinalResponse",
    "GeometricTighteningResponse",
    "GoalCorrectionRequest",
    "HostRevisionDirection",
    "LengthCount",
    "ObservedCategoricalCounts",
    "RepresentativeProposalOutcome",
    "response_schema_for_action",
]
