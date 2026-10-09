"""Deterministic host control for the V13 semantic-reachability experiment.

The LLM supplies categorical labels and proposes natural-language goals.  It
does not decide whether a cohort is trainable, whether an update enriched a
goal, or whether another update is allowed.  Those decisions live here and use
only stable categorical outcomes from two complete passes over one 100-item
working cohort.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
from typing import Any, Literal, Mapping, Sequence

from domains.peptides.audit import QMatrixCell, StableLabel
from domains.peptides.goals import (
    FrozenGoalLengthPolicy,
    GoalCorrectionRequest,
    ObservedCategoricalCounts,
)
from domains.peptides.types import PeptideCandidate


WORKING_COHORT_SIZE = 100
MINIMUM_STABLE_POSITIVES = 20
MINIMUM_STABLE_NEGATIVES = 20
MINIMUM_EXACT_AGREEMENT = 0.50
MAXIMUM_PASS_ABSTAIN_FRACTION = 0.50
MINIMUM_ENRICHMENT = 0.10
MAXIMUM_UPDATES_PER_GOAL = 5
MAXIMUM_PROPOSAL_ATTEMPTS = 4


def _at_least(observed: float, threshold: float) -> bool:
    """Inclusive decimal threshold robust to binary floating-point noise."""

    return observed > threshold or math.isclose(
        observed, threshold, rel_tol=0.0, abs_tol=1e-12
    )


@dataclass(frozen=True)
class LearnabilityDecision:
    """Host decision for one proposed goal on the current working cohort."""

    passed: bool
    stable_positive_count: int
    stable_negative_count: int
    stable_abstain_count: int
    unstable_count: int
    exact_agreement_fraction: float
    maximum_pass_abstain_fraction: float
    failed_conditions: tuple[str, ...]

    def as_mapping(self) -> dict[str, Any]:
        value = asdict(self)
        # Artifacts are JSON, so expose the canonical JSON shape here rather
        # than relying on serialization to silently turn a tuple into a list.
        # This keeps independently recomputed decisions byte-comparable after
        # a write/read round trip.
        value["failed_conditions"] = list(self.failed_conditions)
        return value


@dataclass(frozen=True)
class EnrichmentDecision:
    """Whether a fixed goal moved enough to permit a semantic tightening."""

    action: Literal["tighten", "hold_and_update", "stop_unproductive"]
    entry_positive_fraction: float
    current_positive_fraction: float
    enrichment: float
    minimum_enriched_positive_fraction: float
    updates_at_goal: int
    minimum_enrichment: float
    maximum_updates_per_goal: int
    rationale: str

    def as_mapping(self) -> dict[str, Any]:
        return asdict(self)


def stable_counts(cell: QMatrixCell) -> ObservedCategoricalCounts:
    """Convert a Q-cell into the correction schema's exhaustive partition."""

    summary = cell.summary
    if summary.sample_count != WORKING_COHORT_SIZE:
        raise ValueError("V13 requires exactly 100 categorical outcomes")
    return ObservedCategoricalCounts(
        sample_count=WORKING_COHORT_SIZE,
        positive_count=summary.stable_positive_count,
        negative_count=summary.stable_negative_count,
        abstain_count=summary.stable_abstain_count,
        unstable_count=summary.unstable_count,
    )


def audit_learnability(
    cell: QMatrixCell,
    *,
    minimum_positive: int = MINIMUM_STABLE_POSITIVES,
    minimum_negative: int = MINIMUM_STABLE_NEGATIVES,
    minimum_exact_agreement: float = MINIMUM_EXACT_AGREEMENT,
    maximum_pass_abstain_fraction: float = MAXIMUM_PASS_ABSTAIN_FRACTION,
) -> LearnabilityDecision:
    """Require a useful two-class update buffer without filtering the cohort."""

    counts = stable_counts(cell)
    failures: list[str] = []
    if counts.positive_count < minimum_positive:
        failures.append("stable_positive_count_below_20")
    if counts.negative_count < minimum_negative:
        failures.append("stable_negative_count_below_20")
    if cell.summary.agreement_fraction < minimum_exact_agreement:
        failures.append("exact_agreement_fraction_below_0.50")
    if (
        cell.summary.maximum_pass_abstain_fraction
        > maximum_pass_abstain_fraction
    ):
        failures.append("maximum_pass_abstain_fraction_above_0.50")
    return LearnabilityDecision(
        passed=not failures,
        stable_positive_count=counts.positive_count,
        stable_negative_count=counts.negative_count,
        stable_abstain_count=counts.abstain_count,
        unstable_count=counts.unstable_count,
        exact_agreement_fraction=cell.summary.agreement_fraction,
        maximum_pass_abstain_fraction=(
            cell.summary.maximum_pass_abstain_fraction
        ),
        failed_conditions=tuple(failures),
    )


def correction_request(
    *,
    proposal_attempt: int,
    prior_frontier_criterion: str,
    cell: QMatrixCell,
    semantic_direction_passed: bool,
    length_policy_passed: bool,
    representative_proposed_goal_outcomes: Sequence[Mapping[str, Any]],
    complementary_temporary_negatives: bool = False,
) -> GoalCorrectionRequest:
    """Translate deterministic failures into one bounded controller revision."""

    if not 2 <= proposal_attempt <= MAXIMUM_PROPOSAL_ATTEMPTS:
        raise ValueError("a correction proposal attempt must lie in [2, 4]")
    counts = stable_counts(cell)
    failures: list[str] = []
    if counts.positive_count < MINIMUM_STABLE_POSITIVES:
        failures.append("stable_positive_count_below_20")
    if counts.negative_count < MINIMUM_STABLE_NEGATIVES:
        failures.append("stable_negative_count_below_20")
    if cell.summary.agreement_fraction < MINIMUM_EXACT_AGREEMENT:
        failures.append("exact_agreement_fraction_below_0.50")
    if (
        cell.summary.maximum_pass_abstain_fraction
        > MAXIMUM_PASS_ABSTAIN_FRACTION
    ):
        failures.append("maximum_pass_abstain_fraction_above_0.50")
    if counts.abstain_count + counts.unstable_count > 60:
        failures.append("categorical_judgment_too_ambiguous_or_unstable")
    if not semantic_direction_passed:
        failures.append("semantic_direction_or_nestedness_failed")
    if not length_policy_passed:
        failures.append("proposal_changed_campaign_frozen_length_policy")
    if not failures:
        raise ValueError("correction feedback requires at least one failed condition")

    if not semantic_direction_passed or not length_policy_passed:
        direction = "repair_semantic_direction"
        explanation = (
            "Repair the nested biological relation and repeat the campaign-frozen "
            "sampling-length allocation exactly; length is not a goal edit."
        )
    elif (
        cell.summary.agreement_fraction < MINIMUM_EXACT_AGREEMENT
        or cell.summary.maximum_pass_abstain_fraction
        > MAXIMUM_PASS_ABSTAIN_FRACTION
        or (
            counts.positive_count < MINIMUM_STABLE_POSITIVES
            and counts.abstain_count + counts.unstable_count >= 20
        )
    ):
        direction = "simplify_criterion"
        explanation = (
            "Sparse positives coincide with substantial abstention/instability. "
            "Simplify the operative clauses so a peptide expert can make stable "
            "categorical judgments from the available sequence evidence."
        )
    elif counts.positive_count < MINIMUM_STABLE_POSITIVES:
        direction = "broaden_materially"
        explanation = (
            "Remove or defer one genuine operative requirement. Keep the goal on "
            "the requested side of its source goal and do not target a quota."
        )
    elif counts.negative_count < MINIMUM_STABLE_NEGATIVES:
        direction = "narrow_slightly"
        if complementary_temporary_negatives:
            explanation = (
                "First repair the negative evidence definition so clear failure "
                "of a necessary operative feature is negative rather than "
                "abstaining. Add one small biologically meaningful distinction "
                "only if that coherent complement remains too small; do not "
                "force binary labels or target a quota."
            )
        else:
            explanation = (
                "Add one small biologically meaningful distinction, without "
                "jumping to the final goal or introducing a descriptor cutoff."
            )
    else:
        direction = "simplify_criterion"
        explanation = (
            "Simplify the operative clauses so a peptide expert can make stable "
            "categorical judgments from the available sequence evidence."
        )
    return GoalCorrectionRequest(
        proposal_attempt=proposal_attempt,
        correction_number=proposal_attempt - 1,
        prior_frontier_criterion=prior_frontier_criterion,
        observed_counts=counts,
        representative_proposed_goal_outcomes=list(
            representative_proposed_goal_outcomes
        ),
        failed_conditions=failures,
        host_revision_direction=direction,
        semantic_feedback=explanation,
        outcomes_are_from_same_working_cohort=True,
        exact_class_quota_is_not_the_target=True,
    )


def choose_after_update(
    *,
    entry_cell: QMatrixCell,
    current_cell: QMatrixCell,
    updates_at_goal: int,
    minimum_enrichment: float = MINIMUM_ENRICHMENT,
    minimum_enriched_positive_fraction: float = 0.60,
    maximum_updates_per_goal: int = MAXIMUM_UPDATES_PER_GOAL,
) -> EnrichmentDecision:
    """Hold a goal until enrichment, with a hard five-update futility bound."""

    if entry_cell.goal_id != current_cell.goal_id:
        raise ValueError("enrichment cells must concern the same frozen goal")
    if not 1 <= updates_at_goal <= maximum_updates_per_goal:
        raise ValueError("updates_at_goal is outside the registered bound")
    enrichment = current_cell.q - entry_cell.q
    if _at_least(
        current_cell.q, minimum_enriched_positive_fraction
    ) and _at_least(enrichment, minimum_enrichment):
        action: Literal["tighten", "hold_and_update", "stop_unproductive"] = (
            "tighten"
        )
        rationale = (
            "The fixed goal reached the registered stable-positive prevalence "
            f"floor ({minimum_enriched_positive_fraction:.0%}) and enriched by "
            "at least the registered goal-entry delta; the host may now seek "
            "the smallest newly reachable semantic tightening."
        )
    elif updates_at_goal < maximum_updates_per_goal:
        action = "hold_and_update"
        rationale = (
            "The fixed goal has not yet met both the registered prevalence "
            f"floor ({minimum_enriched_positive_fraction:.0%}) and goal-entry "
            "enrichment gate, so it is held for another update."
        )
    else:
        action = "stop_unproductive"
        rationale = (
            "Five updates failed to enrich the fixed goal by ten percentage "
            "points; tightening would violate the reachability protocol."
        )
    return EnrichmentDecision(
        action=action,
        entry_positive_fraction=entry_cell.q,
        current_positive_fraction=current_cell.q,
        enrichment=enrichment,
        minimum_enriched_positive_fraction=minimum_enriched_positive_fraction,
        updates_at_goal=updates_at_goal,
        minimum_enrichment=minimum_enrichment,
        maximum_updates_per_goal=maximum_updates_per_goal,
        rationale=rationale,
    )


def frozen_length_counts(
    policy: FrozenGoalLengthPolicy | Mapping[str, Any],
) -> dict[int, int]:
    value = FrozenGoalLengthPolicy.model_validate(policy)
    return dict(sorted((row.length, row.count) for row in value.allocations))


def length_policy_matches_campaign(
    policy: FrozenGoalLengthPolicy | Mapping[str, Any],
    campaign_counts: Mapping[int | str, int],
) -> bool:
    expected = dict(sorted((int(key), int(value)) for key, value in campaign_counts.items()))
    return frozen_length_counts(policy) == expected


def categorical_report_by_length(
    *, candidates: Sequence[PeptideCandidate], cell: QMatrixCell
) -> dict[str, Any]:
    """Report raw and per-length categorical statistics; never a score."""

    if len(candidates) != WORKING_COHORT_SIZE:
        raise ValueError("V13 reports require one 100-candidate cohort")
    by_candidate = {row.candidate_id: row for row in candidates}
    if len(by_candidate) != len(candidates):
        raise ValueError("working cohort candidate IDs must be unique")
    if cell.summary.candidate_ids != frozenset(by_candidate):
        raise ValueError("categorical outcomes do not cover the working cohort")

    counts: dict[int, Counter[str]] = defaultdict(Counter)
    for outcome in cell.summary.outcomes:
        length = len(by_candidate[outcome.candidate_id].sequence)
        counts[length][outcome.stable_label.value] += 1
    by_length: dict[str, Any] = {}
    for length, row_counts in sorted(counts.items()):
        total = sum(row_counts.values())
        by_length[str(length)] = {
            "sample_count": total,
            "stable_positive_count": row_counts[StableLabel.POSITIVE.value],
            "stable_negative_count": row_counts[StableLabel.NEGATIVE.value],
            "stable_abstain_count": row_counts[StableLabel.ABSTAIN.value],
            "unstable_count": row_counts[StableLabel.UNSTABLE.value],
            "stable_positive_fraction_all_candidates": (
                row_counts[StableLabel.POSITIVE.value] / total
            ),
        }
    return {
        "raw": cell.as_mapping(include_outcomes=False),
        "by_length": by_length,
        "cardinal_scores_used": False,
        "abstentions_and_disagreements_remain_in_denominator": True,
    }


__all__ = [
    "EnrichmentDecision",
    "LearnabilityDecision",
    "MAXIMUM_PROPOSAL_ATTEMPTS",
    "MAXIMUM_UPDATES_PER_GOAL",
    "MINIMUM_ENRICHMENT",
    "MAXIMUM_PASS_ABSTAIN_FRACTION",
    "MINIMUM_EXACT_AGREEMENT",
    "MINIMUM_STABLE_NEGATIVES",
    "MINIMUM_STABLE_POSITIVES",
    "WORKING_COHORT_SIZE",
    "audit_learnability",
    "categorical_report_by_length",
    "choose_after_update",
    "correction_request",
    "frozen_length_counts",
    "length_policy_matches_campaign",
    "stable_counts",
]
