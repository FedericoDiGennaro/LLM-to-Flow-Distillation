"""Deterministic audits for a moving semantic-goal frontier.

This module deliberately has no model, API, peptide, or prompt dependencies.
It consumes two categorical judgment passes for frozen candidate cohorts and
answers two host-side questions:

* did alignment make the *current* goal materially more prevalent; and
* is a proposed tighter goal newly accessible in the current generator rather
  than already common in its predecessor?

Only categorical labels control these decisions.  Abstentions and cross-pass
disagreements count as not positive in every prevalence estimate.  LLM
confidence and other cardinal scores are intentionally absent.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
import random
from typing import Any, Iterable, Mapping, Sequence


RAW_LABELS = frozenset({"positive", "negative", "abstain"})
DEFAULT_STRATUM = "__all__"


class StableLabel(str, Enum):
    """Cross-pass categorical outcome for one candidate."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    ABSTAIN = "abstain"
    UNSTABLE = "unstable"


@dataclass(frozen=True)
class CategoricalPassCounts:
    """Counts from one complete categorical judgment pass."""

    sample_count: int
    positive_count: int
    negative_count: int
    abstain_count: int

    def __post_init__(self) -> None:
        values = (
            self.sample_count,
            self.positive_count,
            self.negative_count,
            self.abstain_count,
        )
        if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
            raise TypeError("categorical counts must be integers")
        if self.sample_count <= 0 or min(values[1:]) < 0:
            raise ValueError("categorical counts require a non-empty sample")
        if sum(values[1:]) != self.sample_count:
            raise ValueError("categorical counts do not sum to sample_count")

    @property
    def positive_fraction(self) -> float:
        return self.positive_count / self.sample_count

    @property
    def abstain_fraction(self) -> float:
        return self.abstain_count / self.sample_count


@dataclass(frozen=True)
class StableCategoricalOutcome:
    """One candidate's exact cross-pass label and fixed sampling stratum."""

    candidate_id: str
    first_label: str
    repeat_label: str
    stable_label: StableLabel
    stratum: str

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if self.first_label not in RAW_LABELS or self.repeat_label not in RAW_LABELS:
            raise ValueError("raw labels must be positive, negative, or abstain")
        if not self.stratum.strip():
            raise ValueError("stratum must be non-empty")
        expected = (
            StableLabel(self.first_label)
            if self.first_label == self.repeat_label
            else StableLabel.UNSTABLE
        )
        if self.stable_label is not expected:
            raise ValueError("stable_label is inconsistent with the two raw labels")

    @property
    def stable_positive_indicator(self) -> bool:
        return self.stable_label is StableLabel.POSITIVE


@dataclass(frozen=True)
class StableCategoricalSummary:
    """Stable-label arithmetic for two complete passes over one cohort."""

    outcomes: tuple[StableCategoricalOutcome, ...]
    first_counts: CategoricalPassCounts
    repeat_counts: CategoricalPassCounts

    def __post_init__(self) -> None:
        if not self.outcomes:
            raise ValueError("stable summary requires at least one outcome")
        ids = [row.candidate_id for row in self.outcomes]
        if len(ids) != len(set(ids)):
            raise ValueError("stable summary contains duplicate candidate IDs")
        if self.first_counts.sample_count != len(ids):
            raise ValueError("first-pass count differs from outcome count")
        if self.repeat_counts.sample_count != len(ids):
            raise ValueError("repeat-pass count differs from outcome count")
        observed_first = _counts(row.first_label for row in self.outcomes)
        observed_repeat = _counts(row.repeat_label for row in self.outcomes)
        if observed_first != self.first_counts or observed_repeat != self.repeat_counts:
            raise ValueError("stored pass counts differ from outcome labels")

    @property
    def sample_count(self) -> int:
        return len(self.outcomes)

    @property
    def agreement_count(self) -> int:
        return sum(row.stable_label is not StableLabel.UNSTABLE for row in self.outcomes)

    @property
    def agreement_fraction(self) -> float:
        return self.agreement_count / self.sample_count

    @property
    def positive_membership_agreement_count(self) -> int:
        """Number agreeing on positive versus not-positive membership.

        Negative/abstain switching is deliberately ignored.  This is useful
        only for auditing an evidence-sparse final-goal baseline; trainable
        goals continue to require exact three-class repeat agreement.
        """

        return sum(
            (row.first_label == "positive")
            == (row.repeat_label == "positive")
            for row in self.outcomes
        )

    @property
    def positive_membership_agreement_fraction(self) -> float:
        return self.positive_membership_agreement_count / self.sample_count

    def stable_count(self, label: StableLabel) -> int:
        return sum(row.stable_label is label for row in self.outcomes)

    @property
    def stable_positive_count(self) -> int:
        return self.stable_count(StableLabel.POSITIVE)

    @property
    def stable_negative_count(self) -> int:
        return self.stable_count(StableLabel.NEGATIVE)

    @property
    def stable_abstain_count(self) -> int:
        return self.stable_count(StableLabel.ABSTAIN)

    @property
    def unstable_count(self) -> int:
        return self.stable_count(StableLabel.UNSTABLE)

    @property
    def stable_positive_fraction(self) -> float:
        """Stable positives divided by all candidates.

        Stable abstentions and unstable labels remain in the denominator.
        """

        return self.stable_positive_count / self.sample_count

    @property
    def maximum_pass_abstain_fraction(self) -> float:
        return max(
            self.first_counts.abstain_fraction,
            self.repeat_counts.abstain_fraction,
        )

    @property
    def candidate_ids(self) -> frozenset[str]:
        return frozenset(row.candidate_id for row in self.outcomes)

    def by_candidate_id(self) -> dict[str, StableCategoricalOutcome]:
        return {row.candidate_id: row for row in self.outcomes}

    def as_mapping(self, *, include_outcomes: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "sample_count": self.sample_count,
            "agreement_count": self.agreement_count,
            "agreement_fraction": self.agreement_fraction,
            "positive_membership_agreement_count": (
                self.positive_membership_agreement_count
            ),
            "positive_membership_agreement_fraction": (
                self.positive_membership_agreement_fraction
            ),
            "stable_positive_count": self.stable_positive_count,
            "stable_negative_count": self.stable_negative_count,
            "stable_abstain_count": self.stable_abstain_count,
            "unstable_count": self.unstable_count,
            "stable_positive_fraction_all_candidates": self.stable_positive_fraction,
            "maximum_pass_abstain_fraction": self.maximum_pass_abstain_fraction,
            "first_counts": asdict(self.first_counts),
            "repeat_counts": asdict(self.repeat_counts),
        }
        if include_outcomes:
            payload["outcomes"] = [
                {
                    **asdict(row),
                    "stable_label": row.stable_label.value,
                    "stable_positive_indicator": row.stable_positive_indicator,
                }
                for row in self.outcomes
            ]
        return payload


@dataclass(frozen=True)
class QMatrixCell:
    """One empirical ``q(generator, goal)`` matrix cell.

    ``q`` is the stable-positive fraction among all judged candidates.  This
    conservative definition prevents abstention or disagreement filtering from
    inflating semantic prevalence.
    """

    generator_id: str
    goal_id: str
    summary: StableCategoricalSummary

    def __post_init__(self) -> None:
        if not self.generator_id.strip() or not self.goal_id.strip():
            raise ValueError("generator_id and goal_id must be non-empty")

    @property
    def q(self) -> float:
        return self.summary.stable_positive_fraction

    def as_mapping(self, *, include_outcomes: bool = False) -> dict[str, Any]:
        return {
            "generator_id": self.generator_id,
            "goal_id": self.goal_id,
            "q_stable_positive_fraction_all_candidates": self.q,
            "stable_labels": self.summary.as_mapping(
                include_outcomes=include_outcomes
            ),
        }


@dataclass(frozen=True)
class BootstrapInterval:
    """Percentile interval for current minus previous stable-positive rate."""

    point_difference: float
    lower: float
    upper: float
    confidence_level: float
    replicates: int
    seed: int
    stratum_counts: tuple[tuple[str, int], ...]

    def as_mapping(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["stratum_counts"] = dict(self.stratum_counts)
        return payload


@dataclass(frozen=True)
class EmpiricalSubsetAudit:
    """Observed subset relationship between a parent and tighter goal."""

    generator_id: str
    parent_goal_id: str
    tighter_goal_id: str
    parent_stable_positive_count: int
    tighter_stable_positive_count: int
    retained_positive_count: int
    tighter_positive_outside_parent_count: int
    parent_positive_removed_count: int
    nestedness_fraction: float | None
    pruning_fraction: float | None

    def as_mapping(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GateCheck:
    """One deterministic, machine-readable gate condition."""

    name: str
    passed: bool
    observed: int | float | bool | None
    requirement: str

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.requirement.strip():
            raise ValueError("gate checks require a name and requirement")


@dataclass(frozen=True)
class GateDecision:
    """All checks and the resulting host decision."""

    checks: tuple[GateCheck, ...]

    def __post_init__(self) -> None:
        if not self.checks:
            raise ValueError("gate decision requires at least one check")
        names = [check.name for check in self.checks]
        if len(names) != len(set(names)):
            raise ValueError("gate checks must have unique names")

    @property
    def passed(self) -> bool:
        return all(check.passed for check in self.checks)

    @property
    def failed_conditions(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if not check.passed)

    def as_mapping(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed_conditions": list(self.failed_conditions),
            "checks": [asdict(check) for check in self.checks],
        }


@dataclass(frozen=True)
class CurrentGoalGateConfig:
    """Prospectively fixed thresholds for declaring current-goal movement."""

    minimum_current_positive_fraction: float = 0.60
    minimum_positive_fraction_gain: float = 0.15
    bootstrap_lower_bound_must_exceed: float = 0.0
    minimum_repeat_agreement: float = 0.80
    maximum_pass_abstain_fraction: float = 0.50
    bootstrap_replicates: int = 10_000
    bootstrap_confidence_level: float = 0.95
    bootstrap_seed: int = 17_021

    def __post_init__(self) -> None:
        _validate_gate_config(
            fractions={
                "minimum_current_positive_fraction": self.minimum_current_positive_fraction,
                "minimum_positive_fraction_gain": self.minimum_positive_fraction_gain,
                "minimum_repeat_agreement": self.minimum_repeat_agreement,
                "maximum_pass_abstain_fraction": self.maximum_pass_abstain_fraction,
            },
            lower_bound=self.bootstrap_lower_bound_must_exceed,
            replicates=self.bootstrap_replicates,
            confidence_level=self.bootstrap_confidence_level,
            seed=self.bootstrap_seed,
        )


@dataclass(frozen=True)
class NewlyUnlockedGoalGateConfig:
    """Prospectively fixed thresholds for accepting one tighter goal."""

    minimum_current_stable_positive_count: int = 20
    minimum_current_stable_negative_count: int = 25
    maximum_pass_abstain_fraction: float = 0.50
    minimum_repeat_agreement: float = 0.80
    minimum_positive_fraction_gain: float = 0.15
    bootstrap_lower_bound_must_exceed: float = 0.0
    minimum_nestedness_fraction: float = 0.90
    minimum_pruning_fraction: float = 0.10
    bootstrap_replicates: int = 10_000
    bootstrap_confidence_level: float = 0.95
    bootstrap_seed: int = 17_023

    def __post_init__(self) -> None:
        for name in (
            "minimum_current_stable_positive_count",
            "minimum_current_stable_negative_count",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        _validate_gate_config(
            fractions={
                "maximum_pass_abstain_fraction": self.maximum_pass_abstain_fraction,
                "minimum_repeat_agreement": self.minimum_repeat_agreement,
                "minimum_positive_fraction_gain": self.minimum_positive_fraction_gain,
                "minimum_nestedness_fraction": self.minimum_nestedness_fraction,
                "minimum_pruning_fraction": self.minimum_pruning_fraction,
            },
            lower_bound=self.bootstrap_lower_bound_must_exceed,
            replicates=self.bootstrap_replicates,
            confidence_level=self.bootstrap_confidence_level,
            seed=self.bootstrap_seed,
        )


@dataclass(frozen=True)
class InitialBroadeningGateConfig:
    """Frozen gates for showing that G1 is empirically broader than F on p0."""

    minimum_positive_fraction_gain: float = 0.15
    minimum_final_positive_retention: float = 0.90
    bootstrap_lower_bound_must_exceed: float = 0.0
    minimum_repeat_agreement: float = 0.80
    minimum_final_positive_membership_agreement: float = 0.80
    maximum_pass_abstain_fraction: float = 0.50
    bootstrap_replicates: int = 10_000
    bootstrap_confidence_level: float = 0.95
    bootstrap_seed: int = 17_019

    def __post_init__(self) -> None:
        _validate_gate_config(
            fractions={
                "minimum_positive_fraction_gain": self.minimum_positive_fraction_gain,
                "minimum_final_positive_retention": self.minimum_final_positive_retention,
                "minimum_repeat_agreement": self.minimum_repeat_agreement,
                "minimum_final_positive_membership_agreement": (
                    self.minimum_final_positive_membership_agreement
                ),
                "maximum_pass_abstain_fraction": self.maximum_pass_abstain_fraction,
            },
            lower_bound=self.bootstrap_lower_bound_must_exceed,
            replicates=self.bootstrap_replicates,
            confidence_level=self.bootstrap_confidence_level,
            seed=self.bootstrap_seed,
        )


@dataclass(frozen=True)
class CurrentGoalMovementAudit:
    previous_cell: QMatrixCell
    current_cell: QMatrixCell
    enrichment_interval: BootstrapInterval
    decision: GateDecision

    def as_mapping(self) -> dict[str, Any]:
        return {
            "previous": self.previous_cell.as_mapping(),
            "current": self.current_cell.as_mapping(),
            "enrichment": self.enrichment_interval.as_mapping(),
            "decision": self.decision.as_mapping(),
        }


@dataclass(frozen=True)
class NewlyUnlockedGoalAudit:
    previous_tighter_goal_cell: QMatrixCell
    current_tighter_goal_cell: QMatrixCell
    current_parent_goal_cell: QMatrixCell
    enrichment_interval: BootstrapInterval
    empirical_subset: EmpiricalSubsetAudit
    semantic_direction_passed: bool
    decision: GateDecision

    def as_mapping(self) -> dict[str, Any]:
        return {
            "previous_tighter_goal": self.previous_tighter_goal_cell.as_mapping(),
            "current_tighter_goal": self.current_tighter_goal_cell.as_mapping(),
            "current_parent_goal": self.current_parent_goal_cell.as_mapping(),
            "enrichment": self.enrichment_interval.as_mapping(),
            "empirical_subset": self.empirical_subset.as_mapping(),
            "semantic_direction_passed": self.semantic_direction_passed,
            "decision": self.decision.as_mapping(),
        }


@dataclass(frozen=True)
class InitialBroadeningAudit:
    """Same-p0 empirical test that an initial bridge really broadens F."""

    final_goal_cell: QMatrixCell
    broadened_goal_cell: QMatrixCell
    broadening_interval: BootstrapInterval
    final_positive_count: int
    retained_final_positive_count: int
    broadened_positive_outside_final_count: int
    final_positive_retention_fraction: float
    semantic_direction_passed: bool
    decision: GateDecision

    def as_mapping(self) -> dict[str, Any]:
        return {
            "final_goal": self.final_goal_cell.as_mapping(),
            "broadened_goal": self.broadened_goal_cell.as_mapping(),
            "broadening": self.broadening_interval.as_mapping(),
            "final_positive_count": self.final_positive_count,
            "retained_final_positive_count": self.retained_final_positive_count,
            "broadened_positive_outside_final_count": (
                self.broadened_positive_outside_final_count
            ),
            "final_positive_retention_fraction": (
                self.final_positive_retention_fraction
            ),
            "semantic_direction_passed": self.semantic_direction_passed,
            "decision": self.decision.as_mapping(),
        }


def stable_categorical_labels(
    first_labels: Mapping[str, object],
    repeat_labels: Mapping[str, object],
    *,
    strata: Mapping[str, object] | None = None,
) -> StableCategoricalSummary:
    """Intersect two categorical passes without filtering hard cases.

    A candidate is stably positive/negative/abstaining only when both passes
    emit that exact label.  Every disagreement is ``unstable``.
    """

    if not first_labels:
        raise ValueError("label passes must be non-empty")
    if set(first_labels) != set(repeat_labels):
        raise ValueError("label passes must cover identical candidate IDs")
    candidate_ids = set(first_labels)
    if any(
        not isinstance(candidate_id, str) or not candidate_id.strip()
        for candidate_id in candidate_ids
    ):
        raise ValueError("candidate IDs must be non-empty strings")
    if strata is not None and set(strata) != candidate_ids:
        raise ValueError("strata must cover exactly the judged candidate IDs")

    outcomes: list[StableCategoricalOutcome] = []
    first_normalized: list[str] = []
    repeat_normalized: list[str] = []
    for candidate_id in sorted(candidate_ids):
        first = _normalize_raw_label(first_labels[candidate_id])
        repeat = _normalize_raw_label(repeat_labels[candidate_id])
        stratum = (
            DEFAULT_STRATUM
            if strata is None
            else _normalize_stratum(strata[candidate_id])
        )
        stable = StableLabel(first) if first == repeat else StableLabel.UNSTABLE
        outcomes.append(
            StableCategoricalOutcome(
                candidate_id=candidate_id,
                first_label=first,
                repeat_label=repeat,
                stable_label=stable,
                stratum=stratum,
            )
        )
        first_normalized.append(first)
        repeat_normalized.append(repeat)
    return StableCategoricalSummary(
        outcomes=tuple(outcomes),
        first_counts=_counts(first_normalized),
        repeat_counts=_counts(repeat_normalized),
    )


def build_q_matrix_cell(
    *,
    generator_id: str,
    goal_id: str,
    first_labels: Mapping[str, object],
    repeat_labels: Mapping[str, object],
    strata: Mapping[str, object] | None = None,
) -> QMatrixCell:
    """Construct one auditable ``q(generator, goal)`` cell."""

    return QMatrixCell(
        generator_id=generator_id,
        goal_id=goal_id,
        summary=stable_categorical_labels(
            first_labels,
            repeat_labels,
            strata=strata,
        ),
    )


def q_matrix_table(cells: Sequence[QMatrixCell]) -> dict[str, dict[str, float]]:
    """Return ``generator -> goal -> q`` while rejecting duplicate cells."""

    if not cells:
        raise ValueError("q matrix requires at least one cell")
    result: dict[str, dict[str, float]] = {}
    seen: set[tuple[str, str]] = set()
    for cell in cells:
        key = (cell.generator_id, cell.goal_id)
        if key in seen:
            raise ValueError("q matrix contains a duplicate generator/goal cell")
        seen.add(key)
        result.setdefault(cell.generator_id, {})[cell.goal_id] = cell.q
    return result


def stratified_bootstrap_rate_difference(
    previous: QMatrixCell,
    current: QMatrixCell,
    *,
    replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 17_021,
) -> BootstrapInterval:
    """Bootstrap ``q(current)-q(previous)`` within fixed sampling strata.

    Each generator-by-stratum cell is resampled independently at its observed
    size.  Matching stratum counts are required prospectively, preventing a
    changed peptide-length mixture from masquerading as semantic enrichment.
    """

    _validate_bootstrap_parameters(replicates, confidence_level, seed)
    if previous.goal_id != current.goal_id:
        raise ValueError("bootstrap comparison requires one frozen goal")
    if previous.generator_id == current.generator_id:
        raise ValueError("bootstrap comparison requires distinct generators")
    previous_by_stratum = _binary_by_stratum(previous.summary)
    current_by_stratum = _binary_by_stratum(current.summary)
    previous_counts = {name: len(values) for name, values in previous_by_stratum.items()}
    current_counts = {name: len(values) for name, values in current_by_stratum.items()}
    if previous_counts != current_counts:
        raise ValueError(
            "bootstrap cohorts must have identical fixed stratum counts"
        )

    rng = random.Random(seed)
    draws: list[float] = []
    total = sum(previous_counts.values())
    for _ in range(replicates):
        previous_positive = 0
        current_positive = 0
        for stratum in sorted(previous_counts):
            previous_values = previous_by_stratum[stratum]
            current_values = current_by_stratum[stratum]
            count = previous_counts[stratum]
            previous_positive += sum(
                previous_values[rng.randrange(count)] for _ in range(count)
            )
            current_positive += sum(
                current_values[rng.randrange(count)] for _ in range(count)
            )
        draws.append((current_positive - previous_positive) / total)
    draws.sort()
    tail = (1 - confidence_level) / 2
    return BootstrapInterval(
        point_difference=current.q - previous.q,
        lower=_percentile(draws, tail),
        upper=_percentile(draws, 1 - tail),
        confidence_level=confidence_level,
        replicates=replicates,
        seed=seed,
        stratum_counts=tuple(sorted(previous_counts.items())),
    )


def _paired_stratified_goal_rate_difference(
    narrower: QMatrixCell,
    broader: QMatrixCell,
    *,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> BootstrapInterval:
    """Paired bootstrap of q(broader)-q(narrower) on one fixed cohort."""

    _validate_bootstrap_parameters(replicates, confidence_level, seed)
    if narrower.generator_id != broader.generator_id:
        raise ValueError("paired goal comparison requires one generator")
    if narrower.goal_id == broader.goal_id:
        raise ValueError("paired goal comparison requires distinct goals")
    if narrower.summary.candidate_ids != broader.summary.candidate_ids:
        raise ValueError("paired goal comparison requires identical candidate IDs")
    narrow = narrower.summary.by_candidate_id()
    broad = broader.summary.by_candidate_id()
    paired_by_stratum: dict[str, list[tuple[int, int]]] = {}
    for candidate_id in sorted(narrow):
        if narrow[candidate_id].stratum != broad[candidate_id].stratum:
            raise ValueError("paired goal comparison requires fixed strata")
        paired_by_stratum.setdefault(narrow[candidate_id].stratum, []).append(
            (
                int(narrow[candidate_id].stable_positive_indicator),
                int(broad[candidate_id].stable_positive_indicator),
            )
        )
    counts = {name: len(values) for name, values in paired_by_stratum.items()}
    total = sum(counts.values())
    rng = random.Random(seed)
    draws: list[float] = []
    for _ in range(replicates):
        difference = 0
        for stratum in sorted(paired_by_stratum):
            values = paired_by_stratum[stratum]
            count = len(values)
            for _position in range(count):
                narrow_value, broad_value = values[rng.randrange(count)]
                difference += broad_value - narrow_value
        draws.append(difference / total)
    draws.sort()
    tail = (1 - confidence_level) / 2
    return BootstrapInterval(
        point_difference=broader.q - narrower.q,
        lower=_percentile(draws, tail),
        upper=_percentile(draws, 1 - tail),
        confidence_level=confidence_level,
        replicates=replicates,
        seed=seed,
        stratum_counts=tuple(sorted(counts.items())),
    )


def audit_initial_broadening(
    final_goal: QMatrixCell,
    broadened_goal: QMatrixCell,
    *,
    semantic_direction_passed: bool,
    config: InitialBroadeningGateConfig | None = None,
) -> InitialBroadeningAudit:
    """Require G1 to be a reliable empirical superset of F on the same p0 gate."""

    if config is None:
        config = InitialBroadeningGateConfig()
    if final_goal.generator_id != broadened_goal.generator_id:
        raise ValueError("initial broadening requires one generator")
    if not isinstance(semantic_direction_passed, bool):
        raise TypeError("semantic_direction_passed must be boolean")
    interval = _paired_stratified_goal_rate_difference(
        final_goal,
        broadened_goal,
        replicates=config.bootstrap_replicates,
        confidence_level=config.bootstrap_confidence_level,
        seed=config.bootstrap_seed,
    )
    final_by_id = final_goal.summary.by_candidate_id()
    broad_by_id = broadened_goal.summary.by_candidate_id()
    final_positive = {
        candidate_id
        for candidate_id, row in final_by_id.items()
        if row.stable_positive_indicator
    }
    broad_positive = {
        candidate_id
        for candidate_id, row in broad_by_id.items()
        if row.stable_positive_indicator
    }
    retained = final_positive.intersection(broad_positive)
    retention = (
        1.0 if not final_positive else len(retained) / len(final_positive)
    )
    checks = (
        # F is expected to be evidence-sparse.  Its role here is to establish
        # positive-set absence, so negative/abstain switching does not make
        # that baseline unreliable.  Trainable goals below still require
        # exact three-class repeat agreement.
        GateCheck(
            "final_goal_positive_membership_agreement_at_or_above_floor",
            final_goal.summary.positive_membership_agreement_fraction
            >= config.minimum_final_positive_membership_agreement,
            final_goal.summary.positive_membership_agreement_fraction,
            f">= {config.minimum_final_positive_membership_agreement}",
        ),
        *_reliability_checks(
            "broadened_goal",
            broadened_goal,
            minimum_agreement=config.minimum_repeat_agreement,
            maximum_abstain=config.maximum_pass_abstain_fraction,
        ),
        GateCheck(
            "broadened_minus_final_positive_fraction_at_or_above_gain_floor",
            interval.point_difference >= config.minimum_positive_fraction_gain,
            interval.point_difference,
            f">= {config.minimum_positive_fraction_gain}",
        ),
        GateCheck(
            "paired_bootstrap_lower_bound_above_floor",
            interval.lower > config.bootstrap_lower_bound_must_exceed,
            interval.lower,
            f"> {config.bootstrap_lower_bound_must_exceed}",
        ),
        GateCheck(
            "final_positive_retention_at_or_above_floor",
            retention >= config.minimum_final_positive_retention,
            retention,
            f">= {config.minimum_final_positive_retention}",
        ),
        GateCheck(
            "semantic_broadening_direction_passed",
            semantic_direction_passed,
            semantic_direction_passed,
            "is true",
        ),
    )
    return InitialBroadeningAudit(
        final_goal_cell=final_goal,
        broadened_goal_cell=broadened_goal,
        broadening_interval=interval,
        final_positive_count=len(final_positive),
        retained_final_positive_count=len(retained),
        broadened_positive_outside_final_count=len(
            broad_positive.difference(final_positive)
        ),
        final_positive_retention_fraction=retention,
        semantic_direction_passed=semantic_direction_passed,
        decision=GateDecision(tuple(checks)),
    )


def empirical_nestedness_and_pruning(
    parent_goal: QMatrixCell,
    tighter_goal: QMatrixCell,
) -> EmpiricalSubsetAudit:
    """Measure empirical subset retention on the same current-generator cohort."""

    if parent_goal.generator_id != tighter_goal.generator_id:
        raise ValueError("subset audit requires one generator")
    if parent_goal.goal_id == tighter_goal.goal_id:
        raise ValueError("parent and tighter goal IDs must differ")
    if parent_goal.summary.candidate_ids != tighter_goal.summary.candidate_ids:
        raise ValueError("subset audit requires identical candidate cohorts")

    parent_by_id = parent_goal.summary.by_candidate_id()
    tighter_by_id = tighter_goal.summary.by_candidate_id()
    if any(
        parent_by_id[candidate_id].stratum
        != tighter_by_id[candidate_id].stratum
        for candidate_id in parent_by_id
    ):
        raise ValueError("subset audit requires fixed candidate strata")
    parent_positive = {
        candidate_id
        for candidate_id, row in parent_by_id.items()
        if row.stable_label is StableLabel.POSITIVE
    }
    tighter_positive = {
        candidate_id
        for candidate_id, row in tighter_by_id.items()
        if row.stable_label is StableLabel.POSITIVE
    }
    retained = parent_positive.intersection(tighter_positive)
    outside = tighter_positive.difference(parent_positive)
    removed = parent_positive.difference(tighter_positive)
    return EmpiricalSubsetAudit(
        generator_id=parent_goal.generator_id,
        parent_goal_id=parent_goal.goal_id,
        tighter_goal_id=tighter_goal.goal_id,
        parent_stable_positive_count=len(parent_positive),
        tighter_stable_positive_count=len(tighter_positive),
        retained_positive_count=len(retained),
        tighter_positive_outside_parent_count=len(outside),
        parent_positive_removed_count=len(removed),
        nestedness_fraction=(
            len(retained) / len(tighter_positive) if tighter_positive else None
        ),
        pruning_fraction=(
            len(removed) / len(parent_positive) if parent_positive else None
        ),
    )


def audit_current_goal_movement(
    previous: QMatrixCell,
    current: QMatrixCell,
    *,
    config: CurrentGoalGateConfig | None = None,
) -> CurrentGoalMovementAudit:
    """Decide whether one update enriched and saturated its frozen goal."""

    if config is None:
        config = CurrentGoalGateConfig()
    if previous.goal_id != current.goal_id:
        raise ValueError("current-goal movement requires the same frozen goal")
    if previous.generator_id == current.generator_id:
        raise ValueError("current-goal movement requires distinct generators")
    interval = stratified_bootstrap_rate_difference(
        previous,
        current,
        replicates=config.bootstrap_replicates,
        confidence_level=config.bootstrap_confidence_level,
        seed=config.bootstrap_seed,
    )
    checks = (
        *_reliability_checks(
            "previous",
            previous,
            minimum_agreement=config.minimum_repeat_agreement,
            maximum_abstain=config.maximum_pass_abstain_fraction,
        ),
        *_reliability_checks(
            "current",
            current,
            minimum_agreement=config.minimum_repeat_agreement,
            maximum_abstain=config.maximum_pass_abstain_fraction,
        ),
        GateCheck(
            "current_goal_positive_fraction_at_or_above_saturation_floor",
            current.q >= config.minimum_current_positive_fraction,
            current.q,
            f">= {config.minimum_current_positive_fraction}",
        ),
        GateCheck(
            "current_minus_previous_positive_fraction_at_or_above_gain_floor",
            interval.point_difference >= config.minimum_positive_fraction_gain,
            interval.point_difference,
            f">= {config.minimum_positive_fraction_gain}",
        ),
        GateCheck(
            "bootstrap_lower_bound_above_floor",
            interval.lower > config.bootstrap_lower_bound_must_exceed,
            interval.lower,
            f"> {config.bootstrap_lower_bound_must_exceed}",
        ),
    )
    return CurrentGoalMovementAudit(
        previous_cell=previous,
        current_cell=current,
        enrichment_interval=interval,
        decision=GateDecision(tuple(checks)),
    )


def audit_newly_unlocked_goal(
    previous_tighter_goal: QMatrixCell,
    current_tighter_goal: QMatrixCell,
    current_parent_goal: QMatrixCell,
    *,
    semantic_direction_passed: bool,
    config: NewlyUnlockedGoalGateConfig | None = None,
) -> NewlyUnlockedGoalAudit:
    """Accept a tighter goal only when generator movement newly unlocked it."""

    if config is None:
        config = NewlyUnlockedGoalGateConfig()
    if previous_tighter_goal.goal_id != current_tighter_goal.goal_id:
        raise ValueError("enrichment comparison must use one frozen tighter goal")
    if previous_tighter_goal.generator_id == current_tighter_goal.generator_id:
        raise ValueError("newly-unlocked audit requires distinct generators")
    if current_parent_goal.generator_id != current_tighter_goal.generator_id:
        raise ValueError("parent and tighter goals must use the current generator")
    if not isinstance(semantic_direction_passed, bool):
        raise TypeError("semantic_direction_passed must be boolean")

    interval = stratified_bootstrap_rate_difference(
        previous_tighter_goal,
        current_tighter_goal,
        replicates=config.bootstrap_replicates,
        confidence_level=config.bootstrap_confidence_level,
        seed=config.bootstrap_seed,
    )
    subset = empirical_nestedness_and_pruning(
        current_parent_goal,
        current_tighter_goal,
    )
    current_summary = current_tighter_goal.summary
    nestedness_passed = (
        subset.nestedness_fraction is not None
        and subset.nestedness_fraction >= config.minimum_nestedness_fraction
    )
    pruning_passed = (
        subset.pruning_fraction is not None
        and subset.pruning_fraction >= config.minimum_pruning_fraction
    )
    checks = (
        *_reliability_checks(
            "previous_tighter_goal",
            previous_tighter_goal,
            minimum_agreement=config.minimum_repeat_agreement,
            maximum_abstain=config.maximum_pass_abstain_fraction,
        ),
        *_reliability_checks(
            "current_tighter_goal",
            current_tighter_goal,
            minimum_agreement=config.minimum_repeat_agreement,
            maximum_abstain=config.maximum_pass_abstain_fraction,
        ),
        *_reliability_checks(
            "current_parent_goal",
            current_parent_goal,
            minimum_agreement=config.minimum_repeat_agreement,
            maximum_abstain=config.maximum_pass_abstain_fraction,
        ),
        GateCheck(
            "current_tighter_goal_has_enough_stable_positives",
            current_summary.stable_positive_count
            >= config.minimum_current_stable_positive_count,
            current_summary.stable_positive_count,
            f">= {config.minimum_current_stable_positive_count}",
        ),
        GateCheck(
            "current_tighter_goal_has_enough_stable_negatives",
            current_summary.stable_negative_count
            >= config.minimum_current_stable_negative_count,
            current_summary.stable_negative_count,
            f">= {config.minimum_current_stable_negative_count}",
        ),
        GateCheck(
            "current_minus_previous_tighter_goal_fraction_at_or_above_gain_floor",
            interval.point_difference >= config.minimum_positive_fraction_gain,
            interval.point_difference,
            f">= {config.minimum_positive_fraction_gain}",
        ),
        GateCheck(
            "bootstrap_lower_bound_above_floor",
            interval.lower > config.bootstrap_lower_bound_must_exceed,
            interval.lower,
            f"> {config.bootstrap_lower_bound_must_exceed}",
        ),
        GateCheck(
            "empirical_nestedness_at_or_above_floor",
            nestedness_passed,
            subset.nestedness_fraction,
            f">= {config.minimum_nestedness_fraction}",
        ),
        GateCheck(
            "empirical_pruning_at_or_above_floor",
            pruning_passed,
            subset.pruning_fraction,
            f">= {config.minimum_pruning_fraction}",
        ),
        GateCheck(
            "independent_semantic_direction_audit_passed",
            semantic_direction_passed,
            semantic_direction_passed,
            "is true",
        ),
    )
    return NewlyUnlockedGoalAudit(
        previous_tighter_goal_cell=previous_tighter_goal,
        current_tighter_goal_cell=current_tighter_goal,
        current_parent_goal_cell=current_parent_goal,
        enrichment_interval=interval,
        empirical_subset=subset,
        semantic_direction_passed=semantic_direction_passed,
        decision=GateDecision(tuple(checks)),
    )


def _normalize_raw_label(value: object) -> str:
    if isinstance(value, Enum):
        value = value.value
    if not isinstance(value, str):
        raise TypeError("categorical labels must be strings or string-valued enums")
    normalized = value.strip().lower()
    if normalized not in RAW_LABELS:
        raise ValueError("categorical label must be positive, negative, or abstain")
    return normalized


def _normalize_stratum(value: object) -> str:
    if isinstance(value, bool) or value is None:
        raise TypeError("strata must be non-empty strings or finite numbers")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("numeric strata must be finite")
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("strata must be non-empty")
    return normalized


def _counts(labels: Iterable[str]) -> CategoricalPassCounts:
    rows = tuple(labels)
    if not rows:
        raise ValueError("cannot count an empty label pass")
    if any(label not in RAW_LABELS for label in rows):
        raise ValueError("raw categorical label is invalid")
    return CategoricalPassCounts(
        sample_count=len(rows),
        positive_count=sum(label == "positive" for label in rows),
        negative_count=sum(label == "negative" for label in rows),
        abstain_count=sum(label == "abstain" for label in rows),
    )


def _binary_by_stratum(summary: StableCategoricalSummary) -> dict[str, tuple[int, ...]]:
    grouped: dict[str, list[int]] = {}
    for row in summary.outcomes:
        grouped.setdefault(row.stratum, []).append(int(row.stable_positive_indicator))
    return {name: tuple(values) for name, values in grouped.items()}


def _reliability_checks(
    prefix: str,
    cell: QMatrixCell,
    *,
    minimum_agreement: float,
    maximum_abstain: float,
) -> tuple[GateCheck, GateCheck]:
    summary = cell.summary
    return (
        GateCheck(
            f"{prefix}_repeat_agreement_at_or_above_floor",
            summary.agreement_fraction >= minimum_agreement,
            summary.agreement_fraction,
            f">= {minimum_agreement}",
        ),
        GateCheck(
            f"{prefix}_maximum_pass_abstain_fraction_at_or_below_ceiling",
            summary.maximum_pass_abstain_fraction <= maximum_abstain,
            summary.maximum_pass_abstain_fraction,
            f"<= {maximum_abstain}",
        ),
    )


def _validate_gate_config(
    *,
    fractions: Mapping[str, float],
    lower_bound: float,
    replicates: int,
    confidence_level: float,
    seed: int,
) -> None:
    for name, value in fractions.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be numeric")
        if not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"{name} must lie in [0, 1]")
    if isinstance(lower_bound, bool) or not isinstance(lower_bound, (int, float)):
        raise TypeError("bootstrap lower-bound floor must be numeric")
    if not math.isfinite(float(lower_bound)) or not -1 <= float(lower_bound) <= 1:
        raise ValueError("bootstrap lower-bound floor must lie in [-1, 1]")
    _validate_bootstrap_parameters(replicates, confidence_level, seed)


def _validate_bootstrap_parameters(
    replicates: int,
    confidence_level: float,
    seed: int,
) -> None:
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates <= 0:
        raise ValueError("bootstrap replicates must be a positive integer")
    if (
        isinstance(confidence_level, bool)
        or not isinstance(confidence_level, (int, float))
        or not math.isfinite(float(confidence_level))
        or not 0 < float(confidence_level) < 1
    ):
        raise ValueError("bootstrap confidence level must lie in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")


def _percentile(ordered_values: Sequence[float], probability: float) -> float:
    if not ordered_values:
        raise ValueError("percentile requires at least one value")
    position = probability * (len(ordered_values) - 1)
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return ordered_values[lower_index]
    weight = position - lower_index
    return (
        ordered_values[lower_index] * (1 - weight)
        + ordered_values[upper_index] * weight
    )


__all__ = [
    "BootstrapInterval",
    "CategoricalPassCounts",
    "CurrentGoalGateConfig",
    "CurrentGoalMovementAudit",
    "EmpiricalSubsetAudit",
    "GateCheck",
    "GateDecision",
    "InitialBroadeningAudit",
    "InitialBroadeningGateConfig",
    "NewlyUnlockedGoalAudit",
    "NewlyUnlockedGoalGateConfig",
    "QMatrixCell",
    "StableCategoricalOutcome",
    "StableCategoricalSummary",
    "StableLabel",
    "audit_current_goal_movement",
    "audit_initial_broadening",
    "audit_newly_unlocked_goal",
    "build_q_matrix_cell",
    "empirical_nestedness_and_pruning",
    "q_matrix_table",
    "stable_categorical_labels",
    "stratified_bootstrap_rate_difference",
]
