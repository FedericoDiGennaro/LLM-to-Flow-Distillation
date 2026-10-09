"""Single-harness control loop for adaptive semantic goal annealing.

The first prototype deliberately uses an empirical exponential tilt over a
frozen peptide pool instead of claiming to fine-tune PepDFM.  The semantic
controller and generator update are separate so the latter can later be
replaced by real PepDFM guidance without changing the experiment protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from enum import Enum
import hashlib
import json
import math
import random
from typing import Any, Mapping, Protocol, Sequence


CANONICAL_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")


class SemanticLabel(str, Enum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    ABSTAIN = "abstain"


class GoalAction(str, Enum):
    TIGHTEN = "tighten"
    HOLD = "hold"
    REACHED_FINAL = "reached_final"


@dataclass(frozen=True)
class PeptideCandidate:
    candidate_id: str
    sequence: str
    developability_evidence: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not self.sequence or any(
            residue not in CANONICAL_AMINO_ACIDS for residue in self.sequence
        ):
            raise ValueError("sequence must contain canonical amino-acid letters")
        evidence = dict(self.developability_evidence)
        if len(evidence) != len(self.developability_evidence):
            raise ValueError("developability evidence contains duplicate names")
        allowed_names = {
            "non_hemolysis",
            "non_fouling",
            "solubility",
            "half_life",
        }
        if set(evidence).difference(allowed_names):
            raise ValueError("developability evidence contains an unknown property")
        allowed_bands = {"lower_third", "middle_third", "upper_third"}
        if any(value not in allowed_bands for value in evidence.values()):
            raise ValueError("developability evidence contains an unknown qualitative band")
        if evidence and set(evidence) != allowed_names:
            raise ValueError("developability evidence must cover all four properties")


@dataclass(frozen=True)
class SampleJudgment:
    candidate_id: str
    current_goal_label: SemanticLabel
    final_goal_label: SemanticLabel | None
    current_goal_confidence: float
    final_goal_confidence: float | None
    current_goal_rationale: str
    final_goal_rationale: str | None
    current_goal_evidence_for: str | None = None
    current_goal_evidence_against: str | None = None
    current_goal_critical_unknowns: str | None = None
    current_goal_initial_label: SemanticLabel | None = None
    current_goal_self_critique: str | None = None

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if not math.isfinite(self.current_goal_confidence) or not (
            0 <= self.current_goal_confidence <= 1
        ):
            raise ValueError("current confidence must be finite and in [0, 1]")
        if not self.current_goal_rationale.strip():
            raise ValueError("current_goal_rationale must be non-empty")
        final_values = (
            self.final_goal_label,
            self.final_goal_confidence,
            self.final_goal_rationale,
        )
        if any(value is None for value in final_values) and not all(
            value is None for value in final_values
        ):
            raise ValueError("final-goal judgment fields must be all present or all absent")
        if self.final_goal_confidence is not None and (
            not math.isfinite(self.final_goal_confidence)
            or not 0 <= self.final_goal_confidence <= 1
        ):
            raise ValueError("final confidence must be finite and in [0, 1]")
        if self.final_goal_rationale is not None and not (
            self.final_goal_rationale.strip()
        ):
            raise ValueError("final_goal_rationale must be non-empty when present")
        critique_fields = (
            self.current_goal_evidence_for,
            self.current_goal_evidence_against,
            self.current_goal_critical_unknowns,
            self.current_goal_self_critique,
        )
        for value in critique_fields:
            if value is not None and not value.strip():
                raise ValueError("structured rationale fields must be non-empty")


@dataclass(frozen=True)
class JudgeResult:
    current_goal: str
    final_goal: str
    judgments: tuple[SampleJudgment, ...]
    goal_semantic_closeness_to_final: float | None
    goal_closeness_rationale: str
    request_hash: str | None = None
    request_hashes: tuple[str, ...] = ()
    usage: Any = None
    model_version: str | None = None
    thought_summary: str | None = None

    def __post_init__(self) -> None:
        if not self.current_goal.strip() or not self.final_goal.strip():
            raise ValueError("goals must be non-empty")
        if not self.judgments:
            raise ValueError("a judge result requires at least one judgment")
        ids = [item.candidate_id for item in self.judgments]
        if len(ids) != len(set(ids)):
            raise ValueError("judge result contains duplicate candidate IDs")
        if self.goal_semantic_closeness_to_final is not None and (
            not math.isfinite(self.goal_semantic_closeness_to_final)
            or not 0 <= self.goal_semantic_closeness_to_final <= 1
        ):
            raise ValueError("goal semantic closeness must be finite and in [0, 1]")
        if not self.goal_closeness_rationale.strip():
            raise ValueError("goal_closeness_rationale must be non-empty")
        if self.current_goal == self.final_goal:
            for item in self.judgments:
                if item.final_goal_label is None or (
                    item.current_goal_label != item.final_goal_label
                ):
                    raise ValueError(
                        "current and final labels must agree in an isolated "
                        "final-goal judgment"
                    )
        for request_hash in self.request_hashes:
            if len(request_hash) != 64:
                raise ValueError("request hashes must be SHA-256 digests")


@dataclass(frozen=True)
class SampleStatistics:
    sample_count: int
    positive_count: int
    negative_count: int
    abstain_count: int
    positive_fraction_all: float
    positive_fraction_decided: float | None
    abstain_fraction: float
    final_positive_count: int | None
    final_negative_count: int | None
    final_abstain_count: int | None
    final_positive_fraction_all: float | None
    final_positive_fraction_decided: float | None
    llm_goal_semantic_closeness_to_final: float | None

    @classmethod
    def from_judge_result(cls, result: JudgeResult) -> "SampleStatistics":
        def counts(labels: Sequence[SemanticLabel]) -> tuple[int, int, int]:
            return (
                sum(label == SemanticLabel.POSITIVE for label in labels),
                sum(label == SemanticLabel.NEGATIVE for label in labels),
                sum(label == SemanticLabel.ABSTAIN for label in labels),
            )

        current = [item.current_goal_label for item in result.judgments]
        final = [item.final_goal_label for item in result.judgments]
        positive, negative, abstain = counts(current)
        total = len(current)
        decided = positive + negative
        if all(label is None for label in final):
            final_positive = None
            final_negative = None
            final_abstain = None
            final_positive_fraction_all = None
            final_positive_fraction_decided = None
        elif any(label is None for label in final):
            raise ValueError("judge result mixes present and absent final-goal labels")
        else:
            present_final = [label for label in final if label is not None]
            final_positive, final_negative, final_abstain = counts(present_final)
            final_decided = final_positive + final_negative
            final_positive_fraction_all = final_positive / total
            final_positive_fraction_decided = (
                final_positive / final_decided if final_decided else None
            )
        return cls(
            sample_count=total,
            positive_count=positive,
            negative_count=negative,
            abstain_count=abstain,
            positive_fraction_all=positive / total,
            positive_fraction_decided=(positive / decided if decided else None),
            abstain_fraction=abstain / total,
            final_positive_count=final_positive,
            final_negative_count=final_negative,
            final_abstain_count=final_abstain,
            final_positive_fraction_all=final_positive_fraction_all,
            final_positive_fraction_decided=final_positive_fraction_decided,
            llm_goal_semantic_closeness_to_final=(
                result.goal_semantic_closeness_to_final
            ),
        )


@dataclass(frozen=True)
class GoalDirectionCertificate:
    """Structured claim that one prompt edit preserves the task direction."""

    operation: str
    preserves_anticancer_application: bool
    preserves_low_toxicity_direction: bool
    weakens_or_restores_only_evidence_or_specificity: bool
    uses_proxy_waypoint: bool
    proxy_waypoint_is_explicitly_nonfinal: bool
    bridge_to_final_goal_explained: bool
    remaining_final_constraints_identified: bool
    restores_at_least_one_deferred_final_constraint: bool
    final_target_anchor_accepted: bool
    broader_near_miss_anchor_accepted: bool
    toxic_membrane_disruptor_not_mistaken_for_final_success: bool
    irrelevant_peptide_anchor_rejected: bool
    failure_mode: str
    bridge_explanation: str
    next_restriction_plan: str

    def __post_init__(self) -> None:
        if not all(
            value.strip()
            for value in (
                self.operation,
                self.failure_mode,
                self.bridge_explanation,
                self.next_restriction_plan,
            )
        ):
            raise ValueError("direction-certificate text must be non-empty")
        for name in (
            "preserves_anticancer_application",
            "preserves_low_toxicity_direction",
            "weakens_or_restores_only_evidence_or_specificity",
            "uses_proxy_waypoint",
            "proxy_waypoint_is_explicitly_nonfinal",
            "bridge_to_final_goal_explained",
            "remaining_final_constraints_identified",
            "restores_at_least_one_deferred_final_constraint",
            "final_target_anchor_accepted",
            "broader_near_miss_anchor_accepted",
            "toxic_membrane_disruptor_not_mistaken_for_final_success",
            "irrelevant_peptide_anchor_rejected",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"direction certificate {name} must be boolean")


@dataclass(frozen=True)
class GoalProposal:
    proposed_goal: str
    rationale: str
    containment_explanation: str
    contains_final_goal_as_subset: bool
    strictly_broader_than_current_goal: bool
    expected_positive_fraction: float
    request_hash: str | None = None
    usage: Any = None
    model_version: str | None = None
    thought_summary: str | None = None
    observable_positive_example_candidate_id: str | None = None
    observable_negative_example_candidate_id: str | None = None
    non_universal_counterexample_candidate_id: str | None = None
    positive_evidence_definition: str | None = None
    negative_evidence_definition: str | None = None
    self_critique: str | None = None
    direction_certificate: GoalDirectionCertificate | None = None

    def __post_init__(self) -> None:
        for value in (
            self.proposed_goal,
            self.rationale,
            self.containment_explanation,
        ):
            if not value.strip():
                raise ValueError("goal proposal text fields must be non-empty")
        if not self.contains_final_goal_as_subset:
            raise ValueError("a relaxation must claim to contain the final goal")
        if not self.strictly_broader_than_current_goal:
            raise ValueError("a relaxation must claim to be strictly broader")
        if not math.isfinite(self.expected_positive_fraction) or not (
            0 <= self.expected_positive_fraction <= 1
        ):
            raise ValueError("expected_positive_fraction must be in [0, 1]")
        optional_text = (
            self.observable_positive_example_candidate_id,
            self.observable_negative_example_candidate_id,
            self.non_universal_counterexample_candidate_id,
            self.positive_evidence_definition,
            self.negative_evidence_definition,
            self.self_critique,
        )
        for value in optional_text:
            if value is not None and not value.strip():
                raise ValueError("optional relaxation evidence must be non-empty")


@dataclass(frozen=True)
class GoalDecision:
    action: GoalAction
    next_goal: str
    rationale: str
    containment_explanation: str
    contains_final_goal_as_subset: bool
    strictly_narrower_than_current_goal: bool
    reaches_final_goal: bool
    next_goal_semantic_closeness_to_final: float
    request_hash: str | None = None
    usage: Any = None
    model_version: str | None = None
    thought_summary: str | None = None
    direction_certificate: GoalDirectionCertificate | None = None

    def __post_init__(self) -> None:
        for value in (self.next_goal, self.rationale, self.containment_explanation):
            if not value.strip():
                raise ValueError("goal decision text fields must be non-empty")
        if not self.contains_final_goal_as_subset:
            raise ValueError("the next goal must claim to contain the final goal")
        if self.action in {GoalAction.TIGHTEN, GoalAction.REACHED_FINAL} and not (
            self.strictly_narrower_than_current_goal
        ):
            raise ValueError("a tightening decision must claim to be strictly narrower")
        if not math.isfinite(self.next_goal_semantic_closeness_to_final) or not (
            0 <= self.next_goal_semantic_closeness_to_final <= 1
        ):
            raise ValueError("next-goal semantic closeness must be in [0, 1]")
