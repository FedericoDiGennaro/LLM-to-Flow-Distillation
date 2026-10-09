"""Structured controller for the V13 geometric semantic-goal protocol."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable, Literal, Mapping, Sequence

from adm.preferences.artifacts import PreferenceArtifactStore
from adm.preferences.clients import StructuredLLMClient
from domains.peptides.sequence_evidence import (
    candidate_evidence_payload,
    evidence_tool_definition,
    frozen_p0_developability_evidence_tool_definition,
)
from domains.peptides.agent import StructuredSemanticGoalAgent
from domains.peptides.cell_penetration.goals import (
    FrozenGoalLengthPolicy,
    GEOMETRIC_V13_CONTROLLER_PROMPT,
    GEOMETRIC_V13_CORRECTION_ADDENDUM,
    GoalCorrectionRequest,
    response_schema_for_action,
)
from domains.peptides.types import PeptideCandidate
from domains.peptides.cell_penetration.judgments import V4JudgeResult
from domains.peptides.cell_penetration.frontier import FrontierGoal


@dataclass(frozen=True)
class GeometricFrontierProposal:
    """One goal proposal plus a repeated campaign-frozen length policy."""

    action: str
    goal: FrontierGoal
    length_policy: FrozenGoalLengthPolicy
    proposal_attempt: int
    raw_response: Mapping[str, Any]
    metadata: Mapping[str, Any]
    positive_witness_candidate_id: str | None
    negative_witness_candidate_id: str | None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "goal": asdict(self.goal),
            "length_policy": self.length_policy.model_dump(mode="json"),
            "proposal_attempt": self.proposal_attempt,
            "raw_response": dict(self.raw_response),
            "metadata": dict(self.metadata),
            "positive_witness_candidate_id": self.positive_witness_candidate_id,
            "negative_witness_candidate_id": self.negative_witness_candidate_id,
        }


def _result_by_id(result: V4JudgeResult) -> dict[str, Any]:
    return {row.candidate_id: row for row in result.judgments}


def _visible_working_cohort(
    samples: Sequence[PeptideCandidate], result: V4JudgeResult
) -> tuple[list[dict[str, Any]], dict[str, PeptideCandidate]]:
    """Project exactly one 100-candidate cohort into provider-safe item IDs."""

    if len(samples) != 100:
        raise ValueError("V13 controller requires exactly 100 working candidates")
    by_id = {sample.candidate_id: sample for sample in samples}
    if len(by_id) != len(samples):
        raise ValueError("working cohort contains duplicate candidate IDs")
    judgments = _result_by_id(result)
    if set(by_id) != set(judgments):
        raise ValueError("working cohort and categorical judgments differ")
    if any(not 6 <= len(sample.sequence) <= 49 for sample in samples):
        raise ValueError("V13 working peptide length must lie in [6, 49]")

    visible: list[dict[str, Any]] = []
    item_to_sample: dict[str, PeptideCandidate] = {}
    for index, sample in enumerate(sorted(samples, key=lambda row: row.candidate_id), 1):
        item_id = f"item-{index:03d}"
        judgment = judgments[sample.candidate_id]
        item_to_sample[item_id] = sample
        visible.append(
            {
                "item_id": item_id,
                "sequence": sample.sequence,
                "length": len(sample.sequence),
                **candidate_evidence_payload(
                    sample.sequence, sample.developability_evidence
                ),
                "current_goal_label": judgment.label.value,
                "current_goal_rationale": judgment.rationale,
                "current_goal_support": judgment.goal_specific_support,
                "contraindications": judgment.contraindications,
                "critical_unknowns": judgment.critical_unknowns,
            }
        )
    return visible, item_to_sample


class GeometricFrontierController(StructuredSemanticGoalAgent):
    """V13 controller with one initial proposal and at most three corrections.

    The class builds requests only from a single 100-candidate working cohort
    per distribution.  The runner, rather than this role, decides whether the
    observed categorical mix passes and whether another correction is allowed.
    """

    def __init__(
        self,
        *,
        client: StructuredLLMClient,
        artifact_store: PreferenceArtifactStore,
        model: str,
        reasoning_effort: str,
        max_output_tokens: int,
        max_provider_calls: int,
        campaign_length_policy: FrozenGoalLengthPolicy | Mapping[str, Any] | None = None,
        initial_provider_calls: int = 0,
        experiment_context: Mapping[str, Any] | None = None,
        system_prompt: str = GEOMETRIC_V13_CONTROLLER_PROMPT,
        correction_addendum: str = GEOMETRIC_V13_CORRECTION_ADDENDUM,
        response_schema_resolver: Callable[[str], type] = response_schema_for_action,
        external_evidence_contract: Mapping[str, Any] | None = None,
        developability_evidence_tool: Mapping[str, Any] | None = None,
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
        )
        self.experiment_context = dict(experiment_context or {})
        self.system_prompt = system_prompt
        self.correction_addendum = correction_addendum
        self.response_schema_resolver = response_schema_resolver
        self.external_evidence_contract = dict(
            external_evidence_contract
            or {
                "labelled_peptide_dataset_visible": False,
                "reference_peptides_visible": False,
                "database_lookup_visible": False,
                "internet_visible": False,
                "target_specific_activity_predictor_visible": False,
                "use_pretrained_domain_knowledge": True,
            }
        )
        self.developability_evidence_tool = dict(
            developability_evidence_tool
            or frozen_p0_developability_evidence_tool_definition()
        )
        self.campaign_length_policy = (
            None
            if campaign_length_policy is None
            else FrozenGoalLengthPolicy.model_validate(campaign_length_policy)
        )

    def propose(
        self,
        *,
        action: Literal["broaden", "tighten", "reached_final"],
        final_goal: str,
        current_goal: FrontierGoal,
        current_samples: Sequence[PeptideCandidate],
        current_result: V4JudgeResult,
        generator_id: str,
        goal_history: Sequence[Mapping[str, Any]] = (),
        update_count: int = 0,
        proposal_attempt: int = 1,
        correction: GoalCorrectionRequest | Mapping[str, Any] | None = None,
        goal_entry_samples: Sequence[PeptideCandidate] | None = None,
        goal_entry_result: V4JudgeResult | None = None,
        goal_entry_generator_id: str | None = None,
        final_goal_feedback: Mapping[str, Any] | None = None,
        current_repeat_result: V4JudgeResult | None = None,
    ) -> GeometricFrontierProposal:
        if not final_goal.strip() or current_goal.final_goal.strip() != final_goal.strip():
            raise ValueError("controller final-goal binding differs")
        if not 1 <= proposal_attempt <= 4:
            raise ValueError("proposal_attempt must lie in [1, 4]")

        correction_value: GoalCorrectionRequest | None = None
        if proposal_attempt == 1:
            if correction is not None:
                raise ValueError("the initial proposal cannot carry correction feedback")
        else:
            if correction is None:
                raise ValueError("a corrected proposal requires host feedback")
            correction_value = GoalCorrectionRequest.model_validate(correction)
            if correction_value.proposal_attempt != proposal_attempt:
                raise ValueError("correction feedback proposal attempt differs")

        current_visible, current_map = _visible_working_cohort(
            current_samples, current_result
        )
        if final_goal_feedback is not None and getattr(self, "repaired_feedback", False):
            from domains.peptides.cell_penetration.controls import link_final_feedback
            final_goal_feedback = link_final_feedback(final_goal_feedback, current_map)
        if final_goal_feedback is not None:
            if current_repeat_result is None:
                raise ValueError("F-informed controller requires both source-goal passes")
            repeats = _result_by_id(current_repeat_result)
            if set(repeats) != {s.candidate_id for s in current_samples}:
                raise ValueError("source repeat coverage differs")
            for row in current_visible:
                repeat = repeats[current_map[row["item_id"]].candidate_id]
                row["current_goal_repeat_label"] = repeat.label.value
                row["current_goal_repeat_rationale"] = repeat.rationale
                row["stable_current_goal_label"] = (
                    repeat.label.value if repeat.label.value == row["current_goal_label"] else "unstable"
                )
        goal_entry_visible: list[dict[str, Any]] | None = None
        if action == "tighten":
            if (
                goal_entry_samples is None
                or goal_entry_result is None
                or not goal_entry_generator_id
            ):
                raise ValueError(
                    "tightening requires the goal-entry and current working cohorts"
                )
            goal_entry_visible, _ = _visible_working_cohort(
                goal_entry_samples, goal_entry_result
            )
        elif any(
            value is not None
            for value in (
                goal_entry_samples,
                goal_entry_result,
                goal_entry_generator_id,
            )
        ):
            raise ValueError("only tightening may expose a goal-entry cohort")

        system_prompt = self.system_prompt
        if correction_value is not None:
            system_prompt += self.correction_addendum
        request = self._request(
            schema_id=f"pepdfm-geometric-v13-{action}-attempt-{proposal_attempt}",
            system_prompt=system_prompt,
            user_payload={
                "requested_action_chosen_by_host": action,
                "immutable_final_goal": final_goal,
                "current_goal": asdict(current_goal),
                "current_generator_id": generator_id,
                "completed_generator_updates": update_count,
                "accepted_goal_history": [dict(item) for item in goal_history],
                "experiment_context": self.experiment_context,
                "geometric_search_contract": {
                    "positive_set_order": (
                        "Pos(F) subset Pos(G_m) subset ... subset Pos(G_2) subset Pos(G_1)"
                        if getattr(self, "nested_final_goal_chain", False) else
                        "Operational Pos(G_m) subset ... subset Pos(G_1); "
                        "universal biological containment of F is not asserted"
                        if final_goal_feedback is not None else
                        "Pos(F) subset Pos(G_m) subset ... subset Pos(G_2) subset Pos(G_1)"
                    ),
                    "generator_update": "p_k -- train on G_k --> p_(k+1)",
                    "initial_goal_rule": "choose the weakest meaningful reachable rung",
                    "tightening_rule": (
                        "tighten only after G_k enrichment; use the smallest newly "
                        "reachable discriminative increment"
                    ),
                    "maximum_corrections_after_initial_proposal": 3,
                },
                "proposal_attempt": proposal_attempt,
                "correction_feedback": (
                    None
                    if correction_value is None
                    else correction_value.model_dump(mode="json")
                ),
                "working_cohort_size": 100,
                "current_working_cohort": current_visible,
                "goal_entry_generator_id": goal_entry_generator_id,
                "goal_entry_working_cohort": goal_entry_visible,
                "same_cohort_is_used_for_goal_selection_and_training": True,
                "no_independent_gate_cohort": True,
                "generic_sequence_evidence_tool": evidence_tool_definition(),
                "developability_evidence_tool": (
                    self.developability_evidence_tool
                ),
                "external_evidence_contract": self.external_evidence_contract,
                "length_policy_contract": {
                    "campaign_frozen_length_policy": (
                        None
                        if self.campaign_length_policy is None
                        else self.campaign_length_policy.model_dump(mode="json")
                    ),
                    "selected_once_before_p0": self.campaign_length_policy is not None,
                    "freeze_policy_for_entire_campaign": True,
                    "proposal_must_repeat_exact_policy": (
                        self.campaign_length_policy is not None
                    ),
                    "length_is_not_goal_evidence": True,
                },
                "return_exactly_one_proposal": True,
                "cardinal_scores_forbidden": True,
                **({"final_goal_diagnostics": dict(final_goal_feedback)}
                   if final_goal_feedback is not None else {}),
            },
            logical_context={
                "action": action,
                "generator_id": generator_id,
                "goal_entry_generator_id": goal_entry_generator_id,
                "current_goal": current_goal.frontier_criterion,
                "update_count": update_count,
                "proposal_attempt": proposal_attempt,
            },
        )
        schema = self.response_schema_resolver(action)
        self.ensure_provider_capacity(((request, schema),))
        values, metadata = self._execute(request, schema)

        if int(values["proposal_attempt_acknowledged"]) != proposal_attempt:
            metadata = {**metadata, "proposal_attempt_echo_mismatch": {
                "expected": proposal_attempt,
                "returned": int(values["proposal_attempt_acknowledged"]),
                "handling": "request_bound_host_attempt_is_authoritative",
            }}
        criterion = str(values["frontier_criterion"]).strip()
        if correction_value is not None and criterion.casefold() == (
            correction_value.prior_frontier_criterion.strip().casefold()
        ):
            raise ValueError("corrected proposal repeated the prior criterion")
        length_policy = FrozenGoalLengthPolicy.model_validate(values["length_policy"])

        if criterion == final_goal.strip():
            if action != "reached_final":
                raise ValueError("a non-final action returned the exact final goal")
            return GeometricFrontierProposal(
                action=action,
                goal=FrontierGoal.exact_final(final_goal),
                length_policy=length_policy,
                proposal_attempt=proposal_attempt,
                raw_response=dict(values),
                metadata=dict(metadata),
                positive_witness_candidate_id=None,
                negative_witness_candidate_id=None,
            )
        if action == "reached_final":
            raise ValueError("reached_final did not return the immutable final text")

        positive_item = str(values["positive_witness_item_id"])
        negative_item = str(values["negative_witness_item_id"])
        if positive_item not in current_map or negative_item not in current_map:
            raise ValueError("proposal witness is outside the current working cohort")
        positive = current_map[positive_item]
        negative = current_map[negative_item]
        goal = FrontierGoal(
            final_goal=final_goal,
            frontier_criterion=criterion,
            bridge_to_final=str(values["bridge_to_final"]),
            is_final=False,
            positive_evidence_definition=str(values["positive_evidence_definition"]),
            negative_evidence_definition=str(values["negative_evidence_definition"]),
            positive_witness_sequence=positive.sequence,
            negative_witness_sequence=negative.sequence,
            limitations=str(values["limitations"]),
        )
        return GeometricFrontierProposal(
            action=action,
            goal=goal,
            length_policy=length_policy,
            proposal_attempt=proposal_attempt,
            raw_response=dict(values),
            metadata=dict(metadata),
            positive_witness_candidate_id=positive.candidate_id,
            negative_witness_candidate_id=negative.candidate_id,
        )


__all__ = ["GeometricFrontierController", "GeometricFrontierProposal"]
