"""Molecule-specific prompts and strict schemas for categorical LTFD roles."""

from __future__ import annotations

from domains.molecules.prompts import GSK3B as DOMAIN_PROMPTS

from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from adm.preferences.clients import StructuredRequest
from domains.molecules.gsk3b.cards import (
    FINAL_GOAL,
    MoleculeCandidateCard,
    evidence_tool_definition,
)


MOLECULES_GEOMETRIC_JUDGE_SYSTEM_PROMPT = DOMAIN_PROMPTS['LLM-Judge']


MOLECULES_GEOMETRIC_CONTROLLER_PROMPT = DOMAIN_PROMPTS['SetGoal_proposer']


MOLECULES_GEOMETRIC_CORRECTION_ADDENDUM = DOMAIN_PROMPTS['goal_controller']


MOLECULES_MOVING_FRONTIER_AUDITOR_PROMPT = DOMAIN_PROMPTS['semantic_auditor']


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MoleculeJudgmentRow(StrictModel):
    item_id: str = Field(pattern=r"^item-\d{3}$")
    evidence_for: str = Field(min_length=1, max_length=1600)
    evidence_against: str = Field(min_length=1, max_length=1600)
    critical_unknowns: str = Field(min_length=1, max_length=1600)
    initial_label: Literal["positive", "negative", "abstain"]
    self_critique: str = Field(min_length=1, max_length=1600)
    final_label: Literal["positive", "negative", "abstain"]
    rationale: str = Field(min_length=1, max_length=1600)


class MoleculeJudgmentResponse(StrictModel):
    results: list[MoleculeJudgmentRow] = Field(min_length=1, max_length=8)


class MoleculeGoalProposalResponse(StrictModel):
    action: Literal["broaden", "tighten", "hold", "reached_final"]
    proposed_goal: str = Field(min_length=1, max_length=1000)
    rationale: str = Field(min_length=1, max_length=2400)
    semantic_relation: Literal[
        "final_subset_of_proposed",
        "proposed_subset_of_source",
        "unchanged_from_source",
        "equals_final",
    ]
    bridge_to_final_goal: str = Field(min_length=1, max_length=1600)
    deferred_final_requirements: list[str] = Field(max_length=10)
    observable_positive_criterion: str = Field(min_length=1, max_length=1200)
    observable_negative_criterion: str = Field(min_length=1, max_length=1200)
    assessable_from_registered_cards: bool
    numerical_descriptor_cutoff_used: Literal[False]
    hidden_evidence_used: Literal[False]
    receptor_binding_success_claimed: Literal[False]


class MoleculeSemanticAuditResponse(StrictModel):
    decision: Literal["accept", "reject"]
    requested_action_followed: bool
    semantic_direction_valid: bool
    assessable_from_registered_cards: bool
    not_numeric_cutoff_or_synonym: bool
    hidden_evidence_excluded: bool
    structural_proxy_limits_explicit: bool
    scientific_success_not_claimed: bool
    rationale: str = Field(min_length=1, max_length=2400)


def build_judgment_request(
    cards: Sequence[MoleculeCandidateCard],
    *,
    current_goal: str,
    model: str,
    max_output_tokens: int,
    campaign_id: str,
    iteration: int,
    repeat_index: int,
) -> StructuredRequest:
    if not 1 <= len(cards) <= 8:
        raise ValueError("one molecule judgment request requires 1 to 8 cards")
    if repeat_index not in (1, 2):
        raise ValueError("repeat index must be 1 or 2")
    item_ids = [card.item_id for card in cards]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("molecule cards must have distinct item IDs")
    return StructuredRequest.from_payload(
        model=model,
        schema_id="molecules-categorical-judgment-v1",
        system_prompt=MOLECULES_GEOMETRIC_JUDGE_SYSTEM_PROMPT,
        user_payload={
            "immutable_final_goal": FINAL_GOAL,
            "current_goal": current_goal,
            "allowed_labels": ["positive", "negative", "abstain"],
            "independent_repeat_index": repeat_index,
            "repeat_instruction": (
                "Judge independently from the supplied evidence; no earlier "
                "response is available."
            ),
            "evidence_tool_definition": evidence_tool_definition(),
            "candidates": [card.model_dump(mode="json") for card in cards],
        },
        logical_context={
            "campaign_id": campaign_id,
            "iteration": iteration,
            "repeat_index": repeat_index,
            "item_ids": item_ids,
        },
        max_output_tokens=max_output_tokens,
        reasoning_effort="high",
    )


def build_controller_request(
    cards: Sequence[MoleculeCandidateCard],
    source_feedback: Mapping[str, Mapping[str, Any]],
    *,
    final_goal_feedback: Mapping[str, Mapping[str, Any]],
    source_goal_adoption_cohort: Sequence[Mapping[str, Any]] | None = None,
    source_goal: str,
    requested_action: Literal["broaden", "tighten", "hold"],
    model: str,
    max_output_tokens: int,
    campaign_id: str,
    iteration: int,
    proposal_attempt: int,
    previous_failure: Mapping[str, Any] | None = None,
) -> StructuredRequest:
    if not cards or proposal_attempt < 1:
        raise ValueError("controller requires cards and a positive attempt index")
    item_ids = {card.item_id for card in cards}
    if set(source_feedback) != item_ids or set(final_goal_feedback) != item_ids:
        raise ValueError("controller feedback must cover every supplied card")
    prompt = MOLECULES_GEOMETRIC_CONTROLLER_PROMPT
    if proposal_attempt > 1:
        prompt += "\n\n" + MOLECULES_GEOMETRIC_CORRECTION_ADDENDUM
    return StructuredRequest.from_payload(
        model=model,
        schema_id="molecules-geometric-controller-v1",
        system_prompt=prompt,
        user_payload={
            "immutable_final_goal": FINAL_GOAL,
            "source_goal": source_goal,
            "requested_action": requested_action,
            "proposal_attempt": proposal_attempt,
            "visible_cohort": [
                {
                    "card": card.model_dump(mode="json"),
                    "source_goal_two_pass_feedback": dict(
                        source_feedback[card.item_id]
                    ),
                    "final_goal_two_pass_feedback": dict(
                        final_goal_feedback[card.item_id]
                    ),
                }
                for card in cards
            ],
            "source_goal_adoption_cohort": list(
                source_goal_adoption_cohort or []
            ),
            "feedback_contract": {
                "same_current_cohort_for_source_and_final_goal": True,
                "two_independent_passes_per_goal": True,
                "categorical_labels_and_rationales_only": True,
                "cardinal_scores_or_hidden_evaluators_visible": False,
            },
            "previous_failure": dict(previous_failure or {}),
        },
        logical_context={
            "campaign_id": campaign_id,
            "iteration": iteration,
            "proposal_attempt": proposal_attempt,
            "role": "curriculum_controller",
        },
        max_output_tokens=max_output_tokens,
        reasoning_effort="high",
    )


def build_audit_request(
    proposal: MoleculeGoalProposalResponse,
    *,
    source_goal: str,
    requested_action: Literal["broaden", "tighten", "hold"],
    model: str,
    max_output_tokens: int,
    campaign_id: str,
    iteration: int,
    proposal_attempt: int,
) -> StructuredRequest:
    return StructuredRequest.from_payload(
        model=model,
        schema_id="molecules-geometric-semantic-audit-v1",
        system_prompt=MOLECULES_MOVING_FRONTIER_AUDITOR_PROMPT,
        user_payload={
            "immutable_final_goal": FINAL_GOAL,
            "source_goal": source_goal,
            "requested_action": requested_action,
            "proposal": proposal.model_dump(mode="json"),
            "registered_evidence_boundary": evidence_tool_definition(),
        },
        logical_context={
            "campaign_id": campaign_id,
            "iteration": iteration,
            "proposal_attempt": proposal_attempt,
            "role": "independent_auditor",
        },
        max_output_tokens=max_output_tokens,
        reasoning_effort="high",
    )


def validated_rows(
    response: MoleculeJudgmentResponse,
    cards: Sequence[MoleculeCandidateCard],
) -> list[MoleculeJudgmentRow]:
    expected = [card.item_id for card in cards]
    observed = [row.item_id for row in response.results]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise ValueError(
            "judgment response does not cover each molecule card exactly once"
        )
    by_id = {row.item_id: row for row in response.results}
    return [by_id[item_id] for item_id in expected]


def stable_labels(
    first: MoleculeJudgmentResponse,
    second: MoleculeJudgmentResponse,
    cards: Sequence[MoleculeCandidateCard],
) -> dict[str, str]:
    first_rows = validated_rows(first, cards)
    second_rows = validated_rows(second, cards)
    return {
        left.item_id: (
            left.final_label
            if left.final_label == right.final_label
            else "unstable"
        )
        for left, right in zip(first_rows, second_rows)
    }


__all__ = [
    "MOLECULES_GEOMETRIC_CONTROLLER_PROMPT",
    "MOLECULES_GEOMETRIC_CORRECTION_ADDENDUM",
    "MOLECULES_GEOMETRIC_JUDGE_SYSTEM_PROMPT",
    "MOLECULES_MOVING_FRONTIER_AUDITOR_PROMPT",
    "MoleculeGoalProposalResponse",
    "MoleculeJudgmentResponse",
    "MoleculeSemanticAuditResponse",
    "build_audit_request",
    "build_controller_request",
    "build_judgment_request",
    "stable_labels",
    "validated_rows",
]
