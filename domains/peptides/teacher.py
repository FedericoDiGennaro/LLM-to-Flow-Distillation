"""Structured Gemini roles for the held-out moving-frontier experiment."""

from __future__ import annotations

from domains.peptides.prompts import ANTICANCER as DOMAIN_PROMPTS

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from domains.peptides.agent import StructuredSemanticGoalAgent
from domains.peptides.frontier import FrontierGoal


MOVING_FRONTIER_AUDITOR_PROMPT = DOMAIN_PROMPTS["semantic_auditor"]


class MovingFrontierSemanticAuditResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["accept", "reject"]
    requested_action_followed: bool
    direction_toward_final_preserved: bool
    directional_relation: Literal[
        "source_subset_of_proposed", "proposed_subset_of_source"
    ]
    directional_relation_rationale: str = Field(min_length=1)
    exactly_one_final_requirement_restored: bool
    materially_easier_if_broadening: bool
    not_numeric_cutoff_or_synonym: bool
    proxy_limitations_explicit: bool
    multiple_routes_preserved_where_reasonable: bool
    biological_success_not_claimed: bool
    rationale: str = Field(min_length=1)


@dataclass(frozen=True)
class MovingFrontierProposal:
    action: str
    goal: FrontierGoal
    raw_response: Mapping[str, Any]
    metadata: Mapping[str, Any]
    positive_witness_candidate_id: str | None
    negative_witness_candidate_id: str | None

    def as_mapping(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "goal": asdict(self.goal),
            "raw_response": dict(self.raw_response),
            "metadata": dict(self.metadata),
            "positive_witness_candidate_id": self.positive_witness_candidate_id,
            "negative_witness_candidate_id": self.negative_witness_candidate_id,
        }


class MovingFrontierSemanticAuditor(StructuredSemanticGoalAgent):
    """Independent, outcome-blind audit of one frozen goal edit."""

    def __init__(
        self, *, system_prompt: str = MOVING_FRONTIER_AUDITOR_PROMPT, **kwargs
    ):
        super().__init__(**kwargs)
        self.system_prompt = system_prompt

    def audit(
        self,
        *,
        proposal: MovingFrontierProposal,
        source_goal: FrontierGoal,
        final_goal: str,
        goal_history: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        request = self._request(
            schema_id="pepdfm-moving-frontier-semantic-audit-v1",
            system_prompt=self.system_prompt,
            user_payload={
                "immutable_final_goal": final_goal,
                "requested_action": proposal.action,
                "source_goal": asdict(source_goal),
                "proposed_goal": proposal.as_mapping(),
                "prior_accepted_goal_history": [dict(item) for item in goal_history],
                "gate_training_confirmation_outcomes_hidden": True,
                "cardinal_scores_forbidden": True,
            },
            logical_context={
                "action": proposal.action,
                "source_goal": source_goal.frontier_criterion,
                "target_goal": proposal.goal.frontier_criterion,
            },
        )
        self.ensure_provider_capacity(((request, MovingFrontierSemanticAuditResponse),))
        values, metadata = self._execute(request, MovingFrontierSemanticAuditResponse)
        broadening = proposal.action == "broaden"
        accepted = (
            values["decision"] == "accept"
            and values["requested_action_followed"] is True
            and values["direction_toward_final_preserved"] is True
            and values["directional_relation"]
            == (
                "source_subset_of_proposed"
                if broadening
                else "proposed_subset_of_source"
            )
            and values["not_numeric_cutoff_or_synonym"] is True
            and values["proxy_limitations_explicit"] is True
            and values["multiple_routes_preserved_where_reasonable"] is True
            and values["biological_success_not_claimed"] is True
            and (
                values["materially_easier_if_broadening"] is True
                if broadening
                else values["exactly_one_final_requirement_restored"] is True
            )
        )
        return {**dict(values), "accepted": accepted, **dict(metadata)}


__all__ = [
    "MOVING_FRONTIER_AUDITOR_PROMPT",
    "MovingFrontierProposal",
    "MovingFrontierSemanticAuditResponse",
    "MovingFrontierSemanticAuditor",
]
