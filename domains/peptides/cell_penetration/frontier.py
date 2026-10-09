"""Minimal semantic-frontier controller for iterative PepDFM alignment.

This module deliberately keeps two concepts separate:

``frontier_criterion``
    The literal, observable rule used by the categorical candidate judge and
    by the next alignment update.

``bridge_to_final``
    The scientific hypothesis explaining why that temporary search frontier
    might help reach the immutable final goal.  It is logged, but is not part
    of the candidate-labeling rubric.

The host chooses broaden/hold/tighten from categorical counts.  The LLM only
formulates a requested semantic transition; it never supplies a scalar score,
confidence, progress estimate, or transition decision.
"""

from __future__ import annotations


from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any, Mapping


from domains.peptides.sequence_evidence import (
    generic_sequence_evidence,
)




@dataclass(frozen=True)
class FrontierGoal:
    """One operative goal and its separately logged path to the final goal."""

    final_goal: str
    frontier_criterion: str
    bridge_to_final: str
    is_final: bool
    positive_evidence_definition: str | None = None
    negative_evidence_definition: str | None = None
    positive_witness_sequence: str | None = None
    negative_witness_sequence: str | None = None
    limitations: str = "Sequence-level evidence does not establish biological activity or safety."

    def __post_init__(self) -> None:
        for name in ("final_goal", "frontier_criterion", "bridge_to_final", "limitations"):
            if not getattr(self, name).strip():
                raise ValueError(f"{name} must be non-empty")
        definitions = (
            self.positive_evidence_definition,
            self.negative_evidence_definition,
        )
        witnesses = (
            self.positive_witness_sequence,
            self.negative_witness_sequence,
        )
        if self.is_final:
            if self.frontier_criterion.strip() != self.final_goal.strip():
                raise ValueError("a final frontier must equal the immutable final goal")
            if any(value is not None for value in (*definitions, *witnesses)):
                raise ValueError("the exact final goal cannot carry a proxy rubric")
        else:
            if self.frontier_criterion.strip() == self.final_goal.strip():
                raise ValueError("a non-final frontier cannot equal the final goal")
            if not all(value is not None and value.strip() for value in definitions):
                raise ValueError("a non-final frontier requires positive and negative definitions")
            if not all(value is not None and value.strip() for value in witnesses):
                raise ValueError("a non-final frontier requires positive and negative witnesses")

    @classmethod
    def exact_final(cls, final_goal: str) -> "FrontierGoal":
        return cls(
            final_goal=final_goal,
            frontier_criterion=final_goal,
            bridge_to_final="This is the immutable final goal.",
            is_final=True,
        )

    def operative_proxy_specification(self) -> Mapping[str, Any] | None:
        """Return only the judge's rubric, intentionally excluding the bridge."""

        if self.is_final:
            return None
        assert self.positive_evidence_definition is not None
        assert self.negative_evidence_definition is not None
        assert self.positive_witness_sequence is not None
        assert self.negative_witness_sequence is not None
        return {
            "positive_evidence_definition": self.positive_evidence_definition,
            "negative_evidence_definition": self.negative_evidence_definition,
            "positive_witness": {
                "sequence": self.positive_witness_sequence,
                "generic_sequence_evidence": generic_sequence_evidence(
                    self.positive_witness_sequence
                ),
            },
            "negative_witness": {
                "sequence": self.negative_witness_sequence,
                "generic_sequence_evidence": generic_sequence_evidence(
                    self.negative_witness_sequence
                ),
            },
            "apply_definitions_verbatim": True,
            "bridge_to_final_is_not_part_of_the_labeling_rule": True,
        }














































def frontier_as_dict(value: Any) -> Any:
    """Convert v6 records to deterministic JSON-compatible primitives."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        result = {
            key: frontier_as_dict(item) for key, item in asdict(value).items()
        }
        # This post-failure provenance field must not perturb hashes for the
        # original cached proposals that predate it.  Include it only when an
        # actual transport repair occurred.
        if result.get("witness_contract_repair") is None:
            result.pop("witness_contract_repair", None)
        return result
    if isinstance(value, Mapping):
        return {str(key): frontier_as_dict(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [frontier_as_dict(item) for item in value]
    return value
