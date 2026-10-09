"""Deterministic, target-agnostic evidence cards for peptide sequences.

These arithmetic summaries deliberately contain no activity, toxicity,
database, neighbour, assay, or target-specific prediction.  They help an LLM
inspect a sequence consistently without becoming a surrogate reward model.
"""

from __future__ import annotations

from collections import Counter
import math
from typing import Any, Mapping

from domains.peptides.qualitative_evidence import (
    DEVELOPABILITY_EVIDENCE_LIMITATION,
    DEVELOPABILITY_EVIDENCE_VERSION,
)


AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWY")
EISENBERG_HYDROPHOBICITY: Mapping[str, float] = {
    "A": 0.62,
    "C": 0.29,
    "D": -0.90,
    "E": -0.74,
    "F": 1.19,
    "G": 0.48,
    "H": -0.40,
    "I": 1.38,
    "K": -1.50,
    "L": 1.06,
    "M": 0.64,
    "N": -0.78,
    "P": 0.12,
    "Q": -0.85,
    "R": -2.53,
    "S": -0.18,
    "T": -0.05,
    "V": 1.08,
    "W": 0.81,
    "Y": 0.26,
}

EVIDENCE_CARD_VERSION = "generic-sequence-evidence-v1"
EVIDENCE_LIMITATION = (
    "Deterministic sequence-derived arithmetic only; no assay, activity, "
    "selectivity, toxicity, database, neighbour, or target-specific evidence."
)


def evidence_tool_definition() -> dict[str, Any]:
    """Return the frozen assumptions once per provider request.

    Keeping assumptions separate from each candidate card makes the prompt
    smaller and prevents a simple arithmetic summary from looking like a
    biological measurement.
    """
    return {
        "tool_version": EVIDENCE_CARD_VERSION,
        "scope": EVIDENCE_LIMITATION,
        "side_chain_charge_index_formula": "K + R + 0.1*H - D - E",
        "charge_assumptions": (
            "Side-chain counting heuristic only; termini are omitted and no "
            "pH-dependent equilibrium is modeled."
        ),
        "hydrophobicity_scale": "Eisenberg consensus scale",
        "hydrophobic_moment_assumptions": (
            "100 degrees per residue, ideal alpha helix; sequence proxy only, "
            "not a predicted conformation."
        ),
        "fraction_denominator": "full canonical sequence length",
        "normalized_shannon_complexity": (
            "base-2 residue entropy divided by log2(20)"
        ),
    }


def developability_evidence_tool_definition() -> dict[str, Any]:
    """Describe the optional predictor evidence without exposing cardinal values."""
    return {
        "tool_version": DEVELOPABILITY_EVIDENCE_VERSION,
        "scope": DEVELOPABILITY_EVIDENCE_LIMITATION,
        "properties": {
            "non_hemolysis": "higher is predicted to be less hemolytic",
            "non_fouling": "higher is predicted to be less nonspecifically fouling",
            "solubility": "higher is predicted to be more soluble",
            "half_life": "higher is predicted to have a longer half-life",
        },
        "bands": {
            "lower_third": (
                "lower third within the current arm-blind comparison cohort of "
                "valid generated peptides of the same length"
            ),
            "middle_third": (
                "middle third within the current arm-blind comparison cohort of "
                "valid generated peptides of the same length"
            ),
            "upper_third": (
                "upper third within the current arm-blind comparison cohort of "
                "valid generated peptides of the same length"
            ),
        },
        "cohort_size_note": (
            "The number of same-length peptides depends on the campaign stage; "
            "the bands are ordinal within that stage and are not cardinal values."
        ),
        "cardinal_predictor_outputs_visible": False,
        "labelled_dataset_examples_visible": False,
    }


def frozen_p0_developability_evidence_tool_definition() -> dict[str, Any]:
    """Describe bands projected onto one immutable p0 reference.

    This variant is deliberately separate from the historical stage-local tool
    description.  Moving-frontier experiments can therefore compare the same
    qualitative category across generator distributions without changing any
    earlier request hash.
    """

    return {
        "tool_version": f"{DEVELOPABILITY_EVIDENCE_VERSION}-frozen-p0-v1",
        "scope": DEVELOPABILITY_EVIDENCE_LIMITATION,
        "properties": {
            "non_hemolysis": "higher is predicted to be less hemolytic",
            "non_fouling": "higher is predicted to be less nonspecifically fouling",
            "solubility": "higher is predicted to be more soluble",
            "half_life": "higher is predicted to have a longer half-life",
        },
        "bands": {
            "lower_third": (
                "below the first empirical-tertile cutpoint fitted once to "
                "valid p0 peptides of the same length"
            ),
            "middle_third": (
                "between the two empirical-tertile cutpoints fitted once to "
                "valid p0 peptides of the same length"
            ),
            "upper_third": (
                "above the second empirical-tertile cutpoint fitted once to "
                "valid p0 peptides of the same length"
            ),
        },
        "reference_note": (
            "All cutpoints are frozen from the original p0 generator separately "
            "for each peptide length. Later distributions are projected onto "
            "those unchanged cutpoints; their bands are not rebalanced or refit."
        ),
        "interpretation_note": (
            "These are coarse ordinal computational predictions, comparable "
            "across generator distributions but neither calibrated probabilities "
            "nor evidence of anticancer activity or healthy-cell safety."
        ),
        "cardinal_predictor_outputs_visible": False,
        "labelled_dataset_examples_visible": False,
    }


def candidate_evidence_payload(
    sequence: str,
    developability_evidence: tuple[tuple[str, str], ...] = (),
) -> dict[str, Any]:
    """Build the complete evidence payload visible for one candidate."""
    payload: dict[str, Any] = {
        "generic_sequence_evidence": generic_sequence_evidence(sequence)
    }
    if developability_evidence:
        payload["released_mogdfm_developability_evidence"] = dict(
            developability_evidence
        )
    return payload




def _longest_identical_residue_run(sequence: str) -> int:
    longest = 1
    current = 1
    for previous, residue in zip(sequence, sequence[1:]):
        if residue == previous:
            current += 1
            longest = max(longest, current)
        else:
            current = 1
    return longest


def generic_sequence_evidence(sequence: str) -> dict[str, Any]:
    """Return a compact JSON-safe evidence card for one canonical sequence."""
    if not sequence or any(residue not in AMINO_ACIDS for residue in sequence):
        raise ValueError("sequence must contain canonical amino-acid letters")

    counts = Counter(sequence)
    length = len(sequence)
    nonzero_frequencies = [count / length for count in counts.values()]
    complexity = -sum(
        frequency * math.log2(frequency) for frequency in nonzero_frequencies
    ) / math.log2(20)
    angles = [math.radians(index * 100.0) for index in range(length)]
    hydrophobicities = [EISENBERG_HYDROPHOBICITY[aa] for aa in sequence]
    cosine_sum = sum(
        hydrophobicity * math.cos(angle)
        for hydrophobicity, angle in zip(hydrophobicities, angles)
    )
    sine_sum = sum(
        hydrophobicity * math.sin(angle)
        for hydrophobicity, angle in zip(hydrophobicities, angles)
    )
    hydrophobic_moment = math.hypot(cosine_sum, sine_sum) / length

    # Rounded values keep provider payloads compact and prevent meaningless
    # floating-point precision from looking like biological measurement.
    return {
        "sequence_length": length,
        "side_chain_charge_index": round(
            counts["K"]
            + counts["R"]
            + 0.1 * counts["H"]
            - counts["D"]
            - counts["E"],
            3,
        ),
        "basic_fraction": round(
            (counts["K"] + counts["R"] + counts["H"]) / length, 4
        ),
        "acidic_fraction": round((counts["D"] + counts["E"]) / length, 4),
        "hydrophobic_fraction": round(
            sum(counts[aa] for aa in "AILMFWVY") / length, 4
        ),
        "aromatic_fraction": round(sum(counts[aa] for aa in "FWY") / length, 4),
        "glycine_proline_fraction": round(
            (counts["G"] + counts["P"]) / length, 4
        ),
        "normalized_shannon_complexity": round(complexity, 4),
        "alpha_helical_hydrophobic_moment": round(hydrophobic_moment, 4),
        "distinct_residue_count": len(counts),
        "longest_identical_residue_run": _longest_identical_residue_run(sequence),
    }
