"""Sequence-only sampling diagnostics for PepDFM transition gates."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import itertools
import math
from statistics import mean
from typing import Any, Mapping, Sequence

import numpy as np

from domains.peptides.checkpoint import PepDfmCheckpoint


EISENBERG_HYDROPHOBICITY = {
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


def descriptor_values(sequence: str) -> dict[str, float]:
    counts = Counter(sequence)
    length = len(sequence)
    frequencies = np.asarray([count / length for count in counts.values()])
    complexity = -float(np.sum(frequencies * np.log2(frequencies))) / math.log2(20)
    angles = np.deg2rad(np.arange(length) * 100.0)
    hydrophobicities = np.asarray(
        [EISENBERG_HYDROPHOBICITY[residue] for residue in sequence]
    )
    hydrophobic_moment = float(
        np.hypot(
            np.sum(hydrophobicities * np.cos(angles)),
            np.sum(hydrophobicities * np.sin(angles)),
        )
        / length
    )
    return {
        "net_charge_approx": (
            counts["K"] + counts["R"] + 0.1 * counts["H"]
            - counts["D"] - counts["E"]
        ),
        "hydrophobic_fraction": sum(counts[aa] for aa in "AILMFWVY") / length,
        "alpha_helical_hydrophobic_moment": hydrophobic_moment,
        "gly_pro_fraction": (counts["G"] + counts["P"]) / length,
        "normalized_shannon_complexity": complexity,
    }


def average_pairwise_hamming(sequences: Sequence[str]) -> float | None:
    unique = sorted(set(sequences))
    if len(unique) < 2:
        return None
    total = 0.0
    count = 0
    by_length: dict[int, list[str]] = {}
    for sequence in unique:
        by_length.setdefault(len(sequence), []).append(sequence)
    for length, group in sorted(by_length.items()):
        for left, right in itertools.combinations(group, 2):
            total += sum(a != b for a, b in zip(left, right)) / length
            count += 1
    return None if count == 0 else total / count


def amino_acid_distribution(sequences: Sequence[str]) -> dict[str, float]:
    alphabet = "ACDEFGHIKLMNPQRSTVWY"
    counts = Counter("".join(sequences))
    total = sum(counts.values())
    return {
        residue: (counts[residue] / total if total else 0.0)
        for residue in alphabet
    }


def jensen_shannon(
    left: Mapping[str, float], right: Mapping[str, float]
) -> float:
    keys = sorted(set(left).union(right))
    p = np.asarray([left.get(key, 0.0) for key in keys], dtype=float)
    q = np.asarray([right.get(key, 0.0) for key in keys], dtype=float)
    midpoint = 0.5 * (p + q)

    def kl(first: np.ndarray, second: np.ndarray) -> float:
        mask = first > 0
        return float(np.sum(first[mask] * np.log2(first[mask] / second[mask])))

    return 0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)


def summarize_samples(
    rows: list[dict[str, Any]],
    *,
    all_replay_sequences: set[str],
    source_pool_sequences: set[str],
) -> dict[str, Any]:
    valid = [row for row in rows if row["is_valid"]]
    sequences = [str(row["sequence"]) for row in valid]
    unique = set(sequences)
    descriptors = [descriptor_values(sequence) for sequence in sequences]
    descriptor_fields = tuple(descriptor_values("A"))
    replay_count = sum(
        sequence in all_replay_sequences for sequence in sequences
    )
    return {
        "attempt_count": len(rows),
        "valid_count": len(valid),
        "valid_fraction": len(valid) / len(rows),
        "unique_valid_count": len(unique),
        "unique_valid_fraction_of_attempts": len(unique) / len(rows),
        "exact_replay_match_count": replay_count,
        "exact_replay_match_fraction": (
            replay_count / len(sequences) if sequences else None
        ),
        "exact_original_pool_match_count": sum(
            sequence in source_pool_sequences for sequence in sequences
        ),
        "novel_vs_original_pool_fraction": (
            sum(sequence not in source_pool_sequences for sequence in sequences)
            / len(sequences)
            if sequences
            else None
        ),
        "mean_pairwise_normalized_hamming_unique": average_pairwise_hamming(
            sequences
        ),
        "amino_acid_distribution": amino_acid_distribution(sequences),
        "descriptor_means": {
            field: (
                mean(float(row[field]) for row in descriptors)
                if descriptors
                else None
            )
            for field in descriptor_fields
        },
    }


def sample_rows(
    model: PepDfmCheckpoint, sampling: dict[str, Any]
) -> list[dict[str, Any]]:
    samples = model.sample(
        num_samples=int(sampling["num_samples"]),
        length=int(sampling["length"]),
        num_steps=int(sampling["num_steps"]),
        seed=int(sampling["seed"]),
        batch_size=int(sampling["batch_size"]),
        source_distribution=str(sampling["source_distribution"]),
    )
    return [asdict(sample) for sample in samples]


__all__ = ["jensen_shannon", "sample_rows", "summarize_samples"]
