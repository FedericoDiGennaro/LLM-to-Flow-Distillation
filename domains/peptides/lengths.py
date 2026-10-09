"""Strict length-policy utilities for an adaptive PepDFM campaign.

PepDFM's sequence length is an external sampling control: the model fills a
tensor whose length has already been chosen by the harness.  Consequently, a
parent/child movement claim is interpretable only when both generator samples
use the same frozen length policy.  This module represents that policy and
provides small, side-effect-free helpers for categorical-label summaries.

The positive rates below are aggregates of categorical judgments.  They are
not LLM confidence scores or calibrated biological probabilities.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any, TypeAlias


MINIMUM_PEPTIDE_LENGTH = 6
MAXIMUM_PEPTIDE_LENGTH = 49
MINIMUM_ACTIVE_LENGTHS = 1
MAXIMUM_ACTIVE_LENGTHS = 4
POLICY_SAMPLE_COUNT = 100
MINIMUM_SAMPLES_PER_LENGTH = 10
MAXIMUM_SAMPLES_PER_LENGTH = 60
CATEGORICAL_LABELS = ("positive", "negative", "abstain", "unstable")

LengthKey: TypeAlias = int | str
LengthPolicyLike: TypeAlias = "AdaptiveLengthPolicy | Mapping[LengthKey, int]"


def _strict_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _normalize_length_key(value: Any, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer or canonical integer string")
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value and value == value.strip():
        try:
            normalized = int(value)
        except ValueError as error:
            raise TypeError(
                f"{name} must be an integer or canonical integer string"
            ) from error
        if str(normalized) == value:
            return normalized
    raise TypeError(f"{name} must be an integer or canonical integer string")


@dataclass(frozen=True, slots=True, init=False)
class AdaptiveLengthPolicy(Mapping[int, int]):
    """An immutable, canonical 100-sample PepDFM length allocation.

    JSON objects naturally turn integer keys into strings, so both integer keys
    and canonical decimal strings are accepted at construction.  Internally the
    entries are sorted integer pairs and cannot be mutated.
    """

    entries: tuple[tuple[int, int], ...]

    def __init__(self, length_counts: Mapping[LengthKey, int]) -> None:
        if not isinstance(length_counts, Mapping):
            raise TypeError("length policy must be a mapping from length to count")
        normalized: dict[int, int] = {}
        for raw_length, raw_count in length_counts.items():
            length = _normalize_length_key(raw_length, name="peptide length")
            if length in normalized:
                raise ValueError(
                    f"length policy contains duplicate normalized length {length}"
                )
            count = _strict_integer(
                raw_count, name=f"sample count for peptide length {length}"
            )
            normalized[length] = count

        if not MINIMUM_ACTIVE_LENGTHS <= len(normalized) <= MAXIMUM_ACTIVE_LENGTHS:
            raise ValueError(
                "length policy must contain between "
                f"{MINIMUM_ACTIVE_LENGTHS} and {MAXIMUM_ACTIVE_LENGTHS} lengths"
            )
        for length, count in normalized.items():
            if not MINIMUM_PEPTIDE_LENGTH <= length <= MAXIMUM_PEPTIDE_LENGTH:
                raise ValueError(
                    f"peptide length {length} is outside the supported "
                    f"[{MINIMUM_PEPTIDE_LENGTH}, {MAXIMUM_PEPTIDE_LENGTH}] range"
                )
            if not MINIMUM_SAMPLES_PER_LENGTH <= count <= MAXIMUM_SAMPLES_PER_LENGTH:
                raise ValueError(
                    f"sample count for peptide length {length} must lie in "
                    f"[{MINIMUM_SAMPLES_PER_LENGTH}, {MAXIMUM_SAMPLES_PER_LENGTH}]"
                )
        total = sum(normalized.values())
        if total != POLICY_SAMPLE_COUNT:
            raise ValueError(
                f"length-policy counts must sum to {POLICY_SAMPLE_COUNT}, got {total}"
            )
        object.__setattr__(self, "entries", tuple(sorted(normalized.items())))

    def __getitem__(self, length: int) -> int:
        normalized = _normalize_length_key(length, name="peptide length")
        for key, count in self.entries:
            if key == normalized:
                return count
        raise KeyError(length)

    def __iter__(self) -> Iterator[int]:
        return (length for length, _ in self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    @property
    def total_count(self) -> int:
        return sum(count for _, count in self.entries)

    def as_mapping(self, *, string_keys: bool = True) -> dict[str, int] | dict[int, int]:
        """Return a fresh JSON-friendly mapping without exposing mutable state."""

        if string_keys:
            return {str(length): count for length, count in self.entries}
        return dict(self.entries)


def coerce_length_policy(value: LengthPolicyLike) -> AdaptiveLengthPolicy:
    if isinstance(value, AdaptiveLengthPolicy):
        return value
    return AdaptiveLengthPolicy(value)


def expand_length_policy(policy: LengthPolicyLike) -> tuple[int, ...]:
    """Expand a policy to a deterministic round-robin sequence of 100 lengths."""

    normalized = coerce_length_policy(policy)
    remaining = dict(normalized.entries)
    expanded: list[int] = []
    while remaining:
        for length in tuple(sorted(remaining)):
            expanded.append(length)
            remaining[length] -= 1
            if remaining[length] == 0:
                del remaining[length]
    if len(expanded) != POLICY_SAMPLE_COUNT:  # Defensive against future drift.
        raise AssertionError("validated length policy expanded to the wrong size")
    return tuple(expanded)


def _normalize_histogram(
    histogram: Mapping[LengthKey, int], *, name: str
) -> dict[int, int]:
    if not isinstance(histogram, Mapping):
        raise TypeError(f"{name} must be a mapping from length to count")
    normalized: dict[int, int] = {}
    for raw_length, raw_count in histogram.items():
        length = _normalize_length_key(raw_length, name=f"{name} peptide length")
        if length in normalized:
            raise ValueError(f"{name} contains duplicate normalized length {length}")
        count = _strict_integer(raw_count, name=f"{name} count for length {length}")
        if count <= 0:
            raise ValueError(f"{name} count for length {length} must be positive")
        normalized[length] = count
    return normalized


def validate_sampled_histogram(
    policy: LengthPolicyLike,
    observed_histogram: Mapping[LengthKey, int],
) -> dict[int, int]:
    """Validate exact sampled counts and return their canonical representation."""

    expected = dict(coerce_length_policy(policy).entries)
    observed = _normalize_histogram(observed_histogram, name="sampled histogram")
    if observed != expected:
        missing = {key: value for key, value in expected.items() if key not in observed}
        unexpected = {
            key: value for key, value in observed.items() if key not in expected
        }
        wrong_counts = {
            key: {"expected": expected[key], "observed": observed[key]}
            for key in expected.keys() & observed.keys()
            if expected[key] != observed[key]
        }
        raise ValueError(
            "sampled length histogram differs from frozen policy: "
            f"missing={missing}, unexpected={unexpected}, "
            f"wrong_counts={wrong_counts}"
        )
    return observed


def categorical_counts_by_length(
    outcomes: Sequence[tuple[int, str]],
    *,
    policy: LengthPolicyLike | None = None,
) -> dict[int, dict[str, int]]:
    """Count positive/negative/abstain/unstable outcomes within each length."""

    by_length: dict[int, Counter[str]] = {}
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, tuple) or len(outcome) != 2:
            raise TypeError(f"categorical outcome {index} must be a (length, label) tuple")
        raw_length, raw_label = outcome
        length = _strict_integer(raw_length, name=f"categorical outcome {index} length")
        if not MINIMUM_PEPTIDE_LENGTH <= length <= MAXIMUM_PEPTIDE_LENGTH:
            raise ValueError(
                f"categorical outcome {index} has unsupported peptide length {length}"
            )
        if not isinstance(raw_label, str) or raw_label not in CATEGORICAL_LABELS:
            raise ValueError(
                f"categorical outcome {index} label must be one of "
                f"{CATEGORICAL_LABELS}, got {raw_label!r}"
            )
        by_length.setdefault(length, Counter())[raw_label] += 1

    if not by_length:
        raise ValueError("at least one categorical outcome is required")
    histogram = {
        length: sum(counter.values()) for length, counter in by_length.items()
    }
    if policy is not None:
        validate_sampled_histogram(policy, histogram)

    return {
        length: {
            "sample_count": sum(counter.values()),
            **{f"{label}_count": counter[label] for label in CATEGORICAL_LABELS},
        }
        for length, counter in sorted(by_length.items())
    }


def _validated_categorical_counts(
    counts_by_length: Mapping[int, Mapping[str, int]],
) -> dict[int, dict[str, int]]:
    if not isinstance(counts_by_length, Mapping) or not counts_by_length:
        raise ValueError("categorical counts by length must be a non-empty mapping")
    normalized: dict[int, dict[str, int]] = {}
    for raw_length, raw_counts in counts_by_length.items():
        length = _normalize_length_key(raw_length, name="categorical-count length")
        if not MINIMUM_PEPTIDE_LENGTH <= length <= MAXIMUM_PEPTIDE_LENGTH:
            raise ValueError(
                f"categorical-count peptide length {length} is outside the supported "
                f"[{MINIMUM_PEPTIDE_LENGTH}, {MAXIMUM_PEPTIDE_LENGTH}] range"
            )
        if length in normalized:
            raise ValueError(f"categorical counts duplicate normalized length {length}")
        if not isinstance(raw_counts, Mapping):
            raise TypeError(f"categorical counts for length {length} must be a mapping")
        expected_fields = {"sample_count", *(
            f"{label}_count" for label in CATEGORICAL_LABELS
        )}
        if set(raw_counts) != expected_fields:
            raise ValueError(
                f"categorical counts for length {length} require exactly "
                f"{sorted(expected_fields)}"
            )
        values = {
            field: _strict_integer(
                raw_counts[field], name=f"{field} for peptide length {length}"
            )
            for field in expected_fields
        }
        if values["sample_count"] <= 0 or any(
            values[f"{label}_count"] < 0 for label in CATEGORICAL_LABELS
        ):
            raise ValueError(
                f"categorical counts for length {length} must be non-negative "
                "with a positive sample_count"
            )
        label_total = sum(values[f"{label}_count"] for label in CATEGORICAL_LABELS)
        if label_total != values["sample_count"]:
            raise ValueError(
                f"categorical counts for length {length} sum to {label_total}, "
                f"not sample_count={values['sample_count']}"
            )
        normalized[length] = values
    return normalized


def _normalized_reference_weights(
    reference_weights: Mapping[LengthKey, float],
) -> dict[int, float]:
    if not isinstance(reference_weights, Mapping) or not reference_weights:
        raise ValueError("frozen reference weights must be a non-empty mapping")
    normalized: dict[int, float] = {}
    for raw_length, raw_weight in reference_weights.items():
        length = _normalize_length_key(raw_length, name="reference-weight length")
        if not MINIMUM_PEPTIDE_LENGTH <= length <= MAXIMUM_PEPTIDE_LENGTH:
            raise ValueError(
                f"reference-weight peptide length {length} is outside the supported "
                f"[{MINIMUM_PEPTIDE_LENGTH}, {MAXIMUM_PEPTIDE_LENGTH}] range"
            )
        if length in normalized:
            raise ValueError(f"reference weights duplicate normalized length {length}")
        if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
            raise TypeError(f"reference weight for length {length} must be numeric")
        weight = float(raw_weight)
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError(
                f"reference weight for length {length} must be finite and positive"
            )
        normalized[length] = weight
    total = sum(normalized.values())
    return {length: weight / total for length, weight in normalized.items()}


def length_standardized_positive_rate(
    counts_by_length: Mapping[int, Mapping[str, int]],
    *,
    frozen_reference_weights: Mapping[LengthKey, float],
) -> float:
    """Aggregate within-length positive rates under one frozen weighting."""

    counts = _validated_categorical_counts(counts_by_length)
    weights = _normalized_reference_weights(frozen_reference_weights)
    if set(counts) != set(weights):
        raise ValueError(
            "categorical-count and frozen-reference supports must match exactly: "
            f"counts={sorted(counts)}, weights={sorted(weights)}"
        )
    return sum(
        weights[length]
        * counts[length]["positive_count"]
        / counts[length]["sample_count"]
        for length in sorted(weights)
    )


def compare_matched_length_policies(
    *,
    parent_policy: LengthPolicyLike,
    child_policy: LengthPolicyLike,
    parent_counts_by_length: Mapping[int, Mapping[str, int]],
    child_counts_by_length: Mapping[int, Mapping[str, int]],
    frozen_reference_weights: Mapping[LengthKey, float],
) -> dict[str, Any]:
    """Compare categorical yields only under exactly matched length policies."""

    parent = coerce_length_policy(parent_policy)
    child = coerce_length_policy(child_policy)
    if parent != child:
        raise ValueError(
            "parent and child length policies must match exactly before comparison: "
            f"parent={parent.as_mapping()}, child={child.as_mapping()}"
        )

    normalized_parent = _validated_categorical_counts(parent_counts_by_length)
    normalized_child = _validated_categorical_counts(child_counts_by_length)
    validate_sampled_histogram(
        parent,
        {
            length: values["sample_count"]
            for length, values in normalized_parent.items()
        },
    )
    validate_sampled_histogram(
        child,
        {
            length: values["sample_count"]
            for length, values in normalized_child.items()
        },
    )
    parent_rate = length_standardized_positive_rate(
        normalized_parent, frozen_reference_weights=frozen_reference_weights
    )
    child_rate = length_standardized_positive_rate(
        normalized_child, frozen_reference_weights=frozen_reference_weights
    )
    return {
        "matched_length_policy": parent.as_mapping(),
        "frozen_reference_weights": {
            str(length): weight
            for length, weight in sorted(
                _normalized_reference_weights(frozen_reference_weights).items()
            )
        },
        "parent_length_standardized_positive_rate": parent_rate,
        "child_length_standardized_positive_rate": child_rate,
        "child_minus_parent_length_standardized_positive_rate": (
            child_rate - parent_rate
        ),
        "parent_per_length_positive_rate": {
            str(length): values["positive_count"] / values["sample_count"]
            for length, values in sorted(normalized_parent.items())
        },
        "child_per_length_positive_rate": {
            str(length): values["positive_count"] / values["sample_count"]
            for length, values in sorted(normalized_child.items())
        },
    }


__all__ = [
    "AdaptiveLengthPolicy",
    "CATEGORICAL_LABELS",
    "MAXIMUM_ACTIVE_LENGTHS",
    "MAXIMUM_PEPTIDE_LENGTH",
    "MAXIMUM_SAMPLES_PER_LENGTH",
    "MINIMUM_ACTIVE_LENGTHS",
    "MINIMUM_PEPTIDE_LENGTH",
    "MINIMUM_SAMPLES_PER_LENGTH",
    "POLICY_SAMPLE_COUNT",
    "categorical_counts_by_length",
    "coerce_length_policy",
    "compare_matched_length_policies",
    "expand_length_policy",
    "length_standardized_positive_rate",
    "validate_sampled_histogram",
]
