"""Frozen-reference qualitative bands for PepDFM developability evidence.

Historical V12 code recomputed tertiles jointly on each parent/child sample.
That is useful for an arm-blinded comparison, but it makes a label such as
``upper_third`` distribution-relative.  A moving-frontier experiment needs the
visible evidence scale to remain fixed while the generator changes.  This
module fits an empirical reference on the immutable p0 pool and applies it to
all later samples without exposing cardinal predictor outputs to the LLM.
"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Any, Mapping, Sequence


RAW_TO_VISIBLE = {
    "non_hemolysis": "non_hemolysis",
    "non_fouling": "non_fouling",
    "solubility": "solubility",
    "half_life": "half_life_hours",
}


def fit_frozen_reference(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit a length-stratified empirical CDF from private p0 predictor rows."""

    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    seen: set[str] = set()
    for row in rows:
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id in seen:
            raise ValueError("reference rows require unique candidate IDs")
        seen.add(candidate_id)
        length = int(row["length"])
        if length <= 0:
            raise ValueError("reference peptide length must be positive")
        grouped[length].append(row)
    if not grouped:
        raise ValueError("frozen reference requires at least one row")

    groups: dict[str, Any] = {}
    for length, values in sorted(grouped.items()):
        properties: dict[str, list[float]] = {}
        for visible_name, raw_name in RAW_TO_VISIBLE.items():
            observed = sorted(float(row[raw_name]) for row in values)
            if any(not math.isfinite(value) for value in observed):
                raise ValueError(f"non-finite p0 reference value for {raw_name}")
            properties[visible_name] = observed
        groups[str(length)] = {
            "count": len(values),
            "sorted_reference_values": properties,
        }
    return {
        "schema_version": 1,
        "method": "p0_length_stratified_tie_aware_empirical_tertiles",
        "raw_to_visible": dict(RAW_TO_VISIBLE),
        "groups": groups,
        "cardinal_values_private": True,
        "reference_distribution": "released unconditional PepDFM p0",
    }


def frozen_band(value: float, reference: Sequence[float]) -> str:
    """Apply the same tie-aware empirical-percentile convention as V12."""

    scalar = float(value)
    values = tuple(float(item) for item in reference)
    if not values or not math.isfinite(scalar) or any(
        not math.isfinite(item) for item in values
    ):
        raise ValueError("frozen-band inputs must be finite and non-empty")
    lower = sum(item < scalar for item in values)
    tied = sum(item == scalar for item in values)
    percentile = (lower + 0.5 * tied) / len(values)
    if percentile < 1 / 3:
        return "lower_third"
    if percentile < 2 / 3:
        return "middle_third"
    return "upper_third"


def apply_frozen_reference(
    rows: Sequence[Mapping[str, Any]],
    reference: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    """Return LLM-visible qualitative bands keyed by candidate ID."""

    if reference.get("schema_version") != 1 or reference.get("method") != (
        "p0_length_stratified_tie_aware_empirical_tertiles"
    ):
        raise ValueError("unsupported frozen developability reference")
    groups = reference.get("groups")
    if not isinstance(groups, Mapping):
        raise TypeError("frozen reference groups must be an object")
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        candidate_id = str(row.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id in result:
            raise ValueError("application rows require unique candidate IDs")
        length = str(int(row["length"]))
        group = groups.get(length)
        if not isinstance(group, Mapping):
            raise ValueError(f"no frozen p0 reference for peptide length {length}")
        properties = group.get("sorted_reference_values")
        if not isinstance(properties, Mapping):
            raise TypeError("frozen reference properties must be an object")
        result[candidate_id] = {
            visible_name: frozen_band(
                float(row[raw_name]), properties[visible_name]
            )
            for visible_name, raw_name in RAW_TO_VISIBLE.items()
        }
    return result


__all__ = [
    "RAW_TO_VISIBLE",
    "apply_frozen_reference",
    "fit_frozen_reference",
    "frozen_band",
]
