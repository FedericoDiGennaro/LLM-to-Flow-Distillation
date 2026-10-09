"""Lightweight definitions for qualitative MOG-DFM evidence cards."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


DEVELOPABILITY_EVIDENCE_VERSION = "mog-dfm-within-length-tertiles-v1"
DEVELOPABILITY_EVIDENCE_LIMITATION = (
    "Qualitative within-length bands from the released MOG-DFM predictor heads. "
    "They are uncertain computational proxies, not assays, calibrated "
    "probabilities, anticancer evidence, healthy-cell selectivity evidence, or "
    "database retrieval. No labelled example or dataset row is shown."
)


def within_group_tertiles(
    rows: Sequence[Mapping[str, Any]], raw_name: str
) -> dict[str, str]:
    """Return tie-aware empirical tertiles for one pre-stratified group."""
    values = [float(row[raw_name]) for row in rows]
    if not values:
        raise ValueError("tertile group must not be empty")
    result: dict[str, str] = {}
    for row, value in zip(rows, values):
        candidate_id = str(row["candidate_id"])
        if candidate_id in result:
            raise ValueError("tertile group contains duplicate candidate IDs")
        lower = sum(other < value for other in values)
        tied = sum(other == value for other in values)
        percentile = (lower + 0.5 * tied) / len(values)
        result[candidate_id] = (
            "lower_third"
            if percentile < 1 / 3
            else "middle_third" if percentile < 2 / 3 else "upper_third"
        )
    return result
