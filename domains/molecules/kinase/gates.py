"""CPU-only gates shared by execution and offline validation."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

from domains.molecules.kinase.artifacts import read_json, sha256_file


NONCOLLAPSE_CHECKS = frozenset({
    "checkpoint_changed", "prior_policy_satisfied", "child_native_states_unique",
    "exact_native_replay_within_limit", "rdkit_success_within_limit",
})


def at_least(value: float, threshold: float) -> bool:
    """Inclusive numerical boundary, without relaxing a scientific threshold."""
    return math.isfinite(value) and math.isfinite(threshold) and (
        value >= threshold
        or math.isclose(value, threshold, rel_tol=0.0, abs_tol=1e-12)
    )


def noncollapse_valid(receipt: Mapping[str, Any]) -> bool:
    checks = receipt.get("checks")
    return (
        receipt.get("all_engineering_gates_passed") is True
        and isinstance(checks, Mapping)
        and set(checks) == NONCOLLAPSE_CHECKS
        and all(value is True for value in checks.values())
    )


def require_noncollapse(path: Path, *, inputs: Mapping[str, Path]) -> None:
    """A saved failure is not a completed gate; bind a pass to its actual inputs."""
    receipt = read_json(path)
    if not noncollapse_valid(receipt):
        raise ValueError(f"noncollapse gate did not pass: {path}")
    for name, source in inputs.items():
        if receipt.get(f"{name}_sha256") != sha256_file(source):
            raise ValueError(f"noncollapse receipt {name} binding differs: {path}")


def judgments_learnable(
    result: Mapping[str, Any], discovery: Mapping[str, Any],
    *, count_limits: Mapping[str, Any] | None = None,
) -> bool:
    limits = discovery if count_limits is None else count_limits
    fractions = result.get("abstain_fraction_by_pass", [])
    agreement = result["exact_agreement"]
    return (
        result["counts"]["positive"] >= int(limits["minimum_stable_positive_count"])
        and result["counts"]["negative"] >= int(limits["minimum_stable_negative_count"])
        and 0 <= agreement <= 1
        and at_least(agreement, float(discovery["minimum_exact_agreement"]))
        and len(fractions) == 2
        and all(0 <= value <= 1 for value in fractions)
        and at_least(float(discovery["maximum_abstain_fraction_per_pass"]), max(fractions))
    )
