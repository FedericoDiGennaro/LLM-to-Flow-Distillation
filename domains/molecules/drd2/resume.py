"""Append-only, hash-bound protocol amendments at a completed round boundary."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from domains.molecules.drd2.gates import at_least
from domains.molecules.drd2.artifacts import immutable_json, read_json


ALLOWED_CHANGES = {
    "profile_source", "base_config",
    "geometric_control.minimum_new_goal_unlock",
    "curriculum_controller.api_budget.maximum_prompt_tokens",
    "curriculum_controller.api_budget.maximum_estimated_usd",
    "curriculum_controller.api_budget.input_usd_per_million_tokens",
    "curriculum_controller.api_budget.output_usd_per_million_tokens",
    "aggregate_api_budget.maximum_prompt_tokens",
    "aggregate_api_budget.maximum_estimated_usd",
}


def revision_id(config):
    value = config.get("resume_revision", {}).get("id", "")
    if value and not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", value):
        raise ValueError("unsafe resume revision id")
    return value


def _differences(old, new, prefix=""):
    if isinstance(old, dict) and isinstance(new, dict):
        found = set()
        for key in old.keys() | new.keys():
            path = f"{prefix}.{key}" if prefix else key
            found |= _differences(old.get(key), new.get(key), path)
        return found
    return {prefix} if old != new else set()


def bind_configuration(config, root: Path, *, create: bool):
    """Return original config and effective snapshot; never rewrite history."""
    original_path = root / "frozen_config.json"
    amendment = config.get("resume_revision")
    if amendment is None:
        if create:
            immutable_json(original_path, config)
        original = read_json(original_path)
        if original != config:
            raise ValueError("materialized molecules config differs from its profile")
        return original, original_path
    identifier = revision_id(config)
    if not identifier:
        raise ValueError("resume revision requires an id")
    boundary = amendment["effective_from_round"]
    if type(boundary) is not int or boundary < 2:
        raise ValueError("revision must follow at least one completed round")
    if hashlib.sha256(original_path.read_bytes()).hexdigest() != amendment["original_frozen_sha256"]:
        raise ValueError("original frozen config hash differs")
    original = read_json(original_path)
    candidate = {k: v for k, v in config.items() if k != "resume_revision"}
    unexpected = _differences(original, candidate) - ALLOWED_CHANGES
    if unexpected:
        raise ValueError(f"unapproved protocol changes: {sorted(unexpected)}")
    for index in range(1, boundary):
        path = root / "rounds" / f"round-{index:02d}" / "round_result.json"
        row = read_json(path)
        if row.get("round") != index:
            raise ValueError("revision prefix is not contiguous")
    if hashlib.sha256(path.read_bytes()).hexdigest() != amendment["last_completed_round_sha256"]:
        raise ValueError("last completed round hash differs")
    snapshot = root / "revisions" / identifier / "frozen_config.json"
    if not snapshot.exists():
        if not create:
            raise ValueError("resume revision snapshot missing")
        if (root / "campaign_result.json").exists():
            raise ValueError("cannot amend a terminal campaign")
        for round_path in (root / "rounds").glob("round-*"):
            if int(round_path.name.split("-")[-1]) >= boundary:
                if (round_path / "round_result.json").exists() or (round_path / "update").exists():
                    raise ValueError("revision boundary already trained")
        immutable_json(snapshot, config)
    if read_json(snapshot) != config:
        raise ValueError("resume revision snapshot differs")
    return original, snapshot


def control_for_round(config, original, index):
    boundary = config.get("resume_revision", {}).get("effective_from_round", 0)
    return (original if index < boundary else config)["geometric_control"]


def clears_unlock(value, threshold, config, index):
    # Preserve exact historical decisions before the registered amendment.
    boundary = config.get("resume_revision", {}).get("effective_from_round")
    if boundary is not None and index < boundary:
        return value >= threshold
    return at_least(value, threshold)


def goal_passed(control, movement):
    """Apply the task's registered curriculum advancement rule."""
    threshold = float(control['minimum_enriched_positive_fraction'])
    if control.get('enrichment_rule') == 'strict_absolute_fraction':
        return float(movement['child_q']) > threshold
    return (at_least(float(movement['child_q']), threshold)
            and at_least(float(movement['child_minus_adoption']), float(control['minimum_enrichment'])))
