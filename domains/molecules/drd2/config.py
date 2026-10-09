"""Shared immutable configuration helpers for the separate molecules LTFD pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[3]


def deep_merge(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_campaign_config(path: Path) -> dict[str, Any]:
    source = json.loads(path.read_text(encoding="utf-8"))
    base_name = source.get("base_config")
    if base_name is None:
        return source
    base = load_campaign_config((ROOT / str(base_name)).resolve())
    result = deep_merge(base, source.get("overrides", {}))
    replace_sections = source.get("replace_sections", [])
    if not isinstance(replace_sections, list) or not all(
        isinstance(section, str) for section in replace_sections
    ):
        raise ValueError("replace_sections must be a list of section names")
    for section in replace_sections:
        if section not in source.get("overrides", {}):
            raise ValueError(
                f"replace_sections names a section absent from overrides: {section}"
            )
        result[section] = copy.deepcopy(source["overrides"][section])
    for key in (
        "schema_version",
        "experiment_id",
        "execution_profile",
        "run_gnina_after_campaign",
        "preference_source",
        "adaptation_method",
        "execution_authorized",
        "claim_boundary",
    ):
        if key in source:
            result[key] = copy.deepcopy(source[key])
    result["profile_source"] = str(path.resolve())
    result["base_config"] = str((ROOT / str(base_name)).resolve())
    return result


def load_frozen_or_campaign_config(profile: Path, frozen_config: Path) -> dict[str, Any]:
    """Resume from the campaign snapshot if it already exists.

    Later edits to a shared base config must not rewrite a started run.
    """

    if frozen_config.is_file():
        return json.loads(frozen_config.read_text(encoding="utf-8"))
    return load_campaign_config(profile)


def _frozen_contains_overrides(
    frozen: Mapping[str, Any], overrides: Mapping[str, Any]
) -> bool:
    for key, value in overrides.items():
        current = frozen.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            if not _frozen_contains_overrides(current, value):
                return False
        elif current != value:
            return False
    return True


def load_frozen_campaign_bound_to_profile(
    profile: Path, frozen_config: Path
) -> dict[str, Any]:
    """Load the materialized run, checking only this profile's own bindings.

    Shared-base fields that the profile does not set may change later without
    invalidating a completed campaign snapshot.
    """

    frozen = json.loads(frozen_config.read_text(encoding="utf-8"))
    source = json.loads(profile.read_text(encoding="utf-8"))
    for key in (
        "schema_version",
        "experiment_id",
        "execution_profile",
        "run_gnina_after_campaign",
        "preference_source",
        "adaptation_method",
        "execution_authorized",
        "claim_boundary",
    ):
        if key in source and frozen.get(key) != source[key]:
            raise ValueError("materialized molecules config differs from its profile")
    if not _frozen_contains_overrides(frozen, source.get("overrides") or {}):
        raise ValueError("materialized molecules config differs from its profile")
    return frozen


__all__ = [
    "deep_merge",
    "load_campaign_config",
    "load_frozen_campaign_bound_to_profile",
    "load_frozen_or_campaign_config",
]
