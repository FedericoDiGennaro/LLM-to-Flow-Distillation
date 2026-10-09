#!/usr/bin/env python3
"""Select and freeze the one campaign-wide PepDFM length policy for V13."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping

from adm.preferences.gemini_vertex import GeminiUsageBudget, GeminiVertexClient
from adm.preferences.provider import bind_provider, make_structured_client
from domains.peptides.length_policy import select_pre_p0_length_policy
from adm.campaign import resolve_campaign_registration
from adm.artifacts import (
    _model_version_path,
    bind_model_version,
    read_json,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _usage_budget(role: Mapping[str, Any]) -> GeminiUsageBudget:
    value = role["api_budget"]
    return GeminiUsageBudget(
        maximum_prompt_tokens=int(value["maximum_prompt_tokens"]),
        maximum_output_tokens=int(value["maximum_output_tokens"]),
        maximum_estimated_usd=float(value["maximum_estimated_usd"]),
        input_usd_per_million_tokens=float(value["input_usd_per_million_tokens"]),
        output_usd_per_million_tokens=float(value["output_usd_per_million_tokens"]),
        prompt_token_overhead_per_request=int(
            value["prompt_token_overhead_per_request"]
        ),
    )


def _git_state(root: Path) -> tuple[str, bool]:
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).strip()
    )
    return commit, dirty


def _validate_authorization(
    *,
    config: Mapping[str, Any],
    config_path: Path,
    artifact_path: Path,
    output_root: Path,
    authorization_path: Path,
    campaign_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    authorization = read_json(authorization_path)
    campaign = read_json(campaign_path)
    experiment_id = str(config.get("experiment_id", ""))
    commit, dirty = _git_state(repository_root)
    if dirty:
        raise PermissionError("pre-p0 length selection requires a clean commit")
    if (
        campaign.get("status") != "active"
        or authorization.get("status") != "approved"
        or authorization.get("experiment_id") != experiment_id
    ):
        raise PermissionError("V13 campaign is not active and approved")
    resources = resolve_campaign_registration(
        campaign, experiment_id=experiment_id
    ).resources
    policy_record = resources.get("dynamic_length_policy")
    if not isinstance(policy_record, Mapping):
        raise PermissionError("campaign omitted V13 dynamic length policy")
    expected_template = str(config_path.relative_to(repository_root))
    checks = {
        "config_template": expected_template,
        "config_template_sha256": _sha256(config_path),
        "code_commit": commit,
        "output_root": str(output_root.resolve()),
    }
    if any(resources.get(name) != expected for name, expected in checks.items()):
        raise PermissionError("campaign V13 resource binding differs")
    if (
        Path(os.path.expandvars(str(policy_record.get("path", "")))).resolve()
        != artifact_path.resolve()
        or policy_record.get("selection_method")
        != "one Gemini controller call under the registered bounded schema"
    ):
        raise PermissionError("campaign length-policy output binding differs")
    authorized = {
        "config_template_file": expected_template,
        "config_template_file_sha256": _sha256(config_path),
        "output_root": str(output_root.resolve()),
        "code_commit": commit,
    }
    if any(
        authorization.get(name) != expected
        for name, expected in authorized.items()
    ):
        raise PermissionError("Gemini authorization differs from V13 resources")
    deadline = dt.datetime.fromisoformat(
        str(authorization["deadline_utc"]).replace("Z", "+00:00")
    )
    if dt.datetime.now(dt.timezone.utc) > deadline:
        raise PermissionError("Gemini authorization expired")


def execute(
    *,
    config_path: Path,
    artifact_path: Path,
    output_root: Path,
    authorization_path: Path,
    campaign_path: Path,
) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("prompt_protocol", {}).get("version") != (
        "geometric-semantic-reachability-v13"
    ):
        raise ValueError("length selector requires the V13 config template")
    _validate_authorization(
        config=config,
        config_path=config_path,
        artifact_path=artifact_path,
        output_root=output_root,
        authorization_path=authorization_path,
        campaign_path=campaign_path,
    )
    role = config["curriculum_controller"]
    attempt_root = output_root / "api_attempts" / "curriculum_controller"
    attempts_before = len(list(attempt_root.glob("*/attempt_started.json")))
    maximum_calls = int(role["maximum_provider_calls"])
    if attempts_before > maximum_calls:
        raise PermissionError("controller attempt ledger exceeds its ceiling")
    bind_provider(output_root, legacy_glob="**/attempt_started.json")
    client = make_structured_client(
        timeout=float(config["transport"]["request_timeout_seconds"]),
        temperature=0.0,
        attempt_root=attempt_root,
        requests_per_minute=float(
            config["transport"]["requests_per_minute_per_role"]
        ),
        usage_budget=_usage_budget(role),
    )
    version_path = _model_version_path(
        output_root=output_root, requested_model=str(role["model"])
    )
    if version_path.is_file():
        version_binding = read_json(version_path)
        if version_binding.get("requested_model") != role["model"]:
            raise PermissionError("preflight model-version binding differs")
        client.observe_model_version(version_binding.get("observed_model_version"))
    artifact = select_pre_p0_length_policy(
        client=client,
        final_goal=str(config["final_goal"]),
        model=str(role["model"]),
        reasoning_effort=str(role["reasoning_effort"]),
        max_output_tokens=int(role["max_output_tokens"]),
        artifact_path=artifact_path,
        logical_context={
            "experiment_id": config["experiment_id"],
            "selection_stage": "before_p0",
        },
    )
    attempts_after = len(list(attempt_root.glob("*/attempt_started.json")))
    if attempts_after > maximum_calls or attempts_after - attempts_before > 1:
        raise PermissionError("pre-p0 selector exceeded the controller call ceiling")
    if artifact.model_metadata.observed_model_version:
        bind_model_version(
            output_root=output_root,
            role_name="curriculum_controller",
            requested_model=str(role["model"]),
            observed_versions=(artifact.model_metadata.observed_model_version,),
        )
    counts = {
        str(row.length): row.count for row in artifact.length_policy.allocations
    }
    return {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "artifact_path": str(artifact_path.resolve()),
        "artifact_sha256": _sha256(artifact_path),
        "length_counts": counts,
        "sample_count": 100,
        "policy_frozen_for_entire_campaign": True,
        "provider_calls_added": attempts_after - attempts_before,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--active-campaign", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            execute(
                config_path=args.config.resolve(),
                artifact_path=args.artifact.resolve(),
                output_root=args.output_root.resolve(),
                authorization_path=args.authorization.resolve(),
                campaign_path=args.active_campaign.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
