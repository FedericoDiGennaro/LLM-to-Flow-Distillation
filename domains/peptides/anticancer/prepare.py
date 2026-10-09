#!/usr/bin/env python3
"""Emit the two run-control records required by ``domains/peptides/anticancer/jobs/run.sh``.

The authorization and active-campaign records are runtime control inputs rather
than source, so the site launcher that produced them is not part of this branch.
This script emits exactly the fields checked by
``domains.peptides.preflight``,
``domains.peptides.select_lengths``,
``domains.peptides.anticancer.pool``,
``domains.peptides.anticancer.run``, and
``domains.peptides.training``.

Both records bind one fresh output root to the current clean commit and to the
committed config hashes, so they must be regenerated after every commit.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any, Mapping


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
LENGTH_POLICY_SELECTION_METHOD = (
    "one Gemini controller call under the registered bounded schema"
)
ROLE_NAMES = ("candidate_judge", "curriculum_controller", "independent_auditor")
ROLE_KEYS = ("model", "reasoning_effort", "maximum_provider_calls")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _write(path: Path, value: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n")
    return path.resolve()


def build(
    *, output_root: Path, deadline_utc: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the authorization and active-campaign records for one fresh run."""

    template_path = REPOSITORY_ROOT / "domains/peptides/anticancer/configs" / "dpo.json"
    pool_path = REPOSITORY_ROOT / "domains/peptides/anticancer/configs" / "pool.json"
    config = json.loads(template_path.read_text(encoding="utf-8"))
    if _git("status", "--porcelain"):
        raise SystemExit(
            "worktree is dirty; commit first so code_commit stays authoritative"
        )
    commit = _git("rev-parse", "HEAD")
    branch = _git("branch", "--show-current")
    template_name = str(template_path.relative_to(REPOSITORY_ROOT))
    pool_name = str(pool_path.relative_to(REPOSITORY_ROOT))
    template_sha256 = _sha256(template_path)
    pool_sha256 = _sha256(pool_path)
    campaign_output = str((output_root / "campaign").resolve())
    api = dict(config["aggregate_api_budget"])
    compute = dict(config["registered_compute_budget"])

    authorization = {
        "status": "approved",
        "experiment_id": str(config["experiment_id"]),
        "config_template_file": template_name,
        "config_template_file_sha256": template_sha256,
        "pool_config_file": pool_name,
        "pool_config_file_sha256": pool_sha256,
        "output_root": campaign_output,
        "code_commit": commit,
        **{
            role: {key: config[role][key] for key in ROLE_KEYS}
            for role in ROLE_NAMES
        },
        "aggregate_api_budget": api,
        "maximum_preflight_allocations": int(api["maximum_preflight_allocations"]),
        "deadline_utc": deadline_utc,
    }
    campaign = {
        "schema_version": 1,
        "status": "active",
        "allowed_code_roots": [REPOSITORY_ROOT.name],
        "experiment_id": str(config["experiment_id"]),
        "approval_gate": {"status": "approved"},
        "budget": {
            "deadline": deadline_utc,
            "max_jobs": max(
                int(compute["max_jobs"]),
                int(api["maximum_preflight_allocations"]),
            ),
            "max_gpu_hours": float(compute["max_gpu_hours"]),
            "max_parallel_jobs": int(compute["max_parallel_jobs"]),
            "max_paid_api_calls": int(api["maximum_paid_calls"]),
            "maximum_prompt_tokens": int(api["maximum_prompt_tokens"]),
            "maximum_output_tokens": int(api["maximum_output_tokens"]),
            "maximum_estimated_api_usd": float(api["maximum_estimated_usd"]),
        },
        "resources": {
            "config_template": template_name,
            "config_template_sha256": template_sha256,
            "pool_config": pool_name,
            "pool_config_sha256": pool_sha256,
            "derived_config_path": str(
                (output_root / "frozen_config.json").resolve()
            ),
            "output_root": campaign_output,
            "code_commit": commit,
            "code_branch": branch,
            "code_root": REPOSITORY_ROOT.name,
            "dynamic_length_policy": {
                "path": str((output_root / "campaign_length_policy.json").resolve()),
                "selection_method": LENGTH_POLICY_SELECTION_METHOD,
            },
        },
    }
    return authorization, campaign


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--valid-days",
        type=int,
        default=30,
        help="authorization/campaign lifetime in days from now",
    )
    args = parser.parse_args()
    deadline = (
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=args.valid_days)
    ).replace(microsecond=0)
    authorization, campaign = build(
        output_root=args.output_root.resolve(),
        deadline_utc=deadline.isoformat().replace("+00:00", "Z"),
    )
    authorization_path = _write(
        args.out_dir / "gemini_authorization.json", authorization
    )
    campaign_path = _write(args.out_dir / "ACTIVE_CAMPAIGN.json", campaign)
    print(
        json.dumps(
            {
                "experiment_id": authorization["experiment_id"],
                "authorization": str(authorization_path),
                "active_campaign": str(campaign_path),
                "code_commit": authorization["code_commit"],
                "deadline_utc": authorization["deadline_utc"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
