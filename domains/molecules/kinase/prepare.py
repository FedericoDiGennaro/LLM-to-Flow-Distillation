#!/usr/bin/env python3
"""Create runtime authorization records for the separate molecules LTFD pipeline."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

from adm.preferences.clients import canonical_json
from domains.molecules.kinase.config import load_campaign_config
from domains.molecules.kinase.artifacts import immutable_json


ROOT = Path(__file__).resolve().parents[3]


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=ROOT, text=True).strip()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("run", "smoke", "kinase_dpo", "kinase_nft", "kinase_tilt", "kinase_guidance", "kinase_grpo"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--valid-days", type=int, default=30)
    parser.add_argument("--deadline-utc", help="Explicit campaign deadline, if earlier than valid-days")
    args = parser.parse_args()
    config_path = args.config.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"molecules campaign config is missing: {config_path}")
    config = load_campaign_config(config_path)
    dirty = bool(_git("status", "--porcelain"))
    if dirty:
        raise SystemExit("commit changes before binding a reproducible run")
    deadline = (
        dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=args.valid_days)
    ).replace(microsecond=0)
    if args.deadline_utc:
        declared = dt.datetime.fromisoformat(args.deadline_utc.replace("Z", "+00:00"))
        if declared.tzinfo is None or declared <= dt.datetime.now(dt.timezone.utc):
            raise ValueError("authorization deadline must be timezone-aware and future")
        deadline = min(deadline, declared)
    output_root = args.output_root.resolve()
    authorization = {
        "schema_version": 1,
        "status": "approved",
        "authorization_scope": {
            "smoke": "engineering_smoke",
            "run": "full_ltfd",
            "kinase_dpo": "kinase_only_dpo_baseline",
            "kinase_nft": "kinase_only_diffusion_nft_baseline",
            "kinase_tilt": "kinase_only_reward_tilt_baseline",
            "kinase_guidance": "kinase_only_distilled_guidance_baseline",
            "kinase_grpo": "kinase_only_flow_grpo_baseline",
        }[args.mode],
        "experiment_id": config["experiment_id"],
        "config_file": str(config_path),
        "config_file_sha256": _sha256(config_path),
        "resolved_config_sha256": hashlib.sha256(
            canonical_json(config).encode("utf-8")
        ).hexdigest(),
        "output_root": str(output_root),
        "code_commit": _git("rev-parse", "HEAD"),
        "code_branch": _git("branch", "--show-current"),
        "dirty_worktree_allowed": False,
        "worktree_dirty_at_authorization": dirty,
        "aggregate_api_budget": config["aggregate_api_budget"],
        "compute_budget": config["registered_compute_budget"],
        "deadline_utc": deadline.isoformat().replace("+00:00", "Z"),
    }
    campaign = {
        "schema_version": 1,
        "status": "active",
        "experiment_id": config["experiment_id"],
        "approval_gate": {"status": "approved"},
        "output_root": str(output_root),
        "config_file_sha256": authorization["config_file_sha256"],
        "deadline_utc": authorization["deadline_utc"],
        "authorization_scope": authorization["authorization_scope"],
    }
    immutable_json(args.out_dir.resolve() / "gemini_authorization.json", authorization)
    immutable_json(args.out_dir.resolve() / "ACTIVE_CAMPAIGN.json", campaign)
    print(json.dumps(authorization, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
