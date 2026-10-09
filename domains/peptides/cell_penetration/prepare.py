"""Prepare a fresh CPP run using the frozen EXP-0189 scientific inputs."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import subprocess

from adm.artifacts import immutable_json
from domains.peptides.cell_penetration.run import _authorized_length_policy_selection_method

ROOT = Path(__file__).resolve().parents[3]
TASK = Path(__file__).resolve().parent


def materialize(output: Path) -> tuple[dict, dict, dict]:
    output = output.resolve()
    template = json.loads((TASK / "configs/dpo.json").read_text().replace("${ACTFLOW_CPP_OUTPUT_ROOT}", str(output)))
    pool = json.loads((TASK / "configs/pool.json").read_text())
    policy = json.loads((TASK / "configs/length_policy.json").read_text())
    return template, pool, policy


def prepare(output: Path, deadline: str) -> dict:
    output = output.resolve()
    expiry = dt.datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    if expiry.tzinfo is None or expiry <= dt.datetime.now(dt.timezone.utc):
        raise ValueError("deadline must be timezone-aware and in the future")
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
    if git("status", "--porcelain"):
        raise ValueError("commit changes before binding a reproducible run")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("use a fresh output directory; existing runs are immutable")
    template, pool, policy = materialize(output)
    template_path = output / "template.json"
    pool_path = output / "pool_config.json"
    immutable_json(template_path, template)
    immutable_json(pool_path, pool)
    immutable_json(output / "length_policy.json", policy)
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    config_id = template["experiment_id"]
    commit = git("rev-parse", "HEAD")
    resource = {
        "config_template": str(template_path), "config_template_sha256": sha(template_path),
        "pool_config": str(pool_path), "pool_config_sha256": sha(pool_path),
        "derived_config_path": str(output / "frozen_config.json"),
        "output_root": str(output / "campaign"), "code_commit": commit,
        "code_branch": git("branch", "--show-current"), "code_root": ROOT.name,
        "dynamic_length_policy": {"path": str(output / "length_policy.json"),
            "selection_method": _authorized_length_policy_selection_method(template)},
    }
    api = template["aggregate_api_budget"]
    authorization = {"status": "approved", "experiment_id": config_id,
        "config_template_file": str(template_path), "config_template_file_sha256": sha(template_path),
        "pool_config_file": str(pool_path), "pool_config_file_sha256": sha(pool_path),
        "output_root": str(output / "campaign"), "code_commit": commit,
        "deadline_utc": expiry.isoformat(), "aggregate_api_budget": api,
        "maximum_preflight_allocations": api["maximum_preflight_allocations"],
        **{role: {key: template[role][key] for key in ("model", "reasoning_effort", "maximum_provider_calls")}
           for role in ("candidate_judge", "curriculum_controller", "independent_auditor")}}
    compute = template["registered_compute_budget"]
    campaign = {"schema_version": 1, "status": "active", "experiment_id": config_id,
        "allowed_code_roots": [ROOT.name], "approval_gate": {"status": "approved"}, "resources": resource,
        "budget": {"deadline": expiry.isoformat(), "max_jobs": compute["max_jobs"],
            "max_gpu_hours": compute["max_gpu_hours"], "max_parallel_jobs": compute["max_parallel_jobs"],
            "max_paid_api_calls": api["maximum_paid_calls"],
            "maximum_prompt_tokens": api["maximum_prompt_tokens"],
            "maximum_output_tokens": api["maximum_output_tokens"],
            "maximum_estimated_api_usd": api["maximum_estimated_usd"]}}
    immutable_json(output / "authorization.json", authorization)
    immutable_json(output / "ACTIVE_CAMPAIGN.json", campaign)
    return {"output_root": str(output), "experiment_id": config_id, "code_commit": commit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--deadline-utc", required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.output_root, args.deadline_utc), indent=2))


if __name__ == "__main__":
    main()
