#!/usr/bin/env python3
"""Two-pass raw-sequence exact-F judgment for EXP-0107 baseline cohorts."""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

from domains.peptides.anticancer.baselines.acp_direct import prepare as p


def json_value(value):
    if is_dataclass(value):
        return {key: json_value(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if hasattr(value, "value"):
        return json_value(value.value)
    return value


def write_idempotent(path: Path, value) -> None:
    serialized = json.dumps(json_value(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text() != serialized:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    path.write_text(serialized)


def result_record(result) -> dict:
    return {
        "goal": result.goal, "exact_final_goal": result.exact_final_goal,
        "judgments": [json_value(row) for row in result.judgments],
        "request_hashes": list(result.request_hashes), "usage": json_value(result.usage),
        "model_versions": list(result.model_versions), "thought_summaries": list(result.thought_summaries),
        "coverage_recovery_request_hashes": list(result.coverage_recovery_request_hashes),
        "coverage_redispatch_request_hashes": list(result.coverage_redispatch_request_hashes),
    }


def stable_summary(first: dict, second: dict) -> dict:
    left = {row["candidate_id"]: row["label"] for row in first["judgments"]}
    right = {row["candidate_id"]: row["label"] for row in second["judgments"]}
    if set(left) != set(right) or len(left) != 100:
        raise ValueError("two judge passes do not cover the identical 100 candidates")
    outcomes = []
    counts = {"stable_positive": 0, "stable_negative": 0, "stable_abstain": 0, "unstable": 0}
    for candidate_id in sorted(left):
        if left[candidate_id] == right[candidate_id]:
            category = "stable_" + left[candidate_id]
        else:
            category = "unstable"
        counts[category] += 1
        outcomes.append({"candidate_id": candidate_id, "first_label": left[candidate_id],
                         "second_label": right[candidate_id], "stable_outcome": category})
    return {**counts, "sample_count": 100,
            "q_stable_positive_fraction_all_candidates": counts["stable_positive"] / 100,
            "exact_agreement_fraction": (100 - counts["unstable"]) / 100,
            "outcomes": outcomes}


def build_cohorts(config: dict, candidates: list[dict]) -> list[tuple[str, list[dict], str]]:
    cohorts = []
    for arm in p.ARMS:
        for replicate in range(3):
            rows = [row for row in candidates if row["arm_id"] == arm and row["replicate_index"] == replicate]
            cohorts.append((f"baseline-r{replicate}-{arm}", rows, "direct_reward_baseline"))
    control_path = config.get("endpoint_control_candidates")
    if control_path:
        control_path = Path(control_path)
        p.check(control_path, config["endpoint_control_candidates_sha256"])
        controls = p.read_json(control_path)
        for arm in ("initial", "signed84_final", "dpo96_final"):
            rows = [row for row in controls if row["arm_id"] == arm]
            cohorts.append((f"endpoint-control-{arm}", rows, "optional_endpoint_control"))
    for name, rows, _ in cohorts:
        if len(rows) != 100 or any(not row["is_valid"] for row in rows):
            raise ValueError(f"{name} must contain 100 valid unfiltered draws")
    return sorted(cohorts, key=lambda item: hashlib.sha256(
        f"{config['candidate_judge']['cohort_dispatch_seed']}:{item[0]}".encode()).hexdigest())


def validate_pass(record: dict, *, goal: str, expected_ids: set[str]) -> None:
    if record.get("goal") != goal or record.get("exact_final_goal") is not True:
        raise ValueError("cached judgment used a different goal or non-final regime")
    rows = record.get("judgments")
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("cached judgment row count differs")
    observed_ids = [row.get("candidate_id") for row in rows]
    if len(set(observed_ids)) != 100 or set(observed_ids) != expected_ids:
        raise ValueError("cached judgment candidate identities differ")
    if not record.get("request_hashes") or not record.get("model_versions"):
        raise ValueError("cached judgment lacks request or resolved model provenance")


def q_values_by_replicate(summaries: list[dict], arm: str) -> np.ndarray:
    records = sorted(
        (row for row in summaries
         if row["cohort_kind"] == "direct_reward_baseline" and row["arm_id"] == arm),
        key=lambda row: row["replicate_index"],
    )
    if [row["replicate_index"] for row in records] != [0, 1, 2]:
        raise ValueError(f"{arm} requires exactly replicate indices 0, 1, 2")
    return np.asarray([row["q_stable_positive_fraction_all_candidates"] for row in records])


def run(config_path: Path, validation: Path, output: Path) -> dict:
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"):
        raise EnvironmentError("tracked Slurm and exactly GEMINI_API_KEY are required")
    config = p.load_config(config_path); p.audit(config)
    completion = p.read_json(validation / "completion.json")
    p.check(validation / "candidates.json", completion["candidates_sha256"])
    candidates = p.read_json(validation / "candidates.json")
    output.mkdir(parents=True, exist_ok=True)
    source_root = Path(config["source_root"])
    sys.path.insert(0, str(source_root))
    from adm.preferences.artifacts import FilePreferenceArtifactStore
    from adm.preferences.gemini_vertex import GeminiUsageBudget, GeminiVertexClient
    from domains.peptides.types import PeptideCandidate
    from domains.peptides.judgments import V4CandidateJudge, V4_JUDGE_SYSTEM_PROMPT
    attempts = output / "api_attempts"
    initial_calls = len(list(attempts.glob("*/attempt_started.json"))) if attempts.exists() else 0
    role = config["candidate_judge"]
    budget = role["api_budget"]
    client = GeminiVertexClient(timeout=float(role["timeout_seconds"]), temperature=0.0,
        attempt_root=attempts, requests_per_minute=float(role["requests_per_minute"]),
        usage_budget=GeminiUsageBudget(maximum_prompt_tokens=budget["maximum_prompt_tokens"],
            maximum_output_tokens=budget["maximum_output_tokens"], maximum_estimated_usd=budget["maximum_estimated_usd"],
            input_usd_per_million_tokens=budget["input_usd_per_million_tokens"],
            output_usd_per_million_tokens=budget["output_usd_per_million_tokens"],
            prompt_token_overhead_per_request=budget["prompt_token_overhead_per_request"]))
    judge = V4CandidateJudge(client=client, artifact_store=FilePreferenceArtifactStore(output / "api_artifacts"),
        model=role["model"], reasoning_effort=role["reasoning_effort"], max_output_tokens=role["max_output_tokens"],
        max_provider_calls=role["maximum_provider_calls"], initial_provider_calls=initial_calls,
        chunk_size=role["chunk_size"], provider_order_seed=role["provider_order_seed"],
        system_prompt=V4_JUDGE_SYSTEM_PROMPT, use_generic_evidence_cards=False,
        require_model_version=True, maximum_coverage_redispatches_per_chunk=role["maximum_coverage_redispatches_per_chunk"])
    prompt_hash = hashlib.sha256(V4_JUDGE_SYSTEM_PROMPT.encode()).hexdigest()
    if prompt_hash != config["candidate_judge"]["system_prompt_sha256"]:
        raise ValueError("frozen exact-F prompt changed")
    started = {"experiment_id": config["experiment_id"], "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "slurm_job_id": os.environ["SLURM_JOB_ID"], "model": role["model"],
        "reasoning_effort": role["reasoning_effort"], "system_prompt_sha256": prompt_hash,
        "goal": config["final_goal"], "raw_sequences_only": True,
        "two_passes": True, "credential_value_recorded": False,
        "validation_candidates_sha256": completion["candidates_sha256"],
        "endpoint_control_candidates_sha256": config["endpoint_control_candidates_sha256"],
        "config_sha256": p.sha256(config_path),
        "cohort_dispatch_seed": role["cohort_dispatch_seed"]}
    started_path = output / "started.json"
    if started_path.exists():
        existing = p.read_json(started_path)
        for key in ("experiment_id", "model", "reasoning_effort", "system_prompt_sha256", "goal",
                    "raw_sequences_only", "two_passes", "validation_candidates_sha256",
                    "endpoint_control_candidates_sha256", "config_sha256", "cohort_dispatch_seed"):
            if existing.get(key) != started[key]:
                raise ValueError(f"resumed LLM input/config differs: {key}")
    else:
        write_idempotent(started_path, started)
    summaries = []
    observed_model_versions = set()
    for cohort_name, source_rows, cohort_kind in build_cohorts(config, candidates):
        samples = [PeptideCandidate(candidate_id=row["draw_id"], sequence=row["sequence"]) for row in source_rows]
        pass_records = []
        iteration = int.from_bytes(hashlib.sha256(cohort_name.encode()).digest()[:4], "big") & 0x7fffffff
        for repeat_index in (0, 1):
            pass_path = output / "passes" / cohort_name / f"pass-{repeat_index}.json"
            if pass_path.exists():
                record = p.read_json(pass_path)
            else:
                result = judge.judge_current_goal(config["final_goal"], samples, final_goal=config["final_goal"],
                    arm=cohort_name, iteration=iteration, repeat_index=repeat_index)
                record = result_record(result)
                write_idempotent(pass_path, record)
            validate_pass(record, goal=config["final_goal"],
                          expected_ids={sample.candidate_id for sample in samples})
            observed_model_versions.update(record["model_versions"])
            pass_records.append(record)
        stable = stable_summary(pass_records[0], pass_records[1])
        stable.update(cohort_name=cohort_name, cohort_kind=cohort_kind,
                      arm_id=source_rows[0]["arm_id"],
                      replicate_index=source_rows[0].get("replicate_index"))
        write_idempotent(output / "passes" / cohort_name / "stable_summary.json", stable)
        summaries.append(stable)
    aggregate = {}
    baseline_values = q_values_by_replicate(summaries, "unguided_p0")
    for arm in p.ARMS:
        values = q_values_by_replicate(summaries, arm)
        aggregate[arm] = {"mean_q": float(values.mean()), "sd_q_across_sampling_replicates": float(values.std(ddof=1)),
                          "q_values": values.tolist()}
        if arm != "unguided_p0":
            delta = values - baseline_values
            aggregate[arm].update(mean_paired_q_change_vs_unguided=float(delta.mean()),
                                  sd_paired_q_change_vs_unguided=float(delta.std(ddof=1)),
                                  paired_q_changes=delta.tolist())
    if len(observed_model_versions) != 1:
        raise ValueError(f"resolved Gemini model version changed across cohorts: {sorted(observed_model_versions)}")
    report = {"schema_version": 1, "experiment_id": config["experiment_id"], "completed": True,
        "primary_metric": "two-pass stable-positive fraction among all 100 raw-sequence candidates",
        "abstentions_and_disagreements_count_as_not_positive": True,
        "cohorts": summaries, "direct_baseline_aggregate": aggregate,
        "endpoint_controls": [row for row in summaries if row["cohort_kind"] != "direct_reward_baseline"],
        "historical_EXP0084_q_0_06_to_0_19_is_context_only": True,
        "comparison_note": "Fresh EXP-0106 p0/signed/DPO endpoint controls use this same raw-sequence prompt; historical 0.06/0.19 used its frozen campaign evidence context and is not substituted as a matched estimate.",
        "provider_calls_observed": judge.provider_calls, "prompt_sha256": prompt_hash,
        "resolved_model_version": next(iter(observed_model_versions)),
        "cohort_dispatch_order": [row[0] for row in build_cohorts(config, candidates)],
        "claim_boundary": "LLM sequence judgment is a semantic computational endpoint, not an assay or proof of anticancer efficacy."}
    write_idempotent(output / "report.json", report)
    write_idempotent(output / "completion.json", {"completed": True, "report_sha256": p.sha256(output / "report.json"),
        "provider_calls_observed": judge.provider_calls, "completed_at": dt.datetime.now(dt.timezone.utc).isoformat()})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.validation.resolve(), args.output.resolve())
    print(json.dumps({"completed": result["completed"], "provider_calls": result["provider_calls_observed"]}))


if __name__ == "__main__":
    main()
