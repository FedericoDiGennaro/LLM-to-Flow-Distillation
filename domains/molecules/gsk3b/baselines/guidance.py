#!/usr/bin/env python3
"""Train and apply an explicitly oracle-distilled FlowMol3 guidance baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from domains.molecules.gsk3b.oracle import REWARD_ENDPOINT
from domains.molecules.gsk3b.config import load_frozen_or_campaign_config
from domains.molecules.gsk3b.artifacts import immutable_json, read_json, read_jsonl, sha256_file
from domains.molecules.gsk3b.baselines.dpo import mean_attempt_reward, score_pool
from domains.molecules.gsk3b.run import (
    _run, generate_cohort, verify_authorization,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-atoms-hist", type=Path, required=True)
    parser.add_argument("--marginal-dists", type=Path, required=True)
    parser.add_argument("--flowmol-python", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--active-campaign", type=Path, required=True)
    parser.add_argument("--tdc-python", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    args = parser.parse_args()
    profile = args.config.resolve()
    output = args.output_root.resolve()
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "frozen_config.json"
    config = load_frozen_or_campaign_config(profile, frozen)
    if config.get("adaptation_method") != "oracle_distilled_endpoint_guidance":
        raise PermissionError("wrong GSK3B guidance adaptation method")
    verify_authorization(
        profile, config, output, args.authorization.resolve(),
        args.active_campaign.resolve(),
        expected_scopes=("gsk3b_only_distilled_guidance_baseline",),
    )
    immutable_json(frozen, config)
    checkpoint = args.checkpoint.resolve()
    python = args.flowmol_python.absolute()
    batch_size = int(config["fresh_sampling"]["batch_size"])
    timesteps = int(config["fresh_sampling"]["flow_timesteps"])
    guidance = config["distilled_guidance"]
    source = generate_cohort(
        checkpoint=checkpoint, root=output / "distillation-source",
        count=int(guidance["distillation_candidate_count"]),
        batch_size=batch_size, seed=int(config["candidate_pool"]["seed"]),
        n_timesteps=timesteps, role="training", config_path=frozen,
        python=python, n_atoms_hist=args.n_atoms_hist.resolve(),
        marginal_dists=args.marginal_dists.resolve(),
    )
    source_reward = score_pool(
        tdc_python=args.tdc_python,
        pool=source["pool"] / "candidate_pool.jsonl", oracle=args.oracle,
        output=output / "distillation-source-gsk3b", config=config,
    )
    # v1 was trained before the CTMC mask-category contract was made explicit.
    # Keep it immutable for diagnosis and write the corrected surrogate beside it.
    surrogate_root = output / "surrogate-v2"
    surrogate_path = surrogate_root / "surrogate.pt"
    if not surrogate_path.is_file():
        _run([
            str(python), "-m", "domains.molecules.gsk3b.train_surrogate",
            "--samples", str(source["generation"] / "samples.pt"),
            "--reward", str(source_reward), "--output-dir", str(surrogate_root),
            "--seed", str(int(guidance["distillation_seed"])),
            "--steps", str(int(guidance["training_steps"])),
            "--learning-rate", str(float(guidance["learning_rate"])),
            "--holdout-fraction", str(float(guidance["holdout_fraction"])),
            "--minimum-holdout-spearman",
            str(float(guidance["minimum_holdout_spearman"])),
        ], timeout=7200)
    validation = read_json(surrogate_root / "validation.json")
    if validation.get("guidance_gate_passed") is not True:
        raise RuntimeError("distilled classifier did not pass its held-out fidelity gate")
    count = int(config["fresh_sampling"]["num_attempts_per_arm"])
    eval_seed = int(config["fresh_sampling"]["round_1_seeds"][0])
    parent = generate_cohort(
        checkpoint=checkpoint, root=output / "development/parent", count=count,
        batch_size=batch_size, seed=eval_seed, n_timesteps=timesteps,
        role="development", config_path=frozen, python=python,
        n_atoms_hist=args.n_atoms_hist.resolve(),
        marginal_dists=args.marginal_dists.resolve(),
    )
    guided_generation = output / "development/child/generation"
    if not (guided_generation / "samples.pt").is_file():
        _run([
            str(python), "-m", "domains.molecules.gsk3b.sample_guided",
            "--checkpoint", str(checkpoint),
            "--n-atoms-hist", str(args.n_atoms_hist.resolve()),
            "--marginal-dists", str(args.marginal_dists.resolve()),
            "--surrogate", str(surrogate_path),
            "--output-dir", str(guided_generation), "--count", str(count),
            "--batch-size", str(batch_size), "--seed", str(eval_seed),
            "--n-timesteps", str(timesteps),
            "--categorical-scale", str(float(guidance["categorical_scale"])),
            "--coordinate-scale", str(float(guidance["coordinate_scale"])),
            "--priors-from", str(parent["generation"] / "samples.pt"),
        ], timeout=7200)
    guided_pool = output / "development/child/pool"
    if not (guided_pool / "pool_manifest.json").is_file():
        _run([
            str(python), "-m", "domains.molecules.gsk3b.pool",
            "--samples", str(guided_generation / "samples.pt"),
            "--config", str(frozen), "--output-dir", str(guided_pool),
            "--python", str(python), "--role", "development",
        ], timeout=max(3600, 120 * count))
    child_receipt = output / "development/child/cohort_receipt.json"
    immutable_json(child_receipt, {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "generation": str(guided_generation),
        "pool": str(guided_pool),
        "samples_sha256": sha256_file(guided_generation / "samples.pt"),
        "pool_manifest_sha256": sha256_file(guided_pool / "pool_manifest.json"),
        "seed": eval_seed,
        "attempt_count": count,
        "generation_kind": "oracle_distilled_guided_sampling",
    })
    parent_reward = score_pool(
        tdc_python=args.tdc_python,
        pool=parent["pool"] / "candidate_pool.jsonl", oracle=args.oracle,
        output=output / "evaluation/parent-gsk3b", config=config,
        induce_preferences=False,
    )
    child_reward = score_pool(
        tdc_python=args.tdc_python,
        pool=guided_pool / "candidate_pool.jsonl", oracle=args.oracle,
        output=output / "evaluation/child-gsk3b", config=config,
        induce_preferences=False,
    )
    metrics = {
        "parent": mean_attempt_reward(parent_reward),
        "child": mean_attempt_reward(child_reward),
    }
    metrics["child_minus_parent"] = (
        metrics["child"]["mean_reward_per_attempt"]
        - metrics["parent"]["mean_reward_per_attempt"]
    )
    child_rows = read_jsonl(guided_pool / "candidate_pool.jsonl")
    result = {
        "schema_version": 1,
        "artifact_type": "flowmol_gsk3b_distilled_guidance_baseline_v1",
        "experiment_id": config["experiment_id"],
        "method": "gsk3b_oracle_distilled_endpoint_guidance",
        "training_reward": REWARD_ENDPOINT,
        "final_model": str(checkpoint),
        "final_cohort": str((guided_pool / "candidate_pool.jsonl").resolve()),
        "final_cohort_receipt": str(child_receipt.resolve()),
        "surrogate": str(surrogate_path),
        "surrogate_sha256": sha256_file(surrogate_path),
        "surrogate_validation": validation,
        "judgeable_child_count": sum(row.get("status") == "judgeable" for row in child_rows),
        "reward_metrics": metrics,
        "held_out_evaluation_pending": True,
        "gnina_used_during_adaptation": False,
        "solubility_used_during_adaptation": False,
        "provider_calls": 0,
        "scientific_success_claimed": False,
    }
    immutable_json(output / "campaign_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
