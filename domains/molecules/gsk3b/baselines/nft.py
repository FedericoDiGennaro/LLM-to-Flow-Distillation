#!/usr/bin/env python3
"""Run a fixed number of online GSK3B-only DiffusionNFT rounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from domains.molecules.gsk3b.oracle import MODEL_ID, REWARD_ENDPOINT
from domains.molecules.gsk3b.config import load_frozen_or_campaign_config
from domains.molecules.gsk3b.gates import require_noncollapse
from domains.molecules.gsk3b.artifacts import immutable_json, read_json
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
    if config.get("preference_source") != "tdc_gsk3b_only":
        raise PermissionError("GSK3B NFT requires preference_source=tdc_gsk3b_only")
    training = config["fine_tuning"]
    if training.get("kind") != "flowmol3_flow_matching_nft_v1":
        raise PermissionError("GSK3B NFT requires FlowMol3 DiffusionNFT")
    verify_authorization(
        profile, config, output, args.authorization.resolve(),
        args.active_campaign.resolve(),
        expected_scopes=("gsk3b_only_diffusion_nft_baseline",),
    )
    immutable_json(frozen, config)
    released = args.checkpoint.resolve()
    policy = released
    collection = released
    python = args.flowmol_python.absolute()
    batch_size = int(config["fresh_sampling"]["batch_size"])
    n_timesteps = int(config["fresh_sampling"]["flow_timesteps"])
    rounds = int(training["rounds"])
    method = f"gsk3b_diffusion_nft_{rounds}_rounds"
    rollout_count = int(config["candidate_pool"]["candidate_count"])
    round_records = []
    for iteration in range(1, rounds + 1):
        round_root = output / f"rounds/round-{iteration:02d}"
        seed = int(training["rollout_seeds"][iteration - 1])
        rollout_cohort = generate_cohort(
            checkpoint=collection, root=round_root / "rollout-cohort",
            count=rollout_count, batch_size=batch_size, seed=seed,
            n_timesteps=n_timesteps, role="training", config_path=frozen,
            python=python, n_atoms_hist=args.n_atoms_hist.resolve(),
            marginal_dists=args.marginal_dists.resolve(),
        )
        rollout_path = round_root / "rollout.json"
        if not rollout_path.is_file():
            _run([
                str(python), "-m", "domains.molecules.gsk3b.baselines.nft_manifest",
                "rollout", "--samples-dir", str(rollout_cohort["generation"]),
                "--arm", f"nft-round-{iteration:02d}", "--seed", str(seed),
                "--collection-checkpoint", str(collection),
                "--group-size", str(rollout_count), "--output", str(rollout_path),
            ])
        reward_root = round_root / "gsk3b-reward"
        if not (reward_root / "gsk3b_reward.json").is_file():
            _run([
                str(args.tdc_python.absolute()), "-m", "domains.molecules.gsk3b.reward",
                "--rollout", str(rollout_path), "--oracle", str(args.oracle.resolve()),
                "--output-dir", str(reward_root), "--reward-floor",
                str(float(config["gsk3b_reward"]["reward_floor"])),
            ], timeout=7200)
        reward = read_json(reward_root / "gsk3b_reward.json")
        if reward.get("reward_model_id") != MODEL_ID:
            raise ValueError("DiffusionNFT reward model drifted")
        inputs = round_root / "training-inputs"
        if not (inputs / "manifest.json").is_file():
            _run([
                str(python), "-m", "domains.molecules.gsk3b.baselines.nft_manifest",
                "manifest", "--rollout", str(rollout_path),
                "--reward", str(reward_root / "gsk3b_reward.json"),
                "--config", str(profile),
                "--collection-checkpoint", str(collection),
                "--output-dir", str(inputs),
            ])
        update = round_root / "update"
        if not (update / "training_result.json").is_file():
            _run([
                str(python), "-m", "domains.molecules.gsk3b.baselines.nft_campaign",
                "--contract", str(inputs / "contract.json"),
                "--manifest", str(inputs / "manifest.json"),
                "--parent-checkpoint", str(policy),
                "--collection-checkpoint", str(collection),
                "--n-atoms-hist", str(args.n_atoms_hist.resolve()),
                "--marginal-dists", str(args.marginal_dists.resolve()),
                "--output-dir", str(update), "--iteration", str(iteration),
                "--seed", str(int(training["update_seeds"][iteration - 1])),
                "--maximum-relative-parameter-delta",
                str(float(training["maximum_relative_parameter_delta"])),
            ], timeout=28800)
        policy = update / "last.ckpt"
        collection = update / "collection.ckpt"
        record = {
            "round": iteration,
            "rollout_seed": seed,
            "reward_mean": float(reward["samples"] and sum(
                float(row["raw_reward"]) for row in reward["samples"]
            ) / len(reward["samples"])),
            "policy": str(policy),
            "collection": str(collection),
            "training_result": str(update / "training_result.json"),
        }
        immutable_json(round_root / "round_result.json", record)
        round_records.append(record)

    eval_seed = int(config["fresh_sampling"]["round_1_seeds"][0])
    count = int(config["fresh_sampling"]["num_attempts_per_arm"])
    parent_dev = generate_cohort(
        checkpoint=released, root=output / "development/parent", count=count,
        batch_size=batch_size, seed=eval_seed, n_timesteps=n_timesteps,
        role="development", config_path=frozen, python=python,
        n_atoms_hist=args.n_atoms_hist.resolve(),
        marginal_dists=args.marginal_dists.resolve(),
    )
    child_dev = generate_cohort(
        checkpoint=policy, root=output / "development/child", count=count,
        batch_size=batch_size, seed=eval_seed, n_timesteps=n_timesteps,
        role="development", config_path=frozen, python=python,
        n_atoms_hist=args.n_atoms_hist.resolve(),
        marginal_dists=args.marginal_dists.resolve(),
        priors_from=parent_dev["generation"] / "samples.pt",
    )
    noncollapse = output / "noncollapse.json"
    if not noncollapse.is_file():
        _run([
            str(python), "-m", "domains.molecules.gsk3b.validate_samples",
            "--parent-samples", str(parent_dev["generation"] / "samples.pt"),
            "--frozen-p0-samples", str(parent_dev["generation"] / "samples.pt"),
            "--child-samples", str(child_dev["generation"] / "samples.pt"),
            "--parent-checkpoint", str(released), "--child-checkpoint", str(policy),
            "--config", str(frozen), "--output", str(noncollapse),
        ])
    require_noncollapse(noncollapse, inputs={
        "parent_checkpoint": released, "child_checkpoint": policy,
        "parent_samples": parent_dev["generation"] / "samples.pt",
        "frozen_p0_samples": parent_dev["generation"] / "samples.pt",
        "child_samples": child_dev["generation"] / "samples.pt",
    })
    parent_reward = score_pool(
        tdc_python=args.tdc_python,
        pool=parent_dev["pool"] / "candidate_pool.jsonl", oracle=args.oracle,
        output=output / "evaluation/parent-gsk3b", config=config,
        induce_preferences=False,
    )
    child_reward = score_pool(
        tdc_python=args.tdc_python,
        pool=child_dev["pool"] / "candidate_pool.jsonl", oracle=args.oracle,
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
    result = {
        "schema_version": 1,
        "artifact_type": "flowmol_gsk3b_diffusion_nft_baseline_v1",
        "experiment_id": config["experiment_id"],
        "method": method,
        "training_reward": REWARD_ENDPOINT,
        "completed_round_count": rounds,
        "rounds": round_records,
        "final_model": str(policy),
        "final_collection_model": str(collection),
        "final_cohort_receipt": str(
            (output / "development/child/cohort_receipt.json").resolve()
        ),
        "noncollapse": str(noncollapse),
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
