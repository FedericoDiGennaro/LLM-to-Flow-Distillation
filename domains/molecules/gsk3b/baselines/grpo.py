#!/usr/bin/env python3
"""Run the preregistered GSK3B-only FlowGRPO-style FlowMol3 adaptation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from domains.molecules.gsk3b.oracle import REWARD_ENDPOINT
from domains.molecules.gsk3b.config import load_frozen_or_campaign_config
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
    if config.get("adaptation_method") not in {
        "flowgrpo_style_flow_matching_surrogate_v1",
        "flowgrpo_style_flow_matching_surrogate_v2",
    }:
        raise PermissionError("wrong FlowGRPO-style adaptation method")
    verify_authorization(
        profile, config, output, args.authorization.resolve(),
        args.active_campaign.resolve(),
        expected_scopes=("gsk3b_only_flow_grpo_baseline",),
    )
    immutable_json(frozen, config)
    released = args.checkpoint.resolve()
    policy = released
    python = args.flowmol_python.absolute()
    batch_size = int(config["fresh_sampling"]["batch_size"])
    timesteps = int(config["fresh_sampling"]["flow_timesteps"])
    training = config["fine_tuning"]
    rounds = int(training["rounds"])
    rollout_count = int(config["candidate_pool"]["candidate_count"])
    records = []
    for iteration in range(1, rounds + 1):
        root = output / f"rounds/round-{iteration:02d}"
        rollout_seed = int(training["rollout_seeds"][iteration - 1])
        cohort = generate_cohort(
            checkpoint=policy, root=root / "rollout-cohort", count=rollout_count,
            batch_size=batch_size, seed=rollout_seed, n_timesteps=timesteps,
            role="training", config_path=frozen, python=python,
            n_atoms_hist=args.n_atoms_hist.resolve(),
            marginal_dists=args.marginal_dists.resolve(),
        )
        rollout = root / "rollout.json"
        if not rollout.is_file():
            _run([
                str(python), "-m", "domains.molecules.gsk3b.baselines.nft_manifest",
                "rollout", "--samples-dir", str(cohort["generation"]),
                "--arm", f"grpo-round-{iteration:02d}",
                "--seed", str(rollout_seed),
                "--collection-checkpoint", str(policy),
                "--group-size", str(int(training["group_size"])),
                "--output", str(rollout),
            ])
        reward_root = root / "gsk3b-reward"
        if not (reward_root / "gsk3b_reward.json").is_file():
            _run([
                str(args.tdc_python.absolute()), "-m", "domains.molecules.gsk3b.reward",
                "--rollout", str(rollout), "--oracle", str(args.oracle.resolve()),
                "--output-dir", str(reward_root), "--reward-floor",
                str(float(config["gsk3b_reward"]["reward_floor"])),
            ], timeout=7200)
        inputs = root / "training-inputs"
        if not (inputs / "manifest.json").is_file():
            _run([
                str(python), "-m", "domains.molecules.gsk3b.baselines.nft_manifest",
                "manifest", "--rollout", str(rollout),
                "--reward", str(reward_root / "gsk3b_reward.json"),
                "--config", str(profile), "--collection-checkpoint", str(policy),
                "--frozen-config", str(frozen),
                "--output-dir", str(inputs),
            ])
        update = root / "update"
        old = policy
        if not (update / "training_result.json").is_file():
            _run([
                str(python), "-m", "domains.molecules.gsk3b.baselines.grpo_update",
                "--manifest", str(inputs / "manifest.json"),
                "--policy-checkpoint", str(old), "--old-checkpoint", str(old),
                "--base-checkpoint", str(released),
                "--n-atoms-hist", str(args.n_atoms_hist.resolve()),
                "--marginal-dists", str(args.marginal_dists.resolve()),
                "--output-dir", str(update),
                "--seed", str(int(training["update_seeds"][iteration - 1])),
                "--steps", str(int(training["steps"])),
                "--batch-size", str(int(training["batch_size"])),
                "--learning-rate", str(float(training["learning_rate"])),
                "--clip-epsilon", str(float(training["clip_epsilon"])),
                "--kl-beta", str(float(training["kl_beta"])),
                "--log-density-scale", str(float(training["log_density_scale"])),
                "--initial-ratio-atol",
                str(float(training["initial_ratio_absolute_tolerance"])),
                "--maximum-relative-parameter-delta",
                str(float(training["maximum_relative_parameter_delta"])),
            ], timeout=28800)
        policy = update / "last.ckpt"
        update_result = read_json(update / "training_result.json")
        if update_result.get("on_policy_ratio_one_verified") is not True:
            raise RuntimeError("FlowGRPO-style on-policy ratio check failed")
        record = {
            "round": iteration, "rollout_seed": rollout_seed,
            "policy": str(policy),
            "reward": str(reward_root / "gsk3b_reward.json"),
            "training_result": str(update / "training_result.json"),
        }
        immutable_json(root / "round_result.json", record)
        records.append(record)
    count = int(config["fresh_sampling"]["num_attempts_per_arm"])
    eval_seed = int(config["fresh_sampling"]["round_1_seeds"][0])
    parent = generate_cohort(
        checkpoint=released, root=output / "development/parent", count=count,
        batch_size=batch_size, seed=eval_seed, n_timesteps=timesteps,
        role="development", config_path=frozen, python=python,
        n_atoms_hist=args.n_atoms_hist.resolve(),
        marginal_dists=args.marginal_dists.resolve(),
    )
    child = generate_cohort(
        checkpoint=policy, root=output / "development/child", count=count,
        batch_size=batch_size, seed=eval_seed, n_timesteps=timesteps,
        role="development", config_path=frozen, python=python,
        n_atoms_hist=args.n_atoms_hist.resolve(),
        marginal_dists=args.marginal_dists.resolve(),
        priors_from=parent["generation"] / "samples.pt",
    )
    parent_reward = score_pool(
        tdc_python=args.tdc_python, pool=parent["pool"] / "candidate_pool.jsonl",
        oracle=args.oracle, output=output / "evaluation/parent-gsk3b", config=config,
        induce_preferences=False,
    )
    child_reward = score_pool(
        tdc_python=args.tdc_python, pool=child["pool"] / "candidate_pool.jsonl",
        oracle=args.oracle, output=output / "evaluation/child-gsk3b", config=config,
        induce_preferences=False,
    )
    metrics = {"parent": mean_attempt_reward(parent_reward),
               "child": mean_attempt_reward(child_reward)}
    metrics["child_minus_parent"] = (
        metrics["child"]["mean_reward_per_attempt"]
        - metrics["parent"]["mean_reward_per_attempt"]
    )
    result = {
        "schema_version": 1,
        "artifact_type": "flowmol_gsk3b_flowgrpo_style_baseline_v1",
        "experiment_id": config["experiment_id"],
        "method": "gsk3b_flowgrpo_style_flow_matching_surrogate",
        "literal_flow_grpo_reproduction": False,
        "training_reward": REWARD_ENDPOINT,
        "completed_round_count": rounds,
        "rounds": records,
        "final_model": str(policy),
        "final_cohort_receipt": str(
            (output / "development/child/cohort_receipt.json").resolve()
        ),
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
