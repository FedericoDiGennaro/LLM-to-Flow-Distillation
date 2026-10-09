#!/usr/bin/env python3
"""Bind a FlowMol3 rollout to KINASE rewards for DiffusionNFT."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from domains.molecules.kinase.baselines.nft_loss import validate_nft_preflight
from domains.molecules.kinase.oracle import HELD_OUT_EVALUATOR_IDS, MODEL_ID, REWARD_ENDPOINT
from domains.molecules.kinase.config import load_campaign_config
from domains.molecules.kinase.artifacts import immutable_json, read_json, sha256_file
from domains.molecules.kinase.baselines.manifest import build_rollout, join_rewards


def contract(config: Mapping[str, Any], collection: str) -> dict[str, Any]:
    training = config["fine_tuning"]
    if training.get("algorithm") != "diffusion_nft_v1":
        raise ValueError("KINASE NFT requires algorithm=diffusion_nft_v1")
    return {
        "schema_version": 1,
        "adapter_id": "flowmol3_flow_matching_nft_v1",
        "algorithm": "diffusion_nft_v1",
        "arm_role": "reward_optimization_baseline",
        "reward_model_id": MODEL_ID,
        "reward_endpoint": REWARD_ENDPOINT,
        "collection_checkpoint": collection,
        "beta": float(training["beta"]),
        "reward_normalizer": str(training.get("reward_normalizer", "global_std")),
        "soft_update_slope": float(training["soft_update_slope"]),
        "soft_update_maximum": float(training["soft_update_maximum"]),
        "learning_rate": float(training["learning_rate"]),
        "optimizer_steps": int(training["steps"]),
        "batch_size": int(training["batch_size"]),
        "gradient_accumulation_steps": int(training["gradient_accumulation_steps"]),
        "gradient_clip_norm": float(training.get("gradient_clip_norm", 1.0)),
        "minimum_sample_count": int(training["minimum_sample_count"]),
        "minimum_group_count": int(training.get("minimum_group_count", 1)),
        "nft_iteration": int(training.get("nft_iteration", 1)),
        "current_blocker": None,
    }


def manifest(
    rollout: Mapping[str, Any], reward: Mapping[str, Any],
    immutable_final_goal: str,
) -> dict[str, Any]:
    if reward.get("reward_model_id") != MODEL_ID or reward.get("reward_endpoint") != REWARD_ENDPOINT:
        raise ValueError("reward is not the registered KINASE model/endpoint")
    rollout_path = Path(str(reward.get("rollout", "")))
    if (not rollout_path.is_file()
            or reward.get("rollout_sha256") != sha256_file(rollout_path)
            or read_json(rollout_path) != rollout):
        raise ValueError("reward is not bound to this rollout's content")
    if reward.get("failed_candidates_scored") is not True:
        raise ValueError("KINASE reward dropped failed rollout candidates")
    if reward.get("usable_as_held_out_evaluator") is not False:
        raise ValueError("KINASE training reward claimed held-out status")
    samples = join_rewards(rollout, reward)
    return {
        "schema_version": 1,
        "artifact_type": "flowmol_nft_training_manifest",
        "arm": rollout["arm"],
        "seed": rollout["seed"],
        "reward_model_id": MODEL_ID,
        "reward_endpoint": REWARD_ENDPOINT,
        "held_out_evaluator_ids": list(HELD_OUT_EVALUATOR_IDS),
        "collection_checkpoint": rollout["collection_checkpoint"],
        "fresh_rollout": True,
        "training_state_bindings_verified": True,
        "failed_candidates_scored": True,
        "training_authorized": True,
        "scientific_qualification_passed": False,
        "training_goal": REWARD_ENDPOINT,
        "training_goal_kind": "fixed_reward_objective",
        "immutable_final_goal": immutable_final_goal,
        "rollout_path": reward["rollout"],
        "reward_mean": sum(float(row["raw_reward"]) for row in samples) / len(samples),
        "scored_count": sum(bool(row["scored_by_model"]) for row in samples),
        "floored_count": sum(not bool(row["scored_by_model"]) for row in samples),
        "samples": samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="phase", required=True)
    rollout_parser = sub.add_parser("rollout")
    rollout_parser.add_argument("--samples-dir", type=Path, required=True)
    rollout_parser.add_argument("--arm", required=True)
    rollout_parser.add_argument("--seed", type=int, required=True)
    rollout_parser.add_argument("--collection-checkpoint", required=True)
    rollout_parser.add_argument("--group-size", type=int, required=True)
    rollout_parser.add_argument("--output", type=Path, required=True)
    manifest_parser = sub.add_parser("manifest")
    manifest_parser.add_argument("--rollout", type=Path, required=True)
    manifest_parser.add_argument("--reward", type=Path, required=True)
    manifest_parser.add_argument("--config", type=Path, required=True)
    manifest_parser.add_argument("--frozen-config", type=Path)
    manifest_parser.add_argument("--collection-checkpoint", required=True)
    manifest_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.phase == "rollout":
        value = build_rollout(
            samples_dir=args.samples_dir.resolve(), arm=args.arm, seed=args.seed,
            collection_checkpoint=args.collection_checkpoint,
            group_size=args.group_size,
        )
        immutable_json(args.output.resolve(), value)
        print(json.dumps({"sample_count": value["sample_count"]}, indent=2))
        return
    rollout = read_json(args.rollout.resolve())
    reward = read_json(args.reward.resolve())
    config = (read_json(args.frozen_config.resolve()) if args.frozen_config
              else load_campaign_config(args.config.resolve()))
    training_manifest = manifest(rollout, reward, str(config["final_goal"]))
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    method = config.get("adaptation_method")
    if method in {"flowgrpo_style_flow_matching_surrogate_v1", "flowgrpo_style_flow_matching_surrogate_v2"}:
        if config["fine_tuning"]["algorithm"] != method:
            raise ValueError("GRPO adaptation method and training algorithm differ")
        training_manifest["algorithm"] = method
        training_manifest["literal_flow_grpo_reproduction"] = False
    else:
        training_contract = contract(config, args.collection_checkpoint)
        check = validate_nft_preflight(training_contract, training_manifest)
        if not check.ready:
            raise ValueError(str(check.blockers))
        immutable_json(output / "contract.json", training_contract)
    immutable_json(output / "manifest.json", training_manifest)
    print(json.dumps({
        "sample_count": len(training_manifest["samples"]),
        "reward_mean": training_manifest["reward_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()
