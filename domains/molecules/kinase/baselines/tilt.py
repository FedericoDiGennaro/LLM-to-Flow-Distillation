#!/usr/bin/env python3
"""Finite-pool KINASE sampling-importance-resampling baseline for FlowMol3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from domains.molecules.kinase.oracle import REWARD_ENDPOINT
from domains.molecules.kinase.config import load_frozen_or_campaign_config
from domains.molecules.kinase.artifacts import immutable_json, immutable_jsonl, read_json, read_jsonl
from domains.molecules.kinase.baselines.dpo import score_pool
from domains.molecules.kinase.run import generate_cohort, verify_authorization


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
    if config.get("preference_source") != "kinase_link_relaxed_binary_prompt_exact_v1":
        raise PermissionError("KINASE tilt requires preference_source=kinase_link_relaxed_binary_prompt_exact_v1")
    if config.get("adaptation_method") != "finite_pool_sir":
        raise PermissionError("KINASE tilt requires finite_pool_sir")
    verify_authorization(
        profile, config, output, args.authorization.resolve(),
        args.active_campaign.resolve(),
        expected_scopes=("kinase_only_reward_tilt_baseline",),
    )
    immutable_json(frozen, config)
    pool_count = int(config["reward_tilt"]["pool_count"])
    selected_count = int(config["reward_tilt"]["selected_count"])
    cohort = generate_cohort(
        checkpoint=args.checkpoint.resolve(), root=output / "source-pool",
        count=pool_count, batch_size=int(config["fresh_sampling"]["batch_size"]),
        seed=int(config["candidate_pool"]["seed"]),
        n_timesteps=int(config["fresh_sampling"]["flow_timesteps"]),
        role="training", config_path=frozen, python=args.flowmol_python.absolute(),
        n_atoms_hist=args.n_atoms_hist.resolve(),
        marginal_dists=args.marginal_dists.resolve(),
    )
    pool_path = cohort["pool"] / "candidate_pool.jsonl"
    reward_path = score_pool(
        tdc_python=args.tdc_python, pool=pool_path, oracle=args.oracle,
        output=output / "kinase-source-pool", config=config,
        induce_preferences=False,
    )
    reward = read_json(reward_path)
    rows = read_jsonl(pool_path.with_name("attempt_records.jsonl"))
    rows = [{**row, "item_id": row.get("item_id", row["candidate_id"])} for row in rows]
    by_item = {str(row["item_id"]): row for row in reward["samples"]}
    weights = np.asarray([
        float(by_item[str(row["item_id"])]["raw_reward"]) for row in rows
    ], dtype=np.float64)
    if not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise RuntimeError("KINASE tilt source pool has invalid or empty weights")
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(int(config["reward_tilt"]["resampling_seed"]))
    selected_indices = rng.choice(
        len(rows), size=selected_count, replace=True, p=probabilities
    )
    selected = []
    for draw, source_index in enumerate(selected_indices.tolist()):
        source = rows[int(source_index)]
        reward_row = by_item[str(source["item_id"])]
        selected.append({
            **source,
            "tilt_draw_index": draw,
            "source_pool_index": int(source_index),
            "source_pool_probability": float(probabilities[source_index]),
            "source_reward": float(reward_row["raw_reward"]),
            "resampling_seed": int(config["reward_tilt"]["resampling_seed"]),
        })
    immutable_jsonl(output / "selected_cohort.jsonl", selected)
    parent_rewards = weights[:selected_count].tolist()
    child_rewards = [float(row["source_reward"]) for row in selected]
    metrics = {
        "parent": {
            "mean_reward_per_attempt": float(np.mean(parent_rewards)),
            "attempt_count": selected_count,
            "scored_count": sum(bool(by_item[str(row["item_id"])]["scored_by_model"])
                                for row in rows[:selected_count]),
            "floored_count": sum(not bool(by_item[str(row["item_id"])]["scored_by_model"])
                                 for row in rows[:selected_count]),
        },
        "child": {
            "mean_reward_per_attempt": float(np.mean(child_rewards)),
            "attempt_count": selected_count,
            "scored_count": sum(
                bool(by_item[str(rows[int(index)]["item_id"])]["scored_by_model"])
                for index in selected_indices
            ),
            "floored_count": sum(
                not bool(by_item[str(rows[int(index)]["item_id"])]["scored_by_model"])
                for index in selected_indices
            ),
        },
    }
    metrics["child_minus_parent"] = (
        metrics["child"]["mean_reward_per_attempt"]
        - metrics["parent"]["mean_reward_per_attempt"]
    )
    positive = probabilities[probabilities > 0]
    result: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "flowmol_kinase_finite_pool_sir_baseline_v1",
        "experiment_id": config["experiment_id"],
        "method": "kinase_reward_tilt_10x_pool_sir",
        "training_reward": REWARD_ENDPOINT,
        "source_pool_count": pool_count,
        "selected_count": selected_count,
        "with_replacement": True,
        "selected_unique_source_count": len(set(selected_indices.tolist())),
        "normalized_weight_ess": float(1.0 / np.square(probabilities).sum()),
        "maximum_normalized_weight": float(probabilities.max()),
        "minimum_positive_normalized_weight": float(positive.min()),
        "reward_metrics": metrics,
        "selected_cohort": str((output / "selected_cohort.jsonl").resolve()),
        "final_model": None,
        "held_out_evaluation_pending": True,
        "gnina_used_during_adaptation": False,
        "provider_calls": 0,
        "scientific_success_claimed": False,
    }
    immutable_json(output / "campaign_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
