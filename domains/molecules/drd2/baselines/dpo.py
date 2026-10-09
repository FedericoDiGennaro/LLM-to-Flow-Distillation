#!/usr/bin/env python3
"""Run one median-reward DRD2 FlowMol3 DPO update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from domains.molecules.drd2.oracle import EXTREME_LABEL_RULE, PREFERENCE_RULE, REWARD_ENDPOINT
from domains.molecules.drd2.config import load_frozen_or_campaign_config
from domains.molecules.drd2.gates import require_noncollapse
from domains.molecules.drd2.artifacts import immutable_json, read_json
from domains.molecules.drd2.run import (
    _run,
    generate_cohort,
    verify_authorization,
)


def mean_attempt_reward(path: Path) -> dict[str, Any]:
    value = read_json(path)
    samples = list(value["samples"])
    denominator = int(value["sample_count"])
    rewards = [float(row["raw_reward"]) for row in samples]
    if len(rewards) != denominator:
        raise ValueError("DRD2 evaluation dropped generation attempts")
    return {
        "mean_reward_per_attempt": sum(rewards) / denominator,
        "scored_count": int(value["scored_count"]),
        "floored_count": int(value["floored_count"]),
        "attempt_count": denominator,
    }


def score_pool(
    *, tdc_python: Path, pool: Path, oracle: Path, output: Path,
    config: dict[str, Any], induce_preferences: bool = True,
) -> Path:
    target = output / "drd2_reward.json"
    if not target.is_file():
        command = [
            str(tdc_python.absolute()), "-m", "domains.molecules.drd2.reward",
            "--pool", str(pool), "--oracle", str(oracle.resolve()),
            "--output-dir", str(output), "--reward-floor",
            str(float(config["drd2_reward"]["reward_floor"])),
        ]
        if induce_preferences:
            command.extend([
                "--positive-count", str(int(config["drd2_reward"]["positive_count"])),
                "--negative-count", str(int(config["drd2_reward"]["negative_count"])),
            ])
        else:
            command.append("--no-preference-labels")
        _run(command, timeout=7200)
    return target


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
    if config.get("preference_source") != "tdc_olivecrona_drd2_current_fcfp6":
        raise PermissionError("DRD2 DPO requires preference_source=tdc_olivecrona_drd2_current_fcfp6")
    if config["fine_tuning"].get("kind") != "flowmol3_flow_dpo_v1":
        raise PermissionError("DRD2 DPO requires FlowMol3 flow-DPO")
    verify_authorization(
        profile, config, output, args.authorization.resolve(),
        args.active_campaign.resolve(),
        expected_scopes=("drd2_only_dpo_baseline",),
    )
    immutable_json(frozen, config)

    parent = args.checkpoint.resolve()
    flowmol_python = args.flowmol_python.absolute()
    n_timesteps = int(config["fresh_sampling"]["flow_timesteps"])
    batch_size = int(config["fresh_sampling"]["batch_size"])
    p0 = generate_cohort(
        checkpoint=parent, root=output / "p0",
        count=int(config["candidate_pool"]["candidate_count"]),
        batch_size=batch_size, seed=int(config["candidate_pool"]["seed"]),
        n_timesteps=n_timesteps, role="training", config_path=frozen,
        python=flowmol_python, n_atoms_hist=args.n_atoms_hist.resolve(),
        marginal_dists=args.marginal_dists.resolve(),
    )
    reward_path = score_pool(
        tdc_python=args.tdc_python, pool=p0["pool"] / "candidate_pool.jsonl",
        oracle=args.oracle, output=output / "drd2-training", config=config,
    )
    labels = read_json(reward_path)
    if labels.get("label_rule") != EXTREME_LABEL_RULE:
        raise ValueError("DRD2 DPO label rule drifted")
    if labels.get("preference_rule") != PREFERENCE_RULE:
        raise ValueError("DRD2 DPO preference rule drifted")
    if labels.get("visible_to_llm") is not False:
        raise PermissionError("DRD2 reward leaked to the LLM")
    required = {
        "positive": int(config["drd2_reward"]["positive_count"]),
        "negative": int(config["drd2_reward"]["negative_count"]),
    }
    if any(int(labels['counts'][label]) < count for label, count in required.items()):
        immutable_json(output / 'campaign_result.json', {
            'method': 'reward_dpo', 'status': 'no_update_insufficient_reward_classes',
            'counts': labels['counts'], 'training_reward': REWARD_ENDPOINT,
            'final_model': str(parent), 'provider_calls': 0,
        })
        return
    required = {label: int(labels['counts'][label]) for label in required}

    round_root = output / "rounds/round-01"
    training_result = round_root / "update/training_result.json"
    if not training_result.is_file():
        _run([
            str(flowmol_python), "-m", "domains.molecules.drd2.round",
            "--pool", str(p0["pool"] / "candidate_pool.jsonl"),
            "--stable-labels", str(output / "drd2-training/stable_summary.json"),
            "--parent-checkpoint", str(parent),
            "--n-atoms-hist", str(args.n_atoms_hist.resolve()),
            "--marginal-dists", str(args.marginal_dists.resolve()),
            "--config", str(profile), "--output-dir", str(round_root / "update"),
        ], timeout=28800)
    child = round_root / "update/last.ckpt"
    eval_seed = int(config["fresh_sampling"]["round_1_seeds"][0])
    eval_count = int(config["fresh_sampling"]["num_attempts_per_arm"])
    parent_dev = generate_cohort(
        checkpoint=parent, root=round_root / "development/parent",
        count=eval_count, batch_size=batch_size, seed=eval_seed,
        n_timesteps=n_timesteps, role="development", config_path=frozen,
        python=flowmol_python, n_atoms_hist=args.n_atoms_hist.resolve(),
        marginal_dists=args.marginal_dists.resolve(),
    )
    child_dev = generate_cohort(
        checkpoint=child, root=round_root / "development/child",
        count=eval_count, batch_size=batch_size, seed=eval_seed,
        n_timesteps=n_timesteps, role="development", config_path=frozen,
        python=flowmol_python, n_atoms_hist=args.n_atoms_hist.resolve(),
        marginal_dists=args.marginal_dists.resolve(),
        priors_from=parent_dev["generation"] / "samples.pt",
    )
    noncollapse = round_root / "noncollapse.json"
    if not noncollapse.is_file():
        _run([
            str(flowmol_python), "-m", "domains.molecules.drd2.validate_samples",
            "--parent-samples", str(parent_dev["generation"] / "samples.pt"),
            "--frozen-p0-samples", str(p0["generation"] / "samples.pt"),
            "--child-samples", str(child_dev["generation"] / "samples.pt"),
            "--parent-checkpoint", str(parent), "--child-checkpoint", str(child),
            "--config", str(frozen), "--output", str(noncollapse),
        ])
    require_noncollapse(noncollapse, inputs={
        "parent_checkpoint": parent, "child_checkpoint": child,
        "parent_samples": parent_dev["generation"] / "samples.pt",
        "frozen_p0_samples": p0["generation"] / "samples.pt",
        "child_samples": child_dev["generation"] / "samples.pt",
    })
    parent_reward = score_pool(
        tdc_python=args.tdc_python,
        pool=parent_dev["pool"] / "candidate_pool.jsonl", oracle=args.oracle,
        output=round_root / "evaluation/parent-drd2", config=config,
        induce_preferences=False,
    )
    child_reward = score_pool(
        tdc_python=args.tdc_python,
        pool=child_dev["pool"] / "candidate_pool.jsonl", oracle=args.oracle,
        output=round_root / "evaluation/child-drd2", config=config,
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
        "artifact_type": "flowmol_drd2_dpo_baseline_v1",
        "experiment_id": config["experiment_id"],
        "method": "one_step_drd2_reward_dpo",
        "training_reward": REWARD_ENDPOINT,
        "final_model": str(child),
        "final_cohort_receipt": str(
            (round_root / "development/child/cohort_receipt.json").resolve()
        ),
        "training_result": str(training_result),
        "noncollapse": str(noncollapse),
        "training_candidate_count": int(labels["sample_count"]),
        "positive_count": required["positive"],
        "negative_count": required["negative"],
        "pair_count": min(required["positive"] * required["negative"], int(config["fine_tuning"]["maximum_pair_count"])),
        "reward_metrics": metrics,
        "held_out_evaluation_pending": True,
        "gnina_used_during_adaptation": False,
        "provider_calls": 0,
        "scientific_success_claimed": False,
    }
    immutable_json(output / "campaign_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
