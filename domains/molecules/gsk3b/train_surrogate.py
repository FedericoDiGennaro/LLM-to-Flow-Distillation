#!/usr/bin/env python3
"""Distill TDC GSK3B into a differentiable FlowMol3 endpoint surrogate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from domains.molecules.gsk3b.guidance import EndpointRewardSurrogate, molecule_features
from domains.molecules.gsk3b.artifacts import (
    immutable_json,
    read_json,
    read_jsonl,
    sha256_file,
)


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def correlation(left: np.ndarray, right: np.ndarray) -> float:
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--holdout-fraction", type=float, default=0.2)
    parser.add_argument("--minimum-holdout-spearman", type=float, default=0.1)
    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    payload = torch.load(args.samples.resolve(), map_location="cpu", weights_only=False)
    all_molecules = list(payload["molecules"])
    reward = read_json(args.reward.resolve())
    pool_path = Path(str(reward["pool"])).resolve()
    pool_rows = read_jsonl(pool_path)
    pool_by_item = {str(row["item_id"]): row for row in pool_rows}
    if len(pool_by_item) != len(pool_rows):
        raise ValueError("surrogate source pool contains duplicate item ids")
    samples_sha256 = sha256_file(args.samples.resolve())
    molecules = []
    targets_list = []
    selected_indices = []
    for reward_row in reward["samples"]:
        item_id = str(reward_row["item_id"])
        if item_id not in pool_by_item:
            raise ValueError(f"reward item is absent from source pool: {item_id}")
        state = pool_by_item[item_id].get("training_state", {})
        if Path(str(state.get("samples_pt", ""))).resolve() != args.samples.resolve():
            raise ValueError(f"{item_id} is bound to a different samples artifact")
        if str(state.get("samples_pt_sha256", "")) != samples_sha256:
            raise ValueError(f"{item_id} samples artifact hash does not match")
        sample_index = int(state["sample_index"])
        if sample_index < 0 or sample_index >= len(all_molecules):
            raise IndexError(f"{item_id} sample index is out of range")
        selected_indices.append(sample_index)
        molecules.append(all_molecules[sample_index])
        targets_list.append(float(reward_row["raw_reward"]))
    if len(selected_indices) != len(set(selected_indices)):
        raise ValueError("surrogate reward maps multiple items to one generated sample")
    targets = torch.tensor(
        targets_list, dtype=torch.float32
    )
    if not molecules or len(molecules) != len(targets):
        raise ValueError("surrogate source mapping is empty or inconsistent")
    features = molecule_features(molecules).float()
    generator = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(len(targets), generator=generator)
    holdout_count = max(20, int(round(len(targets) * args.holdout_fraction)))
    holdout = order[:holdout_count]
    train = order[holdout_count:]
    mean = features[train].mean(0)
    scale = features[train].std(0, unbiased=False).clamp_min(1e-5)
    normalized = (features - mean) / scale
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EndpointRewardSurrogate(features.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    rng = torch.Generator(device="cpu").manual_seed(args.seed + 1)
    for _step in range(args.steps):
        selected = train[torch.randint(len(train), (min(64, len(train)),), generator=rng)]
        prediction = model(normalized[selected].to(device))
        loss = torch.nn.functional.mse_loss(prediction, targets[selected].to(device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    with torch.no_grad():
        prediction = model(normalized[holdout].to(device)).cpu().numpy()
    truth = targets[holdout].numpy()
    report = {
        "schema_version": 1,
        "artifact_type": "gsk3b_endpoint_surrogate_validation",
        "training_count": len(train),
        "holdout_count": len(holdout),
        "generated_sample_count": len(all_molecules),
        "rewarded_sample_count": len(molecules),
        "excluded_unjudgeable_count": len(all_molecules) - len(molecules),
        "seed": args.seed,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "holdout_rmse": float(np.sqrt(np.mean(np.square(prediction - truth)))),
        "holdout_pearson": correlation(prediction, truth),
        "holdout_spearman": correlation(rankdata(prediction), rankdata(truth)),
        "minimum_holdout_spearman": args.minimum_holdout_spearman,
        "oracle_reward_used_for_distillation": True,
        "solubility_or_docking_used": False,
        "scientific_success_claimed": False,
    }
    report["guidance_gate_passed"] = (
        np.isfinite(report["holdout_spearman"])
        and report["holdout_spearman"] >= args.minimum_holdout_spearman
    )
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / "validation.json", report)
    if not report["guidance_gate_passed"]:
        raise RuntimeError(f"distilled guidance fidelity gate failed: {report}")
    checkpoint = output / "surrogate.pt"
    if checkpoint.exists():
        raise FileExistsError(f"refusing to overwrite {checkpoint}")
    torch.save({
        "schema_version": 1,
        "feature_dim": int(features.shape[1]),
        "hidden_dim": 64,
        "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "feature_mean": mean,
        "feature_scale": scale,
        "validation": report,
        "samples_sha256": sha256_file(args.samples.resolve()),
        "reward_sha256": sha256_file(args.reward.resolve()),
    }, checkpoint)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
