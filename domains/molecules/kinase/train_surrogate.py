#!/usr/bin/env python3
"""Distil a registered terminal reward onto FlowMol soft trajectory states."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from domains.molecules.kinase.guidance import SoftGraphRewardRegressor
from domains.molecules.kinase.artifacts import immutable_json, read_json, sha256_file


def _batch_previous(transitions: list[Mapping[str, Any]], device: torch.device):
    previous = [transition["previous_endpoint"] for transition in transitions]
    if all(value is None for value in previous):
        return None
    if any(value is None for value in previous):
        raise ValueError("a predictor cache batch mixed initial and later states")
    return {
        feature: torch.cat([value[feature] for value in previous], dim=0).to(device)
        for feature in ("x", "a", "c", "e")
    }


def _split_endpoint(
    endpoint: Mapping[str, torch.Tensor],
    *,
    node_counts: list[int],
    upper_edge_counts: list[int],
) -> list[dict[str, torch.Tensor]]:
    node_parts = {
        feature: list(torch.split(endpoint[feature].detach().cpu(), node_counts))
        for feature in ("a", "c")
    }
    edge_parts = list(
        torch.split(endpoint["e"].detach().cpu(), upper_edge_counts)
    )
    return [
        {
            "a": node_parts["a"][index].contiguous(),
            "c": node_parts["c"][index].contiguous(),
            "e": edge_parts[index].contiguous(),
        }
        for index in range(len(node_counts))
    ]


def _precompute_endpoints(
    *,
    generator: Any,
    trajectory_samples: list[Mapping[str, Any]],
    transition_count: int,
    batch_size: int,
    device: torch.device,
) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
    import dgl
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    cache: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
    for step in range(transition_count):
        for start in range(0, len(trajectory_samples), batch_size):
            indices = list(range(start, min(start + batch_size, len(trajectory_samples))))
            transitions = [trajectory_samples[index]["transitions"][step] for index in indices]
            graph = dgl.batch([transition["graph"] for transition in transitions]).to(device)
            node_batch_index, _ = get_batch_idxs(graph)
            upper_edge_mask = get_upper_edge_mask(graph)
            times = torch.tensor(
                [float(transition["time"]) for transition in transitions],
                device=device,
                dtype=graph.ndata["x_t"].dtype,
            )
            previous = _batch_previous(transitions, device)
            with torch.no_grad():
                endpoint = generator.vector_field(
                    graph,
                    times,
                    node_batch_idx=node_batch_index,
                    upper_edge_mask=upper_edge_mask,
                    apply_softmax=True,
                    remove_com=True,
                    prev_dst_dict=previous,
                )
            node_counts = [int(value) for value in graph.batch_num_nodes().tolist()]
            upper_edge_counts = [
                int(value // 2) for value in graph.batch_num_edges().tolist()
            ]
            parts = _split_endpoint(
                endpoint,
                node_counts=node_counts,
                upper_edge_counts=upper_edge_counts,
            )
            for index, part in zip(indices, parts):
                cache[(index, step)] = part
        print(
            json.dumps(
                {
                    "cache_transition_step": step + 1,
                    "transition_count": transition_count,
                    "cached_states": len(cache),
                }
            ),
            flush=True,
        )
    expected = len(trajectory_samples) * transition_count
    if len(cache) != expected:
        raise RuntimeError(f"cached {len(cache)} states, expected {expected}")
    return cache


def _collate(
    *,
    examples: list[tuple[int, int]],
    trajectory_samples: list[Mapping[str, Any]],
    clean_molecules: list[Any],
    transition_count: int,
    endpoint_cache: Mapping[tuple[int, int], Mapping[str, torch.Tensor]],
    normalized_rewards: torch.Tensor,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    import dgl
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    source_graphs = [
        (
            clean_molecules[molecule].g
            if step == transition_count
            else trajectory_samples[molecule]["transitions"][step]["graph"]
        )
        for molecule, step in examples
    ]
    graphs = []
    for source_graph in source_graphs:
        source, destination = source_graph.edges(form="uv")
        graphs.append(
            dgl.graph(
                (source.detach().cpu(), destination.detach().cpu()),
                num_nodes=source_graph.num_nodes(),
            )
        )
    graph = dgl.batch(graphs).to(device)
    node_batch_index, edge_batch_index = get_batch_idxs(graph)
    upper_edge_mask = get_upper_edge_mask(graph)
    source, destination = graph.edges(form="uv")
    endpoints = [endpoint_cache[example] for example in examples]
    return {
        "atom_probabilities": torch.cat([row["a"] for row in endpoints]).to(device),
        "charge_probabilities": torch.cat([row["c"] for row in endpoints]).to(device),
        "bond_probabilities": torch.cat([row["e"] for row in endpoints]).to(device),
        "node_batch_index": node_batch_index,
        "upper_edge_source": source[upper_edge_mask],
        "upper_edge_destination": destination[upper_edge_mask],
        "upper_edge_batch_index": edge_batch_index[upper_edge_mask],
        "times": torch.tensor(
            [
                1.0
                if step == transition_count
                else float(trajectory_samples[molecule]["transitions"][step]["time"])
                for molecule, step in examples
            ],
            device=device,
            dtype=torch.float32,
        ),
        "target": torch.tensor(
            [float(normalized_rewards[molecule]) for molecule, _step in examples],
            device=device,
            dtype=torch.float32,
        ),
    }


def _predict(predictor: SoftGraphRewardRegressor, batch: Mapping[str, torch.Tensor]):
    return predictor(
        atom_probabilities=batch["atom_probabilities"],
        charge_probabilities=batch["charge_probabilities"],
        bond_probabilities=batch["bond_probabilities"],
        node_batch_index=batch["node_batch_index"],
        upper_edge_source=batch["upper_edge_source"],
        upper_edge_destination=batch["upper_edge_destination"],
        upper_edge_batch_index=batch["upper_edge_batch_index"],
        times=batch["times"],
    )


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        return 0.0
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if float(a.std()) == 0.0 or float(b.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--n-atoms-hist", type=Path, required=True)
    parser.add_argument("--marginal-dists", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-batch-size", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FlowMol reward-predictor training requires CUDA")

    config_path = args.config.resolve()
    config = read_json(config_path)
    reward_model_id = str(config.get("preference_source", "")).strip()
    if not reward_model_id:
        raise ValueError("guidance config must register preference_source")
    training_data = config["training_data"]
    training = config["reward_predictor"]
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    if args.cache_batch_size < 1:
        raise ValueError("cache batch size must be positive")

    seed = int(training["training_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")

    from flowmol.models.flowmol import FlowMol

    generator = FlowMol.load_from_checkpoint(
        str(args.parent_checkpoint.resolve()),
        n_atoms_hist_file=str(args.n_atoms_hist.resolve()),
        marginal_dists_file=str(args.marginal_dists.resolve()),
    ).to(device).eval()
    if generator.parameterization != "ctmc":
        raise ValueError("reward guidance requires the released FlowMol CTMC")
    for parameter in generator.parameters():
        parameter.requires_grad_(False)

    trajectory_path = args.trajectory.resolve()
    trajectory = torch.load(trajectory_path, map_location="cpu")
    trajectory_samples = list(trajectory["samples"])
    distinct_count = int(training_data["distinct_oracle_scored_molecules"])
    transition_count = int(training_data["trajectory_states_per_molecule"])
    if len(trajectory_samples) != distinct_count:
        raise ValueError("trajectory molecule count differs from registered data budget")
    if any(len(row["transitions"]) != transition_count for row in trajectory_samples):
        raise ValueError("trajectory state count differs from registered data budget")
    parent_hash = sha256_file(args.parent_checkpoint.resolve())
    if trajectory.get("checkpoint_sha256") != parent_hash:
        raise ValueError("trajectory was not collected from the registered parent")

    samples_path = args.samples.resolve()
    clean_samples = torch.load(samples_path, map_location="cpu")
    clean_molecules = list(clean_samples.get("molecules", []))
    if len(clean_molecules) != distinct_count:
        raise ValueError("clean endpoint count differs from registered data budget")

    reward_path = args.reward.resolve()
    reward = read_json(reward_path)
    if reward.get("reward_model_id") != reward_model_id:
        raise ValueError("unexpected reward model")
    if reward.get("failed_candidates_scored") is not True:
        raise ValueError("reward artifact dropped failed candidates")
    reward_by_id = {
        str(row["candidate_id"]): float(row["raw_reward"])
        for row in reward["samples"]
    }
    candidate_ids = [str(row["candidate_id"]) for row in trajectory_samples]
    if set(candidate_ids) != set(reward_by_id):
        raise ValueError("trajectory and reward candidate sets differ")
    raw_rewards = torch.tensor(
        [reward_by_id[candidate] for candidate in candidate_ids], dtype=torch.float32
    )
    log_scale = float(training["target_log_scale"])
    transformed = torch.log1p(raw_rewards / log_scale)
    target_mean = float(transformed.mean())
    target_std = float(transformed.std(unbiased=False))
    if not target_std > 0:
        raise ValueError("registered reward has no variation")
    normalized_rewards = (transformed - target_mean) / target_std

    endpoint_cache = _precompute_endpoints(
        generator=generator,
        trajectory_samples=trajectory_samples,
        transition_count=transition_count,
        batch_size=args.cache_batch_size,
        device=device,
    )
    from flowmol.data_processing.utils import get_upper_edge_mask

    for molecule_index, molecule in enumerate(clean_molecules):
        graph = molecule.g
        upper_edge_mask = get_upper_edge_mask(graph)
        # CTMC terminal tensors include a final mask column; the denoiser's
        # endpoint distributions contain only the real categories.
        endpoint_cache[(molecule_index, transition_count)] = {
            "a": graph.ndata["a_1"][:, :-1].detach().cpu().float().contiguous(),
            "c": graph.ndata["c_1"][:, :-1].detach().cpu().float().contiguous(),
            "e": graph.edata["e_1"][upper_edge_mask, :-1]
            .detach()
            .cpu()
            .float()
            .contiguous(),
        }
    first_endpoint = endpoint_cache[(0, 0)]
    predictor = SoftGraphRewardRegressor(
        atom_dim=int(first_endpoint["a"].shape[-1]),
        charge_dim=int(first_endpoint["c"].shape[-1]),
        bond_dim=int(first_endpoint["e"].shape[-1]),
        hidden_dim=int(training["hidden_dim"]),
        message_layers=int(training["message_layers"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        predictor.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    batch_size = int(training["batch_size"])
    optimizer_steps = int(training["optimizer_steps"])
    gradient_clip = float(training["gradient_clip_norm"])
    if optimizer_steps > 1600:
        raise ValueError("reward-guidance predictor exceeds the 1,600-step cap")

    split_generator = torch.Generator().manual_seed(int(training_data["split_seed"]))
    permutation = torch.randperm(distinct_count, generator=split_generator).tolist()
    validation_count = int(training_data["validation_molecules"])
    validation_molecules = sorted(permutation[:validation_count])
    training_molecules = sorted(permutation[validation_count:])
    train_examples = [
        (molecule, step)
        for molecule in training_molecules
        for step in range(transition_count + 1)
    ]
    validation_examples = [
        (molecule, step)
        for molecule in validation_molecules
        for step in range(transition_count + 1)
    ]
    rng = random.Random(seed)
    positive_molecules = [
        molecule for molecule in training_molecules if float(raw_rewards[molecule]) > 0
    ]
    zero_molecules = [
        molecule for molecule in training_molecules if float(raw_rewards[molecule]) == 0
    ]
    if not positive_molecules or not zero_molecules:
        raise ValueError("balanced reward-regressor training needs both reward strata")
    positive_fraction = float(training["positive_fraction_per_batch"])
    positive_per_batch = int(round(batch_size * positive_fraction))
    if not 0 < positive_per_batch < batch_size:
        raise ValueError("positive batch fraction must retain both reward strata")
    state_times = [
        float(trajectory_samples[0]["transitions"][step]["time"])
        for step in range(transition_count)
    ] + [1.0]
    state_weights = [max(value * value, 1e-6) for value in state_times]

    def sampled_example(molecules: list[int]) -> tuple[int, int]:
        return (
            rng.choice(molecules),
            rng.choices(range(transition_count + 1), weights=state_weights, k=1)[0],
        )

    metrics: list[dict[str, float | int]] = []
    predictor.train()
    for step_index in range(optimizer_steps):
        selected = [
            sampled_example(positive_molecules) for _ in range(positive_per_batch)
        ] + [
            sampled_example(zero_molecules)
            for _ in range(batch_size - positive_per_batch)
        ]
        batch = _collate(
            examples=selected,
            trajectory_samples=trajectory_samples,
            clean_molecules=clean_molecules,
            transition_count=transition_count,
            endpoint_cache=endpoint_cache,
            normalized_rewards=normalized_rewards,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        prediction = _predict(predictor, batch)
        regression_loss = torch.nn.functional.smooth_l1_loss(
            prediction, batch["target"]
        )
        paired_count = min(positive_per_batch, batch_size - positive_per_batch)
        ranking_loss = torch.nn.functional.softplus(
            -(prediction[:paired_count] - prediction[positive_per_batch : positive_per_batch + paired_count])
        ).mean()
        loss = regression_loss + float(training["pairwise_ranking_loss_weight"]) * ranking_loss
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("non-finite reward-regressor loss")
        loss.backward()
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), gradient_clip)
            .detach()
            .cpu()
        )
        if not math.isfinite(gradient_norm):
            raise FloatingPointError("non-finite reward-regressor gradient")
        optimizer.step()
        if step_index == 0 or (step_index + 1) % 20 == 0:
            row: dict[str, float | int] = {
                "optimizer_step": step_index + 1,
                "loss": float(loss.detach().cpu()),
                "regression_loss": float(regression_loss.detach().cpu()),
                "ranking_loss": float(ranking_loss.detach().cpu()),
                "preclip_gradient_norm": gradient_norm,
            }
            metrics.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    predictor.eval()
    validation_predictions: list[float] = []
    validation_targets: list[float] = []
    validation_raw_predictions: list[float] = []
    validation_raw_targets: list[float] = []
    with torch.no_grad():
        for start in range(0, len(validation_examples), batch_size):
            examples = validation_examples[start : start + batch_size]
            batch = _collate(
                examples=examples,
                trajectory_samples=trajectory_samples,
                clean_molecules=clean_molecules,
                transition_count=transition_count,
                endpoint_cache=endpoint_cache,
                normalized_rewards=normalized_rewards,
                device=device,
            )
            predictions = _predict(predictor, batch)
            validation_predictions.extend(predictions.cpu().tolist())
            validation_targets.extend(batch["target"].cpu().tolist())
            raw_prediction = (
                torch.expm1(predictions.cpu() * target_std + target_mean) * log_scale
            ).clamp_min(0.0)
            validation_raw_predictions.extend(raw_prediction.tolist())
            validation_raw_targets.extend(
                [float(raw_rewards[molecule]) for molecule, _step in examples]
            )
    normalized_mse = float(
        np.mean(
            (np.asarray(validation_predictions) - np.asarray(validation_targets)) ** 2
        )
    )
    raw_mae = float(
        np.mean(
            np.abs(
                np.asarray(validation_raw_predictions)
                - np.asarray(validation_raw_targets)
            )
        )
    )
    clean_indices = [
        index
        for index, (_molecule, state) in enumerate(validation_examples)
        if state == transition_count
    ]
    clean_raw_predictions = [validation_raw_predictions[index] for index in clean_indices]
    clean_raw_targets = [validation_raw_targets[index] for index in clean_indices]

    predictor_path = output_dir / "reward_predictor.pt"
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": "flowmol_soft_graph_reward_regressor",
            "algorithm": config["algorithm"],
            "architecture": predictor.architecture(),
            "state_dict": {
                name: value.detach().cpu()
                for name, value in predictor.state_dict().items()
            },
            "target_transform": {
                "kind": "standardized_log1p",
                "log_scale": log_scale,
                "mean": target_mean,
                "std": target_std,
            },
            "reward_model_id": reward_model_id,
            "parent_checkpoint": str(args.parent_checkpoint.resolve()),
            "parent_checkpoint_sha256": parent_hash,
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
        },
        predictor_path,
    )
    result = {
        "schema_version": 1,
        "artifact_type": "flowmol_reward_guidance_training",
        "algorithm": config["algorithm"],
        "reward_model_id": reward_model_id,
        "distinct_oracle_scored_molecules": distinct_count,
        "oracle_positive_reward_count": int((raw_rewards > 0).sum()),
        "trajectory_states_per_molecule": transition_count,
        "soft_state_observation_count": distinct_count * transition_count,
        "clean_endpoint_observation_count": distinct_count,
        "total_state_observation_count": distinct_count * (transition_count + 1),
        "training_molecule_count": len(training_molecules),
        "validation_molecule_count": len(validation_molecules),
        "split_by_molecule": True,
        "generator_optimizer_steps": 0,
        "predictor_optimizer_steps": optimizer_steps,
        "optimizer_step_cap": 1600,
        "reward_predictor": str(predictor_path.resolve()),
        "reward_predictor_sha256": sha256_file(predictor_path),
        "architecture": predictor.architecture(),
        "target_transform": {
            "kind": "standardized_log1p",
            "log_scale": log_scale,
            "mean": target_mean,
            "std": target_std,
        },
        "validation_normalized_mse": normalized_mse,
        "validation_normalized_pearson": _pearson(
            validation_predictions, validation_targets
        ),
        "validation_raw_reward_mae": raw_mae,
        "validation_raw_reward_pearson": _pearson(
            validation_raw_predictions, validation_raw_targets
        ),
        "validation_clean_endpoint_raw_reward_pearson": _pearson(
            clean_raw_predictions, clean_raw_targets
        ),
        "training_state_sampling": training["state_sampling"],
        "positive_fraction_per_batch": positive_fraction,
        "pairwise_ranking_loss_weight": float(training["pairwise_ranking_loss_weight"]),
        "trajectory": str(trajectory_path),
        "trajectory_sha256": sha256_file(trajectory_path),
        "samples": str(samples_path),
        "samples_sha256": sha256_file(samples_path),
        "reward": str(reward_path),
        "reward_sha256": sha256_file(reward_path),
        "parent_checkpoint": str(args.parent_checkpoint.resolve()),
        "parent_checkpoint_sha256": parent_hash,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "metrics": metrics,
        "held_out_docking_used_for_training_or_selection": False,
        "scientific_success_claimed": False,
    }
    immutable_json(output_dir / "training_report.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
