#!/usr/bin/env python3
"""Run one on-policy FlowGRPO update from a bound FlowMol trajectory set."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn.functional as F

from domains.molecules.drd2.dpo import (
    relative_parameter_delta,
    save_reloadable_checkpoint,
    sha256_state_dict,
    verify_frozen_state,
)
from domains.molecules.drd2.baselines.grpo_loss import (
    categorical_kl_per_graph,
    categorical_log_prob_per_graph,
    clipped_grpo_loss,
    coordinate_transition_mean,
    flowgrpo_noise_std,
    gaussian_log_prob_per_graph,
    group_relative_advantages,
    joint_transition_statistic,
)
from domains.molecules.drd2.oracle import MODEL_ID
from domains.molecules.drd2.artifacts import immutable_json, read_json, sha256_file


def _load_flowmol(
    checkpoint: Path, n_atoms_hist: Path, marginal_dists: Path, device: torch.device
):
    from flowmol.models.flowmol import FlowMol

    return FlowMol.load_from_checkpoint(
        str(checkpoint),
        n_atoms_hist_file=str(n_atoms_hist),
        marginal_dists_file=str(marginal_dists),
    ).to(device)


def _categorical_policy(probabilities: torch.Tensor, temperature: Any) -> torch.Tensor:
    value = torch.as_tensor(
        temperature, device=probabilities.device, dtype=probabilities.dtype
    ).clamp_min(1e-6)
    return F.softmax(probabilities.clamp_min(1e-12).log() / value, dim=-1)


def _batch_previous(transitions: list[Mapping[str, Any]], device: torch.device):
    previous = [transition["previous_endpoint"] for transition in transitions]
    if all(value is None for value in previous):
        return None
    if any(value is None for value in previous):
        raise ValueError("a training batch mixed initial and non-initial transitions")
    return {
        name: torch.cat([value[name] for value in previous], dim=0).to(device)
        for name in ("x", "a", "c", "e")
    }


def _transition_statistics(
    *,
    policy: Any,
    reference: Any,
    transitions: list[Mapping[str, Any]],
    device: torch.device,
    noise_level: float,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    import dgl
    from flowmol.data_processing.utils import get_batch_idxs, get_upper_edge_mask

    graph = dgl.batch([transition["graph"] for transition in transitions]).to(device)
    graph_count = int(graph.batch_size)
    node_batch_idx, edge_batch_idx = get_batch_idxs(graph)
    upper_edge_mask = get_upper_edge_mask(graph)
    times = torch.tensor(
        [float(transition["time"]) for transition in transitions],
        device=device,
        dtype=graph.ndata["x_t"].dtype,
    )
    step_sizes = {float(transition["step_size"]) for transition in transitions}
    if len(step_sizes) != 1:
        raise ValueError("one training batch must share an integration step size")
    step_size = next(iter(step_sizes))
    previous = _batch_previous(transitions, device)
    policy_endpoint = policy.vector_field(
        graph,
        times,
        node_batch_idx=node_batch_idx,
        upper_edge_mask=upper_edge_mask,
        apply_softmax=True,
        remove_com=True,
        prev_dst_dict=previous,
    )
    with torch.no_grad():
        reference_endpoint = reference.vector_field(
            graph,
            times,
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
            prev_dst_dict=previous,
        )

    sigma_graph = flowgrpo_noise_std(
        times, step_size=step_size, noise_level=noise_level
    ).to(device=device, dtype=graph.ndata["x_t"].dtype)
    sigma_node = sigma_graph[node_batch_idx]
    time_node = times[node_batch_idx]
    policy_mean = coordinate_transition_mean(
        graph.ndata["x_t"],
        policy_endpoint["x"],
        time_per_node=time_node,
        step_size=step_size,
        noise_std_per_node=sigma_node,
    )
    with torch.no_grad():
        reference_mean = coordinate_transition_mean(
            graph.ndata["x_t"],
            reference_endpoint["x"],
            time_per_node=time_node,
            step_size=step_size,
            noise_std_per_node=sigma_node,
        )
    next_x = torch.cat([transition["next_x"] for transition in transitions]).to(device)
    transition_std = sigma_node * math.sqrt(step_size)
    coordinate_log_prob = gaussian_log_prob_per_graph(
        next_x,
        policy_mean,
        standard_deviation=transition_std,
        batch_index=node_batch_idx,
        graph_count=graph_count,
    )
    coordinate_kl = gaussian_log_prob_per_graph(
        policy_mean,
        policy_mean,
        standard_deviation=transition_std,
        batch_index=node_batch_idx,
        graph_count=graph_count,
    ) - gaussian_log_prob_per_graph(
        policy_mean,
        reference_mean,
        standard_deviation=transition_std,
        batch_index=node_batch_idx,
        graph_count=graph_count,
    )

    log_prob_components = [coordinate_log_prob]
    kl_components = [coordinate_kl]
    temperature = policy.vector_field.cat_temp_func(times[0])
    for feature in ("a", "c", "e"):
        policy_probabilities = _categorical_policy(policy_endpoint[feature], temperature)
        reference_probabilities = _categorical_policy(
            reference_endpoint[feature], temperature
        )
        batch_index = edge_batch_idx[upper_edge_mask] if feature == "e" else node_batch_idx
        actions = torch.cat(
            [transition[f"action_{feature}"] for transition in transitions]
        ).to(device)
        log_prob_components.append(
            categorical_log_prob_per_graph(
                policy_probabilities,
                actions,
                batch_index=batch_index,
                graph_count=graph_count,
            )
        )
        kl_components.append(
            categorical_kl_per_graph(
                policy_probabilities,
                reference_probabilities,
                batch_index=batch_index,
                graph_count=graph_count,
            )
        )
    return (
        joint_transition_statistic(log_prob_components),
        joint_transition_statistic(kl_components),
        {
            "coordinate_kl": coordinate_kl.detach(),
            "categorical_kl": joint_transition_statistic(kl_components[1:]).detach(),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--reward", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--n-atoms-hist", type=Path, required=True)
    parser.add_argument("--marginal-dists", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FlowMol FlowGRPO training requires CUDA")
    config = read_json(args.config.resolve())
    training = config["flowgrpo"]
    if training.get("algorithm") != "flow_grpo_ode_to_sde_v1":
        raise ValueError("unsupported FlowGRPO algorithm")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    seed = int(training["training_seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda:0")
    parent_path = args.parent_checkpoint.resolve()
    parent_hash = sha256_file(parent_path)
    policy = _load_flowmol(
        parent_path, args.n_atoms_hist.resolve(), args.marginal_dists.resolve(), device
    ).eval()
    reference = _load_flowmol(
        parent_path, args.n_atoms_hist.resolve(), args.marginal_dists.resolve(), device
    ).eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    if sha256_state_dict(policy) != sha256_state_dict(reference):
        raise RuntimeError("FlowGRPO policy/reference did not start identically")

    rollout = torch.load(args.trajectory.resolve(), map_location="cpu")
    if rollout.get("algorithm") not in (None, "flow_grpo_ode_to_sde_v1"):
        raise ValueError("trajectory was produced by another algorithm")
    if rollout.get("checkpoint_sha256") != parent_hash:
        raise ValueError("trajectory collection checkpoint differs from parent")
    samples = list(rollout["samples"])
    if len(samples) != int(training["rollout_count"]):
        raise ValueError("trajectory sample count differs from registered config")
    transition_count = int(training["transition_count"])
    if any(len(sample["transitions"]) != transition_count for sample in samples):
        raise ValueError("trajectory transition count differs from registered config")
    if float(rollout["noise_level"]) != float(training["noise_level"]):
        raise ValueError("trajectory noise level differs from registered config")

    reward = read_json(args.reward.resolve())
    expected_reward_model = str(
        config.get("reward", {}).get("model_id", MODEL_ID)
    )
    if reward.get("reward_model_id") != expected_reward_model:
        raise ValueError("FlowGRPO reward model differs from the registered config")
    if reward.get("failed_candidates_scored") is not True:
        raise ValueError("FlowGRPO reward dropped failed candidates")
    reward_by_id = {
        str(item["candidate_id"]): item for item in reward.get("samples", [])
    }
    candidate_ids = [str(sample["candidate_id"]) for sample in samples]
    if set(candidate_ids) != set(reward_by_id):
        raise ValueError("FlowGRPO reward and trajectory candidate sets differ")
    rewards = torch.tensor(
        [float(reward_by_id[candidate]["raw_reward"]) for candidate in candidate_ids]
    )
    group_ids = [str(sample["group_id"]) for sample in samples]
    advantages = group_relative_advantages(
        rewards,
        group_ids,
        epsilon=float(training.get("advantage_epsilon", 1e-4)),
        clip_max=float(training.get("advantage_clip_max", 5.0)),
    )
    decisive_fraction = float((advantages.abs() > 1e-8).float().mean())
    if decisive_fraction <= 0:
        raise RuntimeError("registered reward produced no group-relative FlowGRPO signal")

    trainable = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    trainable_names = [
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(training["learning_rate"]),
        betas=(0.9, 0.999),
        weight_decay=float(training.get("weight_decay", 1e-4)),
        eps=1e-8,
    )
    batch_size = int(training["training_batch_size"])
    if batch_size < 1 or len(samples) % batch_size:
        raise ValueError("training batch size must divide the rollout count")
    clip_range = float(training["clip_range"])
    beta = float(training["kl_beta"])
    noise_level = float(training["noise_level"])
    gradient_clip = float(training.get("gradient_clip_norm", 1.0))
    inner_epochs = int(training.get("inner_epochs", 1))
    metrics: list[dict[str, Any]] = []
    first_ratio_error: float | None = None
    optimizer_step = 0
    for inner_epoch in range(inner_epochs):
        indices = list(range(len(samples)))
        if inner_epoch:
            random.Random(seed + inner_epoch).shuffle(indices)
        for start in range(0, len(indices), batch_size):
            selected = indices[start : start + batch_size]
            optimizer.zero_grad(set_to_none=True)
            batch_metrics: list[dict[str, float]] = []
            for step in range(transition_count):
                transitions = [samples[index]["transitions"][step] for index in selected]
                new_log_prob, transition_kl, details = _transition_statistics(
                    policy=policy,
                    reference=reference,
                    transitions=transitions,
                    device=device,
                    noise_level=noise_level,
                )
                old_log_prob = torch.tensor(
                    [float(value["old_log_prob"]) for value in transitions],
                    device=device,
                    dtype=new_log_prob.dtype,
                )
                batch_advantage = advantages[selected].to(device=device, dtype=new_log_prob.dtype)
                policy_loss, diagnostics = clipped_grpo_loss(
                    new_log_prob,
                    old_log_prob,
                    batch_advantage,
                    clip_range=clip_range,
                )
                kl_loss = transition_kl.mean()
                loss = policy_loss + beta * kl_loss
                if not bool(torch.isfinite(loss)):
                    raise FloatingPointError("non-finite FlowGRPO loss")
                (loss / transition_count).backward()
                ratio_error = float((diagnostics["ratio"] - 1.0).abs().max().cpu())
                if first_ratio_error is None:
                    first_ratio_error = ratio_error
                batch_metrics.append(
                    {
                        "loss": float(loss.detach().cpu()),
                        "policy_loss": float(policy_loss.detach().cpu()),
                        "transition_kl": float(kl_loss.detach().cpu()),
                        "coordinate_kl": float(details["coordinate_kl"].mean().cpu()),
                        "categorical_kl": float(details["categorical_kl"].mean().cpu()),
                        "ratio_mean": float(diagnostics["ratio"].mean().cpu()),
                        "ratio_max_error": ratio_error,
                        "clip_fraction": float(diagnostics["clip_fraction"].cpu()),
                        "approx_kl": float(diagnostics["approx_kl"].cpu()),
                    }
                )
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(trainable, gradient_clip).detach().cpu()
            )
            optimizer.step()
            optimizer_step += 1
            aggregate = {
                key: float(sum(row[key] for row in batch_metrics) / len(batch_metrics))
                for key in batch_metrics[0]
            }
            aggregate.update(
                {
                    "inner_epoch": inner_epoch + 1,
                    "optimizer_step": optimizer_step,
                    "preclip_gradient_norm": gradient_norm,
                }
            )
            metrics.append(aggregate)
            print(json.dumps(aggregate, sort_keys=True), flush=True)

    if first_ratio_error is None or first_ratio_error > float(
        training.get("maximum_initial_ratio_error", 5e-4)
    ):
        raise RuntimeError(
            f"FlowGRPO collection/training log-prob mismatch: {first_ratio_error}"
        )
    relative_delta = relative_parameter_delta(policy, reference, trainable_names)
    maximum_delta = float(training["maximum_relative_parameter_delta"])
    if not 0 < relative_delta <= maximum_delta:
        raise RuntimeError(
            f"relative FlowGRPO parameter delta {relative_delta:.6g} exceeds {maximum_delta}"
        )
    mutable_state_names = verify_frozen_state(policy, reference, trainable)
    if sha256_file(parent_path) != parent_hash:
        raise RuntimeError("FlowGRPO parent checkpoint changed")

    child = output_dir / "last.ckpt"
    save_reloadable_checkpoint(
        parent_checkpoint=parent_path, policy=policy, destination=child
    )
    reloaded = _load_flowmol(
        child,
        args.n_atoms_hist.resolve(),
        args.marginal_dists.resolve(),
        torch.device("cpu"),
    )
    policy_state = {
        name: value.detach().cpu() for name, value in policy.state_dict().items()
    }
    for name, value in reloaded.state_dict().items():
        if not torch.equal(value.detach().cpu(), policy_state[name]):
            raise RuntimeError(f"released loader changed FlowGRPO child state: {name}")
    result = {
        "schema_version": 1,
        "artifact_type": "flowmol_flowgrpo_update",
        "algorithm": "flow_grpo_ode_to_sde_v1",
        "method_source": {
            "paper": "Flow-GRPO: Training Flow Matching Models via Online RL",
            "arxiv": "2505.05470v5",
            "official_repository": "https://github.com/yifan123/flow_grpo",
        },
        "molecular_adaptation": {
            "coordinates": "same-marginal ODE-to-SDE Gaussian transitions",
            "categorical_features": "FlowMol Campbell CTMC endpoint-token actions",
            "joint_log_probability": "mean_of_coordinate_atom_charge_bond_log_probabilities",
            "training_timesteps": transition_count,
            "inference_timesteps": int(config["evaluation"]["flow_timesteps"]),
        },
        "parent_checkpoint": str(parent_path),
        "parent_checkpoint_sha256": parent_hash,
        "child_checkpoint": str(child),
        "child_checkpoint_sha256": sha256_file(child),
        "rollout_count": len(samples),
        "group_size": int(training["group_size"]),
        "group_count": len(set(group_ids)),
        "reward_model_id": reward["reward_model_id"],
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std(unbiased=False)),
        "reward_decisive_fraction": decisive_fraction,
        "scored_count": int(reward["scored_count"]),
        "floored_count": int(reward["floored_count"]),
        "optimizer_steps": optimizer_step,
        "inner_epochs": inner_epochs,
        "learning_rate": float(training["learning_rate"]),
        "clip_range": clip_range,
        "kl_beta": beta,
        "noise_level": noise_level,
        "first_on_policy_ratio_max_error": first_ratio_error,
        "relative_parameter_delta": relative_delta,
        "maximum_relative_parameter_delta": maximum_delta,
        "trainable_parameter_count": sum(value.numel() for value in trainable),
        "trainable_parameter_names": trainable_names,
        "trainable_state_names_including_aliases": mutable_state_names,
        "originally_frozen_state_preserved": True,
        "checkpoint_reload_verified": True,
        "metrics": metrics,
        "scientific_success_claimed": False,
    }
    immutable_json(output_dir / "training_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
