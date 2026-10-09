#!/usr/bin/env python3
"""Run one bounded FlowGRPO-style update over an immutable FlowMol3 rollout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch

from domains.molecules.gsk3b.dpo import (
    _batch_graphs, attach_prior, interpolate_conditional_path,
    relative_parameter_delta, save_reloadable_checkpoint, sha256_state_dict,
    training_graph_from_sampled, verify_frozen_state,
)
from domains.molecules.gsk3b.baselines.grpo_loss import (
    flowmol_grpo_step, group_relative_advantages, initial_ratio_diagnostic,
)
from domains.molecules.gsk3b.artifacts import immutable_json, read_json, sha256_file
from domains.molecules.gsk3b.baselines.nft_campaign import BoundMoleculeStore, _load_flowmol


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--policy-checkpoint", type=Path, required=True)
    parser.add_argument("--old-checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--n-atoms-hist", type=Path, required=True)
    parser.add_argument("--marginal-dists", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--clip-epsilon", type=float, required=True)
    parser.add_argument("--kl-beta", type=float, required=True)
    parser.add_argument("--log-density-scale", type=float, required=True)
    parser.add_argument("--initial-ratio-atol", type=float, required=True)
    parser.add_argument("--maximum-relative-parameter-delta", type=float, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FlowGRPO-style update requires CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = read_json(args.manifest.resolve())
    if manifest.get("algorithm") not in {"flowgrpo_style_flow_matching_surrogate_v1", "flowgrpo_style_flow_matching_surrogate_v2"}:
        raise ValueError("unsupported GRPO manifest algorithm")
    if Path(manifest["collection_checkpoint"]).resolve() != args.old_checkpoint.resolve():
        raise ValueError("GRPO rollout and old policy checkpoint differ")
    from domains.molecules.gsk3b.oracle import REWARD_ENDPOINT
    if manifest.get("reward_endpoint") != REWARD_ENDPOINT:
        raise ValueError("GRPO manifest reward is not the registered GSK3B endpoint")
    samples = list(manifest["samples"])
    rewards = torch.tensor([float(row["raw_reward"]) for row in samples])
    advantages = group_relative_advantages(
        rewards, [str(row["group_id"]) for row in samples]
    )
    device = torch.device("cuda:0")
    policy = _load_flowmol(
        args.policy_checkpoint.resolve(), args.n_atoms_hist.resolve(),
        args.marginal_dists.resolve(), device,
    )
    old = _load_flowmol(
        args.old_checkpoint.resolve(), args.n_atoms_hist.resolve(),
        args.marginal_dists.resolve(), device,
    )
    base = _load_flowmol(
        args.base_checkpoint.resolve(), args.n_atoms_hist.resolve(),
        args.marginal_dists.resolve(), device,
    )
    if sha256_state_dict(policy) != sha256_state_dict(old):
        raise RuntimeError("on-policy GRPO update must start from the rollout policy")
    # FlowMol does not require train-mode layers for gradient computation.
    # Evaluation mode keeps the on-policy equality check exact even if a
    # future checkpoint adds dropout or running-statistics layers.
    policy.eval()
    old.eval()
    base.eval()
    for model in (old, base):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    names = [name for name, p in policy.named_parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable, lr=args.learning_rate)
    store = BoundMoleculeStore()
    rng = random.Random(args.seed)
    metrics = []
    initial_ratio_checked = False
    initial_ratio_maximum_absolute_deviation = None
    for step in range(args.steps):
        selected = torch.tensor(
            [rng.randrange(len(samples)) for _ in range(args.batch_size)],
            dtype=torch.long,
        )
        molecules = [store.item(samples[int(index)]) for index in selected]
        graphs = [attach_prior(policy, training_graph_from_sampled(mol, policy))
                  for mol in molecules]
        batched = _batch_graphs(graphs).to(device)
        times = torch.rand(args.batch_size, device=device)
        interpolated = interpolate_conditional_path(policy, batched, times)
        loss, values = flowmol_grpo_step(
            policy, old, base, interpolated, times,
            advantages=advantages[selected].to(device),
            clip_epsilon=args.clip_epsilon, kl_beta=args.kl_beta,
            log_density_scale=args.log_density_scale,
        )
        if step == 0:
            passed, maximum_deviation = initial_ratio_diagnostic(
                values["ratio"], absolute_tolerance=args.initial_ratio_atol,
            )
            initial_ratio_maximum_absolute_deviation = maximum_deviation
            if not passed:
                raise RuntimeError(
                    "untouched on-policy importance ratio differs from one: "
                    f"max_abs_deviation={maximum_deviation:.9g}, "
                    f"atol={args.initial_ratio_atol:.9g}"
                )
            initial_ratio_checked = True
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = float(torch.nn.utils.clip_grad_norm_(trainable, 1.0).cpu())
        optimizer.step()
        metrics.append({
            "step": step + 1,
            "loss": float(loss.detach().cpu()),
            "mean_ratio": float(values["ratio"].mean().cpu()),
            "maximum_ratio": float(values["ratio"].max().cpu()),
            "kl_proxy": float(values["kl_proxy"].cpu()),
            "gradient_norm": gradient_norm,
        })
    delta = relative_parameter_delta(policy, old, names)
    if not 0 < delta <= args.maximum_relative_parameter_delta:
        raise RuntimeError(f"FlowGRPO-style parameter delta outside bound: {delta}")
    mutable = verify_frozen_state(policy, old, trainable)
    child = output / "last.ckpt"
    save_reloadable_checkpoint(
        parent_checkpoint=args.policy_checkpoint.resolve(), policy=policy,
        destination=child,
    )
    result = {
        "schema_version": 1,
        "artifact_type": "flowmol_grpo_style_update",
        "algorithm": manifest["algorithm"],
        "literal_flow_grpo_reproduction": False,
        "likelihood_surrogate": "negative_per_molecule_flow_matching_loss",
        "on_policy_ratio_one_verified": initial_ratio_checked,
        "initial_ratio_absolute_tolerance": args.initial_ratio_atol,
        "initial_ratio_maximum_absolute_deviation": (
            initial_ratio_maximum_absolute_deviation
        ),
        "group_relative_advantages": True,
        "clip_epsilon": args.clip_epsilon,
        "kl_beta": args.kl_beta,
        "log_density_scale": args.log_density_scale,
        "optimizer_steps": args.steps,
        "sample_count": len(samples),
        "relative_parameter_delta": delta,
        "trainable_state_names_including_aliases": mutable,
        "checkpoint_reload_required": True,
        "child_checkpoint": str(child),
        "child_checkpoint_sha256": sha256_file(child),
        "metrics": metrics,
        "scientific_success_claimed": False,
    }
    immutable_json(output / "training_result.json", result)
    print(json.dumps({k: result[k] for k in (
        "algorithm", "optimizer_steps", "relative_parameter_delta",
        "on_policy_ratio_one_verified",
    )}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
