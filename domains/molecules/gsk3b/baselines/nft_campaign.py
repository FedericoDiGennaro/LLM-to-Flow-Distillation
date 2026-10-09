#!/usr/bin/env python3
"""Run one bounded FlowMol DiffusionNFT iteration against a GSK3B reward."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch

from domains.molecules.gsk3b.dpo import (
    _batch_graphs,
    attach_prior,
    interpolate_conditional_path,
    relative_parameter_delta,
    save_reloadable_checkpoint,
    sha256_state_dict,
    training_graph_from_sampled,
    verify_frozen_state,
)
from domains.molecules.gsk3b.baselines.nft_loss import (
    optimality_reward,
    soft_update_,
    soft_update_schedule,
    validate_nft_preflight,
)
from domains.molecules.gsk3b.baselines.nft_update import flowmol_nft_step
from domains.molecules.gsk3b.artifacts import immutable_json, read_json, sha256_file
from domains.molecules.gsk3b.validate_samples import native_signature


class BoundMoleculeStore:
    def __init__(self) -> None:
        self.loaded: dict[Path, dict[str, Any]] = {}

    def item(self, sample: Mapping[str, Any]) -> Any:
        state = sample["training_state"]
        path = Path(state["samples_pt"]).resolve()
        expected = str(state["samples_pt_sha256"])
        if path not in self.loaded:
            if sha256_file(path) != expected:
                raise ValueError(f"samples.pt digest changed: {path}")
            # Hash verification above binds this to the campaign-generated
            # FlowMol artifact. It contains SampledMolecule instances, so the
            # PyTorch >=2.6 tensor-only default is not applicable.
            self.loaded[path] = torch.load(
                path, map_location="cpu", weights_only=False
            )
        index = int(state["sample_index"])
        molecule = self.loaded[path]["molecules"][index]
        if native_signature(molecule) != str(sample["content_sha256"]):
            raise ValueError(
                f"candidate {sample['candidate_id']} changed after it was rewarded"
            )
        return molecule


def _load_flowmol(
    checkpoint: Path, n_atoms_hist: Path, marginal_dists: Path, device: torch.device
):
    from flowmol.models.flowmol import FlowMol

    model = FlowMol.load_from_checkpoint(
        str(checkpoint),
        n_atoms_hist_file=str(n_atoms_hist),
        marginal_dists_file=str(marginal_dists),
    )
    return model.to(device)


def _prepare_interpolant(policy: Any, molecules: list[Any], times: torch.Tensor):
    prepared = []
    for molecule in molecules:
        graph = training_graph_from_sampled(molecule, policy)
        graph = attach_prior(policy, graph)
        prepared.append(graph)
    batched = _batch_graphs(prepared).to(times.device)
    return interpolate_conditional_path(policy, batched, times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--collection-checkpoint", type=Path)
    parser.add_argument("--n-atoms-hist", type=Path, required=True)
    parser.add_argument("--marginal-dists", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--iteration", type=int, default=1)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--maximum-relative-parameter-delta", type=float, default=0.02)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FlowMol DiffusionNFT requires CUDA")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and (output_dir / "training_result.json").is_file():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    contract = read_json(args.contract)
    manifest = read_json(args.manifest)
    for field in ("training_goal", "training_goal_kind", "immutable_final_goal"):
        if not isinstance(manifest.get(field), str) or not manifest[field].strip():
            raise ValueError(f"missing training metadata: {field}")
    preflight = validate_nft_preflight(contract, manifest)
    if not preflight.ready or manifest.get("training_authorized") is not True:
        raise PermissionError(f"DiffusionNFT preflight failed: {preflight.blockers}")

    seed = int(args.seed if args.seed is not None else 32751)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    parent = args.parent_checkpoint.resolve()
    collection_path = (args.collection_checkpoint or args.parent_checkpoint).resolve()
    parent_hash = sha256_file(parent)
    device = torch.device("cuda:0")
    policy = _load_flowmol(
        parent, args.n_atoms_hist.resolve(), args.marginal_dists.resolve(), device
    )
    collection = _load_flowmol(
        collection_path,
        args.n_atoms_hist.resolve(),
        args.marginal_dists.resolve(),
        device,
    )
    if args.iteration == 1 and sha256_state_dict(policy) != sha256_state_dict(
        collection
    ):
        raise RuntimeError("iteration 1 collection policy must start as the parent")
    policy.train()
    collection.eval()
    for parameter in collection.parameters():
        parameter.requires_grad_(False)
    trainable = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    trainable_names = [
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    ]

    store = BoundMoleculeStore()
    samples = list(manifest["samples"])
    optimality = optimality_reward(
        torch.tensor([float(sample["raw_reward"]) for sample in samples]),
        [sample["group_id"] for sample in samples],
        normalizer=contract.get("reward_normalizer", "global_std"),
    )
    decisive = float((optimality - 0.5).abs().gt(1e-6).float().mean())
    if decisive <= 0.0:
        raise RuntimeError(
            "GSK3B reward separated no candidate within its group, so this rollout "
            "carries no improvement direction"
        )

    optimizer_steps = int(contract["optimizer_steps"])
    batch_size = int(contract["batch_size"])
    accumulation = int(contract["gradient_accumulation_steps"])
    beta = float(contract["beta"])
    clip_norm = float(contract.get("gradient_clip_norm", 1.0))
    optimizer = torch.optim.Adam(trainable, lr=float(contract["learning_rate"]))
    rng = random.Random(seed)
    metrics: list[dict[str, Any]] = []
    optimizer.zero_grad(set_to_none=True)
    for step in range(optimizer_steps):
        micro_losses: list[float] = []
        micro_gaps: list[float] = []
        for _ in range(accumulation):
            selected = torch.tensor(
                [rng.randrange(len(samples)) for _ in range(batch_size)],
                dtype=torch.long,
            )
            molecules = [store.item(samples[int(index)]) for index in selected]
            times = torch.rand(batch_size, device=device)
            interpolated = _prepare_interpolant(policy, molecules, times)
            if int(interpolated.batch_size) != batch_size:
                raise RuntimeError(
                    "FlowMol interpolant lost per-molecule batch membership"
                )
            loss, values = flowmol_nft_step(
                policy,
                collection,
                interpolated,
                times,
                optimality=optimality[selected].to(device),
                beta=beta,
            )
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("non-finite FlowMol DiffusionNFT loss")
            (loss / accumulation).backward()
            micro_losses.append(float(loss.detach().cpu()))
            micro_gaps.append(
                float(
                    (values["negative_branch_loss"] - values["positive_branch_loss"])
                    .mean()
                    .cpu()
                )
            )
        gradient_norm = float(
            torch.nn.utils.clip_grad_norm_(trainable, clip_norm).cpu()
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        metrics.append(
            {
                "optimizer_step": step + 1,
                "loss": sum(micro_losses) / len(micro_losses),
                "negative_minus_positive_branch_loss": sum(micro_gaps)
                / len(micro_gaps),
                "preclip_gradient_norm": gradient_norm,
            }
        )
        if (step + 1) % 10 == 0:
            print(json.dumps(metrics[-1]), flush=True)

    relative_delta = relative_parameter_delta(policy, collection, trainable_names)
    if not 0 < relative_delta <= args.maximum_relative_parameter_delta:
        raise RuntimeError(
            f"relative FlowMol parameter delta {relative_delta:.6g} exceeds safety bound"
        )
    mutable_state_names = verify_frozen_state(policy, collection, trainable)
    if sha256_file(parent) != parent_hash:
        raise RuntimeError("FlowMol parent checkpoint changed")

    child = output_dir / "last.ckpt"
    save_reloadable_checkpoint(
        parent_checkpoint=parent, policy=policy, destination=child
    )
    eta = soft_update_schedule(
        args.iteration,
        slope=float(contract["soft_update_slope"]),
        maximum=float(contract["soft_update_maximum"]),
    )
    soft_update_(collection, policy, eta=eta)
    collection_ckpt = output_dir / "collection.ckpt"
    save_reloadable_checkpoint(
        parent_checkpoint=collection_path,
        policy=collection,
        destination=collection_ckpt,
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
            raise RuntimeError(f"released loader changed child state: {name}")
    child_hash = sha256_file(child)
    result = {
        "schema_version": 1,
        "artifact_type": "flowmol_diffusion_nft_update",
        "algorithm": contract["algorithm"],
        "arm_role": "reward_optimization_baseline",
        "reward_model_id": manifest["reward_model_id"],
        "reward_endpoint": manifest.get("reward_endpoint"),
        "reward_is_a_learned_surrogate": True,
        "training_goal": manifest["training_goal"],
        "training_goal_kind": manifest["training_goal_kind"],
        "immutable_final_goal": manifest["immutable_final_goal"],
        "iteration": args.iteration,
        "optimizer_steps": optimizer_steps,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation,
        "beta": beta,
        "learning_rate": contract["learning_rate"],
        "reward_normalizer": contract.get("reward_normalizer", "global_std"),
        "soft_update_eta": eta,
        "sample_count": len(samples),
        "group_count": len({sample["group_id"] for sample in samples}),
        "reward_decisive_fraction": decisive,
        "trainable_parameter_names": trainable_names,
        "trainable_state_names_including_aliases": mutable_state_names,
        "trainable_parameter_count": sum(parameter.numel() for parameter in trainable),
        "relative_parameter_delta": relative_delta,
        "originally_frozen_state_preserved": True,
        "checkpoint_reload_verified": True,
        "training_completed": True,
        "held_out_eval_checkpoint": "trained_policy",
        "collection_policy_not_used_for_held_out_eval": True,
        "parent_checkpoint_sha256": parent_hash,
        "child_checkpoint_sha256": child_hash,
        "metrics": metrics,
        "fresh_child_evaluation_required": True,
        "scientific_success_claimed": False,
        "gnina_used": False,
    }
    immutable_json(output_dir / "training_result.json", result)
    print(
        json.dumps(
            {
                "iteration": args.iteration,
                "optimizer_steps": optimizer_steps,
                "final_loss": metrics[-1]["loss"],
                "soft_update_eta": eta,
                "relative_parameter_delta": relative_delta,
                "child_checkpoint_sha256": child_hash,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
