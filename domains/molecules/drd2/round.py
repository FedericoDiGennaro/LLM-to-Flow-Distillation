#!/usr/bin/env python3
"""Train one content-bound FlowMol flow-DPO round from stable categorical labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch

from domains.molecules.drd2.dpo import (
    FLOW_DPO_KIND,
    config_from_campaign,
    deterministic_pair_schedule,
    flow_dpo_loss,
    optimizer_microbatch_count,
    per_example_loss,
    preference_pairs_for_training,
    relative_parameter_delta,
    save_reloadable_checkpoint,
    sha256_state_dict,
    verify_frozen_state,
    write_or_validate_pair_manifest,
)
from domains.molecules.drd2.config import load_campaign_config
from domains.molecules.drd2.artifacts import (
    immutable_json,
    read_json,
    read_jsonl,
    sha256_file,
)


class MoleculeStore:
    def __init__(self) -> None:
        self.loaded: dict[Path, dict[str, Any]] = {}

    def item(self, row: dict[str, Any]) -> Any:
        state = row["training_state"]
        path = Path(state["samples_pt"]).resolve()
        if path not in self.loaded:
            if sha256_file(path) != state["samples_pt_sha256"]:
                raise ValueError(f"FlowMol sample artifact changed: {path}")
            self.loaded[path] = torch.load(
                path, map_location="cpu", weights_only=False
            )
        values = self.loaded[path]
        index = int(state["sample_index"])
        molecules = values.get("molecules", [])
        if index < 0 or index >= len(molecules):
            raise ValueError(f"FlowMol sample index is missing: {index}")
        return molecules[index]


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--stable-labels", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--n-atoms-hist", type=Path, required=True)
    parser.add_argument("--marginal-dists", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("curriculum", "final_f"), default="curriculum"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FlowMol flow-DPO requires CUDA")
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and (output_dir / "training_result.json").is_file():
        raise FileExistsError(f"training result already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    config = load_campaign_config(args.config.resolve())
    if args.stage == "final_f":
        training = config["final_f_continuation"]["fine_tuning"]
    else:
        training = config["fine_tuning"]
    rows = read_jsonl(args.pool.resolve())
    summary = read_json(args.stable_labels.resolve())
    labels = summary["stable_labels"]
    by_item = {row["item_id"]: row for row in rows if row.get("status") == "judgeable"}
    if set(labels) != set(by_item):
        raise ValueError("stable labels and judgeable pool differ")
    positives = sorted(
        item for item, label in labels.items() if label == "positive"
    )
    negatives = sorted(
        item for item, label in labels.items() if label == "negative"
    )
    construction = str(training["pair_construction"])
    ceiling = int(training["maximum_pair_count"])
    seed = int(training["seed"])
    uncapped = (
        construction == "full_cartesian_stable_positive_x_stable_negative"
    )
    if not uncapped and not construction.startswith(
        "full_cartesian_stable_positive_x_stable_negative_capped"
    ):
        raise ValueError(f"unsupported flow-DPO pair_construction: {construction}")
    maximum = None if uncapped else ceiling
    pairs = preference_pairs_for_training(
        positives, negatives, maximum=maximum, seed=seed
    )
    if len(pairs) > ceiling:
        raise RuntimeError(
            f"flow-DPO pair count {len(pairs)} exceeds ceiling {ceiling}"
        )
    pair_manifest = write_or_validate_pair_manifest(
        output_dir / "preference_pairs.jsonl",
        positive_item_ids=positives,
        negative_item_ids=negatives,
        maximum=maximum,
        seed=seed,
    )
    if int(pair_manifest["pair_count"]) != len(pairs):
        raise RuntimeError("flow-DPO pair manifest does not match the training set")
    if uncapped and pair_manifest["construction"] != (
        "full_cartesian_stable_positive_x_stable_negative"
    ):
        raise RuntimeError("flow-DPO did not keep the full Cartesian pair set")
    train_config = config_from_campaign(training)
    accumulation = int(training["gradient_accumulation_steps"])
    microbatch_count = optimizer_microbatch_count(train_config.steps, accumulation)
    schedule = deterministic_pair_schedule(
        pair_count=len(pairs),
        steps=microbatch_count,
        batch_size=train_config.batch_size,
        seed=train_config.seed,
    )
    parent = args.parent_checkpoint.resolve()
    parent_hash = sha256_file(parent)
    torch.manual_seed(train_config.seed)
    np.random.seed(train_config.seed)
    random.seed(train_config.seed)
    torch.cuda.manual_seed_all(train_config.seed)

    device = torch.device("cuda:0")
    policy = _load_flowmol(
        parent,
        args.n_atoms_hist.resolve(),
        args.marginal_dists.resolve(),
        device,
    )
    reference = _load_flowmol(
        parent,
        args.n_atoms_hist.resolve(),
        args.marginal_dists.resolve(),
        device,
    )
    if sha256_state_dict(policy) != sha256_state_dict(reference):
        raise RuntimeError("policy must initialize from its reference parent")
    policy.train()
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    trainable = [
        parameter for parameter in policy.parameters() if parameter.requires_grad
    ]
    trainable_names = [
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    ]
    optimizer = torch.optim.Adam(trainable, lr=train_config.learning_rate)
    store = MoleculeStore()
    visited = [0] * len(pairs)
    metrics = []
    optimizer.zero_grad(set_to_none=True)
    adam_steps = 0
    for micro_step in range(microbatch_count):
        start = micro_step * train_config.batch_size
        indices = schedule[start : start + train_config.batch_size]
        batch = tuple(pairs[index] for index in indices)
        for index in indices:
            visited[index] += 1
        pair_times = torch.rand(len(batch), device=device)
        chosen_molecules = [
            store.item(by_item[pair.preferred_item_id]) for pair in batch
        ]
        rejected_molecules = [
            store.item(by_item[pair.rejected_item_id]) for pair in batch
        ]
        policy_chosen, reference_chosen = per_example_loss(
            policy,
            reference,
            molecules=chosen_molecules,
            pair_times=pair_times,
        )
        policy_rejected, reference_rejected = per_example_loss(
            policy,
            reference,
            molecules=rejected_molecules,
            pair_times=pair_times,
        )
        loss = flow_dpo_loss(
            policy_chosen,
            policy_rejected,
            reference_chosen,
            reference_rejected,
            beta=train_config.beta,
        )
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(
                f"non-finite FlowMol flow-DPO loss at microbatch {micro_step}"
            )
        (loss / accumulation).backward()
        if (micro_step + 1) % accumulation == 0:
            gradient_norm = float(
                torch.nn.utils.clip_grad_norm_(
                    trainable, train_config.gradient_clip_norm
                )
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            adam_steps += 1
        else:
            gradient_norm = 0.0
        implicit_margin = -(
            (policy_chosen - policy_rejected)
            - (reference_chosen - reference_rejected)
        )
        metrics.append(
            {
                "step": micro_step + 1,
                "optimizer_step": adam_steps,
                "loss": float(loss.detach()),
                "preference_accuracy": float(
                    (implicit_margin.detach() > 0).float().mean()
                ),
                "preclip_gradient_norm": gradient_norm,
            }
        )
    if adam_steps != train_config.steps:
        raise RuntimeError(
            f"FlowMol flow-DPO took {adam_steps} Adam steps, expected {train_config.steps}"
        )

    relative_delta = relative_parameter_delta(policy, reference, trainable_names)
    if not 0 < relative_delta <= float(training["maximum_relative_parameter_delta"]):
        raise RuntimeError(f"unsafe FlowMol relative parameter delta: {relative_delta}")
    mutable_names = verify_frozen_state(policy, reference, trainable)
    if sha256_file(parent) != parent_hash:
        raise RuntimeError("FlowMol parent checkpoint changed")

    child = output_dir / "last.ckpt"
    save_reloadable_checkpoint(
        parent_checkpoint=parent,
        policy=policy,
        destination=child,
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
    if min(visited) < 1:
        raise RuntimeError("flow-DPO failed to visit every Cartesian pair")
    result = {
        "schema_version": 1,
        "artifact_type": "flowmol_dpo_round",
        "objective": FLOW_DPO_KIND,
        "ltfd_stage": args.stage,
        "training_completed": True,
        "training_hyperparameters": {
            "learning_rate": train_config.learning_rate,
            "optimizer_steps": train_config.steps,
            "pair_microbatch_size": train_config.batch_size,
            "gradient_accumulation_steps": accumulation,
            "effective_pair_batch_size": train_config.batch_size * accumulation,
            "beta": train_config.beta,
            "gradient_clip_norm": train_config.gradient_clip_norm,
            "reference_policy": train_config.reference_policy,
            # Record the frozen campaign policy, including any deterministic
            # cap.  FlowDPOConfig describes the loss implementation and uses
            # the underlying Cartesian construction; it does not carry the
            # campaign-level subsampling policy.
            "pair_construction": construction,
        },
        "optimizer_steps": adam_steps,
        "microbatch_count": microbatch_count,
        "gradient_accumulation_steps": accumulation,
        "pair_count": len(pairs),
        "pair_manifest": pair_manifest,
        "visited_pair_count": sum(count > 0 for count in visited),
        "all_pairs_visited": True,
        "minimum_pair_visit_count": min(visited),
        "relative_parameter_delta": relative_delta,
        "originally_frozen_state_preserved": True,
        "mutable_state_names_including_aliases": mutable_names,
        "checkpoint_reload_verified": True,
        "parent_checkpoint_sha256": parent_hash,
        "child_checkpoint_sha256": child_hash,
        "metrics": metrics,
        "fresh_child_evaluation_required": True,
        "scientific_improvement_claimed": False,
        "gnina_used": False,
    }
    immutable_json(output_dir / "training_result.json", result)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "optimizer_steps",
                    "pair_count",
                    "visited_pair_count",
                    "relative_parameter_delta",
                    "checkpoint_reload_verified",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
