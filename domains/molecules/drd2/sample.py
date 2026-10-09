#!/usr/bin/env python3
"""Sample FlowMol3 while retaining native graphs and priors for flow-DPO."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from rdkit import Chem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-atoms-hist", type=Path, required=True)
    parser.add_argument("--marginal-dists", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=30602)
    parser.add_argument("--n-timesteps", type=int, default=250)
    parser.add_argument(
        "--priors-from",
        type=Path,
        help="Reuse retained FlowMol priors from a parent samples.pt",
    )
    return parser


def _prior_from_graph(graph: Any, *, fake_atoms: bool) -> dict[str, Any]:
    missing = [
        name
        for name in ("x_0", "a_0", "c_0")
        if name not in graph.ndata
    ]
    if missing or "e_0" not in graph.edata:
        raise ValueError("sampled graph is missing retained FlowMol priors")
    return {
        "x_0": graph.ndata["x_0"].detach().cpu().contiguous(),
        "a_0": graph.ndata["a_0"].detach().cpu().contiguous(),
        "c_0": graph.ndata["c_0"].detach().cpu().contiguous(),
        "e_0": graph.edata["e_0"].detach().cpu().contiguous(),
        "n_atoms": int(graph.num_nodes()),
        "fake_atoms": bool(fake_atoms),
    }


def _batched_prior(priors: list[dict[str, Any]], device: str) -> dict[str, Any]:
    fake_atoms = {bool(row["fake_atoms"]) for row in priors}
    if len(fake_atoms) != 1:
        raise ValueError("parent priors disagree about fake atoms")
    return {
        "x_0": torch.cat([row["x_0"] for row in priors]).to(device),
        "a_0": torch.cat([row["a_0"] for row in priors]).to(device),
        "c_0": torch.cat([row["c_0"] for row in priors]).to(device),
        "e_0": torch.cat([row["e_0"] for row in priors]).to(device),
        "fake_atoms": next(iter(fake_atoms)),
    }


def main() -> None:
    args = build_parser().parse_args()
    if args.count < 1 or args.batch_size < 1 or args.n_timesteps < 1:
        raise ValueError("count, batch size, and timestep count must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("FlowMol generation requires CUDA")
    samples_path = args.output_dir.resolve() / "samples.pt"
    if samples_path.exists():
        raise FileExistsError(f"refusing to overwrite {samples_path}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sdf_dir = args.output_dir / "sdf"
    # A preempted allocation may leave the directory but cannot leave a valid
    # samples.pt completion artifact. Permit that exact recovery path.
    sdf_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    from flowmol.models.flowmol import FlowMol

    model = FlowMol.load_from_checkpoint(
        str(args.checkpoint.resolve()),
        n_atoms_hist_file=str(args.n_atoms_hist.resolve()),
        marginal_dists_file=str(args.marginal_dists.resolve()),
    )
    model = model.to("cuda:0").eval()

    parent_priors = None
    if args.priors_from is not None:
        parent = torch.load(
            args.priors_from.resolve(), map_location="cpu", weights_only=False
        )
        parent_priors = parent.get("priors")
        if not isinstance(parent_priors, list) or len(parent_priors) != args.count:
            raise ValueError("parent samples do not contain matching retained priors")

    molecules = []
    with torch.inference_mode():
        if parent_priors is None:
            while len(molecules) < args.count:
                current = min(args.batch_size, args.count - len(molecules))
                molecules.extend(
                    model.sample_random_sizes(
                        current,
                        device="cuda:0",
                        n_timesteps=args.n_timesteps,
                    )
                )
        else:
            offset = 0
            while len(molecules) < args.count:
                current = min(args.batch_size, args.count - len(molecules))
                chunk = parent_priors[offset : offset + current]
                n_atoms = torch.tensor(
                    [int(row["n_atoms"]) for row in chunk],
                    device="cuda:0",
                )
                molecules.extend(
                    model.sample(
                        n_atoms,
                        device="cuda:0",
                        n_timesteps=args.n_timesteps,
                        prior=_batched_prior(chunk, "cuda:0"),
                    )
                )
                offset += current

    priors = [
        _prior_from_graph(molecule.g, fake_atoms=bool(model.fake_atoms))
        for molecule in molecules
    ]
    sdf_records = []
    for index, molecule in enumerate(molecules):
        sdf_path = sdf_dir / f"sample_{index}.sdf"
        rdkit_mol = getattr(molecule, "rdkit_mol", None)
        if rdkit_mol is not None:
            writer = Chem.SDWriter(str(sdf_path))
            writer.SetKekulize(False)
            writer.write(rdkit_mol)
            writer.close()
            sdf_records.append(str(sdf_path.resolve()))

    artifact = {
        "molecules": molecules,
        "priors": priors,
        "metadata": {
            "artifact_type": "flowmol_generator_samples",
            "schema_version": 1,
            "checkpoint": str(args.checkpoint.resolve()),
            "n_atoms_hist_file": str(args.n_atoms_hist.resolve()),
            "marginal_dists_file": str(args.marginal_dists.resolve()),
            "sample_count": args.count,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "flow_timesteps": args.n_timesteps,
            "native_flowmol_graphs_retained": True,
            "native_flowmol_priors_retained": True,
            "matched_priors_from": (
                None
                if args.priors_from is None
                else str(args.priors_from.resolve())
            ),
            "valid_sdf_count": len(sdf_records),
            "sdf_files": sdf_records,
        },
    }
    torch.save(artifact, samples_path)
    print(json.dumps(artifact["metadata"], indent=2, sort_keys=True))


# Stochastic trajectory collection for FlowGRPO and reward guidance.


import argparse
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from rdkit import Chem

from domains.molecules.drd2.baselines.grpo_loss import (
    categorical_log_prob_per_graph,
    coordinate_transition_mean,
    flowgrpo_noise_std,
    gaussian_log_prob_per_graph,
    joint_transition_statistic,
)
from domains.molecules.drd2.artifacts import immutable_json, sha256_file


def _minimal_state_graph(graph: Any) -> Any:
    import dgl

    source, destination = graph.edges(form="uv")
    state = dgl.graph(
        (source.detach().clone(), destination.detach().clone()),
        num_nodes=graph.num_nodes(),
        device=graph.device,
    )
    for name in ("x_t", "a_t", "c_t"):
        state.ndata[name] = graph.ndata[name].detach().clone()
    state.edata["e_t"] = graph.edata["e_t"].detach().clone()
    if int(graph.batch_size) > 1:
        state.set_batch_num_nodes(graph.batch_num_nodes().detach().clone())
        state.set_batch_num_edges(graph.batch_num_edges().detach().clone())
    return state


def _split_optional_endpoint(
    endpoint: dict[str, torch.Tensor] | None,
    *,
    node_counts: list[int],
    upper_edge_counts: list[int],
) -> list[dict[str, torch.Tensor] | None]:
    if endpoint is None:
        return [None] * len(node_counts)
    node_values = {
        name: list(torch.split(endpoint[name].detach().cpu(), node_counts))
        for name in ("x", "a", "c")
    }
    edge_values = list(
        torch.split(endpoint["e"].detach().cpu(), upper_edge_counts)
    )
    return [
        {
            "x": node_values["x"][index],
            "a": node_values["a"][index],
            "c": node_values["c"][index],
            "e": edge_values[index],
        }
        for index in range(len(node_counts))
    ]


def _categorical_policy(probabilities: torch.Tensor, temperature: Any) -> torch.Tensor:
    value = torch.as_tensor(
        temperature, device=probabilities.device, dtype=probabilities.dtype
    ).clamp_min(1e-6)
    return F.softmax(probabilities.clamp_min(1e-12).log() / value, dim=-1)


def _categorical_transition(
    *,
    probabilities: torch.Tensor,
    current: torch.Tensor,
    mask_index: int,
    alpha: torch.Tensor,
    alpha_prime: torch.Tensor,
    step_size: float,
    stochasticity: float,
    last_step: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample FlowMol's Campbell CTMC action with policy-independent masks."""

    actions = torch.distributions.Categorical(probabilities).sample()
    tokens = current.argmax(dim=-1).clone()
    unmask_probability = step_size * (
        alpha_prime + stochasticity * alpha
    ) / (1.0 - alpha).clamp_min(1e-6)
    unmask_probability = unmask_probability.clamp(0.0, 1.0)
    will_unmask = torch.rand(tokens.shape[0], device=tokens.device) < unmask_probability
    will_unmask &= tokens == mask_index
    if not last_step:
        mask_probability = min(max(step_size * stochasticity, 0.0), 1.0)
        will_mask = torch.rand(tokens.shape[0], device=tokens.device) < mask_probability
        will_mask &= tokens != mask_index
        tokens[will_mask] = mask_index
    tokens[will_unmask] = actions[will_unmask]
    return F.one_hot(tokens, num_classes=probabilities.shape[-1] + 1).float(), actions


def _center_per_graph(values: torch.Tensor, graph: Any, node_batch_idx: torch.Tensor) -> torch.Tensor:
    import dgl

    with graph.local_scope():
        graph.ndata["_flowgrpo_noise"] = values
        centres = dgl.readout_nodes(graph, "_flowgrpo_noise", op="mean")
    return values - centres[node_batch_idx]


def _collect_batch(
    model: Any,
    *,
    n_atoms: torch.Tensor,
    global_offset: int,
    group_size: int,
    transition_count: int,
    noise_level: float,
    categorical_stochasticity: float,
) -> tuple[list[dict[str, Any]], list[Any], list[dict[str, Any]]]:
    import dgl
    from flowmol.analysis.molecule_builder import SampledMolecule
    from flowmol.data_processing.utils import (
        build_edge_idxs,
        get_batch_idxs,
        get_upper_edge_mask,
    )

    device = n_atoms.device
    graphs = []
    for count in n_atoms:
        edges = build_edge_idxs(count)
        graphs.append(dgl.graph((edges[0], edges[1]), num_nodes=count, device=device))
    graph = dgl.batch(graphs)
    upper_edge_mask = get_upper_edge_mask(graph)
    node_batch_idx, edge_batch_idx = get_batch_idxs(graph)
    graph = model.sample_prior(graph, node_batch_idx, upper_edge_mask)
    for feature in model.canonical_feat_order:
        source = graph.edata if feature == "e" else graph.ndata
        source[f"{feature}_t"] = source[f"{feature}_0"].detach().clone()

    batch_size = int(graph.batch_size)
    node_counts = [int(value) for value in graph.batch_num_nodes().tolist()]
    upper_edge_counts = [int(value // 2) for value in graph.batch_num_edges().tolist()]
    records = [
        {
            "candidate_id": f"flowgrpo-{global_offset + index:04d}",
            "group_id": f"flowgrpo-g{(global_offset + index) // group_size:04d}",
            "transitions": [],
        }
        for index in range(batch_size)
    ]
    times = torch.linspace(0.0, 1.0, transition_count + 1, device=device)
    alpha = model.interpolant_scheduler.alpha_t(times)
    alpha_prime = model.interpolant_scheduler.alpha_t_prime(times)
    previous_endpoint = None
    for step in range(transition_count):
        time = times[step]
        next_time = times[step + 1]
        step_size = float((next_time - time).item())
        model_time = torch.full((batch_size,), time, device=device)
        state_graph = _minimal_state_graph(graph)
        split_previous = _split_optional_endpoint(
            previous_endpoint,
            node_counts=node_counts,
            upper_edge_counts=upper_edge_counts,
        )
        endpoint = model.vector_field(
            graph,
            model_time,
            node_batch_idx=node_batch_idx,
            upper_edge_mask=upper_edge_mask,
            apply_softmax=True,
            remove_com=True,
            prev_dst_dict=previous_endpoint,
        )

        sigma = flowgrpo_noise_std(
            time, step_size=step_size, noise_level=noise_level
        ).to(device=device, dtype=graph.ndata["x_t"].dtype)
        sigma_nodes = sigma.expand(graph.num_nodes())
        coordinate_mean = coordinate_transition_mean(
            graph.ndata["x_t"],
            endpoint["x"],
            time_per_node=model_time[node_batch_idx],
            step_size=step_size,
            noise_std_per_node=sigma_nodes,
        )
        coordinate_noise = _center_per_graph(
            torch.randn_like(coordinate_mean), graph, node_batch_idx
        )
        coordinate_action = coordinate_mean + sigma * math.sqrt(step_size) * coordinate_noise

        categorical_probabilities: dict[str, torch.Tensor] = {}
        categorical_actions: dict[str, torch.Tensor] = {}
        categorical_next: dict[str, torch.Tensor] = {}
        for feature_index, feature in enumerate(model.canonical_feat_order):
            if feature == "x":
                continue
            probabilities = _categorical_policy(
                endpoint[feature], model.vector_field.cat_temp_func(time)
            )
            source = graph.edata if feature == "e" else graph.ndata
            current = source[f"{feature}_t"]
            if feature == "e":
                current = current[upper_edge_mask]
            next_value, action = _categorical_transition(
                probabilities=probabilities,
                current=current,
                mask_index=model.vector_field.mask_idxs[feature],
                alpha=alpha[step, feature_index],
                alpha_prime=alpha_prime[step, feature_index],
                step_size=step_size,
                stochasticity=categorical_stochasticity,
                last_step=step == transition_count - 1,
            )
            categorical_probabilities[feature] = probabilities
            categorical_actions[feature] = action
            categorical_next[feature] = next_value

        coordinate_log_prob = gaussian_log_prob_per_graph(
            coordinate_action,
            coordinate_mean,
            standard_deviation=(sigma * math.sqrt(step_size)).expand(graph.num_nodes()),
            batch_index=node_batch_idx,
            graph_count=batch_size,
        )
        log_prob_components = [coordinate_log_prob]
        for feature in ("a", "c", "e"):
            batch_index = (
                edge_batch_idx[upper_edge_mask] if feature == "e" else node_batch_idx
            )
            log_prob_components.append(
                categorical_log_prob_per_graph(
                    categorical_probabilities[feature],
                    categorical_actions[feature],
                    batch_index=batch_index,
                    graph_count=batch_size,
                )
            )
        old_log_prob = joint_transition_statistic(log_prob_components)

        state_parts = dgl.unbatch(state_graph.to("cpu"))
        next_x_parts = list(torch.split(coordinate_action.detach().cpu(), node_counts))
        action_parts = {
            "a": list(torch.split(categorical_actions["a"].detach().cpu(), node_counts)),
            "c": list(torch.split(categorical_actions["c"].detach().cpu(), node_counts)),
            "e": list(
                torch.split(categorical_actions["e"].detach().cpu(), upper_edge_counts)
            ),
        }
        for index in range(batch_size):
            records[index]["transitions"].append(
                {
                    "step": step,
                    "time": float(time.item()),
                    "step_size": step_size,
                    "graph": state_parts[index],
                    "previous_endpoint": split_previous[index],
                    "next_x": next_x_parts[index],
                    "action_a": action_parts["a"][index],
                    "action_c": action_parts["c"][index],
                    "action_e": action_parts["e"][index],
                    "old_log_prob": float(old_log_prob[index].detach().cpu()),
                }
            )

        graph.ndata["x_t"] = coordinate_action.detach()
        graph.ndata["a_t"] = categorical_next["a"].detach()
        graph.ndata["c_t"] = categorical_next["c"].detach()
        edge_next = torch.zeros_like(graph.edata["e_t"])
        edge_next[upper_edge_mask] = categorical_next["e"]
        edge_next[~upper_edge_mask] = categorical_next["e"]
        graph.edata["e_t"] = edge_next.detach()
        previous_endpoint = {name: value.detach() for name, value in endpoint.items()}

    for feature in model.canonical_feat_order:
        source = graph.edata if feature == "e" else graph.ndata
        source[f"{feature}_1"] = source[f"{feature}_t"]
    graph.edata["ue_mask"] = upper_edge_mask
    graph = graph.to("cpu")
    molecules = [
        SampledMolecule(
            item,
            model.atom_type_map,
            ctmc_mol=True,
            fake_atoms=model.fake_atoms,
            exclude_charges=model.exclude_charges,
            explicit_aromaticity=model.explicit_aromaticity,
        )
        for item in dgl.unbatch(graph)
    ]
    priors = []
    for molecule in molecules:
        priors.append(
            {
                "x_0": molecule.g.ndata["x_0"].detach().cpu().contiguous(),
                "a_0": molecule.g.ndata["a_0"].detach().cpu().contiguous(),
                "c_0": molecule.g.ndata["c_0"].detach().cpu().contiguous(),
                "e_0": molecule.g.edata["e_0"].detach().cpu().contiguous(),
                "n_atoms": int(molecule.g.num_nodes()),
                "fake_atoms": bool(model.fake_atoms),
            }
        )
    return records, molecules, priors


def rollout_main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-atoms-hist", type=Path, required=True)
    parser.add_argument("--marginal-dists", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--group-size", type=int, required=True)
    parser.add_argument("--transition-count", type=int, default=10)
    parser.add_argument("--noise-level", type=float, default=0.7)
    parser.add_argument("--categorical-stochasticity", type=float, default=8.0)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FlowMol FlowGRPO rollout collection requires CUDA")
    if min(args.count, args.batch_size, args.group_size, args.transition_count) < 1:
        raise ValueError("rollout dimensions must be positive")
    if args.count % args.group_size:
        raise ValueError("rollout count must be divisible by the GRPO group size")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    sdf_dir = output_dir / "sdf"
    sdf_dir.mkdir()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    from flowmol.models.flowmol import FlowMol

    model = FlowMol.load_from_checkpoint(
        str(args.checkpoint.resolve()),
        n_atoms_hist_file=str(args.n_atoms_hist.resolve()),
        marginal_dists_file=str(args.marginal_dists.resolve()),
    ).to("cuda:0").eval()
    if model.parameterization != "ctmc":
        raise ValueError("the registered FlowGRPO adapter requires FlowMol CTMC")

    all_records: list[dict[str, Any]] = []
    all_molecules: list[Any] = []
    all_priors: list[dict[str, Any]] = []
    with torch.no_grad():
        while len(all_records) < args.count:
            current = min(args.batch_size, args.count - len(all_records))
            n_atoms = model.sample_n_atoms(current).to("cuda:0")
            records, molecules, priors = _collect_batch(
                model,
                n_atoms=n_atoms,
                global_offset=len(all_records),
                group_size=args.group_size,
                transition_count=args.transition_count,
                noise_level=args.noise_level,
                categorical_stochasticity=args.categorical_stochasticity,
            )
            all_records.extend(records)
            all_molecules.extend(molecules)
            all_priors.extend(priors)
            print(
                json.dumps({"collected": len(all_records), "requested": args.count}),
                flush=True,
            )

    samples_path = output_dir / "samples.pt"
    torch.save(
        {
            "molecules": all_molecules,
            "priors": all_priors,
            "metadata": {
                "artifact_type": "flowmol_flowgrpo_samples",
                "sample_count": args.count,
                "seed": args.seed,
                "training_transition_count": args.transition_count,
            },
        },
        samples_path,
    )
    trajectory_path = output_dir / "trajectories.pt"
    torch.save(
        {
            "schema_version": 1,
            "artifact_type": "flowmol_flowgrpo_rollout_trajectories",
            "algorithm": "flow_grpo_ode_to_sde_v1",
            "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
            "seed": args.seed,
            "group_size": args.group_size,
            "transition_count": args.transition_count,
            "noise_level": args.noise_level,
            "categorical_stochasticity": args.categorical_stochasticity,
            "samples": all_records,
        },
        trajectory_path,
    )

    rollout_samples = []
    valid = 0
    for index, (record, molecule) in enumerate(zip(all_records, all_molecules)):
        sdf_path = sdf_dir / f"sample_{index}.sdf"
        rdkit_mol = getattr(molecule, "rdkit_mol", None)
        if rdkit_mol is not None:
            writer = Chem.SDWriter(str(sdf_path))
            writer.SetKekulize(False)
            writer.write(rdkit_mol)
            writer.close()
            valid += 1
        rollout_samples.append(
            {
                "candidate_id": record["candidate_id"],
                "group_id": record["group_id"],
                "sdf": str(sdf_path.resolve()) if sdf_path.is_file() else None,
                "assembled": sdf_path.is_file(),
                "trajectory_index": index,
            }
        )
    rollout = {
        "schema_version": 1,
        "artifact_type": "flowmol_flowgrpo_rollout",
        "algorithm": "flow_grpo_ode_to_sde_v1",
        "collection_checkpoint": str(args.checkpoint.resolve()),
        "collection_checkpoint_sha256": sha256_file(args.checkpoint.resolve()),
        "trajectory_path": str(trajectory_path.resolve()),
        "trajectory_sha256": sha256_file(trajectory_path),
        "samples_path": str(samples_path.resolve()),
        "sample_count": args.count,
        "assembled_count": valid,
        "group_size": args.group_size,
        "group_count": args.count // args.group_size,
        "transition_count": args.transition_count,
        "noise_level": args.noise_level,
        "categorical_stochasticity": args.categorical_stochasticity,
        "seed": args.seed,
        "samples": rollout_samples,
    }
    immutable_json(output_dir / "rollout.json", rollout)
    print(json.dumps(rollout, indent=2, sort_keys=True))



if __name__ == "__main__":
    import sys
    if "--rollout" in sys.argv:
        sys.argv.remove("--rollout")
        rollout_main()
    else:
        main()
