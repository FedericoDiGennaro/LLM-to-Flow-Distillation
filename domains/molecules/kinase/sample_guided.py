#!/usr/bin/env python3
"""Sample released FlowMol3 with a frozen soft-graph reward regressor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from rdkit import Chem

from domains.molecules.kinase.guidance import (
    SoftGraphRewardRegressor,
    guided_categorical_probabilities,
)
from domains.molecules.kinase.artifacts import read_json, sha256_file


def _prior_from_graph(graph: Any, *, fake_atoms: bool) -> dict[str, Any]:
    return {
        "x_0": graph.ndata["x_0"].detach().cpu().contiguous(),
        "a_0": graph.ndata["a_0"].detach().cpu().contiguous(),
        "c_0": graph.ndata["c_0"].detach().cpu().contiguous(),
        "e_0": graph.edata["e_0"].detach().cpu().contiguous(),
        "n_atoms": int(graph.num_nodes()),
        "fake_atoms": bool(fake_atoms),
    }


def _load_predictor(path: Path, device: torch.device):
    artifact = torch.load(path, map_location="cpu")
    if artifact.get("artifact_type") != "flowmol_soft_graph_reward_regressor":
        raise ValueError("unexpected guidance predictor artifact")
    predictor = SoftGraphRewardRegressor(**artifact["architecture"])
    predictor.load_state_dict(artifact["state_dict"], strict=True)
    predictor.to(device).eval()
    for parameter in predictor.parameters():
        parameter.requires_grad_(False)
    return predictor, artifact


def _guided_integrate(
    *,
    model: Any,
    predictor: SoftGraphRewardRegressor,
    graph: Any,
    node_batch_index: torch.Tensor,
    upper_edge_mask: torch.Tensor,
    n_timesteps: int,
    stochasticity: float,
    high_confidence_threshold: float,
    strengths: dict[str, float],
    schedule_power: float,
    maximum_logit_shift: float,
) -> tuple[Any, dict[str, float]]:
    from flowmol.data_processing.utils import get_edge_batch_idxs

    vector_field = model.vector_field
    edge_batch_index = get_edge_batch_idxs(graph)
    source, destination = graph.edges(form="uv")
    upper_source = source[upper_edge_mask]
    upper_destination = destination[upper_edge_mask]
    upper_edge_batch = edge_batch_index[upper_edge_mask]
    times = torch.linspace(0.0, 1.0, n_timesteps, device=graph.device)
    alpha = model.interpolant_scheduler.alpha_t(times)
    alpha_prime = model.interpolant_scheduler.alpha_t_prime(times)
    for feature in model.canonical_feat_order:
        data = graph.edata if feature == "e" else graph.ndata
        data[f"{feature}_t"] = data[f"{feature}_0"]

    previous_endpoint = None
    totals = {
        "prediction_mean": 0.0,
        "atom_logit_shift_rms": 0.0,
        "charge_logit_shift_rms": 0.0,
        "bond_logit_shift_rms": 0.0,
    }
    steps = 0
    for step_index in range(1, times.shape[0]):
        next_time = times[step_index]
        time = times[step_index - 1]
        last_step = step_index == times.shape[0] - 1
        model_times = torch.full(
            (int(graph.batch_size),), time, device=graph.device
        )
        with torch.no_grad():
            endpoint = vector_field(
                graph,
                t=model_times,
                node_batch_idx=node_batch_index,
                upper_edge_mask=upper_edge_mask,
                apply_softmax=True,
                remove_com=True,
                prev_dst_dict=previous_endpoint,
            )
        guided, diagnostics = guided_categorical_probabilities(
            predictor=predictor,
            endpoint=endpoint,
            node_batch_index=node_batch_index,
            upper_edge_source=upper_source,
            upper_edge_destination=upper_destination,
            upper_edge_batch_index=upper_edge_batch,
            times=model_times,
            temperature=vector_field.cat_temp_func(time),
            strengths=strengths,
            schedule_power=schedule_power,
            maximum_logit_shift=maximum_logit_shift,
        )
        for key in totals:
            totals[key] += diagnostics[key]
        steps += 1

        step_size = next_time - time
        with torch.no_grad():
            x_t = graph.ndata["x_t"]
            x_1 = endpoint["x"]
            velocity = vector_field.vector_field(
                x_t, x_1, alpha[step_index - 1, 0], alpha_prime[step_index - 1, 0]
            )
            graph.ndata["x_t"] = x_t + step_size * velocity
            graph.ndata["x_1_pred"] = x_1.detach().clone()

            for feature_index, feature in enumerate(model.canonical_feat_order):
                if feature == "x":
                    continue
                data = graph.edata if feature == "e" else graph.ndata
                current = data[f"{feature}_t"].argmax(dim=-1)
                if feature == "e":
                    current = current[upper_edge_mask]
                next_value, sampled_endpoint = vector_field.campbell_step(
                    p_1_given_t=guided[feature],
                    xt=current,
                    stochasticity=stochasticity,
                    hc_thresh=high_confidence_threshold,
                    alpha_t=alpha[step_index - 1, feature_index],
                    alpha_t_prime=alpha_prime[step_index - 1, feature_index],
                    dt=step_size,
                    batch_size=int(graph.batch_size),
                    batch_num_nodes=(
                        graph.batch_num_edges() // 2
                        if feature == "e"
                        else graph.batch_num_nodes()
                    ),
                    n_classes=vector_field.n_cat_feats[feature] + 1,
                    mask_index=vector_field.mask_idxs[feature],
                    last_step=last_step,
                    batch_idx=(
                        upper_edge_batch if feature == "e" else node_batch_index
                    ),
                )
                if feature == "e":
                    full_next = torch.zeros_like(graph.edata["e_t"])
                    full_next[upper_edge_mask] = next_value
                    full_next[~upper_edge_mask] = next_value
                    next_value = full_next
                    full_endpoint = torch.zeros_like(graph.edata["e_t"])
                    full_endpoint[upper_edge_mask] = sampled_endpoint
                    full_endpoint[~upper_edge_mask] = sampled_endpoint
                    sampled_endpoint = full_endpoint
                data[f"{feature}_t"] = next_value
                data[f"{feature}_1_pred"] = sampled_endpoint
        # Preserve the released model's native self-conditioning semantics.  The
        # guide changes transition probabilities, while the observed next state
        # carries that change into subsequent denoising calls.
        previous_endpoint = {
            feature: value.detach() for feature, value in endpoint.items()
        }

    for feature in model.canonical_feat_order:
        data = graph.edata if feature == "e" else graph.ndata
        data[f"{feature}_1"] = data[f"{feature}_t"]
    return graph, {key: value / max(steps, 1) for key, value in totals.items()}


def _sample_batch(
    *,
    model: Any,
    predictor: SoftGraphRewardRegressor,
    n_atoms: torch.Tensor,
    n_timesteps: int,
    stochasticity: float,
    high_confidence_threshold: float,
    strengths: dict[str, float],
    schedule_power: float,
    maximum_logit_shift: float,
) -> tuple[list[Any], list[dict[str, Any]], dict[str, float]]:
    import dgl
    from flowmol.analysis.molecule_builder import SampledMolecule
    from flowmol.data_processing.utils import (
        build_edge_idxs,
        get_batch_idxs,
        get_upper_edge_mask,
    )

    graphs = []
    for count in n_atoms:
        edges = build_edge_idxs(count)
        graphs.append(
            dgl.graph((edges[0], edges[1]), num_nodes=count, device=n_atoms.device)
        )
    graph = dgl.batch(graphs)
    upper_edge_mask = get_upper_edge_mask(graph)
    node_batch_index, _ = get_batch_idxs(graph)
    with torch.no_grad():
        graph = model.sample_prior(graph, node_batch_index, upper_edge_mask)
    graph, diagnostics = _guided_integrate(
        model=model,
        predictor=predictor,
        graph=graph,
        node_batch_index=node_batch_index,
        upper_edge_mask=upper_edge_mask,
        n_timesteps=n_timesteps,
        stochasticity=stochasticity,
        high_confidence_threshold=high_confidence_threshold,
        strengths=strengths,
        schedule_power=schedule_power,
        maximum_logit_shift=maximum_logit_shift,
    )
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
    priors = [
        _prior_from_graph(molecule.g, fake_atoms=bool(model.fake_atoms))
        for molecule in molecules
    ]
    return molecules, priors, diagnostics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--predictor", type=Path, required=True)
    parser.add_argument("--n-atoms-hist", type=Path, required=True)
    parser.add_argument("--marginal-dists", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-timesteps", type=int, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("FlowMol reward-guided sampling requires CUDA")
    if min(args.count, args.batch_size, args.n_timesteps) < 1:
        raise ValueError("sampling dimensions must be positive")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)
    sdf_dir = output_dir / "sdf"
    sdf_dir.mkdir()

    config_path = args.config.resolve()
    config = read_json(config_path)
    guidance = config["guidance"]
    registered_sampling = config["sampling"]
    if args.count > int(registered_sampling["attempts_per_seed"]):
        raise ValueError("sample count exceeds the registered per-seed budget")
    if args.n_timesteps != int(registered_sampling["flow_timesteps"]):
        raise ValueError("sampling timestep count differs from registered protocol")
    if int(args.seed) not in [int(value) for value in registered_sampling["sampling_seeds"]]:
        # Small engineering smokes use a separate fixed seed and fewer than 256
        # attempts; scientific evaluation seeds remain exactly registered.
        if args.count >= int(registered_sampling["attempts_per_seed"]):
            raise ValueError("full evaluation used an unregistered sampling seed")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda:0")
    from flowmol.models.flowmol import FlowMol

    checkpoint = args.checkpoint.resolve()
    model = FlowMol.load_from_checkpoint(
        str(checkpoint),
        n_atoms_hist_file=str(args.n_atoms_hist.resolve()),
        marginal_dists_file=str(args.marginal_dists.resolve()),
    ).to(device).eval()
    if model.parameterization != "ctmc":
        raise ValueError("reward guidance requires FlowMol CTMC")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    predictor_path = args.predictor.resolve()
    predictor, predictor_artifact = _load_predictor(predictor_path, device)
    if predictor_artifact["parent_checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("guidance predictor was trained against another generator")
    if predictor_artifact["config_sha256"] != sha256_file(config_path):
        raise ValueError("guidance predictor config differs from sampling config")
    expected_dimensions = {
        "atom_dim": int(model.n_atom_types),
        "charge_dim": int(model.n_atom_charges),
        "bond_dim": int(model.n_bond_types),
    }
    for name, value in expected_dimensions.items():
        if int(predictor_artifact["architecture"][name]) != value:
            raise ValueError(f"guidance predictor {name} differs from FlowMol")

    strengths = {
        "a": float(guidance["atom_logit_strength"]),
        "c": float(guidance["charge_logit_strength"]),
        "e": float(guidance["bond_logit_strength"]),
    }
    all_molecules: list[Any] = []
    all_priors: list[dict[str, Any]] = []
    batch_diagnostics: list[dict[str, float]] = []
    while len(all_molecules) < args.count:
        current = min(args.batch_size, args.count - len(all_molecules))
        with torch.no_grad():
            n_atoms = model.sample_n_atoms(current).to(device)
        molecules, priors, diagnostics = _sample_batch(
            model=model,
            predictor=predictor,
            n_atoms=n_atoms,
            n_timesteps=args.n_timesteps,
            stochasticity=float(registered_sampling["stochasticity"]),
            high_confidence_threshold=float(
                registered_sampling["high_confidence_threshold"]
            ),
            strengths=strengths,
            schedule_power=float(guidance["schedule_power"]),
            maximum_logit_shift=float(guidance["maximum_logit_shift"]),
        )
        all_molecules.extend(molecules)
        all_priors.extend(priors)
        batch_diagnostics.append(diagnostics)
        print(
            json.dumps({"sampled": len(all_molecules), "requested": args.count}),
            flush=True,
        )

    sdf_records = []
    for index, molecule in enumerate(all_molecules):
        path = sdf_dir / f"sample_{index}.sdf"
        rdkit_mol = getattr(molecule, "rdkit_mol", None)
        if rdkit_mol is not None:
            writer = Chem.SDWriter(str(path))
            writer.SetKekulize(False)
            writer.write(rdkit_mol)
            writer.close()
            sdf_records.append(str(path.resolve()))
    diagnostic_mean = {
        key: float(np.mean([row[key] for row in batch_diagnostics]))
        for key in batch_diagnostics[0]
    }
    artifact = {
        "molecules": all_molecules,
        "priors": all_priors,
        "metadata": {
            "schema_version": 1,
            "artifact_type": "flowmol_reward_guided_samples",
            "algorithm": config["algorithm"],
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256_file(checkpoint),
            "reward_predictor": str(predictor_path),
            "reward_predictor_sha256": sha256_file(predictor_path),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "sample_count": args.count,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "flow_timesteps": args.n_timesteps,
            "generator_optimizer_steps": 0,
            "guidance": guidance,
            "guidance_diagnostics": diagnostic_mean,
            "native_flowmol_graphs_retained": True,
            "native_flowmol_priors_retained": True,
            "valid_sdf_count": len(sdf_records),
            "sdf_files": sdf_records,
        },
    }
    torch.save(artifact, output_dir / "samples.pt")
    print(json.dumps(artifact["metadata"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
