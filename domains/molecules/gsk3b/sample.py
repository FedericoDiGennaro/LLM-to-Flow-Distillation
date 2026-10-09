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


if __name__ == "__main__":
    main()
