#!/usr/bin/env python3
"""Sample FlowMol3 under a frozen distilled GSK3B endpoint classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random

import numpy as np
import torch
from rdkit import Chem

from domains.molecules.gsk3b.guidance import (
    guided_endpoint_predictions,
    load_surrogate,
)
from domains.molecules.gsk3b.sample import _batched_prior, _prior_from_graph
from domains.molecules.gsk3b.artifacts import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--n-atoms-hist", type=Path, required=True)
    parser.add_argument("--marginal-dists", type=Path, required=True)
    parser.add_argument("--surrogate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--n-timesteps", type=int, default=250)
    parser.add_argument("--categorical-scale", type=float, required=True)
    parser.add_argument("--coordinate-scale", type=float, required=True)
    parser.add_argument("--priors-from", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("distilled FlowMol3 guidance requires CUDA")
    output = args.output_dir.resolve()
    samples_path = output / "samples.pt"
    if samples_path.exists():
        raise FileExistsError(f"refusing to overwrite {samples_path}")
    output.mkdir(parents=True, exist_ok=True)
    sdf_dir = output / "sdf"
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
    ).to("cuda:0").eval()
    surrogate, mean, scale, surrogate_record = load_surrogate(
        str(args.surrogate.resolve()), torch.device("cuda:0")
    )
    parent_priors = None
    if args.priors_from is not None:
        parent = torch.load(
            args.priors_from.resolve(), map_location="cpu", weights_only=False
        )
        parent_priors = parent.get("priors")
        if not isinstance(parent_priors, list) or len(parent_priors) != args.count:
            raise ValueError("guided sampler needs one retained prior per candidate")
    molecules = []
    with guided_endpoint_predictions(
        model.vector_field, surrogate, mean, scale,
        categorical_scale=args.categorical_scale,
        coordinate_scale=args.coordinate_scale,
    ):
        while len(molecules) < args.count:
            current = min(args.batch_size, args.count - len(molecules))
            if parent_priors is None:
                molecules.extend(model.sample_random_sizes(
                    current, device="cuda:0", n_timesteps=args.n_timesteps,
                ))
            else:
                offset = len(molecules)
                chunk = parent_priors[offset:offset + current]
                n_atoms = torch.tensor(
                    [int(row["n_atoms"]) for row in chunk], device="cuda:0"
                )
                molecules.extend(model.sample(
                    n_atoms, device="cuda:0", n_timesteps=args.n_timesteps,
                    prior=_batched_prior(chunk, "cuda:0"),
                ))
    priors = [
        _prior_from_graph(molecule.g, fake_atoms=bool(model.fake_atoms))
        for molecule in molecules
    ]
    sdf_files = []
    for index, molecule in enumerate(molecules):
        path = sdf_dir / f"sample_{index}.sdf"
        rdkit_mol = getattr(molecule, "rdkit_mol", None)
        if rdkit_mol is not None:
            writer = Chem.SDWriter(str(path))
            writer.SetKekulize(False)
            writer.write(rdkit_mol)
            writer.close()
            sdf_files.append(str(path.resolve()))
    metadata = {
        "schema_version": 1,
        "artifact_type": "flowmol_gsk3b_distilled_guidance_samples",
        "checkpoint": str(args.checkpoint.resolve()),
        "surrogate": str(args.surrogate.resolve()),
        "surrogate_sha256": sha256_file(args.surrogate.resolve()),
        "surrogate_validation": surrogate_record["validation"],
        "sample_count": args.count,
        "valid_sdf_count": len(sdf_files),
        "batch_size": args.batch_size,
        "seed": args.seed,
        "flow_timesteps": args.n_timesteps,
        "categorical_scale": args.categorical_scale,
        "coordinate_scale": args.coordinate_scale,
        "guidance_rule": "mirror_descent_on_categorical_endpoint_probabilities_and_gradient_shift_of_coordinate_endpoint",
        "matched_priors_from": (
            None if args.priors_from is None else str(args.priors_from.resolve())
        ),
        "native_flowmol_graphs_retained": True,
        "native_flowmol_priors_retained": True,
        "sdf_files": sdf_files,
    }
    torch.save({"molecules": molecules, "priors": priors, "metadata": metadata}, samples_path)
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
