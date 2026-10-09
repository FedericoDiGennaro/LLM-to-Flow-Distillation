#!/usr/bin/env python3
"""Validate matched-prior FlowMol child generation and engineering gates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from domains.molecules.gsk3b.artifacts import immutable_json, read_json, sha256_file


def _tensor_digest(digest: Any, name: str, value: torch.Tensor) -> None:
    tensor = value.detach().cpu().contiguous()
    digest.update(name.encode("utf-8"))
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(str(tuple(tensor.shape)).encode("ascii"))
    digest.update(tensor.numpy().tobytes())


def native_signature(molecule: Any) -> str:
    graph = getattr(molecule, "g", None)
    if graph is None:
        raise ValueError("sampled molecule is missing its native FlowMol graph")
    digest = hashlib.sha256()
    for field in ("x_1", "a_1", "c_1"):
        value = graph.ndata.get(field)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"FlowMol graph is missing node field {field}")
        _tensor_digest(digest, field, value)
    edges = graph.edata.get("e_1")
    if not isinstance(edges, torch.Tensor):
        raise ValueError("FlowMol graph is missing edge field e_1")
    _tensor_digest(digest, "e_1", edges)
    return digest.hexdigest()


def family_signature(molecule: Any) -> str:
    graph = getattr(molecule, "g", None)
    if graph is None:
        raise ValueError("sampled molecule is missing its native FlowMol graph")
    atom_types = graph.ndata["a_1"].detach().cpu().argmax(dim=-1).tolist()
    return hashlib.sha256(
        json.dumps(
            {
                "atom_count": int(graph.num_nodes()),
                "atom_type_multiset": sorted(int(value) for value in atom_types),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()


def prior_signature(prior: dict[str, Any]) -> str:
    digest = hashlib.sha256()
    for field in ("x_0", "a_0", "c_0", "e_0"):
        value = prior.get(field)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"retained prior is missing {field}")
        _tensor_digest(digest, field, value)
    digest.update(str(int(prior["n_atoms"])).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-samples", type=Path, required=True)
    parser.add_argument("--frozen-p0-samples", type=Path)
    parser.add_argument("--child-samples", type=Path, required=True)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument("--child-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    parent_path = args.parent_samples.resolve()
    frozen_p0_path = (
        parent_path
        if args.frozen_p0_samples is None
        else args.frozen_p0_samples.resolve()
    )
    parent = torch.load(parent_path, map_location="cpu", weights_only=False)
    frozen_p0 = torch.load(frozen_p0_path, map_location="cpu", weights_only=False)
    child = torch.load(
        args.child_samples.resolve(), map_location="cpu", weights_only=False
    )
    config = read_json(args.config.resolve())

    parent_molecules = parent.get("molecules", [])
    child_molecules = child.get("molecules", [])
    parent_priors = parent.get("priors", [])
    child_priors = child.get("priors", [])
    if not parent_molecules or len(parent_molecules) != len(child_molecules):
        raise ValueError("parent and child sample counts differ")
    if len(parent_priors) != len(child_priors) or len(parent_priors) != len(
        parent_molecules
    ):
        raise ValueError("retained FlowMol priors are incomplete")
    if [prior_signature(row) for row in parent_priors] != [
        prior_signature(row) for row in child_priors
    ]:
        raise ValueError("parent and child generation did not use identical priors")

    parent_native = [native_signature(molecule) for molecule in parent_molecules]
    child_native = [native_signature(molecule) for molecule in child_molecules]
    parent_families = {family_signature(molecule) for molecule in parent_molecules}
    frozen_p0_families = {
        family_signature(molecule) for molecule in frozen_p0.get("molecules", [])
    }
    if not frozen_p0_families:
        raise ValueError("frozen p0 contains no FlowMol families")
    child_families = {family_signature(molecule) for molecule in child_molecules}
    overlap = len(set(parent_native) & set(child_native)) / len(child_native)
    child_unique_fraction = len(set(child_native)) / len(child_native)
    parent_family_ratio = len(child_families) / max(len(parent_families), 1)
    frozen_p0_family_ratio = len(child_families) / len(frozen_p0_families)
    child_sdf_count = int(child.get("metadata", {}).get("valid_sdf_count", 0))
    rdkit_fraction = child_sdf_count / len(child_molecules)
    gates = config["checkpoint_gates"]
    parent_hash = sha256_file(args.parent_checkpoint.resolve())
    child_hash = sha256_file(args.child_checkpoint.resolve())
    checks = {
        "checkpoint_changed": child_hash != parent_hash,
        "matched_priors": True,
        "child_native_states_unique": child_unique_fraction == 1.0,
        "exact_native_replay_within_limit": overlap
        <= float(gates["maximum_exact_native_state_replay_fraction"]),
        "rdkit_success_within_limit": rdkit_fraction
        >= float(gates["minimum_rdkit_success_fraction"]),
    }
    result = {
        "schema_version": 1,
        "artifact_type": "flowmol_child_noncollapse_smoke",
        "sample_count_per_arm": len(child_molecules),
        "parent_checkpoint_sha256": parent_hash,
        "child_checkpoint_sha256": child_hash,
        "parent_samples_sha256": sha256_file(parent_path),
        "frozen_p0_samples_sha256": sha256_file(frozen_p0_path),
        "child_samples_sha256": sha256_file(args.child_samples.resolve()),
        "child_unique_native_fraction": child_unique_fraction,
        "parent_family_count": len(parent_families),
        "frozen_p0_family_count": len(frozen_p0_families),
        "child_family_count": len(child_families),
        "child_parent_family_count_ratio": parent_family_ratio,
        "child_frozen_p0_family_count_ratio": frozen_p0_family_ratio,
        "exact_parent_child_native_replay_fraction": overlap,
        "child_rdkit_success_fraction": rdkit_fraction,
        "checks": checks,
        "all_engineering_gates_passed": all(checks.values()),
        "scientific_preference_used": False,
        "scientific_improvement_claimed": False,
        "training_authorized": False,
        "gnina_used": False,
    }
    immutable_json(args.output.resolve(), result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["all_engineering_gates_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
