#!/usr/bin/env python3
"""Qualify one generated molecule SDF with inexpensive RDKit checks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from domains.molecules.drd2.evidence import mol_from_sdf, qualify_molecule
from domains.molecules.drd2.artifacts import immutable_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdf", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    source = args.sdf.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    mol = mol_from_sdf(str(source))
    qualification = qualify_molecule(mol)
    record = {
        "schema_version": 1,
        "artifact_type": "molecule_rdkit_qualification",
        "source_sdf": str(source),
        "source_sdf_sha256": sha256_file(source),
        "qualification_passed": qualification["qualification_passed"],
        "status": qualification["status"],
        "reason": qualification.get("reason"),
        "atom_count": qualification.get("atom_count"),
        "training_authorized": False,
        "gnina_used": False,
        "claim_boundary": (
            "An RDKit parse, connectivity and 3D-conformer diagnostic only; "
            "not evidence of DRD2 binding, pose, affinity, or experimental activity."
        ),
    }
    immutable_json(output_dir / "qualification.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
