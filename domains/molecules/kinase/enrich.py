#!/usr/bin/env python3
"""Compute the frozen inexpensive molecule evidence card inputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from domains.molecules.kinase.evidence import compute_evidence, mol_from_sdf
from domains.molecules.kinase.artifacts import immutable_json, read_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    qualification = read_json(args.qualification.resolve())
    if qualification.get("qualification_passed") is not True:
        raise ValueError("cannot enrich an unqualified molecule")
    sdf = Path(qualification["source_sdf"]).resolve()
    if sha256_file(sdf) != qualification["source_sdf_sha256"]:
        raise ValueError("qualified SDF changed")
    mol = mol_from_sdf(str(sdf))
    evidence = compute_evidence(mol)
    sdf_hash = sha256_file(sdf)
    record = {
        "schema_version": 1,
        "artifact_type": "molecule_cheap_candidate_evidence",
        "candidate_id": f"mol-{sdf_hash[:16]}",
        "qualification": evidence["qualification"],
        "identity": evidence["identity"],
        "descriptors": evidence["descriptors"],
        "functional_groups": evidence["functional_groups"],
        "geometry": evidence["geometry"],
        "private_provenance": {
            "source_sdf": str(sdf),
            "source_sdf_sha256": sdf_hash,
            "qualification": str(args.qualification.resolve()),
            "qualification_sha256": sha256_file(args.qualification.resolve()),
        },
        "docking_used": False,
        "gnina_used": False,
        "llm_tools_invoked": False,
        "training_authorized": False,
    }
    immutable_json(output_dir / "evidence_record.json", record)
    print(json.dumps(record, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
