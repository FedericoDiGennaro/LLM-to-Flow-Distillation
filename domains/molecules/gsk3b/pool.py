#!/usr/bin/env python3
"""Prepare one FlowMol cohort with cheap evidence and blinded cards."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
from typing import Any

import torch

from domains.molecules.gsk3b.cards import candidate_card
from domains.molecules.gsk3b.artifacts import (
    immutable_json,
    immutable_jsonl,
    read_json,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[3]


def _run(command: list[str], timeout: int) -> None:
    environment = dict(os.environ)
    environment.pop("GEMINI_API_KEY", None)
    environment.pop("GOOGLE_API_KEY", None)
    for name in ("LITELLM_API_KEY", "OPENAI_API_KEY", "API_KEY_FILE"):
        environment.pop(name, None)
    subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        check=True,
        timeout=timeout,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--role", choices=("training", "development"), required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    read_json(args.config.resolve())
    samples_path = args.samples.resolve()
    samples_sha256 = sha256_file(samples_path)
    # This is a locally generated, SHA-256-bound FlowMol artifact containing
    # SampledMolecule/DGL objects, not an untrusted weights-only checkpoint.
    samples = torch.load(samples_path, map_location="cpu", weights_only=False)
    molecules = samples.get("molecules", [])
    if not isinstance(molecules, list) or not molecules:
        raise ValueError("FlowMol generation artifacts have no molecules")

    attempts: list[dict[str, Any]] = []
    judgeable: list[dict[str, Any]] = []
    for index, molecule in enumerate(molecules):
        candidate_id = f"{args.role}-attempt-{index:04d}"
        candidate_root = output_dir / "candidates" / candidate_id
        row: dict[str, Any] = {
            "candidate_id": candidate_id,
            "role": args.role,
            "sample_index": index,
            "training_state": {
                "samples_pt": str(samples_path),
                "samples_pt_sha256": samples_sha256,
                "sample_index": index,
                "native_graph_nodes": int(getattr(molecule, "g").num_nodes())
                if getattr(molecule, "g", None) is not None
                else None,
            },
        }
        sdf = samples_path.parent / "sdf" / f"sample_{index}.sdf"
        if not sdf.is_file():
            row["status"] = "unjudgeable_missing_sdf"
            attempts.append(row)
            continue
        try:
            _run(
                [
                    str(args.python.absolute()),
                    "-m",
                    "domains.molecules.gsk3b.structure",
                    "--sdf",
                    str(sdf),
                    "--output-dir",
                    str(candidate_root / "qualification"),
                ],
                120,
            )
            qualification_path = candidate_root / "qualification/qualification.json"
            qualification = read_json(qualification_path)
            if qualification.get("qualification_passed") is not True:
                row["status"] = "unjudgeable_structural_qualification"
                row["qualification"] = str(qualification_path)
                attempts.append(row)
                continue
            _run(
                [
                    str(args.python.absolute()),
                    "-m",
                    "domains.molecules.gsk3b.enrich",
                    "--qualification",
                    str(qualification_path),
                    "--output-dir",
                    str(candidate_root / "evidence"),
                ],
                120,
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            row["status"] = "unjudgeable_evidence_failure"
            row["failure_type"] = type(error).__name__
            attempts.append(row)
            continue

        evidence_path = candidate_root / "evidence/evidence_record.json"
        item_id = f"item-{len(judgeable):03d}"
        card = candidate_card(
            read_json(evidence_path),
            item_id=item_id,
        ).model_dump(mode="json")
        card_path = candidate_root / "candidate_card.json"
        immutable_json(card_path, card)
        row.update(
            {
                "status": "judgeable",
                "item_id": item_id,
                "qualification": str(qualification_path),
                "qualification_sha256": sha256_file(qualification_path),
                "evidence": str(evidence_path),
                "evidence_sha256": sha256_file(evidence_path),
                "card": card,
                "card_path": str(card_path),
                "card_sha256": sha256_file(card_path),
            }
        )
        attempts.append(row)
        judgeable.append(row)

    immutable_jsonl(output_dir / "attempt_records.jsonl", attempts)
    immutable_jsonl(output_dir / "candidate_pool.jsonl", judgeable)
    manifest = {
        "schema_version": 1,
        "artifact_type": "molecules_geometric_pool_v1",
        "role": args.role,
        "generation_attempt_count": len(attempts),
        "judgeable_count": len(judgeable),
        "unjudgeable_count": len(attempts) - len(judgeable),
        "samples_pt": str(samples_path),
        "samples_pt_sha256": samples_sha256,
        "candidate_pool": str(output_dir / "candidate_pool.jsonl"),
        "candidate_pool_sha256": sha256_file(output_dir / "candidate_pool.jsonl"),
        "all_generation_attempts_retained_in_denominator": True,
        "gnina_used": False,
        "training_authorized": False,
    }
    immutable_json(output_dir / "pool_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if not judgeable:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
