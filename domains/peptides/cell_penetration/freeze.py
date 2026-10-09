#!/usr/bin/env python3
"""Bind a V13 template to its LLM policy and deterministic runtime artifacts."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from domains.peptides.cell_penetration.length_policy import (
    load_pre_p0_length_policy_artifact,
)
from domains.peptides.config_io import (
    ZERO_SHA256,
    immutable_write,
    read_json,
    sha256_file,
)


def _registered_path(value: Any) -> Path:
    return Path(os.path.expandvars(str(value))).resolve()


def _zero_bound(record: Any, *, name: str, path: Path) -> dict[str, Any]:
    if not isinstance(record, Mapping):
        raise TypeError(f"{name} must be an object")
    if record.get("sha256") != ZERO_SHA256:
        raise ValueError(f"{name} requires a zero SHA-256 placeholder")
    if _registered_path(record.get("path")) != path.resolve():
        raise ValueError(f"{name} path differs from generated artifact")
    return dict(record)


def freeze(
    *,
    template_path: Path,
    candidate_path: Path,
    evidence_reference_path: Path,
    length_policy_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    template = read_json(template_path)
    if template.get("prompt_protocol", {}).get("version") != (
        "geometric-semantic-reachability-v13"
    ):
        raise ValueError("config template is not V13")
    candidate = _zero_bound(
        template.get("candidate_pool"), name="candidate_pool", path=candidate_path
    )
    evidence = _zero_bound(
        template.get("fresh_sampling", {}).get("frozen_evidence_reference"),
        name="frozen_evidence_reference",
        path=evidence_reference_path,
    )
    policy_record = _zero_bound(
        template.get("campaign_length_policy"),
        name="campaign_length_policy",
        path=length_policy_path,
    )
    artifact = load_pre_p0_length_policy_artifact(length_policy_path)
    counts = {
        str(row.length): row.count for row in artifact.length_policy.allocations
    }
    if candidate.get("length_counts") not in ({}, None):
        raise ValueError("template candidate length_counts must be empty")
    sampling = template.get("fresh_sampling")
    if not isinstance(sampling, Mapping) or sampling.get(
        "length_counts_per_arm"
    ) not in ({}, None):
        raise ValueError("template fresh-sampling length policy must be empty")

    frozen = dict(template)
    frozen["candidate_pool"] = {
        **candidate,
        "sha256": sha256_file(candidate_path),
        "length_counts": counts,
    }
    frozen["campaign_length_policy"] = {
        **policy_record,
        "sha256": sha256_file(length_policy_path),
        "length_counts": counts,
        "selected_before_p0": True,
        "frozen_for_entire_campaign": True,
    }
    frozen["fresh_sampling"] = {
        **dict(sampling),
        "length_counts_per_arm": counts,
        "frozen_evidence_reference": {
            **evidence,
            "sha256": sha256_file(evidence_reference_path),
        },
    }
    immutable_write(
        output_path,
        json.dumps(frozen, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    manifest = {
        "schema_version": 1,
        "transformation": "V13 bounded dynamic-artifact hash binding only",
        "template_path": str(template_path.resolve()),
        "template_sha256": sha256_file(template_path),
        "frozen_config_path": str(output_path.resolve()),
        "frozen_config_sha256": sha256_file(output_path),
        "candidate_pool_sha256": sha256_file(candidate_path),
        "frozen_evidence_reference_sha256": sha256_file(evidence_reference_path),
        "campaign_length_policy_sha256": sha256_file(length_policy_path),
        "external_dataset_context_visible": False,
        "length_counts": counts,
    }
    manifest_path = output_path.with_name("frozen_config_manifest.json")
    immutable_write(
        manifest_path,
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--evidence-reference", type=Path, required=True)
    parser.add_argument("--length-policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            freeze(
                template_path=args.template.resolve(),
                candidate_path=args.candidates.resolve(),
                evidence_reference_path=args.evidence_reference.resolve(),
                length_policy_path=args.length_policy.resolve(),
                output_path=args.output.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
