#!/usr/bin/env python3
"""Generate one 100-peptide p0 cohort under a frozen LLM-chosen length policy.

The length policy is selected before p0 sampling and remains fixed across the
entire V13 parent/child trajectory.  This avoids attributing an easier length
mixture to PepDFM fine-tuning.  Cardinal developability predictor values stay
private; Gemini receives only p0-frozen within-length qualitative bands.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import torch

from domains.peptides.lengths import AdaptiveLengthPolicy
from domains.peptides.frozen_evidence import (
    apply_frozen_reference,
    fit_frozen_reference,
)
from domains.peptides.model import (
    PEPDFM_CHECKPOINT_SHA256,
    PEPDFM_REVISION,
    PepDfmAssets,
)
from domains.peptides.checkpoint import PepDfmCheckpoint, sha256_file
from domains.peptides.qualitative_evidence import (
    DEVELOPABILITY_EVIDENCE_LIMITATION,
)
from domains.peptides.pool_io import (
    _predict_private_rows,
    immutable_write,
    jsonl_text,
    read_json,
)
from adm.campaign import resolve_campaign_registration


def _load_policy(path: Path) -> tuple[AdaptiveLengthPolicy, Mapping[str, Any]]:
    value = read_json(path)
    content_hash = str(value.get("artifact_content_sha256", ""))
    content = dict(value)
    content.pop("artifact_content_sha256", None)
    observed_content_hash = hashlib.sha256(
        json.dumps(
            content,
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    if content_hash != observed_content_hash:
        raise ValueError("length-policy artifact content hash differs")
    policy_value = value.get("length_policy")
    if not isinstance(policy_value, Mapping):
        raise TypeError("length-policy artifact omitted length_policy")
    allocations = policy_value.get("allocations")
    if not isinstance(allocations, list) or not allocations:
        raise TypeError("length-policy artifact omitted allocations")
    counts: dict[int, int] = {}
    for row in allocations:
        if not isinstance(row, Mapping):
            raise TypeError("length-policy allocation must be an object")
        length = int(row.get("length", -1))
        count = int(row.get("count", -1))
        if length in counts:
            raise ValueError("length-policy artifact contains duplicate lengths")
        counts[length] = count
    policy = AdaptiveLengthPolicy(counts)
    if (
        value.get("protocol") != "pre-p0-campaign-length-policy-v13"
        or value.get("policy_frozen_for_entire_campaign") is not True
        or value.get("sample_count") != 100
    ):
        raise ValueError("length-policy artifact is not frozen for 100 samples")
    return policy, value


def _validate_registration(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    policy_path: Path,
    campaign_path: Path,
) -> None:
    campaign = read_json(campaign_path)
    if campaign.get("status") != "active":
        raise PermissionError("ACTIVE_CAMPAIGN.json is not active")
    experiment_id = str(config.get("experiment_id", ""))
    if not experiment_id:
        raise PermissionError("campaign and V13 p0 experiment IDs differ")
    resources = resolve_campaign_registration(
        campaign, experiment_id=experiment_id
    ).resources
    if resources.get("pool_config_sha256") != sha256_file(config_path):
        raise PermissionError("V13 p0 static resource binding differs")
    policy_record = resources.get("dynamic_length_policy")
    if not isinstance(policy_record, Mapping):
        raise PermissionError("campaign omitted dynamic length-policy authorization")
    if (
        Path(os.path.expandvars(str(policy_record.get("path", "")))).resolve()
        != policy_path.resolve()
        or policy_record.get("selection_method")
        != "one Gemini controller call under the registered bounded schema"
    ):
        raise PermissionError("V13 length-policy path/method differs")


def execute(
    *,
    config_path: Path,
    policy_path: Path,
    output_root: Path,
    asset_dir: Path,
    campaign_path: Path,
) -> dict[str, Any]:
    config = read_json(config_path)
    if config.get("schema_version") != 1 or config.get("protocol_version") != (
        "geometric-semantic-reachability-v13"
    ):
        raise ValueError("p0 pool config is not the V13 protocol")
    _validate_registration(
        config_path=config_path,
        config=config,
        policy_path=policy_path,
        campaign_path=campaign_path,
    )
    policy, policy_value = _load_policy(policy_path)
    paths = {
        "visible": output_root / "candidate_pool.jsonl",
        "raw": output_root / "raw_samples_private.jsonl",
        "predictors": output_root / "predictor_scores_private.jsonl",
        "reference": output_root / "frozen_evidence_reference_private.json",
        "manifest": output_root / "pool_manifest.json",
    }
    if paths["manifest"].is_file():
        manifest = read_json(paths["manifest"])
        expected = {
            "candidate_pool_sha256": sha256_file(paths["visible"]),
            "raw_samples_sha256": sha256_file(paths["raw"]),
            "private_predictor_scores_sha256": sha256_file(paths["predictors"]),
            "frozen_evidence_reference_sha256": sha256_file(paths["reference"]),
            "length_policy_sha256": sha256_file(policy_path),
            "config_sha256": sha256_file(config_path),
        }
        if any(manifest.get(name) != value for name, value in expected.items()):
            raise FileExistsError("completed V13 p0 pool differs from its manifest")
        return manifest

    output_root.mkdir(parents=True, exist_ok=True)
    device = str(config["device"])
    assets = PepDfmAssets.resolve(asset_dir)
    model = PepDfmCheckpoint(assets, device=device)
    generated: list[dict[str, Any]] = []
    base_seed = int(config["seed"])
    batch_size = int(config.get("sampling_batch_size", 60))
    for length, count in policy.entries:
        seed = base_seed + length
        samples = model.sample(
            num_samples=count,
            length=length,
            num_steps=int(config["num_steps"]),
            seed=seed,
            batch_size=min(count, batch_size),
            source_distribution=str(config["source_distribution"]),
        )
        for local_index, sample in enumerate(samples):
            if not sample.is_valid or sample.sequence is None:
                raise RuntimeError(f"PepDFM emitted an invalid length-{length} sample")
            generated.append(
                {
                    "candidate_id": f"p0-l{length:02d}-{local_index:04d}",
                    "generator_id": "p0",
                    "sequence": sample.sequence,
                    "length": length,
                    "sampling_seed": seed,
                    "sample_index": sample.sample_index,
                    "token_ids": list(sample.token_ids),
                    "is_valid": True,
                }
            )
    observed = Counter(int(row["length"]) for row in generated)
    if dict(sorted(observed.items())) != policy.as_mapping(string_keys=False):
        raise RuntimeError("generated p0 histogram differs from frozen length policy")
    sequences = [str(row["sequence"]) for row in generated]
    if len(generated) != 100 or len(sequences) != len(set(sequences)):
        raise RuntimeError("V13 p0 must contain exactly 100 unique valid sequences")

    private = _predict_private_rows(
        generated,
        asset_dir=asset_dir,
        device=device,
        batch_size=int(config["predictor_batch_size"]),
    )
    frozen_reference = fit_frozen_reference(private)
    evidence = apply_frozen_reference(private, frozen_reference)
    visible = [
        {
            "candidate_id": str(row["candidate_id"]),
            "generator_id": "p0",
            "sequence": str(row["sequence"]),
            "length": int(row["length"]),
            "is_valid": True,
            "developability_evidence": evidence[str(row["candidate_id"])],
        }
        for row in generated
    ]
    immutable_write(paths["visible"], jsonl_text(visible))
    immutable_write(paths["raw"], jsonl_text(generated))
    immutable_write(paths["predictors"], jsonl_text(private))
    immutable_write(
        paths["reference"],
        json.dumps(frozen_reference, allow_nan=False, indent=2, sort_keys=True)
        + "\n",
    )
    manifest = {
        "schema_version": 1,
        "protocol_version": "geometric-semantic-reachability-v13",
        "experiment_id": config["experiment_id"],
        "generator_id": "p0",
        "generator": "released unconditional PepDFM p0",
        "sample_count": 100,
        "unique_count": 100,
        "length_histogram": policy.as_mapping(),
        "length_policy_path": str(policy_path.resolve()),
        "length_policy_sha256": sha256_file(policy_path),
        "length_policy_controller_rationale": policy_value.get(
            "length_policy", {}
        ).get("rationale"),
        "pepdfm_revision": PEPDFM_REVISION,
        "pepdfm_checkpoint_sha256": PEPDFM_CHECKPOINT_SHA256,
        "visible_evidence_limitation": DEVELOPABILITY_EVIDENCE_LIMITATION,
        "cardinal_predictor_outputs_visible": False,
        "external_dataset_context_visible": False,
        "candidate_pool_path": str(paths["visible"].resolve()),
        "candidate_pool_sha256": sha256_file(paths["visible"]),
        "raw_samples_path": str(paths["raw"].resolve()),
        "raw_samples_sha256": sha256_file(paths["raw"]),
        "private_predictor_scores_path": str(paths["predictors"].resolve()),
        "private_predictor_scores_sha256": sha256_file(paths["predictors"]),
        "frozen_evidence_reference_path": str(paths["reference"].resolve()),
        "frozen_evidence_reference_sha256": sha256_file(paths["reference"]),
        "config_sha256": sha256_file(config_path),
        "torch_version": torch.__version__,
    }
    immutable_write(
        paths["manifest"],
        json.dumps(manifest, allow_nan=False, indent=2, sort_keys=True) + "\n",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--length-policy", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--active-campaign", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            execute(
                config_path=args.config.resolve(),
                policy_path=args.length_policy.resolve(),
                output_root=args.output_root.resolve(),
                asset_dir=args.asset_dir.resolve(),
                campaign_path=args.active_campaign.resolve(),
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
