#!/usr/bin/env python3
"""Fetch pinned gated PepDFM and released MOG-DFM baseline assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download

from domains.peptides.model import (
    PEPDFM_REPO_ID,
    PEPDFM_REVISION,
    PepDfmAssets,
    sha256_file,
)
from domains.peptides.asset_manifest import validate_asset_manifest


DEFAULT_ASSET_DIR = Path(".cache/upstream/mog-dfm")
ALLOW_PATTERNS = (
    "README.md",
    "PepDFM_unconditional_sample.py",
    "PepDFM_multi_objective_generation.py",
    "ckpt/peptide/cnn_epoch200_lr0.0001_embed512_hidden256_loss3.1051.ckpt",
    "classifier_ckpt/best_model_hemolysis.json",
    "classifier_ckpt/best_model_nonfouling.json",
    "classifier_ckpt/best_model_solubility.json",
    "classifier_ckpt/binding_affinity_pooled.pt",
    "classifier_ckpt/binding_affinity_unpooled.pt",
    "classifier_ckpt/wt_halflife.pt",
    "classifier_ckpt/wt_nonfouling.pt",
    "classifier_code/**",
    "models/peptide_classifiers.py",
    "models/peptide_models.py",
    "flow_matching/**",
    "utils/**",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, default=DEFAULT_ASSET_DIR)
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Verify the pinned local assets without contacting Hugging Face.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.local_files_only:
        assets = PepDfmAssets.resolve(args.asset_dir)
        manifest_path = assets.root / "asset_manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"asset manifest is missing: {manifest_path}")
        validation = validate_asset_manifest(manifest_path, assets.root)
        print(
            f"Verified all {validation['verified_file_count']} pinned local PepDFM "
            f"files at {assets.root}; manifest: {manifest_path}"
        )
        return
    identity = HfApi().whoami()
    snapshot_download(
        repo_id=PEPDFM_REPO_ID,
        revision=PEPDFM_REVISION,
        allow_patterns=list(ALLOW_PATTERNS),
        local_dir=args.asset_dir,
    )
    assets = PepDfmAssets.resolve(args.asset_dir)
    files = sorted(
        path
        for path in assets.root.rglob("*")
        if (
            path.is_file()
            and ".cache" not in path.relative_to(assets.root).parts
            and path.relative_to(assets.root) != Path("asset_manifest.json")
        )
    )
    manifest = {
        "schema_version": 1,
        "repository_id": PEPDFM_REPO_ID,
        "revision": PEPDFM_REVISION,
        "authenticated_user": identity["name"],
        "license_acceptance_required": True,
        "files": [
            {
                "path": str(path.relative_to(assets.root)),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in files
        ],
    }
    manifest_path = assets.root / "asset_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_asset_manifest(manifest_path, assets.root)
    total_bytes = sum(item["size_bytes"] for item in manifest["files"])
    print(
        f"Verified {len(files)} PepDFM files ({total_bytes / 1024 / 1024:.2f} MiB) "
        f"at {assets.root} for Hugging Face user {identity['name']}"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
