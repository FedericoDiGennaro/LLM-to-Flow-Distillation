#!/usr/bin/env python3
"""Sample a fixed PepDFM attempt budget and select an exact unique pool."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any








def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()








def validate_asset_manifest(path: Path, asset_root: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("unsupported PepDFM asset manifest")
    if manifest.get("repository_id") != "ChatterjeeLab/MOG-DFM":
        raise ValueError("PepDFM asset manifest repository mismatch")
    if manifest.get("revision") != "fe115ca03bb2dab2f5aac2f2fbaae56e65c876d6":
        raise ValueError("PepDFM asset manifest revision mismatch")
    if manifest.get("license_acceptance_required") is not True:
        raise ValueError("PepDFM asset manifest omits the gated-license declaration")
    if not str(manifest.get("authenticated_user", "")).strip():
        raise ValueError("PepDFM asset manifest omits the authenticated user")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("PepDFM asset manifest has no files")
    observed_paths: set[str] = set()
    verified = []
    asset_root = asset_root.resolve()
    for row in files:
        if not isinstance(row, dict):
            raise ValueError("PepDFM asset manifest file row is invalid")
        relative = row.get("path")
        size = row.get("size_bytes")
        expected_hash = row.get("sha256")
        if not isinstance(relative, str) or not relative or relative in observed_paths:
            raise ValueError("PepDFM asset manifest paths are invalid or duplicated")
        candidate = (asset_root / relative).resolve()
        if not candidate.is_relative_to(asset_root) or not candidate.is_file():
            raise ValueError(f"PepDFM asset manifest path is missing or escapes root: {relative}")
        if isinstance(size, bool) or not isinstance(size, int) or candidate.stat().st_size != size:
            raise ValueError(f"PepDFM asset size mismatch: {relative}")
        if not isinstance(expected_hash, str) or sha256_file(candidate) != expected_hash:
            raise ValueError(f"PepDFM asset hash mismatch: {relative}")
        observed_paths.add(relative)
        verified.append({"path": relative, "size_bytes": size, "sha256": expected_hash})
    actual_paths = {
        str(candidate.relative_to(asset_root))
        for candidate in asset_root.rglob("*")
        if (
            candidate.is_file()
            and ".cache" not in candidate.relative_to(asset_root).parts
            and candidate.relative_to(asset_root) != Path("asset_manifest.json")
        )
    }
    if actual_paths != observed_paths:
        missing = sorted(observed_paths - actual_paths)
        unmanifested = sorted(actual_paths - observed_paths)
        raise ValueError(
            "PepDFM asset manifest file set mismatch: "
            f"missing={missing}, unmanifested={unmanifested}"
        )
    return {
        "manifest_sha256": sha256_file(path),
        "authenticated_user": manifest["authenticated_user"],
        "verified_file_count": len(verified),
        "verified_total_bytes": sum(row["size_bytes"] for row in verified),
    }
