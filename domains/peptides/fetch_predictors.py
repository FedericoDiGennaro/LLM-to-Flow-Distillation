#!/usr/bin/env python3
"""Fetch the exact public AntiCP2 and HemoPI2 releases used by the baselines."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import subprocess
import tempfile


SOURCES = {
    "anticp2": (
        "https://github.com/raghavagps/anticp2.git",
        "627105e6a1d9d3e134d9cd7af7aaec9a58fb4489",
    ),
    "hemopi2": (
        "https://github.com/raghavagps/HemoPI2.git",
        "2b67a5c85422b25ae847100ebaa81ad586950928",
    ),
}
ANTICP2_HASHES = {
    "dpc_extra_model": "b37028ba1b8b6c26b7ac332f98b149ca8698892f8a13e10cae1c397b7e803cd6",
    "aac_extra_model": "d0a1f19fc088ca787fd12eee7dd73d5b9e9de5aafd6949451087141e56e47124",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def checkout(url: str, revision: str, destination: Path) -> None:
    subprocess.run(["git", "clone", "--no-checkout", url, str(destination)], check=True)
    subprocess.run(["git", "-C", str(destination), "checkout", "--detach", revision], check=True)
    observed = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != revision:
        raise RuntimeError(f"revision mismatch: {observed} != {revision}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root.resolve()
    if root.exists():
        raise FileExistsError(f"refusing to overwrite predictor assets: {root}")
    with tempfile.TemporaryDirectory(prefix="peptide-predictors-") as temporary:
        temporary = Path(temporary)
        checkout(*SOURCES["anticp2"], temporary / "anticp2")
        checkout(*SOURCES["hemopi2"], temporary / "hemopi2")
        root.mkdir(parents=True)
        (root / "anticp2").mkdir()
        for name, expected in ANTICP2_HASHES.items():
            source = temporary / "anticp2" / name
            if digest(source) != expected:
                raise RuntimeError(f"AntiCP2 hash mismatch for {name}")
            shutil.copy2(source, root / "anticp2" / name)
        shutil.copytree(temporary / "hemopi2" / "Model", root / "hemopi2" / "Model")
    print(f"Verified public predictor assets under {root}")


if __name__ == "__main__":
    main()
