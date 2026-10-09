#!/usr/bin/env python3
"""Install the pinned FlowMol3 runtime, checkpoint, and official load files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import urllib.request

from domains.molecules.kinase.artifacts import immutable_json


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG = ROOT / "domains/molecules/kinase/configs/flowmol_generator_smoke.json"
DEFAULT_MANIFEST = ROOT / "domains/molecules/pixi.toml"
ASSET_MANIFEST = ROOT / "domains/molecules/kinase/configs/flowmol3_assets.json"
FLOWMOL_REPOSITORY = "https://github.com/Dunni3/FlowMol.git"
FLOWMOL_REVISION = "40b42309da7247013a6254bbf567f3a8ef8d97c7"
MODEL_BASE_URL = (
    "https://bits.csb.pitt.edu/files/FlowMol/trained_models_v3.1/flowmol3"
)

OFFICIAL_GEOM_SUPPORT_FILES = (
    "train_data_n_atoms_histogram.pt",
    "train_data_marginal_dists.pt",
    "train_data_valencies_kekulized.json",
)


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file() and destination.stat().st_size > 0:
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with urllib.request.urlopen(url, timeout=300) as response, temporary.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def configure_source(source: Path, revision: str) -> None:
    if not (source / ".git").is_dir():
        source.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", FLOWMOL_REPOSITORY, str(source)],
            check=True,
        )
    status = subprocess.check_output(
        ["git", "-C", str(source), "status", "--porcelain"], text=True
    ).splitlines()
    if status:
        raise RuntimeError(f"unexpected changes in FlowMol source: {status}")
    head = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if head != revision:
        subprocess.run(
            ["git", "-C", str(source), "checkout", "--detach", revision],
            check=True,
        )
    observed = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != revision:
        raise RuntimeError(f"FlowMol revision mismatch: {observed}")


def official_inference_support(source: Path) -> dict[str, str]:
    """Use the GEOM-Drugs statistics committed by the pinned FlowMol release."""

    data_dir = source / "data/geom_full_kekulized"
    missing = [name for name in OFFICIAL_GEOM_SUPPORT_FILES if not (data_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            "pinned FlowMol source is missing official GEOM support files: "
            + ", ".join(missing)
        )
    histogram = data_dir / OFFICIAL_GEOM_SUPPORT_FILES[0]
    marginal = data_dir / OFFICIAL_GEOM_SUPPORT_FILES[1]
    valencies = data_dir / OFFICIAL_GEOM_SUPPORT_FILES[2]
    return {
        "n_atoms_hist_file": str(histogram.resolve()),
        "n_atoms_hist_sha256": file_digest(histogram, "sha256"),
        "marginal_dists_file": str(marginal.resolve()),
        "marginal_dists_sha256": file_digest(marginal, "sha256"),
        "valency_file": str(valencies.resolve()),
        "valency_sha256": file_digest(valencies, "sha256"),
        "n_atoms_histogram_source": "official_pinned_flowmol3_geom_full_kekulized",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--ltfd-home",
        type=Path,
        default=Path(".cache/flowmol3"),
    )
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    asset_manifest = json.loads(ASSET_MANIFEST.read_text(encoding="utf-8"))
    revision = str(config["generator"]["revision"])
    if revision != FLOWMOL_REVISION:
        raise ValueError(f"FlowMol revision is not the frozen pin: {revision}")
    asset_root = args.ltfd_home.resolve() / "assets/flowmol3"
    source = asset_root / "source"
    release = asset_root / "release"
    model_dir = release / config["release"]["model_relative_path"]
    checkpoint = release / config["release"]["checkpoint_relative_path"]

    pixi = shutil.which("pixi")
    if pixi is None:
        raise FileNotFoundError("pixi is not on PATH")
    pixi_environment = dict(os.environ)
    pixi_cache = args.ltfd_home.resolve() / "cache/pixi"
    pixi_cache.mkdir(parents=True, exist_ok=True)
    pixi_environment.setdefault("PIXI_CACHE_DIR", str(pixi_cache))
    install = [pixi, "install", "--manifest-path", str(args.manifest.resolve())]
    if (args.manifest.resolve().parent / "pixi.lock").is_file():
        install.insert(2, "--locked")
    subprocess.run(install, cwd=ROOT, env=pixi_environment, check=True)

    configure_source(source, revision)
    download(f"{MODEL_BASE_URL}/config.yaml", model_dir / "config.yaml")
    download(f"{MODEL_BASE_URL}/checkpoints/last.ckpt", checkpoint)
    if not checkpoint.is_file() or checkpoint.stat().st_size < 1_000_000:
        raise FileNotFoundError(f"FlowMol3 checkpoint is missing: {checkpoint}")
    expected_downloads = {
        model_dir / "config.yaml": asset_manifest["config_sha256"],
        checkpoint: asset_manifest["checkpoint_sha256"],
    }
    for path, expected_sha256 in expected_downloads.items():
        observed_sha256 = file_digest(path, "sha256")
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"downloaded FlowMol3 asset hash mismatch: {path}: {observed_sha256}"
            )
    runtime_python = args.manifest.resolve().parent / ".pixi/envs/default/bin/python"
    support = official_inference_support(source)
    support_hashes = asset_manifest["official_support_files"]
    for relative, expected_sha256 in support_hashes.items():
        path = source / relative
        observed_sha256 = file_digest(path, "sha256")
        if observed_sha256 != expected_sha256:
            raise ValueError(
                f"pinned FlowMol3 support hash mismatch: {path}: {observed_sha256}"
            )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = f"{ROOT}:{source}"
    environment["DGLBACKEND"] = "pytorch"
    subprocess.run(
        [
            str(runtime_python),
            "-c",
            (
                "import dgl, pytorch_lightning, rdkit, torch; "
                "print(torch.__version__, pytorch_lightning.__version__, "
                "rdkit.__version__)"
            ),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
    )

    receipt = {
        "schema_version": 1,
        "artifact_type": "flowmol_euler_setup_receipt",
        "source": str(source.resolve()),
        "source_revision": revision,
        "runtime_python": str(runtime_python.resolve()),
        "runtime_manifest": str(args.manifest.resolve()),
        "runtime_manifest_sha256": file_digest(args.manifest.resolve(), "sha256"),
        "release_root": str(release.resolve()),
        "model_dir": str(model_dir.resolve()),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_digest(checkpoint, "sha256"),
        "config_yaml": str((model_dir / "config.yaml").resolve()),
        "config_yaml_sha256": file_digest(model_dir / "config.yaml", "sha256"),
        **support,
        "generator_gpu_tested": False,
    }
    receipt_path = asset_root / "setup_receipt.json"
    immutable_json(receipt_path, receipt)
    print(f"FlowMol setup ready: {receipt_path}")


if __name__ == "__main__":
    main()
