"""Pinned HemoPI2 RF adapter. No fitting, sequence design, ESM, or network calls.

Running with --allow-model-load explicitly authorizes loading the pinned official
pickle assets. Without that flag this program refuses feature/model execution.
Upstream: raghavagps/HemoPI2 commit 2b67a5c85422b25ae847100ebaa81ad586950928.
GPL-3.0; preserve upstream LICENSE.txt when redistributing this adapter/assets.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

SOURCE_URL = "https://webs.iiitd.edu.in/raghava/hemopi2/download/Model.zip"
ARCHIVE_SHA256 = "743e1967b2b646075cef17bbf2dbfba3e3ee9ad8ad30464f4369de2c946f483a"
REVISION = "2b67a5c85422b25ae847100ebaa81ad586950928"
AA = frozenset("ACDEFGHIKLMNPQRSTVWY")
HASHES = {
    "composition_calculate.py": "c75b4b079264cbf2ec89d95e3a0fe1633a8611be84d7ea373d02c2a148902562",
    "composition_calculate_hemopi2_2.py": "8f031f0697f3699a0ae2386163d7abcd36d2e133b0e6e160fe8d512e03ec8be3",
    "hemopi2_ml_clf.sav": "63973510af1b883c505e0c475297b4f9edf07b5c7bcd91546ffcb4fdec62dac5",
    "HemoPI2_reg.sav": "72b8afc63ed7803955aae970ab93627bb710fab17db027bf6110a0237abb6955",
    "Data/Grantham.csv": "87baa46a74c6510d466011358973d340d943844aeb8ce5fb46541db97aaa883c",
    "Data/PhysicoChemical.csv": "f8e94cf9464539ddd0e9e97a20da2fe0ffbdcb8a48c1a632d7117153bea11c04",
    "Data/Schneider-Wrede.csv": "5ae5c4e83e41490464ac0fc120e209de2e87a2b37b8ed3350b028b7760b89a56",
    "Data/aa_attr_group.csv": "8cfa34e00a569810ab9af6c2562e27bb3dc2064eac0f24ca2920faa7ea90b4d9",
    "Data/atom.csv": "16de8b333ff6dbc7539ea9d1d77a114ff6db09a6c53ddfb9ce7153aa08abaaa5",
    "Data/bonds.csv": "9ded52e577d0f44d2a1d9cc5d932c89d312a973cf47aa538743b3b44c770f915",
    "Data/data": "ee61f3c40c87ccacd1b2070faad781ad0ce2e923a52b74d2a9836e2a5884061d",
}


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def write_new(path, value):
    with Path(path).open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def parse_fasta(text):
    rows, header, pieces = [], None, []
    def append():
        if header is not None:
            sequence = "".join(pieces)
            if not sequence:
                status = "invalid_empty_sequence"
            elif not set(sequence) <= AA:
                status = "invalid_noncanonical_sequence"
            elif len(sequence) > 40:
                status = "unsupported_length_over_40_no_truncation"
            elif len(sequence) < 8:
                status = "unsupported_length_under_8"
            else:
                status = "eligible"
            rows.append({"row_id": header, "sequence": sequence,
                         "length": len(sequence), "status": status})
    for line in text.splitlines():
        if line.startswith(">"):
            append()
            header = line[1:].strip().split()[0] if line[1:].strip() else ""
            if not header:
                raise ValueError("FASTA record has no row ID")
            pieces = []
        elif line.strip():
            if header is None:
                raise ValueError("Input must be FASTA with explicit row IDs")
            pieces.append(line.strip())
    append()
    if not rows:
        raise ValueError("FASTA has no records")
    if len({r["row_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate row IDs are ambiguous; sequence duplicates are allowed")
    return rows


def classifier_postprocess(probabilities, threshold):
    import numpy as np
    raw = np.asarray(probabilities, dtype=float)
    if not np.isfinite(raw).all() or ((raw < 0) | (raw > 1)).any():
        raise ValueError("Invalid classifier probabilities")
    published = raw.round(3)
    return [{"hemolysis_probability_raw": float(a), "published_ml_score": float(b),
             "published_prediction": "Hemolytic" if b >= threshold else "Non-Hemolytic"}
            for a, b in zip(raw, published)]


def regression_postprocess(predictions):
    import numpy as np
    latent = np.asarray(predictions, dtype=float)
    transformed = np.exp(-latent)
    if not np.isfinite(latent).all() or not np.isfinite(transformed).all():
        raise ValueError("Nonfinite regression output")
    return [{"regression_latent_raw": float(a), "hc50_transform_raw": float(b),
             "published_HC50": float(c)}
            for a, b, c in zip(latent, transformed, transformed.round(3))]


def verify_assets(assets):
    if assets.name != "Model":
        raise ValueError("Asset directory must be named Model (upstream data paths are relative)")
    for relative, expected in HASHES.items():
        if sha256_bytes((assets / relative).read_bytes()) != expected:
            raise ValueError(f"Asset hash mismatch: {relative}")


def run(args):
    if not args.allow_model_load:
        raise PermissionError("Parent review and --allow-model-load required before executing extractors/loading models")
    import numpy as np
    import pandas as pd
    import pickle
    import sklearn
    if sklearn.__version__ != "1.3.1":
        raise RuntimeError(f"Exact sklearn 1.3.1 required; found {sklearn.__version__}")
    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be within [0,1]")
    assets, output = args.assets.resolve(strict=True), args.output.resolve()
    verify_assets(assets)
    source_bytes = args.input_fasta.read_bytes()
    rows = parse_fasta(source_bytes.decode("utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    (output / "input.fasta").write_bytes(source_bytes)
    write_new(output / "sequence_status.json", rows)
    eligible = [row for row in rows if row["status"] == "eligible"]
    manifest = {
        "source_url": SOURCE_URL, "source_archive_sha256": ARCHIVE_SHA256,
        "upstream_wrapper_revision": REVISION, "asset_sha256": HASHES,
        "input_sha256": sha256_bytes(source_bytes), "input_count": len(rows),
        "eligible_count": len(eligible), "skipped_count": len(rows) - len(eligible),
        "mode": args.mode, "threshold": args.threshold,
        "threshold_source": "Pinned hemopi2_classification.py lines 460-467: RF default 0.45; score rounded to 3 decimals before >= threshold. README differs.",
        "regression_transform": "Pinned hemopi2_regression.py ML_run: exp(-model.predict(X)); then round to 3 decimals. No invented assay validation.",
        "packages": {name: importlib.metadata.version(name) for name in
                     ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl", "tqdm")},
        "python": sys.version, "python_executable": sys.executable,
        "adapter_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "scientific_claim": "Released predictor inference only; no biological validation; no fitting or candidate selection.",
        "commands": [], "complete": False,
    }
    write_new(output / "started.json", manifest)
    if not eligible:
        raise ValueError("No eligible rows; statuses saved without silent replacement")
    results = {row["row_id"]: dict(row) for row in eligible}
    specs = [
        ("classification", "composition_calculate.py", "hemopi2_ml_clf.sav", 1190),
        ("regression", "composition_calculate_hemopi2_2.py", "HemoPI2_reg.sav", 1167),
    ]
    for mode, script, model_file, n_features in specs:
        if args.mode not in ("both", mode):
            continue
        work = output / mode
        work.mkdir()
        local_input, features_path = work / "input.seq", work / "features.csv"
        local_input.write_text("\n".join(row["sequence"] for row in eligible) + "\n")
        command = [sys.executable, str(assets / script), str(local_input), str(work), str(features_path)]
        manifest["commands"].append({"argv": command, "cwd": str(assets.parent)})
        with (work / "extractor.stdout.log").open("x") as stdout, (work / "extractor.stderr.log").open("x") as stderr:
            subprocess.run(command, cwd=assets.parent, stdout=stdout, stderr=stderr, check=True)
        frame = pd.read_csv(features_path)
        if frame.shape != (len(eligible), n_features) or not np.isfinite(frame.to_numpy(dtype=float)).all():
            raise ValueError(f"Wrong/nonfinite {mode} features: {frame.shape}")
        # Hash exactly the bytes deserialized; no legacy conversion or fallback.
        model_bytes = (assets / model_file).read_bytes()
        if sha256_bytes(model_bytes) != HASHES[model_file]:
            raise ValueError("Model changed after audit")
        model = pickle.loads(model_bytes)
        if list(frame.columns) != list(model.feature_names_in_):
            raise ValueError(f"{mode} feature names/order differ from trained model")
        if mode == "classification":
            if list(model.classes_) != [0, 1]:
                raise ValueError("Unexpected classifier label order")
            values = classifier_postprocess(model.predict_proba(frame)[:, 1], args.threshold)
        else:
            values = regression_postprocess(model.predict(frame))
        for row, value in zip(eligible, values):
            results[row["row_id"]].update(value)
        manifest[f"{mode}_features_sha256"] = sha256_bytes(features_path.read_bytes())
        manifest[f"{mode}_feature_columns"] = list(frame.columns)
    final = [results.get(row["row_id"], row) for row in rows]
    write_new(output / "predictions.json", final)
    manifest.update(complete=True, predictions_sha256=sha256_bytes((output / "predictions.json").read_bytes()))
    write_new(output / "completion.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fasta", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New, nonexistent output directory")
    parser.add_argument("--mode", choices=("classification", "regression", "both"), default="both")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--allow-model-load", action="store_true")
    result = run(parser.parse_args())
    print(json.dumps({"complete": result["complete"], "eligible_count": result["eligible_count"]}))


if __name__ == "__main__":
    main()
