#!/usr/bin/env python3
"""Validate and receipt an isolated runtime for the frozen TDC GSK3B model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sklearn

from domains.molecules.gsk3b.oracle import (
    COMPATIBLE_SKLEARN,
    MODEL_ID,
    ORACLE_FILENAME,
    ORACLE_URL,
    REFERENCE_SCORE,
    REFERENCE_SMILES,
    load_oracle,
    score_smiles,
)
from domains.molecules.gsk3b.artifacts import immutable_json, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    args = parser.parse_args()
    root = args.asset_root.resolve()
    oracle = root / ORACLE_FILENAME
    overlay = args.overlay.resolve()
    if sklearn.__version__ != COMPATIBLE_SKLEARN:
        raise RuntimeError(
            f"GSK3B runtime needs sklearn {COMPATIBLE_SKLEARN}, got {sklearn.__version__}"
        )
    if not str(Path(sklearn.__file__).resolve()).startswith(str(overlay)):
        raise RuntimeError("scikit-learn did not load from the isolated GSK3B overlay")
    model = load_oracle(oracle)
    score = score_smiles(REFERENCE_SMILES, model)
    if score != REFERENCE_SCORE:
        raise RuntimeError("official TDC GSK3B reference probe failed")
    receipt = {
        "schema_version": 1,
        "artifact_type": "tdc_gsk3b_runtime",
        "reward_model_id": MODEL_ID,
        "oracle": str(oracle),
        "oracle_url": ORACLE_URL,
        "oracle_sha256": sha256_file(oracle),
        "python": str(args.python.resolve()),
        "overlay": str(overlay),
        "sklearn_version": sklearn.__version__,
        "model_type": type(model).__name__,
        "reference_smiles": REFERENCE_SMILES,
        "reference_score": score,
        "reference_probe_passed": True,
        "visible_to_llm": False,
        "usable_as_held_out_evaluator": False,
    }
    immutable_json(root / "runtime_receipt.json", receipt)
    print(json.dumps(receipt, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
