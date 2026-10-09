#!/usr/bin/env python3
"""Load the released target-independent predictor stack and score one peptide."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from domains.peptides.predictors import (
    MogDfmPredictorAssets,
    MogDfmPredictorSuite,
    encode_esm2_sequence,
)
from adm.artifacts import immutable_json
from domains.peptides.predictor_evidence import (
    ENCODER_CHECKPOINT_TARGET,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    suite = MogDfmPredictorSuite(
        MogDfmPredictorAssets.resolve(args.asset_dir),
        ENCODER_CHECKPOINT_TARGET,
        device=device,
    ).to(device)
    tokens = encode_esm2_sequence("ACDEFGHIK", device=device)
    scores = suite.developability(tokens).as_columns()
    flattened = {name: values[0] for name, values in scores.items()}
    if any(not math.isfinite(value) for value in flattened.values()):
        raise RuntimeError("released predictor smoke produced a non-finite value")
    immutable_json(
        args.output.resolve(),
        {
            "schema_version": 1,
            "status": "passed",
            "device": str(device),
            "test_sequence": "ACDEFGHIK",
            "target_specific_affinity_evaluated": False,
            "target_independent_outputs": flattened,
        },
    )
    print(f"MOG-DFM predictor runtime smoke passed: {args.output.resolve()}")


if __name__ == "__main__":
    main()
