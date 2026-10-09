#!/usr/bin/env python3
"""Generate the immutable 700-candidate p0 pool for the moving frontier.

The private p0 predictor values define the only qualitative evidence reference
used throughout the campaign.  Later generators are projected onto this frozen
reference; their own marginal distributions never redefine ``lower/middle/
upper_third``.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import torch

from domains.peptides.frozen_evidence import (
    apply_frozen_reference,
    fit_frozen_reference,
)
from domains.peptides.predictors import (
    MOG_DFM_PREDICTOR_SHA256,
    MogDfmPredictorAssets,
    MogDfmPredictorSuite,
)
from domains.peptides.qualitative_evidence import (
    DEVELOPABILITY_EVIDENCE_LIMITATION,
)
from domains.peptides.model import (
    PEPDFM_CHECKPOINT_SHA256,
    PEPDFM_REVISION,
    PepDfmAssets,
)
from domains.peptides.checkpoint import PepDfmCheckpoint
from domains.peptides.predictor_evidence import (
    ENCODER_CHECKPOINT_TARGET,
)






def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON object required: {path}")
    return value


def immutable_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise FileExistsError(f"immutable artifact differs: {path}")
        return
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_text(encoding="utf-8") != text:
                raise FileExistsError(f"immutable artifact differs: {path}")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(
        json.dumps(dict(row), allow_nan=False, sort_keys=True, separators=(",", ":"))
        + "\n"
        for row in rows
    )




def _predict_private_rows(
    generated: Sequence[Mapping[str, Any]],
    *,
    asset_dir: Path,
    device: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    if batch_size <= 0:
        raise ValueError("predictor batch size must be positive")
    predictor = MogDfmPredictorSuite(
        MogDfmPredictorAssets.resolve(asset_dir),
        ENCODER_CHECKPOINT_TARGET,
        device=torch.device(device),
    ).to(torch.device(device))
    result: list[dict[str, Any]] = []
    for length in sorted({int(row["length"]) for row in generated}):
        group = [row for row in generated if int(row["length"]) == length]
        columns = {
            "non_hemolysis": [],
            "non_fouling": [],
            "solubility": [],
            "half_life_hours": [],
        }
        for start in range(0, len(group), batch_size):
            batch = group[start : start + batch_size]
            tokens = torch.tensor(
                [row["token_ids"] for row in batch],
                dtype=torch.long,
                device=torch.device(device),
            )
            observed = predictor.developability(tokens).as_columns()
            if set(observed) != set(columns):
                raise RuntimeError("released predictor columns changed")
            for name, values in observed.items():
                if len(values) != len(batch) or any(
                    not math.isfinite(float(value)) for value in values
                ):
                    raise RuntimeError(f"invalid predictor output for {name}")
                columns[name].extend(float(value) for value in values)
        for index, row in enumerate(group):
            result.append(
                {
                    "candidate_id": str(row["candidate_id"]),
                    "sequence": str(row["sequence"]),
                    "length": length,
                    **{name: columns[name][index] for name in columns},
                }
            )
    return result
