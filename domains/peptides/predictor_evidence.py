#!/usr/bin/env python3
"""Generate a balanced variable-length PepDFM pool and attach qualitative evidence.

Raw predictor outputs are preserved in a private audit artifact.  The LLM-facing
pool contains only within-length tertile bands, never cardinal outputs, labelled
examples, database neighbours, or anticancer predictions.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from domains.peptides.predictors import (
    MOG_DFM_PREDICTOR_SHA256,
    MogDfmPredictorAssets,
    MogDfmPredictorSuite,
)
from domains.peptides.qualitative_evidence import (
    DEVELOPABILITY_EVIDENCE_LIMITATION,
    DEVELOPABILITY_EVIDENCE_VERSION,
    within_group_tertiles,
)
from domains.peptides.model import (
    PEPDFM_CHECKPOINT_SHA256,
    PEPDFM_REVISION,
    PepDfmAssets,
)
from domains.peptides.checkpoint import PepDfmCheckpoint


# The target is required only to instantiate the released checkpoint that embeds
# the shared ESM-2 encoder.  This script calls ``developability`` and therefore
# never evaluates or exposes the target-specific affinity head.
ENCODER_CHECKPOINT_TARGET = (
    "GSHMIEPNVISVRLFKRKVGGLGFLVKERVSKPPVIISDLIRGGAAEQSGLIQAGDIILAVNDRP"
    "LVDLSYDSALEVLRGIASETHVVLILRGPEGFTTHLETTFTGDGTPKTIRVTQPLGPPTKAV"
)


def attach_joint_within_length_developability_evidence(
    rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    asset_dir: Path,
    device: str,
    predictor_batch_size: int,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Attach arm-blind qualitative evidence to freshly sampled rows.

    Bands are computed jointly over both arms within each peptide length.  The
    raw outputs are returned for a private audit artifact and must never be
    included in an LLM-facing candidate record.
    """

    if set(rows_by_arm) != {"parent", "child"}:
        raise ValueError("developability annotation requires parent and child arms")
    if predictor_batch_size <= 0:
        raise ValueError("predictor_batch_size must be positive")
    enriched = {
        arm: [dict(row) for row in rows]
        for arm, rows in rows_by_arm.items()
    }
    indexed: list[dict[str, Any]] = []
    for arm in ("parent", "child"):
        for position, row in enumerate(enriched[arm]):
            if row.get("is_valid") is not True:
                continue
            sequence = str(row.get("sequence", "")).strip().upper()
            token_ids = row.get("token_ids")
            # Persisted JSON rows contain lists, whereas freshly sampled
            # ``PepDfmSample`` dataclasses become dictionaries with tuples via
            # ``dataclasses.asdict``.  They encode the same token sequence and
            # must be accepted identically by the post-training annotator.
            if (
                not sequence
                or not isinstance(token_ids, (list, tuple))
                or not token_ids
                or any(
                    isinstance(token_id, bool) or not isinstance(token_id, int)
                    for token_id in token_ids
                )
            ):
                raise ValueError("valid fresh samples require sequence and token_ids")
            indexed.append(
                {
                    "candidate_id": f"{arm}-{position:04d}",
                    "arm": arm,
                    "position": position,
                    "sequence": sequence,
                    "length": len(sequence),
                    "token_ids": list(token_ids),
                }
            )

    predictor_assets = MogDfmPredictorAssets.resolve(asset_dir)
    predictor = MogDfmPredictorSuite(
        predictor_assets,
        ENCODER_CHECKPOINT_TARGET,
        device=torch.device(device),
    ).to(torch.device(device))
    raw_rows: list[dict[str, Any]] = []
    for length in sorted({int(row["length"]) for row in indexed}):
        group = [row for row in indexed if row["length"] == length]
        columns = {
            "non_hemolysis": [],
            "non_fouling": [],
            "solubility": [],
            "half_life_hours": [],
        }
        for start in range(0, len(group), predictor_batch_size):
            batch = group[start : start + predictor_batch_size]
            tokens = torch.tensor(
                [row["token_ids"] for row in batch],
                dtype=torch.long,
                device=torch.device(device),
            )
            observed = predictor.developability(tokens).as_columns()
            for name, values in observed.items():
                if len(values) != len(batch) or any(
                    not math.isfinite(value) for value in values
                ):
                    raise RuntimeError(f"invalid predictor output for {name}")
                columns[name].extend(values)
        for index, row in enumerate(group):
            raw_rows.append(
                {
                    key: row[key]
                    for key in ("candidate_id", "arm", "position", "sequence", "length")
                }
                | {name: columns[name][index] for name in columns}
            )

    raw_to_visible = {
        "non_hemolysis": "non_hemolysis",
        "non_fouling": "non_fouling",
        "solubility": "solubility",
        "half_life": "half_life_hours",
    }
    evidence_by_id: dict[str, dict[str, str]] = {
        str(row["candidate_id"]): {} for row in raw_rows
    }
    for length in sorted({int(row["length"]) for row in raw_rows}):
        group = [row for row in raw_rows if row["length"] == length]
        for visible_name, raw_name in raw_to_visible.items():
            observed = within_group_tertiles(group, raw_name)
            for candidate_id, band in observed.items():
                evidence_by_id[candidate_id][visible_name] = band
    for row in raw_rows:
        arm = str(row["arm"])
        position = int(row["position"])
        candidate_id = str(row["candidate_id"])
        enriched[arm][position]["length"] = int(row["length"])
        enriched[arm][position]["developability_evidence"] = evidence_by_id[
            candidate_id
        ]
    return enriched, raw_rows
