#!/usr/bin/env python3
"""Score immutable FlowMol3 candidates with the frozen TDC GSK3B reward."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping

from domains.molecules.gsk3b.oracle import (
    EXTREME_LABEL_RULE,
    HELD_OUT_EVALUATOR_IDS,
    MODEL_ID,
    PREFERENCE_RULE,
    REWARD_ENDPOINT,
    canonical_smiles,
    induce_extreme_labels,
    load_oracle,
    mol_from_smiles,
    score_mol,
)
from domains.molecules.gsk3b.artifacts import immutable_json, read_json, read_jsonl, sha256_file


LIMITATIONS = (
    "TDC GSK3B is a random-forest probability over radius-2 Morgan bits, "
    "trained from historical ligand labels. It is a baseline optimization "
    "reward only: it is not calibrated affinity, ATP-pocket pose evidence, "
    "aqueous solubility, selectivity, safety, or experimental ground truth."
)


def _without_external_keys() -> None:
    os.environ.pop("GEMINI_API_KEY", None)
    os.environ.pop("GOOGLE_API_KEY", None)
    for name in ("LITELLM_API_KEY", "OPENAI_API_KEY", "API_KEY_FILE"):
        os.environ.pop(name, None)


def smiles_from_pool_row(row: Mapping[str, Any]) -> tuple[str, str] | None:
    evidence = row.get("evidence")
    if not evidence:
        return None
    record = read_json(Path(str(evidence)))
    identity = record.get("identity") or {}
    smiles = identity.get("isomeric_smiles")
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    digest = str(record.get("content_sha256") or row.get("content_sha256") or "")
    if not digest:
        digest = sha256_file(Path(str(evidence)))
    return smiles, digest


def smiles_from_sdf(path: Path) -> tuple[str, str] | None:
    from rdkit import Chem

    if not path.is_file():
        return None
    supplier = Chem.SDMolSupplier(str(path), removeHs=False, sanitize=False)
    mol = next(iter(supplier), None) if supplier is not None else None
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return Chem.MolToSmiles(mol, isomericSmiles=True), sha256_file(path)


def score_candidate(smiles: str, digest: str, model: Any) -> dict[str, Any] | None:
    mol = mol_from_smiles(smiles)
    if mol is None:
        return None
    probability = score_mol(mol, model)
    return {
        "raw_reward": probability,
        "predicted_gsk3b_probability": probability,
        "canonical_smiles": canonical_smiles(mol),
        "content_sha256": digest,
    }


def shared(oracle: Path, reward_floor: float) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "reward_model_id": MODEL_ID,
        "reward_endpoint": REWARD_ENDPOINT,
        "held_out_evaluator_ids": list(HELD_OUT_EVALUATOR_IDS),
        "oracle_path": str(oracle),
        "oracle_sha256": sha256_file(oracle),
        "reward_floor": reward_floor,
        "evidence_role": "baseline_training_reward_only",
        "visible_to_llm": False,
        "usable_as_held_out_evaluator": False,
        "usable_for_baseline_training_or_selection": True,
        "limitations": LIMITATIONS,
    }


def score_pool(
    pool: Path,
    oracle: Path,
    output: Path,
    reward_floor: float,
    positive_count: int,
    negative_count: int,
    induce_preferences: bool,
) -> dict[str, Any]:
    rows = read_jsonl(pool)
    item_ids = [str(row["item_id"]) for row in rows]
    judgeable_item_ids = [
        str(row["item_id"]) for row in rows if row.get("status") == "judgeable"
    ]
    if len(item_ids) != len(set(item_ids)):
        raise ValueError("candidate pool contains duplicate item ids")
    model = load_oracle(oracle)
    predictions: dict[str, dict[str, Any]] = {}
    for row in rows:
        extracted = smiles_from_pool_row(row)
        if extracted is None:
            continue
        scored = score_candidate(*extracted, model)
        if scored is not None:
            predictions[str(row["item_id"])] = scored
    labels = (
        induce_extreme_labels(
            judgeable_item_ids,
            predictions,
            positive_count=positive_count,
            negative_count=negative_count,
        )
        if induce_preferences else {}
    )
    samples = []
    for item_id in item_ids:
        prediction = predictions.get(item_id)
        if prediction is None:
            samples.append({
                "item_id": item_id,
                "raw_reward": reward_floor,
                "scored_by_model": False,
                "floored_reason": "candidate could not be converted to a valid molecule",
            })
        else:
            samples.append({
                "item_id": item_id,
                **prediction,
                "raw_reward": max(reward_floor, float(prediction["raw_reward"])),
                "scored_by_model": True,
            })
    counts = {
        key: sum(label == key for label in labels.values())
        for key in ("positive", "negative", "abstain", "unstable")
    }
    result = {
        **shared(oracle, reward_floor),
        "artifact_type": (
            "gsk3b_extreme_dpo_preference_labels"
            if induce_preferences else "gsk3b_scalar_pool_reward"
        ),
        "pool": str(pool),
        "label_rule": EXTREME_LABEL_RULE,
        "preference_rule": PREFERENCE_RULE,
        "sample_count": len(samples),
        "scored_count": len(predictions),
        "floored_count": len(samples) - len(predictions),
        "counts": counts,
        "stable_labels": labels,
        "samples": samples,
    }
    immutable_json(output / "gsk3b_reward.json", result)
    if induce_preferences:
        immutable_json(output / "stable_summary.json", result)
    return result


def score_rollout(
    rollout_path: Path,
    oracle: Path,
    output: Path,
    reward_floor: float,
) -> dict[str, Any]:
    rollout = read_json(rollout_path)
    raw_samples = rollout.get("samples")
    if not isinstance(raw_samples, list) or not raw_samples:
        raise ValueError("rollout must contain samples")
    model = load_oracle(oracle)
    samples = []
    seen: set[str] = set()
    for row in raw_samples:
        candidate_id = str(row["candidate_id"])
        if candidate_id in seen:
            raise ValueError(f"duplicate rollout candidate: {candidate_id}")
        seen.add(candidate_id)
        extracted = None
        if row.get("smiles"):
            extracted = (str(row["smiles"]), str(row.get("content_sha256") or ""))
        elif row.get("sdf"):
            sdf_path = Path(str(row["sdf"]))
            if sha256_file(sdf_path) != row.get("sdf_sha256"):
                raise ValueError(f"rollout SDF binding changed: {candidate_id}")
            extracted = smiles_from_sdf(sdf_path)
        scored = None if extracted is None else score_candidate(*extracted, model)
        base = {"candidate_id": candidate_id, "group_id": str(row["group_id"])}
        if scored is None:
            samples.append({
                **base,
                "raw_reward": reward_floor,
                "scored_by_model": False,
                "floored_reason": "candidate could not be converted to a valid molecule",
            })
        else:
            samples.append({
                **base,
                **scored,
                "raw_reward": max(reward_floor, float(scored["raw_reward"])),
                "scored_by_model": True,
            })
    result = {
        **shared(oracle, reward_floor),
        "artifact_type": "gsk3b_scalar_reward",
        "rollout": str(rollout_path),
        "rollout_sha256": sha256_file(rollout_path),
        "sample_count": len(samples),
        "scored_count": sum(bool(row["scored_by_model"]) for row in samples),
        "floored_count": sum(not bool(row["scored_by_model"]) for row in samples),
        "group_count": len({row["group_id"] for row in samples}),
        "failed_candidates_scored": True,
        "samples": samples,
    }
    immutable_json(output / "gsk3b_reward.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--pool", type=Path)
    source.add_argument("--rollout", type=Path)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reward-floor", type=float, default=0.0)
    parser.add_argument("--positive-count", type=int, default=20)
    parser.add_argument("--negative-count", type=int, default=20)
    parser.add_argument("--no-preference-labels", action="store_true")
    args = parser.parse_args()
    _without_external_keys()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    oracle = args.oracle.resolve()
    if args.pool:
        result = score_pool(
            args.pool.resolve(), oracle, output, args.reward_floor,
            args.positive_count, args.negative_count,
            not args.no_preference_labels,
        )
        summary = {"counts": result["counts"]}
    else:
        result = score_rollout(args.rollout.resolve(), oracle, output, args.reward_floor)
        summary = {"group_count": result["group_count"]}
    print(json.dumps({
        "reward_endpoint": REWARD_ENDPOINT,
        "sample_count": result["sample_count"],
        "scored_count": result["scored_count"],
        "floored_count": result["floored_count"],
        **summary,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
