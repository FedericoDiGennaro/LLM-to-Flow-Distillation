"""Frozen TDC GSK3B training-reward adapter.

The TDC asset is a scikit-learn 0.23 random forest over radius-2, 2048-bit
Morgan fingerprints.  It is a cheap *baseline training reward*, not an
experimental oracle, calibrated affinity, docking score, or held-out test.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Mapping, Sequence

import numpy as np


REWARD_ENDPOINT = "tdc_gsk3b_random_forest_probability"
MODEL_ID = "tdc_gsk3b_current_rf_morgan2_2048"
HELD_OUT_EVALUATOR_IDS = (
    "gnina_gsk3b_1q5k",
    "admet_ai_solubility_aqsol_db",
)
FINGERPRINT_RADIUS = 2
FINGERPRINT_SIZE = 2048
ORACLE_FILENAME = "gsk3b_current.pkl"
ORACLE_DATAVERSE_ID = "6413412"
ORACLE_URL = f"https://dataverse.harvard.edu/api/access/datafile/{ORACLE_DATAVERSE_ID}"
COMPATIBLE_SKLEARN = "1.2.2"
REFERENCE_SMILES = "CC(C)(C)[C@H]1CCc2c(sc(NC(=O)COc3ccc(Cl)cc3)c2C(N)=O)C1"
REFERENCE_SCORE = 0.03
PREFERENCE_RULE = "higher_tdc_gsk3b_probability_is_chosen"
EXTREME_LABEL_RULE = "top_20_and_bottom_20_distinct_canonical_smiles_hash_tiebreak"


def mol_from_smiles(smiles: str):
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    return mol


def canonical_smiles(mol: Any) -> str:
    from rdkit import Chem

    return str(Chem.MolToSmiles(mol, isomericSmiles=True))


def fingerprint_from_mol(mol: Any) -> np.ndarray:
    """Return the exact bit-vector representation used by TDC's GSK3B class."""

    from rdkit import DataStructs
    from rdkit.Chem import AllChem

    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol, FINGERPRINT_RADIUS, nBits=FINGERPRINT_SIZE
    )
    values = np.zeros((FINGERPRINT_SIZE,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return values.reshape(1, -1)


def load_oracle(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"TDC GSK3B pickle is missing: {path}")
    with path.open("rb") as handle:
        model = pickle.load(handle)
    if not hasattr(model, "predict_proba"):
        raise TypeError("TDC GSK3B asset must expose predict_proba")
    observed = score_smiles(REFERENCE_SMILES, model)
    if observed is None or abs(observed - REFERENCE_SCORE) > 1e-12:
        raise RuntimeError(
            "TDC GSK3B reference probe differs: "
            f"observed={observed}, expected={REFERENCE_SCORE}"
        )
    return model


def score_mol(mol: Any, model: Any) -> float:
    return float(model.predict_proba(fingerprint_from_mol(mol))[0, 1])


def score_smiles(smiles: str, model: Any) -> float | None:
    mol = mol_from_smiles(smiles)
    return None if mol is None else score_mol(mol, model)


def induce_extreme_labels(
    item_ids: Sequence[str],
    predictions: Mapping[str, Mapping[str, Any]],
    *,
    positive_count: int = 20,
    negative_count: int = 20,
) -> dict[str, str]:
    """Label score extremes while admitting at most one copy of each molecule.

    The RF is quantized in 0.01 increments, so ties are common.  The immutable
    candidate content hash is an outcome-independent deterministic tiebreak.
    This rule is frozen prospectively rather than changed after seeing scores.
    """

    if positive_count < 1 or negative_count < 1:
        raise ValueError("top and bottom counts must be positive")
    labels = {str(item): "abstain" for item in item_ids}
    unique: dict[str, tuple[str, float, str]] = {}
    for item_id in item_ids:
        row = predictions.get(str(item_id))
        if not row:
            continue
        smiles = str(row["canonical_smiles"])
        candidate = (
            str(item_id),
            float(row["predicted_gsk3b_probability"]),
            str(row["content_sha256"]),
        )
        previous = unique.get(smiles)
        if previous is None or candidate[2] < previous[2]:
            unique[smiles] = candidate
    ranked = sorted(unique.values(), key=lambda row: (row[1], row[2], row[0]))
    required = positive_count + negative_count
    if len(ranked) < required:
        raise ValueError(
            f"GSK3B DPO needs {required} distinct scored molecules, got {len(ranked)}"
        )
    negative = ranked[:negative_count]
    positive = ranked[-positive_count:]
    if negative[-1][1] > positive[0][1]:
        raise AssertionError("extreme classes overlap in score ordering")
    for item_id, _score, _digest in negative:
        labels[item_id] = "negative"
    for item_id, _score, _digest in positive:
        labels[item_id] = "positive"
    return labels


__all__ = [
    "COMPATIBLE_SKLEARN",
    "EXTREME_LABEL_RULE",
    "FINGERPRINT_RADIUS",
    "FINGERPRINT_SIZE",
    "HELD_OUT_EVALUATOR_IDS",
    "MODEL_ID",
    "ORACLE_DATAVERSE_ID",
    "ORACLE_FILENAME",
    "ORACLE_URL",
    "PREFERENCE_RULE",
    "REFERENCE_SCORE",
    "REFERENCE_SMILES",
    "REWARD_ENDPOINT",
    "canonical_smiles",
    "fingerprint_from_mol",
    "induce_extreme_labels",
    "load_oracle",
    "mol_from_smiles",
    "score_mol",
    "score_smiles",
]
