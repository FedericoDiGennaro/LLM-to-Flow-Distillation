"""Olivecrona/TDC DRD2 FCFP6 probability multiplied by the MolLogP [1, 3] gate."""

from __future__ import annotations
from pathlib import Path
from typing import Any, Mapping, Sequence, Iterable

def mol_from_smiles(smiles):
    from rdkit import Chem
    return Chem.MolFromSmiles(smiles)

def canonical_smiles(mol):
    from rdkit import Chem
    return Chem.MolToSmiles(mol, isomericSmiles=True)

def score_smiles(smiles, model=None):
    mol = mol_from_smiles(smiles)
    return None if mol is None else score_mol(mol, model)

import pickle
MODEL_ID = 'tdc_olivecrona_drd2_current_fcfp6'
REWARD_ENDPOINT = 'tdc_drd2_probability_times_mollogp_1_to_3'
HELD_OUT_EVALUATOR_IDS = ('gnina_drd2_6cm4', 'vina_drd2_6cm4', 'rdock_drd2_6cm4')
FINGERPRINT_RADIUS = 3
FINGERPRINT_SIZE = 2048
ORACLE_FILENAME = 'drd2_current.pkl'
ORACLE_DATAVERSE_ID = '6413411'
ORACLE_URL = f'https://dataverse.harvard.edu/api/access/datafile/{ORACLE_DATAVERSE_ID}'
COMPATIBLE_SKLEARN = '1.3.2'
LOGP_LOW, LOGP_HIGH = 1.0, 3.0
PREFERENCE_RULE = 'higher_combined_drd2_mollogp_reward_is_chosen'
EXTREME_LABEL_RULE = 'median_split_of_scored_drd2_logp_reward'

def load_oracle(path):
    with Path(path).open('rb') as handle:
        model = pickle.load(handle)
    if not hasattr(model, 'predict_proba'):
        raise TypeError('DRD2 model must expose predict_proba')
    return model

def score_mol(mol, model):
    return combined_reward(float(model.predict_proba(fingerprints_from_mol(mol))[0, 1]), mollogp(mol))

def induce_extreme_labels(item_ids, predictions, *, positive_count=2, negative_count=2):
    scores = {str(k): float(v['raw_reward']) for k, v in predictions.items() if str(k) in item_ids}
    if len(set(scores.values())) < 2:
        return {str(item): 'abstain' for item in item_ids}
    return induce_preference_labels(item_ids, scores)


def logp_range_modifier(logp: float, *, low: float = LOGP_LOW, high: float = LOGP_HIGH) -> float:
    """Return 1 inside the registered MolLogP window and 0 outside it."""

    if not (low <= high):
        raise ValueError("MolLogP window must have low <= high")
    return 1.0 if low <= float(logp) <= high else 0.0

def combined_reward(drd2_probability: float, logp: float) -> float:
    """Maximize predicted DRD2 activity only inside the registered LogP window."""

    probability = float(drd2_probability)
    if not 0.0 <= probability <= 1.0:
        raise ValueError("DRD2 probability must lie in [0, 1]")
    return probability * logp_range_modifier(logp)

def fingerprints_from_mol(mol: Any):
    """TDC DRD2 FCFP6 counts folded into 2048 bins (oracle.fingerprints_from_mol)."""

    import numpy as np
    from rdkit.Chem import AllChem

    fingerprint = AllChem.GetMorganFingerprint(
        mol, FINGERPRINT_RADIUS, useCounts=True, useFeatures=True
    )
    folded = np.zeros((1, FINGERPRINT_SIZE), np.int32)
    for index, value in fingerprint.GetNonzeroElements().items():
        folded[0, int(index) % FINGERPRINT_SIZE] += int(value)
    return folded

def mollogp(mol: Any) -> float:
    from rdkit.Chem import Crippen

    return float(Crippen.MolLogP(mol))

def induce_preference_labels(
    item_ids: Sequence[str],
    scores: Mapping[str, float],
) -> dict[str, str]:
    """Prefer the higher combined DRD2×LogP reward via a balanced median split.

    Scored candidates below the median are DPO negatives; those above are
    positives. An odd-sized scored set leaves the median item as abstain.
    Unscored items stay abstain and never enter Cartesian pairs.
    """

    labels = {str(item): "abstain" for item in item_ids}
    ranked = sorted(
        ((str(item), float(scores[item])) for item in item_ids if item in scores),
        key=lambda item: (item[1], item[0]),
    )
    if len(ranked) < 2:
        raise ValueError("DRD2-LogP DPO needs at least two scored candidates")
    if len({score for _item, score in ranked}) < 2:
        raise ValueError("DRD2-LogP DPO needs variation in the combined reward")
    midpoint = len(ranked) // 2
    odd = len(ranked) % 2 == 1
    for index, (item, _score) in enumerate(ranked):
        if index < midpoint:
            labels[item] = "negative"
        elif odd and index == midpoint:
            labels[item] = "abstain"
        else:
            labels[item] = "positive"
    if "positive" not in labels.values() or "negative" not in labels.values():
        raise ValueError(
            "DRD2-LogP median split did not produce both preference classes"
        )
    return labels
