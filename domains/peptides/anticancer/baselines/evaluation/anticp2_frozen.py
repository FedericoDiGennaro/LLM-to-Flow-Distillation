#!/usr/bin/env python3
"""Frozen diagnostic evaluation of EXP-0106/0107 cohorts with AntiCP 2.0.

The released AntiCP 2.0 estimators were serialized with scikit-learn 0.19.1.
This loader restores their tree states under modern scikit-learn without
changing tree splits or leaf counts.  The compatibility path is checked
against the score shipped in AntiCP 2.0's example output.

This is an orthogonal-model diagnostic, not a clean independent biological
endpoint: AntiCP 2.0 and the EXP-0106 ridge proxy use the same provider data.
"""

from __future__ import annotations

import csv
import hashlib
import json
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.ensemble import _forest
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.tree import _classes, _tree


AA = "ACDEFGHIKLMNPQRSTVWY"
BOOTSTRAP_SEED = 107_200
BOOTSTRAP_REPLICATES = 10_000
PUBLISHED_THRESHOLD = 0.5


class _LegacyTreeProxy:
    def __new__(cls, *args):
        instance = super().__new__(cls)
        instance.constructor_args = args
        return instance

    def __setstate__(self, state):
        self.state = state


class _LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "sklearn.tree._tree" and name == "Tree":
            return _LegacyTreeProxy
        return super().find_class(module, name)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_legacy_forest(path: Path, feature_count: int):
    # Names used by scikit-learn 0.19.1 pickles.
    sys.modules["sklearn.ensemble.forest"] = _forest
    sys.modules["sklearn.tree.tree"] = _classes
    with path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InconsistentVersionWarning)
            model = _LegacyUnpickler(handle, encoding="latin1").load()

    for estimator in model.estimators_:
        proxy = estimator.tree_
        state = dict(proxy.state)
        old_nodes = state["nodes"]
        nodes = np.empty(old_nodes.shape, dtype=_tree.NODE_DTYPE)
        for field in old_nodes.dtype.names:
            nodes[field] = old_nodes[field]
        nodes["missing_go_to_left"] = 0
        state["nodes"] = nodes
        tree = _tree.Tree(*proxy.constructor_args)
        tree.__setstate__(state)
        estimator.tree_ = tree
        estimator.n_features_in_ = feature_count
        estimator.monotonic_cst = None
    return model


def aac_features(sequences: list[str]) -> np.ndarray:
    return np.asarray(
        [[sequence.count(residue) / len(sequence) * 100 for residue in AA]
         for sequence in sequences],
        dtype=np.float32,
    )


def dpc_features(sequences: list[str]) -> np.ndarray:
    return np.asarray(
        [[sequence.count(left + right) / (len(sequence) - 1) * 100
          for left in AA for right in AA]
         for sequence in sequences],
        dtype=np.float32,
    )


def legacy_predict_proba(model, features: np.ndarray) -> np.ndarray:
    """Reproduce sklearn 0.19 forest probability averaging from leaf counts."""
    accumulator = np.zeros((len(features), 2), dtype=np.float64)
    for estimator in model.estimators_:
        leaves = estimator.apply(features)
        values = estimator.tree_.value[leaves, 0, :].astype(np.float64)
        totals = values.sum(axis=1, keepdims=True)
        accumulator += np.divide(
            values,
            totals,
            out=np.zeros_like(values),
            where=totals != 0,
        )
    return accumulator / len(model.estimators_)
