#!/usr/bin/env python3
"""Validate the RDKit kinase structural verifier and write its runtime receipt."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import rdkit
from domains.molecules.kinase import oracle
from domains.molecules.kinase.artifacts import immutable_json, sha256_file

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True)
    args = parser.parse_args()
    assert oracle.score_smiles('Nc1ncccc1C1CCNCC1') == 1
    assert oracle.score_smiles('c1ccccc1') == 0
    path = Path(oracle.__file__).resolve()
    immutable_json(args.asset_root.resolve() / 'runtime_receipt.json', {
        'reward_model_id': oracle.MODEL_ID, 'oracle': str(path),
        'oracle_sha256': sha256_file(path), 'python': sys.executable,
        'rdkit_version': rdkit.__version__, 'visible_to_llm': False,
    })

if __name__ == '__main__':
    main()
