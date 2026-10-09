#!/usr/bin/env python3
"""Fetch the public TDC DRD2 model and create its isolated scoring environment."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys
import urllib.request
from domains.molecules.drd2.oracle import COMPATIBLE_SKLEARN, MODEL_ID, ORACLE_FILENAME, ORACLE_URL
from domains.molecules.drd2.artifacts import immutable_json, sha256_file

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', type=Path, required=True)
    parser.add_argument('--python', type=Path, default=Path(sys.executable), help='Python 3.10 interpreter for the isolated scoring environment')
    args = parser.parse_args()
    root = args.asset_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    oracle = root / ORACLE_FILENAME
    if not oracle.exists():
        temporary = root / (ORACLE_FILENAME + '.download')
        urllib.request.urlretrieve(ORACLE_URL, temporary)
        temporary.replace(oracle)
    environment = root / 'runtime'
    python = environment / 'bin/python'
    if not python.exists():
        subprocess.run([str(args.python.absolute()), '-m', 'venv', str(environment)], check=True)
    subprocess.run([str(python), '-m', 'pip', 'install', 'numpy<2',
                    'scikit-learn==' + COMPATIBLE_SKLEARN, 'rdkit'], check=True)
    subprocess.run([str(python), '-c',
        'import sys; from pathlib import Path; from domains.molecules.drd2.oracle import load_oracle, score_smiles; '
        'model=load_oracle(Path(sys.argv[1])); value=score_smiles("CCc1ccccc1", model); '
        'assert value is not None and 0 <= value <= 1', str(oracle)], check=True)
    immutable_json(root / 'runtime_receipt.json', {
        'reward_model_id': MODEL_ID, 'oracle': str(oracle), 'oracle_url': ORACLE_URL,
        'oracle_sha256': sha256_file(oracle), 'python': str(python),
        'sklearn_version': COMPATIBLE_SKLEARN, 'visible_to_llm': False,
    })

if __name__ == '__main__':
    main()
