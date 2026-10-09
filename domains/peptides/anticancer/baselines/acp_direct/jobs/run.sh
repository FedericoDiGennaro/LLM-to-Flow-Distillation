#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gpus=4
#SBATCH --time=06:00:00
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
BASELINE_ROOT="$ROOT/domains/peptides/anticancer/baselines/acp_direct"
OUTPUT="${1:?output root required}"
export PEPDFM_ASSET_DIR="${2:?PepDFM asset directory required}"
PEPTIDE_PYTHON="${3:?PepDFM Python required}"
export HEMOPI2_ASSETS="${4:?released HemoPI2 Model directory required}"
export HEMOPI2_OVERLAY="${5:?scikit-learn 1.3.1 overlay required}"
TORCHDIFFEQ_OVERLAY="${6:?torchdiffeq overlay required}"
export ANTICP2_ASSETS="${7:?released AntiCP2 asset directory required}"
cd "$BASELINE_ROOT"
unset GEMINI_API_KEY GOOGLE_API_KEY
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
GENERATION="$OUTPUT/generation"
VALIDATION="$OUTPUT/validation"
mkdir -p "$GENERATION"
export PYTHONPATH="$ROOT:$TORCHDIFFEQ_OVERLAY"
pids=()
for replicate in 0 1 2; do
    env -u GEMINI_API_KEY -u GOOGLE_API_KEY srun --exclusive --nodes=1 --ntasks=1 \
        --cpus-per-task=3 --gpus=1 "$PEPTIDE_PYTHON" prepare.py generate \
        --config config.json --replicate-index "$replicate" \
        --output "$GENERATION/replicate-$replicate" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
"$PEPTIDE_PYTHON" prepare.py merge \
    --config config.json --output "$GENERATION"
"$PEPTIDE_PYTHON" validate.py \
    --config config.json --generation "$GENERATION" --output "$VALIDATION"
