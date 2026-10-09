#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=160G
#SBATCH --gpus=4
#SBATCH --time=06:00:00
set -euo pipefail
ROOT="${SLURM_SUBMIT_DIR:?submit from the repository root}"
BASELINE_ROOT="$ROOT/domains/peptides/anticancer/baselines/acp_dpo_nft"
export LTFD_BASELINE_OUTPUT="${1:?output root required}"
export PEPDFM_ASSET_DIR="${2:?PepDFM asset directory required}"
PEPTIDE_PYTHON="${3:?PepDFM Python required}"
export ANTICP2_ASSETS="${4:?released AntiCP2 asset directory required}"
export HEMOPI2_ASSETS="${5:?released HemoPI2 Model directory required}"
export HEMOPI2_OVERLAY="${6:?scikit-learn 1.3.1 overlay required}"
TORCHDIFFEQ_OVERLAY="${7:?torchdiffeq overlay required}"
cd "$BASELINE_ROOT"
unset GEMINI_API_KEY GOOGLE_API_KEY
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONNOUSERSITE=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$ROOT:$TORCHDIFFEQ_OVERLAY"
OUTPUT="$LTFD_BASELINE_OUTPUT"
"$PEPTIDE_PYTHON" baseline.py audit
"$PEPTIDE_PYTHON" -m unittest test_baseline
pids=()
for seed in 12301 12302 12303; do
    srun --exclusive --nodes=1 --ntasks=1 --cpus-per-task=3 --gpus=1 \
        "$PEPTIDE_PYTHON" baseline.py train --seed "$seed" \
        > "$OUTPUT/seed_${seed}_${SLURM_JOB_ID}.log" 2>&1 &
    pids+=("$!")
done
worker_failure=0
for pid in "${pids[@]}"; do
    wait "$pid" || worker_failure=1
done
"$PEPTIDE_PYTHON" baseline.py report
exit "$worker_failure"
