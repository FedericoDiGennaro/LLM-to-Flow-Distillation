#!/bin/bash
#SBATCH --job-name=pepdfm-anticancer-e2e
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gpus=1
#SBATCH --time=20:00:00

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?Submit from the repository root}"
OUTPUT_ROOT="${1:?fresh experiment output root required}"
ASSET_DIR="${2:?pinned PepDFM asset directory required}"
PEPDFM_PYTHON="${3:?exact PepDFM Python interpreter required}"
DEPENDENCY_OVERLAY="${4:?frozen torchdiffeq overlay required}"
AUTHORIZATION="${5:?frozen Gemini authorization required}"
ACTIVE_CAMPAIGN="${6:?active campaign required}"
API_PYTHON="${7:?Gemini/API Python interpreter required}"
XGBOOST_OVERLAY="${8:?frozen xgboost overlay required}"

POOL_CONFIG="$ROOT/domains/peptides/anticancer/configs/pool.json"
CONFIG_TEMPLATE="$ROOT/domains/peptides/anticancer/configs/dpo.json"

CONFIG="$OUTPUT_ROOT/frozen_config.json"
P0_ROOT="$OUTPUT_ROOT/p0"
CAMPAIGN_OUTPUT="$OUTPUT_ROOT/campaign"
LENGTH_POLICY="$OUTPUT_ROOT/campaign_length_policy.json"
EVIDENCE_REFERENCE="$P0_ROOT/frozen_evidence_reference_private.json"

for executable in "$PEPDFM_PYTHON" "$API_PYTHON"; do
    if [[ ! -x "$executable" ]]; then
        echo "Required Python interpreter is not executable: $executable" >&2
        exit 2
    fi
done
if [[ ! -d "$DEPENDENCY_OVERLAY" ]]; then
    echo "DEPENDENCY_OVERLAY is missing: $DEPENDENCY_OVERLAY" >&2
    exit 2
fi
if [[ ! -d "$XGBOOST_OVERLAY/xgboost" ]]; then
    echo "XGBOOST_OVERLAY is incomplete: $XGBOOST_OVERLAY" >&2
    exit 2
fi
if [[ -z "${GEMINI_API_KEY:-}" || -n "${GOOGLE_API_KEY:-}" ]]; then
    echo "The secure submitter must provide exactly GEMINI_API_KEY" >&2
    exit 2
fi

trap 'unset GEMINI_API_KEY GOOGLE_API_KEY' EXIT
cd "$ROOT"
export PYTHONPATH="$ROOT:$DEPENDENCY_OVERLAY:$XGBOOST_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0
export PYTHONDONTWRITEBYTECODE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PEPDFM_E2E_OUTPUT_ROOT="$OUTPUT_ROOT"
export TMPDIR="${TMPDIR:-/tmp}/pepdfm-e2e-dpo-${SLURM_JOB_ID:?}"
mkdir -p "$TMPDIR" "$OUTPUT_ROOT" "$P0_ROOT" "$CAMPAIGN_OUTPUT"

"$API_PYTHON" -m domains.peptides.preflight \
    --config "$CONFIG_TEMPLATE" \
    --output-root "$CAMPAIGN_OUTPUT" \
    --pool-config "$POOL_CONFIG" \
    --authorization "$AUTHORIZATION" \
    --active-campaign "$ACTIVE_CAMPAIGN"

"$API_PYTHON" -m domains.peptides.select_lengths \
    --config "$CONFIG_TEMPLATE" \
    --artifact "$LENGTH_POLICY" \
    --output-root "$CAMPAIGN_OUTPUT" \
    --authorization "$AUTHORIZATION" \
    --active-campaign "$ACTIVE_CAMPAIGN"

env -u GEMINI_API_KEY -u GOOGLE_API_KEY "$PEPDFM_PYTHON" -m domains.peptides.fetch_assets \
    --asset-dir "$ASSET_DIR" \
    --local-files-only
env -u GEMINI_API_KEY -u GOOGLE_API_KEY "$PEPDFM_PYTHON" -c \
    'import torch; assert torch.cuda.is_available(), "CUDA GPU unavailable"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))'
env -u GEMINI_API_KEY -u GOOGLE_API_KEY "$PEPDFM_PYTHON" -c \
    'import xgboost; assert xgboost.__version__ == "2.1.4"; print("xgboost", xgboost.__version__)'
env -u GEMINI_API_KEY -u GOOGLE_API_KEY "$PEPDFM_PYTHON" -m domains.peptides.check_predictors \
    --asset-dir "$ASSET_DIR" \
    --output "$OUTPUT_ROOT/smoke/predictor_runtime.json"

env -u GEMINI_API_KEY -u GOOGLE_API_KEY "$PEPDFM_PYTHON" -m domains.peptides.anticancer.pool \
    --config "$POOL_CONFIG" \
    --length-policy "$LENGTH_POLICY" \
    --output-root "$P0_ROOT" \
    --asset-dir "$ASSET_DIR" \
    --active-campaign "$ACTIVE_CAMPAIGN"

"$API_PYTHON" -m domains.peptides.anticancer.freeze \
    --template "$CONFIG_TEMPLATE" \
    --candidates "$P0_ROOT/candidate_pool.jsonl" \
    --evidence-reference "$EVIDENCE_REFERENCE" \
    --length-policy "$LENGTH_POLICY" \
    --output "$CONFIG"

"$API_PYTHON" -m domains.peptides.anticancer.run \
    --config "$CONFIG" \
    --output-root "$CAMPAIGN_OUTPUT" \
    --asset-dir "$ASSET_DIR" \
    --authorization "$AUTHORIZATION" \
    --active-campaign "$ACTIVE_CAMPAIGN" \
    --pepdfm-python "$PEPDFM_PYTHON" \
    --dependency-overlay "$DEPENDENCY_OVERLAY"

env -u GEMINI_API_KEY -u GOOGLE_API_KEY "$API_PYTHON" \
    -m domains.peptides.anticancer.validate \
    --config "$CONFIG" \
    --output-root "$CAMPAIGN_OUTPUT"
