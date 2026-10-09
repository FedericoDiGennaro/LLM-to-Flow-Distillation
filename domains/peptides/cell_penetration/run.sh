#!/usr/bin/env bash
# Convenience wrapper for a fresh run inside an allocated Slurm GPU job.
set -euo pipefail
: "${SLURM_JOB_ID:?run this workflow in an allocated Slurm job}"
: "${ACTFLOW_CPP_OUTPUT_ROOT:?fresh CPP output directory required}"
: "${PEPDFM_PYTHON:?PepDFM runtime interpreter required}"
: "${PEPDFM_ASSET_DIR:?PepDFM asset directory required}"
extra_args=()
if [[ -n "${PEPDFM_DEPENDENCY_OVERLAY:-}" ]]; then
    extra_args+=(--dependency-overlay "$PEPDFM_DEPENDENCY_OVERLAY")
fi
exec "${ACTFLOW_API_PYTHON:-python}" -m adm.cli run cell-penetration \
    --output-dir "$ACTFLOW_CPP_OUTPUT_ROOT" \
    --assets "$PEPDFM_ASSET_DIR" \
    --runtime-python "$PEPDFM_PYTHON" "${extra_args[@]}"
