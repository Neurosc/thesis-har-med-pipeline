#!/usr/bin/env bash
#
# Stage 1 unattended chain — maximal_nocensor, SERVER side.
#
# Runs: preflight -> denoise (78 runs) -> extract spheres -> extract glasser360
#       -> build qinparcels -> commit + push.
#
# Designed for nohup. Every stage logs to _stage1_nocensor_run/ inside the repo,
# and the chain ABORTS on the first failure rather than continuing with partial
# data (set -euo pipefail). On abort it writes and pushes STAGE1_FAILED.txt so
# the local Stage 2 orchestrator stops waiting instead of hanging until its
# timeout -- i.e. a failure at 3am is visible in the morning, not a silent hang.
#
# The conda env is NOT activated here: guessing a conda install path is exactly
# the kind of silent failure this chain must not have. Activate it yourself,
# then launch. The script verifies and refuses to run otherwise.
#
# Usage (from the repo root, env already activated):
#     conda activate fmri
#     nohup bash run_stage1_nocensor.sh > _stage1_nocensor_run/nohup_stdout.txt 2>&1 &
#     tail -f _stage1_nocensor_run/stage1_master.txt
#
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

LOGDIR="$REPO/_stage1_nocensor_run"
mkdir -p "$LOGDIR"
MASTER="$LOGDIR/stage1_master.txt"
SENTINEL_OK="$LOGDIR/STAGE1_COMPLETE.txt"
SENTINEL_FAIL="$LOGDIR/STAGE1_FAILED.txt"

# Clear sentinels from any previous run so a stale one cannot be misread.
rm -f "$SENTINEL_OK" "$SENTINEL_FAIL"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$MASTER"; }

fail() {
    local stage="$1"
    {
        echo "STAGE 1 FAILED"
        echo "stage    : $stage"
        echo "time     : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "log      : $LOGDIR/${stage}.txt"
        echo
        echo "Last 40 lines of that stage log:"
        tail -40 "$LOGDIR/${stage}.txt" 2>/dev/null || echo "(no stage log)"
    } > "$SENTINEL_FAIL"
    log "!!! FAILED at stage: $stage"
    # Push the failure marker so the local orchestrator stops waiting.
    git add -- "$LOGDIR" >/dev/null 2>&1 || true
    git commit -q -m "Stage 1 FAILED at ${stage} (maximal_nocensor)" >/dev/null 2>&1 || true
    git push -q origin master >/dev/null 2>&1 || log "(could not push failure marker)"
    exit 1
}

run_stage() {
    local name="$1"; shift
    log "START  $name"
    if ! "$@" > "$LOGDIR/${name}.txt" 2>&1; then
        fail "$name"
    fi
    log "DONE   $name"
}

log "================= STAGE 1 CHAIN: maximal_nocensor ================="
log "repo: $REPO"

# ── Guard: conda env ──────────────────────────────────────────────────────────
if ! python -c "import nibabel, nilearn, pandas, numpy" 2>/dev/null; then
    echo "ABORT: python cannot import nibabel/nilearn/pandas/numpy." | tee -a "$MASTER"
    echo "Activate the env first:  conda activate fmri" | tee -a "$MASTER"
    exit 1
fi
log "env OK (CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-unset})"

# ── Guard: the preset must exist on THIS checkout ─────────────────────────────
if ! python -c "
import sys; sys.path.insert(0, '01_denoising/scripts')
from denoise_pipelines import PIPELINE_PRESETS
sys.exit(0 if 'maximal_nocensor' in PIPELINE_PRESETS else 1)
" 2>/dev/null; then
    echo "ABORT: PIPELINE_PRESETS has no 'maximal_nocensor'. Run 'git pull' first." | tee -a "$MASTER"
    exit 1
fi
log "preset maximal_nocensor present"

# ── The chain ─────────────────────────────────────────────────────────────────
run_stage 01_preflight \
    python 02_timeseries_extraction/scripts/00_preflight_maximal_nocensor.py

run_stage 02_denoise \
    python 01_denoising/scripts/01_denoise_all.py --pipeline maximal_nocensor

# `env VAR=x cmd` rather than a `VAR=x run_stage ...` prefix: for shell FUNCTIONS
# the persistence and export semantics of an assignment prefix differ between
# bash modes, and a silently-unset PIPELINES here would extract detrend+maximal
# instead -- overwriting nothing, but producing the wrong pipeline entirely.
run_stage 03_extract_spheres \
    env PIPELINES=maximal_nocensor \
    python 02_timeseries_extraction/scripts/01_extract_sphere_timeseries.py

run_stage 04_extract_glasser360 \
    env PIPELINES=maximal_nocensor SUBJECT_SET=pipeline \
    python 02_timeseries_extraction/scripts/glasser_g1/01_extract_glasser_cortex.py --allow-resample

run_stage 05_build_qinparcels \
    env PIPELINES=maximal_nocensor \
    python 02_timeseries_extraction/scripts/02_build_qin_parcel_timeseries.py

# ── Sanity: did the extractions actually produce files? ───────────────────────
log "verifying outputs"
n_sph=$(find 02_timeseries_extraction/results/qinspheres/maximal_nocensor -name '*_timeseries.csv' 2>/dev/null | wc -l)
n_g360=$(find 02_timeseries_extraction/results/glasser360/maximal_nocensor -name '*_timeseries.csv' 2>/dev/null | wc -l)
n_par=$(find 02_timeseries_extraction/results/qinparcels/maximal_nocensor -name '*_timeseries.csv' 2>/dev/null | wc -l)
log "  qinspheres CSVs : $n_sph   (expect 39 x 2 x 6 = 468)"
log "  glasser360 CSVs : $n_g360  (expect 39 x 2 = 78)"
log "  qinparcels CSVs : $n_par   (expect 468)"
if [ "$n_sph" -eq 0 ] || [ "$n_g360" -eq 0 ] || [ "$n_par" -eq 0 ]; then
    echo "An extraction produced ZERO files." > "$LOGDIR/06_verify.txt"
    fail 06_verify
fi

# ── Sentinel + push ───────────────────────────────────────────────────────────
{
    echo "STAGE 1 COMPLETE"
    echo "time            : $(date '+%Y-%m-%d %H:%M:%S')"
    echo "qinspheres CSVs : $n_sph"
    echo "glasser360 CSVs : $n_g360"
    echo "qinparcels CSVs : $n_par"
} > "$SENTINEL_OK"

log "committing outputs"
git add -- 02_timeseries_extraction/results "$LOGDIR" || fail 07_gitadd
git commit -q -m "maximal_nocensor: timeseries for both atlases (n=39) + Stage 1 logs" \
    || log "(nothing new to commit)"

log "pushing"
git push -q origin master || fail 08_push

log "================= STAGE 1 COMPLETE ================="
log "Local Stage 2 orchestrator can now proceed."
