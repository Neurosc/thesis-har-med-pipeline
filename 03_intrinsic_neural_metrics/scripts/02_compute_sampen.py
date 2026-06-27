#!/usr/bin/env python3
"""
Compute Sample Entropy (SampEn) per ROI for the three-pipeline DMT-MED design.

Loops the three denoising pipelines (detrend/glm/maximal) × six Qin-sphere layers,
reading the qinspheres timeseries and writing one SampEn CSV per (pipeline, layer,
subject, session) plus a per-pipeline long table. Sample = 39 (all minus sub-12),
matching the extraction; the 35-subject inclusion filter is applied later at stats.

Companion to 01_compute_acw.jl — same inputs, same design, NoGSR (all pipelines).

Layers: intero · extero · mental (self) | visual · motor · auditory (nonself)

Steps for each run × layer:
  1. Load the sphere timeseries CSV (timepoints × ROIs)
  2. Drop the first 6 dummy volumes (matches the ACW pipeline)
  3. Linear detrend each ROI column (slow-drift removal; bandpass — where applied —
     already happened during denoising)
  4. Compute SampEn per ROI with EntropyHub (m=1, tau=1, r=0.3, log base 2)

Inputs  : 02_timeseries_extraction/results/qinspheres/{pipeline}/{layer}/{sub}_{ses}_{layer}_timeseries.csv
Outputs : 03_intrinsic_neural_metrics/results/sampen/{pipeline}/{layer}/{sub}_{ses}_{layer}_sampen.csv
          03_intrinsic_neural_metrics/results/sampen/{pipeline}/sampen_long_{pipeline}.csv
          03_intrinsic_neural_metrics/figures/sampen_detrend_check_{pipeline}_{sub}_{ses}_{layer}.png  (one QC)

Run (local or server):
  python 03_intrinsic_neural_metrics/scripts/02_compute_sampen.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend
import matplotlib
matplotlib.use("Agg")  # always save to file — no interactive display
import matplotlib.pyplot as plt

try:
    import EntropyHub as EH
except ImportError:
    sys.exit("EntropyHub not installed. Run:  pip install EntropyHub")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from utils.subject_filter import get_pipeline_subjects

# ── SampEn parameters (Keskin et al. Temporal Imprecision conventions) ──────────
M       = 1    # embedding dimension
TAU     = 1    # time delay (in TRs)
R       = 0.3  # tolerance — absolute value, not relative to SD (no z-scoring)
LOGBASE = 2    # log base 2 (Northoff-lab convention)

# ── Dataset constants ──────────────────────────────────────────────────────────
DUMMY_VOLUMES = 6  # volumes dropped at the start (matches the ACW pipeline)

PIPELINES = ["detrend", "glm", "maximal"]
SUBJECTS  = get_pipeline_subjects()       # 39 (all minus sub-12)
SESSIONS  = ["ses-01", "ses-02"]
LAYERS    = ["intero", "extero", "mental", "visual", "motor", "auditory"]

# ── Paths ──────────────────────────────────────────────────────────────────────
TS_BASE     = REPO_ROOT / "02_timeseries_extraction" / "results" / "qinspheres"
SAMPEN_BASE = REPO_ROOT / "03_intrinsic_neural_metrics" / "results" / "sampen"
FIG_DIR     = REPO_ROOT / "03_intrinsic_neural_metrics" / "figures"
PARTS_TSV   = REPO_ROOT / "participants.tsv"


def load_drug_groups():
    """Return a dict: subject ID -> drug condition ('placebo' or 'verum')."""
    df = pd.read_csv(PARTS_TSV, sep="\t")
    return dict(zip(df["participant_id"].astype(str), df["condition"].astype(str)))


def load_timeseries(csv_path):
    """Load a sphere timeseries CSV → (roi_ids, array of shape (n_tp, n_roi))."""
    df = pd.read_csv(csv_path)
    roi_ids = [str(c) for c in df.columns[1:]]
    ts      = df.iloc[:, 1:].to_numpy(dtype=float)
    return roi_ids, ts


def compute_sampen(ts, roi_ids):
    """SampEn for each ROI column. Returns (values, bad_rois)."""
    values, bad_rois = [], []
    for i, roi in enumerate(roi_ids):
        col = ts[:, i]
        if not np.all(np.isfinite(col)):
            values.append(np.nan)
            bad_rois.append(roi)
            continue
        try:
            Samp, _, _ = EH.SampEn(col, m=M, tau=TAU, r=R, Logx=LOGBASE)
            val = float(Samp[M])
        except Exception:
            val = np.nan
        if not np.isfinite(val):
            bad_rois.append(roi)
        values.append(val)
    return np.array(values, dtype=float), bad_rois


def save_qc_plot(ts_raw_col, ts_det_col, pipeline, subject, session, layer, roi):
    """Save a before/after detrend plot for one ROI to check preprocessing."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax_raw, ax_det) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    ax_raw.plot(ts_raw_col, color="#444444", lw=1.0)
    ax_raw.set_title(f"{pipeline} | {subject} {session} | {layer} ROI {roi} — raw (after dummy removal)")
    ax_raw.set_ylabel("BOLD signal")

    ax_det.plot(ts_det_col, color="#1f77b4", lw=1.0)
    ax_det.axhline(0, color="k", lw=0.6, ls="--")
    ax_det.set_title("After linear detrend")
    ax_det.set_xlabel("Timepoint (TR)")
    ax_det.set_ylabel("BOLD signal")

    fig.tight_layout()
    out = FIG_DIR / f"sampen_detrend_check_{pipeline}_{subject}_{session}_{layer}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  [QC] Detrend figure saved: {out}")


def first_run_sanity_check(values, ts_raw, ts_det, pipeline, subject, session, layer, roi_ids):
    """Print basic statistics and save a QC plot for the first processed run."""
    finite = values[np.isfinite(values)]
    print(f"\n  First-run sanity check ({pipeline} {subject} {session} {layer})")
    print(f"  SampEn: n={finite.size}, min={finite.min():.3f}, max={finite.max():.3f}, "
          f"mean={finite.mean():.3f}, sd={finite.std():.3f}")

    out_of_range = finite[(finite < 0) | (finite > 5)]
    if out_of_range.size:
        print(f"  [FLAG] {out_of_range.size} values outside [0, 5] — check tolerance R")
    else:
        print("  [OK] All SampEn values within [0, 5] (expected fMRI range ~0.5–3.0)")

    first_valid = int(np.flatnonzero(np.isfinite(values))[0])
    save_qc_plot(ts_raw[:, first_valid], ts_det[:, first_valid],
                 pipeline, subject, session, layer, roi_ids[first_valid])
    print()


def process_pipeline(pipeline, drug_groups, first_run_done):
    """Run SampEn for every layer × subject × session of one pipeline.

    Returns (completed, failed, first_run_done).
    """
    ts_base     = TS_BASE / pipeline
    sampen_base = SAMPEN_BASE / pipeline
    long_csv    = sampen_base / f"sampen_long_{pipeline}.csv"
    sampen_base.mkdir(parents=True, exist_ok=True)

    all_rows  = []
    completed = 0
    failed    = 0
    total     = len(LAYERS) * len(SUBJECTS) * len(SESSIONS)

    print(f"\n{'='*60}\nPIPELINE: {pipeline}\n{'='*60}")

    for layer in LAYERS:
        out_dir = sampen_base / layer
        out_dir.mkdir(parents=True, exist_ok=True)
        for subject in SUBJECTS:
            for session in SESSIONS:
                run_num  = completed + failed + 1
                label    = f"[{pipeline}][{run_num}/{total}] {subject} {session} {layer}"
                csv_path = ts_base / layer / f"{subject}_{session}_{layer}_timeseries.csv"
                out_csv  = out_dir / f"{subject}_{session}_{layer}_sampen.csv"

                if not csv_path.is_file():
                    print(f"{label} ... MISSING")
                    failed += 1
                    continue

                try:
                    roi_ids, ts_raw = load_timeseries(csv_path)
                    ts_raw = ts_raw[DUMMY_VOLUMES:]
                    ts_det = detrend(ts_raw, axis=0)
                    values, bad_rois = compute_sampen(ts_det, roi_ids)
                except Exception as e:
                    print(f"{label} ... ERROR ({type(e).__name__}: {e})")
                    failed += 1
                    continue

                pd.DataFrame({"roi_id": roi_ids, "sampen": values}).to_csv(out_csv, index=False)

                group = drug_groups.get(subject, "unknown")
                for roi, val in zip(roi_ids, values):
                    all_rows.append({"pipeline": pipeline, "subject": subject,
                                     "session": session, "drug_group": group,
                                     "layer": layer, "roi_id": roi, "sampen": val})

                n_bad = len(bad_rois)
                print(f"{label} ... OK ({len(roi_ids)} ROIs, mean={np.nanmean(values):.3f}, bad={n_bad})")
                if n_bad:
                    shown = ", ".join(bad_rois[:10]) + (" ..." if n_bad > 10 else "")
                    print(f"    [FLAG] {n_bad} ROIs returned NaN/Inf: {shown}")

                if not first_run_done:
                    first_run_sanity_check(values, ts_raw, ts_det, pipeline,
                                           subject, session, layer, roi_ids)
                    first_run_done = True

                completed += 1

    long_df = pd.DataFrame(all_rows, columns=["pipeline", "subject", "session",
                                              "drug_group", "layer", "roi_id", "sampen"])
    long_df.to_csv(long_csv, index=False)
    print(f"\n  {pipeline}: {completed} done, {failed} missing/failed. "
          f"Long table ({len(long_df)} rows) -> {long_csv}")

    return completed, failed, first_run_done


def main():
    print("=" * 70)
    print("02_compute_sampen.py — Sample Entropy, three pipelines × six layers")
    print(f"  SampEn params: m={M}, tau={TAU}, r={R} (absolute), log base={LOGBASE}")
    print(f"  Dummy volumes discarded: {DUMMY_VOLUMES}")
    print(f"  Pipelines: {PIPELINES}  Subjects: {len(SUBJECTS)}  "
          f"Sessions: {len(SESSIONS)}  Layers: {len(LAYERS)}")
    print("=" * 70)

    drug_groups = load_drug_groups()
    SAMPEN_BASE.mkdir(parents=True, exist_ok=True)

    total_done = total_fail = 0
    first_run_done = False
    for pipeline in PIPELINES:
        done, fail, first_run_done = process_pipeline(pipeline, drug_groups, first_run_done)
        total_done += done
        total_fail += fail

    grand_total = len(PIPELINES) * len(LAYERS) * len(SUBJECTS) * len(SESSIONS)
    print("\n" + "=" * 70)
    print(f"Done. Completed: {total_done}/{grand_total} runs, Failed/missing: {total_fail}")
    print("=" * 70)


if __name__ == "__main__":
    main()
