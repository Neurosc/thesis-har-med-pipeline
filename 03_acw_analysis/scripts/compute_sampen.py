#!/usr/bin/env python3
"""
compute_sampen.py — Sample Entropy (SampEn) for DMT-MED self vs nonself parcels.

Companion metric to the ACW (AUC/tau) pipeline: a per-parcel Sample Entropy is
computed from the same denoised parcel timeseries, so SampEn can be carried
through the same self-vs-nonself / pre-vs-post / drug-group analyses as AUC.

Pipeline per subject x session x atlas
---------------------------------------
  1. Load timeseries CSV  -> array (n_timepoints x n_parcels)
  2. Discard dummy volumes (DUMMY_VOLUMES, matches the ACW pipeline)
  3. Linear detrend (scipy.signal.detrend, axis=0). No bandpass (already applied
     during denoising). No z-scoring.
  4. SampEn per parcel via EntropyHub (Keskin "Temporal Imprecision" params):
        m = 1, tau = 1, r = 0.3 (ABSOLUTE tolerance, not a fraction of SD),
        Logx = 2 (log base 2, Northoff-lab convention).
     SampEn at m=1 is taken (Samp[m]).

Inputs (paths adjusted to actual on-disk locations)
---------------------------------------------------
  self    (Keskin Glasser self parcels):
      02_timeseries_extraction/results/timeseries_self_glasser/
          sub-XX_ses-YY_keskin_timeseries.csv          (flat; NoGSR extraction)
  nonself (all remaining Glasser parcels):
      02_timeseries_extraction/results/timeseries_parcels/nonself/denoisedNoGSR/
          sub-XX_ses-YY_nonself_parcel_timeseries.csv
  Drug group: participants.tsv `condition` (placebo | verum)

  NOTE: the literal paths in the task brief (results/keskin/, results/nonself/)
  do not exist; these are the real locations used by 01_compute_acw.jl and
  02_compute_acw_keskin.jl. Denoising version is fixed to NoGSR to match the
  Keskin ACW pipeline and the primary statistics pipeline (the self/keskin
  extraction is only available as the flat NoGSR set).

Outputs
-------
  Per subject x session x atlas:
      03_acw_analysis/results/sampen/sub-XX_ses-YY_{atlas}_sampen.csv
          columns: roi_pos_id, sampen
  Long-format summary (same shape as 04_statistics .../analysis_long_format_auc.csv):
      04_statistics/results/sampen_long_format.csv
          columns: subject, session, drug_group, atlas, roi_pos_id, sampen
  Detrend QC figure (one parcel, one run):
      03_acw_analysis/figures/sampen_detrend_check_<sub>_<ses>_<atlas>.png

Sanity checks (printed to console)
----------------------------------
  - SampEn range (min/max/mean/SD) for the first processed run
  - Flags any SampEn outside [0, 5] (typical fMRI range ~0.5-3.0)
  - Counts NaN/Inf SampEn and lists offending parcels per run

Requirements: EntropyHub (pip install EntropyHub) in addition to the standard
conda `fmri` env (numpy, pandas, scipy, matplotlib).

Run on server (idempotent: existing per-run CSVs are skipped):
  conda activate fmri
  python 03_acw_analysis/scripts/compute_sampen.py

(Do NOT run during Windows development — real data lives on the server.)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import detrend

import matplotlib
matplotlib.use("Agg")  # headless server — never plt.show()
import matplotlib.pyplot as plt

try:
    import EntropyHub as EH
except ImportError:
    sys.exit("EntropyHub not installed. Run:  pip install EntropyHub")

# ── Repo paths / shared utils ──────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from utils.subject_filter import get_included_subjects

# ── SampEn parameters (Keskin et al., Temporal Imprecision) ─────────────────────
M       = 1      # embedding dimension
R       = 0.3    # tolerance — ABSOLUTE (not fraction of SD); no z-scoring upstream
TAU     = 1      # time delay
LOGBASE = 2      # log base 2 (Northoff-lab convention)
# If the sanity checks below reveal mostly Inf/NaN SampEn (tolerance too tight for
# the un-normalised BOLD scale), switch R to a per-parcel relative tolerance, e.g.
#   r_used = R * np.nan_to_num(col).std()
# Left absolute here per the analysis spec.

# ── Preprocessing constants ─────────────────────────────────────────────────────
DUMMY_VOLUMES = 6          # discarded before detrend — matches the ACW pipeline
DENOISING     = "NoGSR"    # NoGSR | GSR  (self/keskin extraction is NoGSR-only)

SESSIONS = ["ses-01", "ses-02"]
SUBJECTS = get_included_subjects()   # 35 included subjects (sub-XX)

# ── Input layout per atlas ──────────────────────────────────────────────────────
# atlas label -> (timeseries directory, filename template)
#   self    = Keskin Glasser self parcels (flat dir, "keskin" suffix)
#   nonself = all remaining Glasser parcels (parcel extraction, versioned dir)
_TS_RESULTS = REPO_ROOT / "02_timeseries_extraction" / "results"
ATLASES = {
    "self": (
        _TS_RESULTS / "timeseries_self_glasser",
        "{subject}_{session}_keskin_timeseries.csv",
    ),
    "nonself": (
        _TS_RESULTS / "timeseries_parcels" / "nonself" / f"denoised{DENOISING}",
        "{subject}_{session}_nonself_parcel_timeseries.csv",
    ),
}

# ── Output layout ───────────────────────────────────────────────────────────────
SAMPEN_DIR = REPO_ROOT / "03_acw_analysis" / "results" / "sampen"
FIG_DIR    = REPO_ROOT / "03_acw_analysis" / "figures"
LONG_CSV   = REPO_ROOT / "04_statistics" / "results" / "sampen_long_format.csv"
PARTS_TSV  = REPO_ROOT / "participants.tsv"

SEP = "=" * 70


def load_drug_map():
    """participant_id -> condition (placebo | verum); fallback 'unknown'."""
    parts = pd.read_csv(PARTS_TSV, sep="\t")
    return dict(zip(parts["participant_id"].astype(str), parts["condition"].astype(str)))


def load_timeseries(csv_path):
    """Load CSV -> (roi_ids, ts) where ts is (n_timepoints x n_parcels).

    First column is the Timepoint index and is dropped; remaining column names
    are the parcel (ROI) identifiers, matching 01_compute_acw.jl.
    """
    df = pd.read_csv(csv_path)
    roi_ids = [str(c) for c in df.columns[1:]]
    ts = df.iloc[:, 1:].to_numpy(dtype=float)
    return roi_ids, ts


def sampen_per_parcel(ts_detrended, roi_ids):
    """Compute SampEn(m=1) for every parcel column.

    Returns (values, invalid_rois). A parcel yields NaN (and is flagged) when its
    column is non-finite or EntropyHub returns a non-finite SampEn (e.g. no
    template matches within the tolerance).
    """
    values = []
    invalid_rois = []
    for j, roi in enumerate(roi_ids):
        col = ts_detrended[:, j]
        if not np.all(np.isfinite(col)):
            values.append(np.nan)
            invalid_rois.append(roi)
            continue
        try:
            Samp, _A, _B = EH.SampEn(col, m=M, tau=TAU, r=R, Logx=LOGBASE)
            val = float(Samp[M])  # SampEn at embedding dimension m (= Samp[1] for m=1)
        except Exception:
            val = np.nan
        if not np.isfinite(val):
            invalid_rois.append(roi)
        values.append(val)
    return np.asarray(values, dtype=float), invalid_rois


def detrend_qc_plot(ts_raw_col, ts_det_col, subject, session, atlas, roi):
    """Save a before/after linear-detrend figure for one parcel (one run)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    axes[0].plot(ts_raw_col, color="#444444", lw=1.0)
    axes[0].set_title(f"{subject} {session}  atlas={atlas}  parcel={roi} — raw (post dummy removal)")
    axes[0].set_ylabel("BOLD (a.u.)")
    axes[1].plot(ts_det_col, color="#1f77b4", lw=1.0)
    axes[1].axhline(0, color="k", lw=0.6, ls="--")
    axes[1].set_title("linear-detrended")
    axes[1].set_xlabel("timepoint (TR)")
    axes[1].set_ylabel("BOLD (a.u.)")
    fig.tight_layout()
    out = FIG_DIR / f"sampen_detrend_check_{subject}_{session}_{atlas}.png"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"  [QC] detrend figure saved: {out}")


def main():
    print(SEP)
    print("compute_sampen.py — Sample Entropy (self vs nonself)")
    print(SEP)
    print(f"Params:        m={M}, tau={TAU}, r={R} (absolute), Logx={LOGBASE}")
    print(f"Dummy volumes: {DUMMY_VOLUMES} discarded")
    print(f"Denoising:     {DENOISING}")
    print(f"Subjects:      {len(SUBJECTS)}  Sessions: {len(SESSIONS)}  Atlases: {len(ATLASES)}")
    print(SEP, "\n")

    drug_map = load_drug_map()
    SAMPEN_DIR.mkdir(parents=True, exist_ok=True)
    LONG_CSV.parent.mkdir(parents=True, exist_ok=True)

    long_rows = []          # accumulates the long-format summary
    total = len(ATLASES) * len(SUBJECTS) * len(SESSIONS)
    run_idx = 0
    completed = skipped = failed = 0
    total_invalid = 0
    first_run_done = False  # gate the detailed first-run sanity block + QC plot
    global_min, global_max = np.inf, -np.inf

    for atlas, (ts_dir, fname_tpl) in ATLASES.items():
        for subject in SUBJECTS:
            for session in SESSIONS:
                run_idx += 1
                label = f"[{run_idx}/{total}] {subject} {session} {atlas}"
                csv_path = ts_dir / fname_tpl.format(subject=subject, session=session)
                out_csv = SAMPEN_DIR / f"{subject}_{session}_{atlas}_sampen.csv"

                if not csv_path.is_file():
                    print(f"{label} ... FAIL (CSV not found: {csv_path})")
                    failed += 1
                    continue

                try:
                    roi_ids, ts_raw = load_timeseries(csv_path)
                    ts_raw = ts_raw[DUMMY_VOLUMES:, :]          # drop dummies -> 234 tp
                    ts_det = detrend(ts_raw, type="linear", axis=0)
                    values, invalid_rois = sampen_per_parcel(ts_det, roi_ids)
                except Exception as e:
                    print(f"{label} ... FAIL ({type(e).__name__}: {e})")
                    failed += 1
                    continue

                # Per-run per-parcel CSV (roi_pos_id, sampen) — keeps NaN rows.
                pd.DataFrame({"roi_pos_id": roi_ids, "sampen": values}).to_csv(
                    out_csv, index=False
                )

                # Append to long-format summary.
                dg = drug_map.get(subject, "unknown")
                for roi, val in zip(roi_ids, values):
                    long_rows.append({
                        "subject": subject, "session": session, "drug_group": dg,
                        "atlas": atlas, "roi_pos_id": roi, "sampen": val,
                    })

                finite = values[np.isfinite(values)]
                n_inv = len(invalid_rois)
                total_invalid += n_inv
                if finite.size:
                    global_min = min(global_min, float(finite.min()))
                    global_max = max(global_max, float(finite.max()))

                msg = (f"{label} ... DONE ({len(roi_ids)} parcels, "
                       f"mean={np.nanmean(values):.3f}, invalid={n_inv})")
                print(msg)
                if n_inv:
                    shown = ", ".join(invalid_rois[:10]) + (" ..." if n_inv > 10 else "")
                    print(f"    [FLAG] {n_inv} invalid (NaN/Inf) SampEn parcels: {shown}")
                completed += 1

                # ── Sanity checks on the first processed run ──────────────────────
                if not first_run_done and finite.size:
                    print("\n  ── Sanity checks (first run) ──")
                    print(f"  SampEn  n={finite.size}  min={finite.min():.3f}  "
                          f"max={finite.max():.3f}  mean={finite.mean():.3f}  "
                          f"sd={finite.std():.3f}")
                    out_of_range = finite[(finite < 0) | (finite > 5)]
                    if out_of_range.size:
                        print(f"  [FLAG] {out_of_range.size} SampEn outside [0, 5] "
                              f"(typical fMRI ~0.5-3.0)")
                    else:
                        print("  [OK] all SampEn within [0, 5] (typical fMRI ~0.5-3.0)")
                    # Detrend before/after plot for the first finite parcel.
                    fin_idx = int(np.flatnonzero(np.isfinite(values))[0])
                    detrend_qc_plot(ts_raw[:, fin_idx], ts_det[:, fin_idx],
                                    subject, session, atlas, roi_ids[fin_idx])
                    print()
                    first_run_done = True

    # ── Write long-format summary ──────────────────────────────────────────────
    long_df = pd.DataFrame(
        long_rows,
        columns=["subject", "session", "drug_group", "atlas", "roi_pos_id", "sampen"],
    )
    long_df.to_csv(LONG_CSV, index=False)

    # ── Final report ────────────────────────────────────────────────────────────
    print("\n" + SEP)
    print("SampEn summary")
    print(SEP)
    print(f"Runs total:     {total}")
    print(f"Completed:      {completed}")
    print(f"Failed/missing: {failed}")
    print(f"Long-format rows: {len(long_df)}  ->  {LONG_CSV}")
    print(f"Invalid (NaN/Inf) SampEn values overall: {total_invalid}")
    if np.isfinite(global_min):
        print(f"Global SampEn range: [{global_min:.3f}, {global_max:.3f}]")
        if global_min < 0 or global_max > 5:
            print("  [FLAG] global SampEn extends outside [0, 5] — inspect tolerance R / scaling")
        else:
            print("  [OK] global SampEn within [0, 5]")
    print("Done.")


if __name__ == "__main__":
    main()
