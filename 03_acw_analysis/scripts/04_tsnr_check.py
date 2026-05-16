#!/usr/bin/env python3
"""
04_tsnr_check.py — τ ≈ 1 s spike investigation: per-ROI tSNR

Tests whether the 46 nonself Glasser ROIs that dominate the τ ≈ 1 s spike
have low temporal SNR due to susceptibility dropout (temporal poles, OFC,
frontal poles — regions affected by sinuses and petrous bone).

RUN ON SERVER (NIfTIs not on local Windows machine):
  conda activate fmri
  cd /BICNAS2/group-northoff/jkokino/codes/har_med_codes
  python 03_acw_analysis/scripts/04_tsnr_check.py

After completion, transfer results to local:
  scp jkokino@10.156.156.21:/BICNAS2/group-northoff/jkokino/codes/har_med_codes/
      03_acw_analysis/results/tsnr/tsnr_per_roi.tsv
      <local>/thesis-har-med-pipeline/03_acw_analysis/results/tsnr/
"""

import sys
import time
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy import stats

# ── Repo root (script is at 03_acw_analysis/scripts/) ──────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.subject_filter import get_included_subjects

# ── Paths ────────────────────────────────────────────────────────────────────
FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
ATLAS_PATH    = (REPO_ROOT / "02_timeseries_extraction" / "results" / "atlases"
                 / "nonself_atlas_native.nii.gz")
SPIKE_ROIS_TSV = REPO_ROOT / "03_acw_analysis" / "results" / "tsnr" / "spike_rois.tsv"
OUT_DIR       = REPO_ROOT / "03_acw_analysis" / "results" / "tsnr"
OUT_LONG      = OUT_DIR / "tsnr_per_roi.tsv"
OUT_SUMMARY   = OUT_DIR / "tsnr_summary.tsv"

# ── Constants ─────────────────────────────────────────────────────────────────
SUBJECTS      = get_included_subjects()   # 35 included subjects
SESSIONS      = ["ses-01", "ses-02"]
DUMMY_VOLUMES = 6                         # discard first 6 volumes (TR=1.8 s, HRF stabilization)

TOTAL_RUNS    = len(SUBJECTS) * len(SESSIONS)


def bold_path(sub, ses):
    return (FMRIPREP_ROOT / sub / ses / "func"
            / f"{sub}_{ses}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")


# ── Idempotency ───────────────────────────────────────────────────────────────
if OUT_LONG.exists():
    print(f"[skip] {OUT_LONG} already exists. Delete to rerun.")
    sys.exit(0)

# ── Check required inputs ─────────────────────────────────────────────────────
for f in [ATLAS_PATH, SPIKE_ROIS_TSV]:
    if not f.exists():
        raise FileNotFoundError(f"Missing required input: {f}")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Load atlas once ───────────────────────────────────────────────────────────
print("Loading atlas...")
atlas_img  = nib.load(str(ATLAS_PATH))
atlas_data = atlas_img.get_fdata()
roi_ids    = np.unique(atlas_data)
roi_ids    = roi_ids[roi_ids > 0].astype(int)   # exclude background (0)
N_ROIS     = len(roi_ids)
print(f"  Atlas shape: {atlas_data.shape}, ROIs: {N_ROIS}")

# Precompute boolean masks for each ROI (avoids repeated comparisons)
roi_masks = {int(r): (atlas_data == r) for r in roi_ids}

# ── Main loop ─────────────────────────────────────────────────────────────────
rows = []
run_idx = 0

for sub in SUBJECTS:
    for ses in SESSIONS:
        run_idx += 1
        bold_file = bold_path(sub, ses)
        label     = f"[{run_idx}/{TOTAL_RUNS}] {sub} {ses}"

        if not bold_file.exists():
            print(f"{label} ... SKIP (BOLD not found: {bold_file})")
            continue

        t0 = time.time()
        bold_img  = nib.load(str(bold_file))
        bold_data = bold_img.get_fdata()             # (x, y, z, t)

        if bold_data.shape[3] <= DUMMY_VOLUMES:
            print(f"{label} ... SKIP (only {bold_data.shape[3]} volumes, too few after dummy removal)")
            continue

        bold_data = bold_data[:, :, :, DUMMY_VOLUMES:]   # discard first 6 volumes

        for roi_id in roi_ids:
            mask       = roi_masks[roi_id]
            vox_ts     = bold_data[mask]                  # (n_voxels, t)
            vox_mean   = vox_ts.mean(axis=1)
            vox_std    = vox_ts.std(axis=1, ddof=1)

            # Exclude voxels with zero std (constant signal, e.g. outside brain)
            valid      = vox_std > 0
            if not np.any(valid):
                tsnr = np.nan
            else:
                vox_tsnr = vox_mean[valid] / vox_std[valid]
                tsnr     = float(np.mean(vox_tsnr))

            rows.append({
                "subject": sub,
                "session": ses,
                "roi_id":  int(roi_id),
                "tsnr":    round(tsnr, 4) if not np.isnan(tsnr) else np.nan,
            })

        elapsed = round(time.time() - t0, 1)
        print(f"{label} ... DONE ({elapsed}s, {N_ROIS} ROIs)")

# ── Write long-format TSV ─────────────────────────────────────────────────────
long_df = pd.DataFrame(rows, columns=["subject", "session", "roi_id", "tsnr"])
long_df.to_csv(OUT_LONG, sep="\t", index=False)
print(f"\nLong TSV written: {OUT_LONG}  ({len(long_df)} rows)")

# ── tSNR summary per ROI ──────────────────────────────────────────────────────
summary = (
    long_df.groupby("roi_id")["tsnr"]
    .agg(
        mean_tsnr   = "mean",
        median_tsnr = "median",
        std_tsnr    = "std",
        n_subjects  = "count",
    )
    .reset_index()
    .round({"mean_tsnr": 3, "median_tsnr": 3, "std_tsnr": 3})
)
summary.to_csv(OUT_SUMMARY, sep="\t", index=False)
print(f"Summary TSV written: {OUT_SUMMARY}  ({len(summary)} rows)")

# ── Analysis: spike ROIs vs non-spike ROIs ────────────────────────────────────
spike_df  = pd.read_csv(SPIKE_ROIS_TSV, sep="\t")
spike_set = set(spike_df["roi_id"].astype(int))

N_SPIKE     = len(spike_set)
N_NON_SPIKE = N_ROIS - N_SPIKE
N_RUNS      = long_df[["subject", "session"]].drop_duplicates().shape[0]

summary["is_spike"] = summary["roi_id"].isin(spike_set)
spike_tsnr     = summary.loc[ summary["is_spike"], "median_tsnr"]
non_spike_tsnr = summary.loc[~summary["is_spike"], "median_tsnr"]

med_spike     = spike_tsnr.median()
med_non_spike = non_spike_tsnr.median()
ratio         = med_spike / med_non_spike if med_non_spike != 0 else float("nan")

u_stat, p_val = stats.mannwhitneyu(
    spike_tsnr, non_spike_tsnr, alternative="less"
)

n_dropout = int((spike_tsnr < 30).sum())

print()
print("─── tSNR diagnosis ───")
print(f"N subjects × sessions: {N_RUNS}")
print(f"N ROIs total:          {N_ROIS}")
print(f"N spike ROIs:          {N_SPIKE}")
print(f"N non-spike ROIs:      {N_NON_SPIKE}")
print()
print("Median tSNR (across ROI medians):")
print(f"  Spike ROIs:          {med_spike:.1f}")
print(f"  Non-spike ROIs:      {med_non_spike:.1f}")
print(f"  Ratio:               {ratio:.2f}")
print()
print("Mann-Whitney U test (spike < non-spike, one-sided):")
print(f"  U = {u_stat:.0f}, p = {p_val:.2e}")
print()
print("Interpretation:")
print("  - tSNR < 30 indicates signal dropout (Murphy et al. 2007)")
print(f"  - Spike ROIs with median tSNR < 30: {n_dropout} of {N_SPIKE}")
if n_dropout > N_SPIKE // 2 and p_val < 0.05:
    conclusion = "SUPPORTS the hypothesis that τ ≈ 1 s reflects susceptibility dropout artifact."
elif p_val < 0.05:
    conclusion = "Partially supports the hypothesis (significant tSNR difference but fewer than half below threshold)."
else:
    conclusion = "DOES NOT support the tSNR dropout hypothesis (p ≥ 0.05)."
print(f"  - {conclusion}")
