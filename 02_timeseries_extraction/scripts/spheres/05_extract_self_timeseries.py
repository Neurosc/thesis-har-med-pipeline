#!/usr/bin/env python3
"""
Self ROI timeseries extraction — DMT-MED dataset.

Extracts mean BOLD timeseries for each self-referential ROI (Qin et al. 2020)
across all 35 included subjects × 2 sessions × 3 BOLD versions = 210 runs.

BOLD versions:
  raw          fMRIPrep preprocessed (no denoising)
  denoisedNoGSR  Denoised, no global signal regression
  denoisedGSR    Denoised, with global signal regression

Atlas: self_atlas_native.nii.gz (37 ROIs, Qin et al. 2020,
       resampled to native BOLD grid 1.72x1.72x2.00mm)

Outputs
-------
  02_timeseries_extraction/results/timeseries_self/{version}/
      sub-XX_ses-YY_self_timeseries.csv  (rows=timepoints, cols=ROI numbers)
  02_timeseries_extraction/results/timeseries_self/_extraction_log.tsv

Run on server:
  conda activate fmri
  python 02_timeseries_extraction/scripts/05_extract_self_timeseries.py
"""

import sys
import csv
import time
import datetime
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.subject_filter import get_included_subjects

# ─── Paths ────────────────────────────────────────────────────────────────────

FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
DENOISING_DIR = REPO_ROOT / "01_preprocessing" / "02_denoising" / "results"
ATLAS_PATH    = (REPO_ROOT / "02_timeseries_extraction" / "results" / "atlases"
                 / "self_atlas_native.nii.gz")
OUT_ROOT      = REPO_ROOT / "02_timeseries_extraction" / "results" / "timeseries_self"
LOG_PATH      = OUT_ROOT / "_extraction_log.tsv"

SUBJECTS = get_included_subjects()          # 35 included subjects
SESSIONS = ["ses-01", "ses-02"]
VERSIONS = ["raw", "denoisedNoGSR", "denoisedGSR"]

LOG_COLS = [
    "subject", "session", "version",
    "n_rois", "n_timepoints", "status", "elapsed_sec", "timestamp",
]


# ─── Path helpers ─────────────────────────────────────────────────────────────

def bold_path(sub, ses, version):
    if version == "raw":
        return (FMRIPREP_ROOT / sub / ses / "func"
                / f"{sub}_{ses}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    elif version == "denoisedNoGSR":
        return DENOISING_DIR / f"{sub}_{ses}_task-rest_desc-denoisedNoGSR_bold.nii.gz"
    else:  # denoisedGSR
        return DENOISING_DIR / f"{sub}_{ses}_task-rest_desc-denoisedGSR_bold.nii.gz"


def out_csv_path(sub, ses, version):
    return OUT_ROOT / version / f"{sub}_{ses}_self_timeseries.csv"


# ─── Extraction ───────────────────────────────────────────────────────────────

def extract_roi_timeseries(bold_file, roi_atlas_file):
    """
    Extract timeseries for each ROI from the atlas.
    Returns dict {roi_num (int): timeseries (1-D array, length T)}.
    """
    bold_img  = nib.load(str(bold_file))
    bold_data = bold_img.get_fdata()   # (x, y, z, T)

    print(f"    BOLD shape: {bold_data.shape}")

    roi_img   = nib.load(str(roi_atlas_file))
    roi_atlas = roi_img.get_fdata()    # (x, y, z)

    print(f"    ROI atlas shape: {roi_atlas.shape}")

    roi_numbers = np.unique(roi_atlas)
    roi_numbers = roi_numbers[roi_numbers > 0]  # exclude background

    print(f"    Found {len(roi_numbers)} ROIs")

    roi_timeseries = {}
    for roi_num in roi_numbers:
        roi_mask = (roi_atlas == roi_num)
        n_voxels = int(np.sum(roi_mask))

        if n_voxels == 0:
            print(f"      Warning: ROI {int(roi_num)} has 0 voxels!")
            continue

        timeseries = bold_data[roi_mask].mean(axis=0)
        roi_timeseries[int(roi_num)] = timeseries

        if int(roi_num) <= 3:
            print(f"      ROI {int(roi_num)}: {n_voxels} voxels, "
                  f"{len(timeseries)} timepoints")

    return roi_timeseries


# ─── Log helper ───────────────────────────────────────────────────────────────

def _append_log(row):
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    for version in VERSIONS:
        (OUT_ROOT / version).mkdir(parents=True, exist_ok=True)

    if not ATLAS_PATH.exists():
        raise FileNotFoundError(
            f"Atlas not found: {ATLAS_PATH}\n"
            "Run 01_create_self_atlas.sh first."
        )

    runs    = [(sub, ses, ver) for sub in SUBJECTS for ses in SESSIONS for ver in VERSIONS]
    n_total = len(runs)   # 35 × 2 × 3 = 210

    print(f"Self timeseries extraction: {len(SUBJECTS)} subjects × "
          f"{len(SESSIONS)} sessions × {len(VERSIONS)} versions = {n_total} runs")
    print(f"Atlas:       {ATLAS_PATH}")
    print(f"Output root: {OUT_ROOT}\n")

    n_completed = 0
    n_skipped   = 0
    n_failed    = 0
    failed_runs = []

    for run_idx, (sub, ses, version) in enumerate(runs, start=1):
        csv_out = out_csv_path(sub, ses, version)
        prefix  = f"[{run_idx:03d}/{n_total}] {sub} {ses} {version}"

        if csv_out.exists():
            print(f"{prefix} ... SKIPPED")
            n_skipped += 1
            continue

        bold = bold_path(sub, ses, version)
        t0   = time.time()
        try:
            if not bold.exists():
                raise FileNotFoundError(f"BOLD not found: {bold}")

            print(f"{prefix} ...")
            roi_ts = extract_roi_timeseries(bold, ATLAS_PATH)

            df = pd.DataFrame(roi_ts)
            df.index.name = "Timepoint"
            df.to_csv(str(csv_out))

            elapsed = time.time() - t0
            n_rois  = len(roi_ts)
            n_tp    = int(next(iter(roi_ts.values())).shape[0]) if roi_ts else 0
            print(f"    Saved: {csv_out.name}  "
                  f"({n_rois} ROIs, {n_tp} timepoints, {elapsed:.0f}s)")

            _append_log({
                "subject":      sub,
                "session":      ses,
                "version":      version,
                "n_rois":       n_rois,
                "n_timepoints": n_tp,
                "status":       "DONE",
                "elapsed_sec":  f"{elapsed:.1f}",
                "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
            })
            n_completed += 1

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"    ERROR: {exc}")
            _append_log({
                "subject":      sub,
                "session":      ses,
                "version":      version,
                "n_rois":       "",
                "n_timepoints": "",
                "status":       f"FAILED: {exc}",
                "elapsed_sec":  f"{elapsed:.1f}",
                "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
            })
            n_failed += 1
            failed_runs.append(f"{sub}_{ses}_{version}: {exc}")

    print(f"\n─── Extraction summary ───")
    print(f"Total runs attempted:    {n_total}")
    print(f"Completed:               {n_completed}")
    print(f"Skipped (already exist): {n_skipped}")
    print(f"Failed:                  {n_failed}")
    if failed_runs:
        print("\nFailed runs:")
        for r in failed_runs:
            print(f"  {r}")


if __name__ == "__main__":
    main()
