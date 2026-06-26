#!/usr/bin/env python3
"""
Parcel-based timeseries extraction — DMT-MED dataset (Keskin method).

Extracts mean BOLD timeseries per Glasser parcel for each atlas across all
35 included subjects × 2 sessions × 3 BOLD versions × 5 atlases = 1,050 CSVs.

Optimization: each BOLD is loaded once per (subject, session, version); all 5
atlas extractions share that loaded array, reducing BOLD loads from 1,050 → 210.

BOLD versions:
  raw            fMRIPrep preprocessed (no denoising)
  denoisedNoGSR  Denoised, no global signal regression
  denoisedGSR    Denoised, with global signal regression

Atlases (Glasser MMP1.0 parcels, native BOLD grid):
  self           all self-referential parcels (combined)
  nonself        all remaining Glasser parcels
  interoceptive  Keskin interoceptive self parcels
  exteroceptive  Keskin exteroceptive self parcels
  mental         Keskin mental-self parcels

Outputs
-------
  02_timeseries_extraction/results/timeseries_parcels/{atlas}/{version}/
      sub-XX_ses-YY_{atlas}_parcel_timeseries.csv  (rows=timepoints, cols=parcel IDs)
  02_timeseries_extraction/results/timeseries_parcels/_extraction_log.tsv

Run on server:
  conda activate fmri
  python 02_timeseries_extraction/scripts/parcels/02_extract_parcel_timeseries.py
"""

import gc
import sys
import csv
import time
import datetime
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from utils.subject_filter import get_included_subjects

# ─── Paths ────────────────────────────────────────────────────────────────────

FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
DENOISING_DIR = REPO_ROOT / "01_preprocessing" / "02_denoising" / "results"
ATLAS_DIR     = REPO_ROOT / "02_timeseries_extraction" / "results" / "atlases"

ATLASES = {
    "self":          ATLAS_DIR / "glasser_self_atlas_native.nii.gz",
    "nonself":       ATLAS_DIR / "glasser_nonself_atlas_native.nii.gz",
    "interoceptive": ATLAS_DIR / "glasser_self_interoceptive_native.nii.gz",
    "exteroceptive": ATLAS_DIR / "glasser_self_exteroceptive_native.nii.gz",
    "mental":        ATLAS_DIR / "glasser_self_mental_native.nii.gz",
}

OUT_ROOT = REPO_ROOT / "02_timeseries_extraction" / "results" / "timeseries_parcels"
LOG_PATH = OUT_ROOT / "_extraction_log.tsv"

SUBJECTS = get_included_subjects()   # 35 included subjects
SESSIONS = ["ses-01", "ses-02"]
VERSIONS = ["raw", "denoisedNoGSR", "denoisedGSR"]

LOG_COLS = [
    "subject", "session", "version", "atlas",
    "n_parcels", "n_timepoints", "status", "elapsed_sec", "timestamp",
]

N_TOTAL = len(SUBJECTS) * len(SESSIONS) * len(VERSIONS) * len(ATLASES)  # 1050


# ─── Path helpers ─────────────────────────────────────────────────────────────

def bold_path(sub, ses, version):
    if version == "raw":
        return (FMRIPREP_ROOT / sub / ses / "func"
                / f"{sub}_{ses}_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
    elif version == "denoisedNoGSR":
        return DENOISING_DIR / f"{sub}_{ses}_task-rest_desc-denoisedNoGSR_bold.nii.gz"
    else:
        return DENOISING_DIR / f"{sub}_{ses}_task-rest_desc-denoisedGSR_bold.nii.gz"


def out_csv_path(sub, ses, atlas_name, version):
    return OUT_ROOT / atlas_name / version / f"{sub}_{ses}_{atlas_name}_parcel_timeseries.csv"


# ─── Extraction ───────────────────────────────────────────────────────────────

def extract_parcel_timeseries(bold_data, atlas_path):
    """
    Extract mean BOLD timeseries per parcel.
    bold_data: pre-loaded 4-D array (x, y, z, T).
    Returns dict {parcel_id (int): timeseries (1-D array, length T)}.
    """
    roi_img   = nib.load(str(atlas_path))
    roi_atlas = roi_img.get_fdata()   # (x, y, z)

    print(f"      Atlas shape: {roi_atlas.shape}")

    parcel_ids = np.unique(roi_atlas)
    parcel_ids = parcel_ids[parcel_ids > 0]

    print(f"      Found {len(parcel_ids)} parcels")

    roi_timeseries = {}
    for pid in parcel_ids:
        mask     = (roi_atlas == pid)
        n_voxels = int(np.sum(mask))

        if n_voxels == 0:
            print(f"        Warning: parcel {int(pid)} has 0 voxels!")
            continue

        ts = bold_data[mask].mean(axis=0)
        roi_timeseries[int(pid)] = ts

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
    for atlas_name in ATLASES:
        for version in VERSIONS:
            (OUT_ROOT / atlas_name / version).mkdir(parents=True, exist_ok=True)

    missing = [name for name, p in ATLASES.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing atlases: {missing}\n"
            "Run 01_create_glasser_self_nonself.py first."
        )

    print(f"Parcel timeseries extraction: {len(SUBJECTS)} subjects × "
          f"{len(SESSIONS)} sessions × {len(VERSIONS)} versions × "
          f"{len(ATLASES)} atlases = {N_TOTAL} runs")
    print(f"Atlas dir:   {ATLAS_DIR}")
    print(f"Output root: {OUT_ROOT}\n")

    n_completed = 0
    n_skipped   = 0
    n_failed    = 0
    failed_runs = []
    run_index   = 0

    for sub in SUBJECTS:
        for ses in SESSIONS:
            for version in VERSIONS:
                bold      = bold_path(sub, ses, version)
                bold_data = None   # loaded on demand, shared across all 5 atlases

                for atlas_name, atlas_path in ATLASES.items():
                    run_index += 1
                    csv_out = out_csv_path(sub, ses, atlas_name, version)
                    prefix  = (f"[{run_index:04d}/{N_TOTAL}] "
                               f"{sub} {ses} {atlas_name} {version}")

                    if csv_out.exists():
                        print(f"{prefix} ... SKIPPED")
                        n_skipped += 1
                        continue

                    t0 = time.time()
                    try:
                        if not bold.exists():
                            raise FileNotFoundError(f"BOLD not found: {bold}")

                        if bold_data is None:
                            print(f"  Loading BOLD: {bold.name}")
                            bold_img  = nib.load(str(bold))
                            bold_data = bold_img.get_fdata()
                            print(f"    BOLD shape: {bold_data.shape}")

                        print(f"{prefix} ...")
                        parcel_ts = extract_parcel_timeseries(bold_data, atlas_path)

                        df = pd.DataFrame(parcel_ts)
                        df.index.name = "Timepoint"
                        df.to_csv(str(csv_out))

                        elapsed   = time.time() - t0
                        n_parcels = len(parcel_ts)
                        n_tp      = int(next(iter(parcel_ts.values())).shape[0]) if parcel_ts else 0
                        print(f"      Saved: {csv_out.name}  "
                              f"({n_parcels} parcels, {n_tp} timepoints, {elapsed:.0f}s)")

                        _append_log({
                            "subject":      sub,
                            "session":      ses,
                            "version":      version,
                            "atlas":        atlas_name,
                            "n_parcels":    n_parcels,
                            "n_timepoints": n_tp,
                            "status":       "DONE",
                            "elapsed_sec":  f"{elapsed:.1f}",
                            "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
                        })
                        n_completed += 1

                    except Exception as exc:
                        elapsed = time.time() - t0
                        print(f"      ERROR: {exc}")
                        _append_log({
                            "subject":      sub,
                            "session":      ses,
                            "version":      version,
                            "atlas":        atlas_name,
                            "n_parcels":    "",
                            "n_timepoints": "",
                            "status":       f"FAILED: {exc}",
                            "elapsed_sec":  f"{elapsed:.1f}",
                            "timestamp":    datetime.datetime.now().isoformat(timespec="seconds"),
                        })
                        n_failed += 1
                        failed_runs.append(f"{sub}_{ses}_{atlas_name}_{version}: {exc}")

                # free BOLD memory before loading next version
                if bold_data is not None:
                    del bold_data
                    gc.collect()

    print(f"\n─── Extraction summary ───")
    print(f"Total runs attempted:  {N_TOTAL}")
    print(f"Completed:             {n_completed}")
    print(f"Skipped:               {n_skipped}")
    print(f"Failed:                {n_failed}")
    if failed_runs:
        print("\nFailed runs:")
        for r in failed_runs:
            print(f"  {r}")


if __name__ == "__main__":
    main()
