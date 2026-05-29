#!/usr/bin/env python3
"""
08_extract_keskin_timeseries.py
Extract mean BOLD timeseries for Keskin self parcels (denoisedNoGSR only).

Analogous to 04_extract_nonself_timeseries.py but targets the Keskin sphere
atlas.  Only denoisedNoGSR is extracted — the version used for ACW analysis.

35 subjects × 2 sessions = 70 runs.

Outputs
-------
  02_timeseries_extraction/results/timeseries_keskin/denoisedNoGSR/
      sub-XX_ses-YY_keskin_timeseries.csv  (rows=timepoints, cols=Glasser IDs)
  02_timeseries_extraction/results/timeseries_keskin/_extraction_log.tsv

Run on SERVER:
  conda activate fmri
  cd /BICNAS2/group-northoff/jkokino/codes/har_med_codes
  python 02_timeseries_extraction/scripts/08_extract_keskin_timeseries.py
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

DENOISING_DIR = REPO_ROOT / "01_preprocessing" / "02_denoising" / "results"
ATLAS_PATH    = (REPO_ROOT / "02_timeseries_extraction" / "results" / "atlases"
                 / "keskin_self_atlas_native.nii.gz")
OUT_ROOT      = REPO_ROOT / "02_timeseries_extraction" / "results" / "timeseries_keskin"
LOG_PATH      = OUT_ROOT / "_extraction_log.tsv"

VERSION  = "denoisedNoGSR"
SUBJECTS = get_included_subjects()
SESSIONS = ["ses-01", "ses-02"]

LOG_COLS = [
    "subject", "session", "n_rois", "n_timepoints",
    "status", "elapsed_sec", "timestamp",
]


# ─── Helpers ─────────────────────────────────────────────────────────────────

def bold_path(sub, ses):
    return DENOISING_DIR / f"{sub}_{ses}_task-rest_desc-denoisedNoGSR_bold.nii.gz"


def out_csv_path(sub, ses):
    return OUT_ROOT / VERSION / f"{sub}_{ses}_keskin_timeseries.csv"


def extract_roi_timeseries(bold_file, atlas_file):
    """Return dict {roi_num (int): timeseries (1-D array)}."""
    bold_data = nib.load(str(bold_file)).get_fdata()   # (x, y, z, T)
    roi_atlas = nib.load(str(atlas_file)).get_fdata()  # (x, y, z)

    print(f"    BOLD shape: {bold_data.shape}   Atlas shape: {roi_atlas.shape}")

    roi_numbers = np.unique(roi_atlas)
    roi_numbers = roi_numbers[roi_numbers > 0]
    print(f"    Found {len(roi_numbers)} ROIs in atlas")

    roi_ts = {}
    for roi_num in roi_numbers:
        mask    = roi_atlas == roi_num
        n_vox   = int(np.sum(mask))
        if n_vox == 0:
            print(f"      Warning: ROI {int(roi_num)} has 0 voxels — skipped")
            continue
        roi_ts[int(roi_num)] = bold_data[mask].mean(axis=0)

    return roi_ts


def _append_log(row):
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=LOG_COLS, delimiter="\t")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not ATLAS_PATH.exists():
        sys.exit(
            f"Atlas not found: {ATLAS_PATH}\n"
            "Run 07_create_keskin_atlas.sh on the server first."
        )

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUT_ROOT / VERSION).mkdir(parents=True, exist_ok=True)

    runs    = [(sub, ses) for sub in SUBJECTS for ses in SESSIONS]
    n_total = len(runs)  # 35 × 2 = 70

    print(f"Keskin timeseries extraction ({VERSION}): "
          f"{len(SUBJECTS)} subjects × {len(SESSIONS)} sessions = {n_total} runs")
    print(f"Atlas:       {ATLAS_PATH}")
    print(f"Output root: {OUT_ROOT}\n")

    n_done    = 0
    n_skipped = 0
    n_failed  = 0
    failures  = []

    for idx, (sub, ses) in enumerate(runs, start=1):
        csv_out = out_csv_path(sub, ses)
        prefix  = f"[{idx:03d}/{n_total}] {sub} {ses}"

        if csv_out.exists():
            print(f"{prefix} ... SKIPPED")
            n_skipped += 1
            continue

        bold = bold_path(sub, ses)
        t0   = time.time()
        try:
            if not bold.exists():
                raise FileNotFoundError(f"BOLD not found: {bold}")

            print(f"{prefix} ...")
            roi_ts  = extract_roi_timeseries(bold, ATLAS_PATH)
            elapsed = time.time() - t0

            df = pd.DataFrame(roi_ts)
            df.index.name = "Timepoint"
            df.to_csv(str(csv_out))

            n_rois = len(roi_ts)
            n_tp   = int(next(iter(roi_ts.values())).shape[0]) if roi_ts else 0
            print(f"    Saved: {csv_out.name}  ({n_rois} ROIs, {n_tp} tp, {elapsed:.0f}s)")

            _append_log({"subject": sub, "session": ses,
                         "n_rois": n_rois, "n_timepoints": n_tp,
                         "status": "DONE", "elapsed_sec": f"{elapsed:.1f}",
                         "timestamp": datetime.datetime.now().isoformat(timespec="seconds")})
            n_done += 1

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"    ERROR: {exc}")
            _append_log({"subject": sub, "session": ses,
                         "n_rois": "", "n_timepoints": "",
                         "status": f"FAILED: {exc}",
                         "elapsed_sec": f"{elapsed:.1f}",
                         "timestamp": datetime.datetime.now().isoformat(timespec="seconds")})
            n_failed += 1
            failures.append(f"{sub}_{ses}: {exc}")

    print(f"\n─── Extraction summary ───")
    print(f"Total:    {n_total}")
    print(f"Done:     {n_done}")
    print(f"Skipped:  {n_skipped}")
    print(f"Failed:   {n_failed}")
    if failures:
        print("\nFailed runs:")
        for f in failures:
            print(f"  {f}")


if __name__ == "__main__":
    main()
