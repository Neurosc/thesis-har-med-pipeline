#!/usr/bin/env python3
"""
Batch denoising pipeline for all included subjects (35 × 2 sessions = 70 runs).

Loops over all included subjects (sub-06, sub-08, sub-12, sub-26, sub-36 excluded)
and both sessions, calling denoise_run() from denoise_core.py for each run.

Usage (on server):
  conda activate fmri
  python 01_preprocessing/02_denoising/scripts/denoise_batch.py

Outputs
-------
  01_preprocessing/02_denoising/results/sub-XX_ses-YY_task-rest_desc-denoisedGSR_bold.nii.gz
  01_preprocessing/02_denoising/results/sub-XX_ses-YY_task-rest_desc-denoisedNoGSR_bold.nii.gz
  01_preprocessing/02_denoising/results/_batch_log.tsv   (appended after each run)

Existing outputs are skipped (both GSR and NoGSR must exist to skip).
"""

import sys
import time
import datetime
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from denoise_core import (
    denoise_run, build_output_paths,
    TR, FD_THRESH, BP_LOW, BP_HIGH, LS_OVERSAMPLE_FAC,
)
from utils.subject_filter import get_included_subjects

# ─── Paths ────────────────────────────────────────────────────────────────────

FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
OUT_DIR  = Path(__file__).resolve().parents[1] / "results"
LOG_PATH = OUT_DIR / "_batch_log.tsv"

SESSIONS = ["ses-01", "ses-02"]

LOG_COLS = [
    "subject", "session", "n_frames", "n_censored", "pct_censored",
    "post_std_GSR", "post_std_NoGSR", "status", "elapsed_seconds", "timestamp",
]


# ─── Log helper ──────────────────────────────────────────────────────────────

def _append_log(row: dict) -> None:
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Override subject list — set to None for the standard 35-subject run.
    # Currently set to the four previously excluded subjects (sub-06, sub-08,
    # sub-26, sub-36) to generate their denoised BOLDs. sub-12 stays excluded.
    # Existing outputs are skipped automatically. Reset to None when done.
    SUBJECTS_OVERRIDE = ["sub-06", "sub-08", "sub-26", "sub-36"]

    subjects = SUBJECTS_OVERRIDE if SUBJECTS_OVERRIDE is not None else get_included_subjects()
    runs     = [(s, ses) for s in subjects for ses in SESSIONS]
    n_total  = len(runs)

    print(f"Batch denoising: {len(subjects)} subjects × {len(SESSIONS)} sessions "
          f"= {n_total} runs")
    print(f"Parameters: TR={TR}s  FD>{FD_THRESH}mm  BP={BP_LOW}–{BP_HIGH}Hz  "
          f"LS_ofac={LS_OVERSAMPLE_FAC}")
    print(f"Output dir: {OUT_DIR}")
    print(f"Log:        {LOG_PATH}\n")

    n_completed = 0
    n_skipped   = 0
    n_failed    = 0
    failed_runs = []
    batch_start = time.time()

    for run_idx, (subject, session) in enumerate(runs, start=1):
        out = build_output_paths(subject, session, OUT_DIR)
        prefix = f"[{run_idx:02d}/{n_total}] {subject} {session}"

        # Skip if both outputs already exist
        if out["gsr"].exists() and out["nogsr"].exists():
            print(f"{prefix} ... SKIPPED (outputs exist)")
            n_skipped += 1
            continue

        t0 = time.time()
        try:
            result = denoise_run(
                subject       = subject,
                session       = session,
                fmriprep_root = FMRIPREP_ROOT,
                output_dir    = OUT_DIR,
                verbose       = False,
            )
            elapsed = time.time() - t0
            print(f"{prefix} ... DONE in {elapsed:.0f}s  "
                  f"censored={result['n_censored']}/{result['n_frames']} "
                  f"({result['pct_censored']:.1f}%)  "
                  f"std+GSR={result['post_std_GSR']:.2f}  "
                  f"std-GSR={result['post_std_NoGSR']:.2f}")
            _append_log({
                "subject":        subject,
                "session":        session,
                "n_frames":       result["n_frames"],
                "n_censored":     result["n_censored"],
                "pct_censored":   f"{result['pct_censored']:.2f}",
                "post_std_GSR":   f"{result['post_std_GSR']:.4f}",
                "post_std_NoGSR": f"{result['post_std_NoGSR']:.4f}",
                "status":         "DONE",
                "elapsed_seconds": f"{elapsed:.1f}",
                "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
            })
            n_completed += 1

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"{prefix} ... FAILED in {elapsed:.0f}s  ERROR: {exc}")
            _append_log({
                "subject":        subject,
                "session":        session,
                "n_frames":       "",
                "n_censored":     "",
                "pct_censored":   "",
                "post_std_GSR":   "",
                "post_std_NoGSR": "",
                "status":         f"FAILED: {exc}",
                "elapsed_seconds": f"{elapsed:.1f}",
                "timestamp":      datetime.datetime.now().isoformat(timespec="seconds"),
            })
            n_failed += 1
            failed_runs.append(f"{subject} {session}")

    total_elapsed = time.time() - batch_start
    h = int(total_elapsed // 3600)
    m = int((total_elapsed % 3600) // 60)

    print(f"\n{'─'*55}")
    print(f"Batch complete: {n_completed} completed, {n_skipped} skipped, "
          f"{n_failed} failed  /  Total elapsed: {h}h {m}m")
    if failed_runs:
        print(f"Failed runs:")
        for r in failed_runs:
            print(f"  {r}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
