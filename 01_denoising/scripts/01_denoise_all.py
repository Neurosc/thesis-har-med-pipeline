#!/usr/bin/env python3
"""
Batch denoising — runs ONE of the two pipelines over the n=39 sample.

Two pipelines (select with --pipeline): detrend | maximal. All share the
same sample of 39 subjects (all minus sub-12) × 2 sessions = 78 runs, so the only
thing that differs between them is the denoising. See denoise_pipelines.PIPELINE_PRESETS.
(maximal_nocensor is an optional control preset, not part of the main flow.)

Usage (on server):
  conda activate fmri
  python 01_denoising/scripts/01_denoise_all.py --pipeline detrend
  python 01_denoising/scripts/01_denoise_all.py --pipeline maximal

Outputs (one self-identifying folder per pipeline)
-------
  01_denoising/results/<pipeline>/
      sub-XX_ses-YY_task-rest_desc-<pipeline>_bold.nii.gz
      _batch_log.tsv   (appended after each run)

Existing outputs are skipped — but only if complete. A file that is present yet
short (an interrupted earlier run) is deleted and re-denoised, so re-running this
script is always safe and always converges on a full set.
"""

import sys
import time
import argparse
import datetime
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from denoise_pipelines import (
    denoise_run, build_output_path, PIPELINE_PRESETS,
    TR, FD_THRESH, BP_LOW, BP_HIGH, LS_OVERSAMPLE_FAC,
)
from utils.subject_filter import get_pipeline_subjects
from utils.nifti_check import is_complete

# ─── Paths ────────────────────────────────────────────────────────────────────

FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

SESSIONS = ["ses-01", "ses-02"]

LOG_COLS = [
    "subject", "session", "pipeline", "n_frames", "n_censored", "pct_censored",
    "post_std", "status", "elapsed_seconds", "timestamp",
]


# ─── Log helper ──────────────────────────────────────────────────────────────

def _append_log(log_path: Path, row: dict) -> None:
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", required=True, choices=list(PIPELINE_PRESETS),
                        help="Which denoising pipeline to run.")
    args = parser.parse_args()
    pipeline = args.pipeline

    out_dir  = RESULTS_ROOT / pipeline
    log_path = out_dir / "_batch_log.tsv"
    out_dir.mkdir(parents=True, exist_ok=True)

    subjects = get_pipeline_subjects()          # n = 39 (all minus sub-12)
    runs     = [(s, ses) for s in subjects for ses in SESSIONS]
    n_total  = len(runs)
    desc     = PIPELINE_PRESETS[pipeline]["desc"]

    print(f"Batch denoising — pipeline '{pipeline}'")
    print(f"Sample: {len(subjects)} subjects × {len(SESSIONS)} sessions = {n_total} runs")
    print(f"Steps: {PIPELINE_PRESETS[pipeline]}")
    print(f"Parameters: TR={TR}s  FD>{FD_THRESH}mm  BP={BP_LOW}–{BP_HIGH}Hz  "
          f"LS_ofac={LS_OVERSAMPLE_FAC}")
    print(f"Output dir: {out_dir}")
    print(f"Log:        {log_path}\n")

    n_completed = 0
    n_skipped   = 0
    n_failed    = 0
    failed_runs = []
    batch_start = time.time()

    for run_idx, (subject, session) in enumerate(runs, start=1):
        out_path = build_output_path(subject, session, out_dir, desc)
        prefix   = f"[{run_idx:02d}/{n_total}] {subject} {session}"

        if is_complete(out_path):
            print(f"{prefix} ... SKIPPED (output complete)")
            n_skipped += 1
            continue

        if out_path.exists():
            # Present but short — a leftover from an interrupted run. Redo it.
            print(f"{prefix} ... INCOMPLETE output, re-denoising")
            out_path.unlink()

        t0 = time.time()
        try:
            result = denoise_run(
                subject       = subject,
                session       = session,
                fmriprep_root = FMRIPREP_ROOT,
                output_dir    = out_dir,
                pipeline      = pipeline,
                verbose       = False,
            )
            elapsed = time.time() - t0
            print(f"{prefix} ... DONE in {elapsed:.0f}s  "
                  f"censored={result['n_censored']}/{result['n_frames']} "
                  f"({result['pct_censored']:.1f}%)  "
                  f"post_std={result['post_std']:.2f}")
            _append_log(log_path, {
                "subject":         subject,
                "session":         session,
                "pipeline":        pipeline,
                "n_frames":        result["n_frames"],
                "n_censored":      result["n_censored"],
                "pct_censored":    f"{result['pct_censored']:.2f}",
                "post_std":        f"{result['post_std']:.4f}",
                "status":          "DONE",
                "elapsed_seconds": f"{elapsed:.1f}",
                "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
            })
            n_completed += 1

        except Exception as exc:
            elapsed = time.time() - t0
            print(f"{prefix} ... FAILED in {elapsed:.0f}s  ERROR: {exc}")
            _append_log(log_path, {
                "subject":         subject,
                "session":         session,
                "pipeline":        pipeline,
                "n_frames":        "",
                "n_censored":      "",
                "pct_censored":    "",
                "post_std":        "",
                "status":          f"FAILED: {exc}",
                "elapsed_seconds": f"{elapsed:.1f}",
                "timestamp":       datetime.datetime.now().isoformat(timespec="seconds"),
            })
            n_failed += 1
            failed_runs.append(f"{subject} {session}")

    total_elapsed = time.time() - batch_start
    h = int(total_elapsed // 3600)
    m = int((total_elapsed % 3600) // 60)

    print(f"\n{'─'*55}")
    print(f"Pipeline '{pipeline}' complete: {n_completed} completed, "
          f"{n_skipped} skipped, {n_failed} failed  /  Total elapsed: {h}h {m}m")
    if failed_runs:
        print(f"Failed runs:")
        for r in failed_runs:
            print(f"  {r}")
    print(f"{'─'*55}")


if __name__ == "__main__":
    main()
