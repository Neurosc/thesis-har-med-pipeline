#!/usr/bin/env python3
"""
Single-subject denoising test (sub-01, ses-01) for one pipeline.

Runs one of the three pipelines (detrend | glm | maximal) on sub-01 ses-01 with
verbose stage diagnostics + sanity checks. Use to eyeball a pipeline before the
full 78-run batch. Core logic lives in denoise_core.py.

Usage (on server):
  conda activate fmri
  python 01_denoising/scripts/denoise_single_subject.py --pipeline maximal
"""

import sys
import argparse
import numpy as np
import nibabel as nib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from denoise_core import (
    denoise_run, sanity_check, PIPELINE_PRESETS,
    TR, FD_THRESH, BP_LOW, BP_HIGH, LS_OVERSAMPLE_FAC,
)

# ─── Hardcoded paths for sub-01 ses-01 ───────────────────────────────────────

FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"

SUBJECT = "sub-01"
SESSION = "ses-01"


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pipeline", default="maximal", choices=list(PIPELINE_PRESETS),
                        help="Which denoising pipeline to test (default: maximal).")
    args = parser.parse_args()
    pipeline = args.pipeline
    out_dir  = RESULTS_ROOT / pipeline

    result = denoise_run(
        subject       = SUBJECT,
        session       = SESSION,
        fmriprep_root = FMRIPREP_ROOT,
        output_dir    = out_dir,
        pipeline      = pipeline,
        verbose       = True,
    )

    cfg = PIPELINE_PRESETS[pipeline]
    summary = (
        f"\n─── Denoising summary: {SUBJECT}_{SESSION} | pipeline '{pipeline}' ───\n"
        f"Steps:                   {cfg}\n"
        f"Total frames:            {result['n_frames']}\n"
        f"High-motion frames:      {result['n_censored']} "
        f"({result['pct_censored']:.1f}%)"
        f"{'  [censored]' if cfg['do_censor'] else '  [NOT censored — kept]'}\n"
        f"Bandpass:                "
        f"{f'{BP_LOW}-{BP_HIGH} Hz via LS freq mask (ofac={LS_OVERSAMPLE_FAC})' if cfg['do_ls'] else 'none'}\n"
        f"Post-denoise std:        {result['post_std']:.2f}\n"
        f"Output: {result['out_path']}\n"
    )
    print(summary)

    log_path = out_dir / f"{SUBJECT}_{SESSION}_denoising_log.txt"
    log_path.write_text(summary)

    # Sanity check on the saved output (reload from disk)
    print("Reloading output for sanity check...")
    mask_path = (FMRIPREP_ROOT / SUBJECT / SESSION / "func"
                 / f"{SUBJECT}_{SESSION}_task-rest_space-MNI152NLin2009cAsym"
                   "_desc-brain_mask.nii.gz")
    mask_flat = nib.load(mask_path).get_fdata().astype(bool).ravel()

    out_data = nib.load(result["out_path"]).get_fdata(dtype=np.float32)
    sanity_check(f"{SUBJECT} {SESSION} {pipeline}", out_data, mask_flat)


if __name__ == "__main__":
    main()
