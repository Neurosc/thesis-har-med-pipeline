#!/usr/bin/env python3
"""
Single-subject denoising pipeline for DMT-MED fMRI data.
Subject: sub-01, ses-01

Implements Goldberg et al. (2024) strategy:
  - Single GLM with WM, CSF, 6 motion, 6 derivatives, spike regressors (+/- GSR)
  - Lomb-Scargle interpolation + bandpass: faithful Python port of CBIG_preproc_censor.m
    (Jingwei Li, Yeo Lab; Power et al. 2014 Eq. 3-4 supplementary).
    Bandpass 0.01-0.1 Hz applied via frequency-domain masking inside LS (no Butterworth).

Two output versions (both include spike regressors + interpolation):
  +GSR +censor : desc-denoisedGSR_bold.nii.gz
  -GSR +censor : desc-denoisedNoGSR_bold.nii.gz

Core pipeline logic lives in denoise_core.py (shared with denoise_batch.py).
"""

import sys
import numpy as np
import nibabel as nib
from pathlib import Path

# ─── Repo / path setup ────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from denoise_core import (
    denoise_run, sanity_check,
    TR, FD_THRESH, BP_LOW, BP_HIGH, LS_OVERSAMPLE_FAC,
)

# ─── Hardcoded paths for sub-01 ses-01 ───────────────────────────────────────

FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
OUT_DIR = Path(__file__).resolve().parents[1] / "results"

SUBJECT = "sub-01"
SESSION = "ses-01"


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    result = denoise_run(
        subject       = SUBJECT,
        session       = SESSION,
        fmriprep_root = FMRIPREP_ROOT,
        output_dir    = OUT_DIR,
        verbose       = True,
    )

    n_base = 14
    summary = (
        f"\n─── Denoising summary: {SUBJECT}_{SESSION} ───\n"
        f"Total frames:            {result['n_frames']}\n"
        f"High-motion censored:    {result['n_censored']}"
        f" ({result['pct_censored']:.1f}%)\n"
        f"Regressors (no GSR):     {n_base} + {result['n_censored']} spikes\n"
        f"Regressors (+GSR):       {n_base + 1} + {result['n_censored']} spikes\n"
        f"Bandpass:                {BP_LOW}–{BP_HIGH} Hz via LS freq mask"
        f" (ofac={LS_OVERSAMPLE_FAC})\n"
        f"Post-denoise std (+GSR): {result['post_std_GSR']:.2f}\n"
        f"Post-denoise std (-GSR): {result['post_std_NoGSR']:.2f}\n"
        f"Output: {result['gsr_path']}\n"
        f"Output: {result['nogsr_path']}\n"
    )
    print(summary)

    log_path = OUT_DIR / f"{SUBJECT}_{SESSION}_denoising_log.txt"
    log_path.write_text(summary)

    # Sanity checks on saved outputs (reload from disk to free memory during processing)
    print("Reloading outputs for sanity check...")
    mask_path = (FMRIPREP_ROOT / SUBJECT / SESSION / "func"
                 / f"{SUBJECT}_{SESSION}_task-rest_space-MNI152NLin2009cAsym"
                   "_desc-brain_mask.nii.gz")
    mask_flat = nib.load(mask_path).get_fdata().astype(bool).ravel()

    gsr_data   = nib.load(result["gsr_path"]).get_fdata(dtype=np.float32)
    nogsr_data = nib.load(result["nogsr_path"]).get_fdata(dtype=np.float32)
    pre_std    = result["post_std_NoGSR"]  # use NoGSR as rough pre-std proxy

    sanity_check(f"{SUBJECT} {SESSION} +GSR",  gsr_data,   pre_std, mask_flat)
    sanity_check(f"{SUBJECT} {SESSION} -GSR",  nogsr_data, pre_std, mask_flat)


if __name__ == "__main__":
    main()
