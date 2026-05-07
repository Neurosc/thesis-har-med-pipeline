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
"""

import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent

BOLD_PATH = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
    "/sub-01/ses-01/func"
    "/sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
)
MASK_PATH = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
    "/sub-01/ses-01/func"
    "/sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
)
CONFOUNDS_PATH = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
    "/sub-01/ses-01/func"
    "/sub-01_ses-01_task-rest_desc-confounds_timeseries.tsv.gz"
)

OUT_DIR = Path(__file__).resolve().parent / "results"

# ─── Parameters ───────────────────────────────────────────────────────────────

TR        = 1.8   # seconds (verified from BOLD JSON header, NOT 1.5)
FD_THRESH = 0.3   # mm — Goldberg et al. 2024 recommendation for ACW
BP_LOW    = 0.01  # Hz
BP_HIGH   = 0.1   # Hz

SUBJECT = "sub-01"
SESSION = "ses-01"
TASK    = "task-rest"

LS_OVERSAMPLE_FAC = 8
LS_BATCH_SIZE     = 5000   # voxels per batch to avoid OOM

# fMRIPrep confound column names
WM_CSF_COLS = ["white_matter", "csf"]
MOTION_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
DERIV_COLS  = [
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1",   "rot_y_derivative1",   "rot_z_derivative1",
]
GSR_COL = "global_signal"


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _detrend(X: np.ndarray) -> np.ndarray:
    """Remove mean + linear trend from each column of X (column = one time series)."""
    n = X.shape[0]
    t = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    D = np.column_stack([np.ones(n), t])
    betas, _, _, _ = np.linalg.lstsq(D, X, rcond=None)
    return X - D @ betas


def _standardize(X: np.ndarray) -> np.ndarray:
    """Z-score each column; constant columns kept as-is."""
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (X - mu) / sd


def _build_regressors(
    confounds_df: pd.DataFrame,
    spike_idx: np.ndarray,
    include_gsr: bool,
) -> np.ndarray:
    """
    Assemble (n_times, n_regressors) nuisance matrix.
    NaNs replaced with 0 before detrending / standardization.
    """
    cols = WM_CSF_COLS + MOTION_COLS + DERIV_COLS
    if include_gsr:
        cols = cols + [GSR_COL]

    X = confounds_df[cols].to_numpy(dtype=np.float64)
    X = np.where(np.isnan(X), 0.0, X)

    n_times = X.shape[0]
    if len(spike_idx) > 0:
        spikes = np.zeros((n_times, len(spike_idx)), dtype=np.float64)
        for k, frame in enumerate(spike_idx):
            spikes[frame, k] = 1.0
        X = np.hstack([X, spikes])

    return X


def _cbig_lomb_scargle_bandpass(
    in_series: np.ndarray,       # (num_in_time, nChannel) — good-frame data
    in_sample_time: np.ndarray,  # (num_in_time,) — times of good frames
    out_sample_time: np.ndarray, # (num_out_time,) — times of all frames
    outliers: np.ndarray,        # (num_out_time,) int — 1=censored, 0=good
    low_f: float = 0.01,
    high_f: float = 0.1,
    oversample_fac: int = 8,
) -> tuple:
    """
    Faithful Python port of CBIG_preproc_censor.m (Jingwei Li, Yeo Lab CBIG).

    Lomb-Scargle interpolation of censored frames with simultaneous bandpass
    filtering via frequency-domain masking (Power et al. 2014, Eq. 3-4 suppl.).

    Frequency grid:  f = 1/(T*ofac) : 1/(T*ofac) : N_uncen/(2*T)
    Phase shift tau: makes cos/sin orthogonal on the irregular time grid.
    Coefficients:    diagonal projection (Eq. 3) — no cross-frequency coupling.
    Reconstruction:  Eq. 4; bandpass by zeroing out-of-band coefficients.
    Std correction:  stdfac = input_std / output_std at good frames, applied twice.

    Returns
    -------
    out_bp     : (num_out_time, nChannel) — bandpass-filtered reconstruction
    interm_out : (num_out_time, nChannel) — full-spectrum reconstruction
    """
    num_in_time, nChannel = in_series.shape

    # Mean-subtract (MATLAB: input_mean = mean(in_series,1))
    input_mean = in_series.mean(axis=0)              # (nChannel,)
    # N-normalised std to match MATLAB std(x,1)
    input_std  = in_series.std(axis=0, ddof=0)       # (nChannel,)
    data       = in_series - input_mean               # (num_in_time, nChannel)

    # Frequency grid (MATLAB colon: df : df : N/(2*T))
    T      = in_sample_time.max() - in_sample_time.min()
    df     = 1.0 / (T * oversample_fac)
    # n_freq = floor(N_uncen * ofac / 2) — exact integer equivalent
    n_freq = int(num_in_time * oversample_fac / 2)
    f      = np.arange(1, n_freq + 1) * df           # (num_freq_bin,)
    w      = 2.0 * np.pi * f                          # (num_freq_bin,)

    # Phase shift tau per frequency (Scargle 1982)
    # tau = atan2(sum(sin(2wt)), sum(cos(2wt))) / (2w)
    phase_2w = 2.0 * w[:, None] * in_sample_time[None, :]   # (num_freq_bin, num_in_time)
    tau = (np.arctan2(np.sin(phase_2w).sum(axis=1),
                      np.cos(phase_2w).sum(axis=1))
           / (2.0 * w))                                      # (num_freq_bin,)

    # Sinusoidal basis at input (good) times
    arg_in = w[:, None] * (in_sample_time[None, :] - tau[:, None])  # (num_freq_bin, num_in_time)
    cterm  = np.cos(arg_in)   # (num_freq_bin, num_in_time)
    sterm  = np.sin(arg_in)   # (num_freq_bin, num_in_time)

    # Eq. 3 — diagonal projection coefficients
    cos_denom = (cterm ** 2).sum(axis=1, keepdims=True)   # (num_freq_bin, 1)
    sin_denom = (sterm ** 2).sum(axis=1, keepdims=True)   # (num_freq_bin, 1)
    cos_denom = np.where(cos_denom == 0.0, 1.0, cos_denom)
    sin_denom = np.where(sin_denom == 0.0, 1.0, sin_denom)

    # (num_freq_bin, num_in_time) @ (num_in_time, nChannel) = (num_freq_bin, nChannel)
    cos_coeff = (cterm @ data) / cos_denom   # (num_freq_bin, nChannel)
    sin_coeff = (sterm @ data) / sin_denom   # (num_freq_bin, nChannel)
    del cterm, sterm, arg_in, phase_2w

    # Sinusoidal basis at output (all) times
    arg_out   = w[:, None] * (out_sample_time[None, :] - tau[:, None])  # (num_freq_bin, num_out_time)
    cterm_new = np.cos(arg_out)   # (num_freq_bin, num_out_time)
    sterm_new = np.sin(arg_out)   # (num_freq_bin, num_out_time)
    del arg_out

    # Eq. 4 — full-spectrum reconstruction
    # (num_out_time, num_freq_bin) @ (num_freq_bin, nChannel) = (num_out_time, nChannel)
    interm_out = cterm_new.T @ cos_coeff + sterm_new.T @ sin_coeff   # (num_out_time, nChannel)

    # Std correction at good frames (stdfac = input_std / output_std)
    good_mask = (outliers == 0)
    out_std   = interm_out[good_mask].std(axis=0, ddof=0)
    out_std   = np.where(out_std == 0.0, 1.0, out_std)
    interm_out = interm_out * (input_std / out_std)[None, :]

    # Bandpass: reconstruct using only in-band coefficients (zero out rest)
    fpass        = ~((f < low_f) | (f > high_f))               # (num_freq_bin,) True = passband
    out_bp       = (cterm_new[fpass].T @ cos_coeff[fpass]
                    + sterm_new[fpass].T @ sin_coeff[fpass])    # (num_out_time, nChannel)

    out_std_bp   = out_bp[good_mask].std(axis=0, ddof=0)
    out_std_bp   = np.where(out_std_bp == 0.0, 1.0, out_std_bp)
    out_bp       = out_bp * (input_std / out_std_bp)[None, :]

    return out_bp, interm_out   # both (num_out_time, nChannel)


# ─── Sanity check helper ─────────────────────────────────────────────────────

_RED   = "\033[91m"
_RESET = "\033[0m"


def _sanity_check(
    label: str,
    post_data: np.ndarray,
    pre_std: float,
    mask_flat: np.ndarray,
) -> None:
    """
    Print a structured sanity-check block for one denoised output.

    Checks (all within brain mask):
      1. NaN / Inf values
      2. Std not exploded (post > 10x pre)
      3. No voxels with zero variance
      4. Mean approximately zero (detrending succeeded)
    """
    n_times  = post_data.shape[3]
    masked   = post_data.reshape(-1, n_times).astype(np.float64)[mask_flat]
    post_std = float(np.std(masked))
    abs_mean = float(np.abs(np.mean(masked)))

    checks = {}
    checks["nan_inf"]      = bool(np.isnan(masked).any() or np.isinf(masked).any())
    checks["std_exploded"] = post_std > 10 * pre_std
    checks["n_zero_var"]   = int(np.sum(np.std(masked, axis=1) == 0))
    checks["bad_mean"]     = abs_mean > 100

    def _fail(msg):
        return f"{_RED}{msg}{_RESET}"

    SEP = "─" * 39
    print(f"\n─── Sanity check: {label} ───")
    print(f"{'Output shape:':<30} {post_data.shape}")

    if checks["nan_inf"]:
        print(f"{'NaN/Inf check:':<30} {_fail('FAIL: output contains NaN/Inf values')}")
    else:
        print(f"{'NaN/Inf check:':<30} PASSED")

    if checks["std_exploded"]:
        print(f"{'Std magnitude check:':<30} "
              f"{_fail(f'FAIL: post std ({post_std:.1f}) exceeds 10x pre ({pre_std:.1f}) — pipeline likely broken')}")
    else:
        print(f"{'Std magnitude check:':<30} PASSED (post={post_std:.1f}, pre={pre_std:.1f})")

    n_zero_var = checks["n_zero_var"]
    if n_zero_var > 0:
        print(f"{'Voxel variance check:':<30} "
              f"{_fail(f'WARNING: {n_zero_var} voxels have zero variance after denoising')}")
    else:
        print(f"{'Voxel variance check:':<30} PASSED")

    if checks["bad_mean"]:
        print(f"{'Mean centering check:':<30} "
              f"{_fail(f'WARNING: post-denoising mean ({abs_mean:.1f}) is far from zero — detrending may have failed')}")
    else:
        print(f"{'Mean centering check:':<30} PASSED (abs mean={abs_mean:.2f})")

    print(SEP)


# ─── Main pipeline function ───────────────────────────────────────────────────

def denoise(
    bold_data: np.ndarray,
    mask_flat: np.ndarray,
    confounds_df: pd.DataFrame,
    spike_idx: np.ndarray,
    include_gsr: bool,
) -> np.ndarray:
    """
    Denoise one run inside the brain mask; voxels outside are left at zero.

    Steps
    -----
    1. Build & prepare regressor matrix (NaN→0, detrend, z-score)
    2. Detrend BOLD (mean + linear trend per voxel)
    3. Single GLM regression (numpy.linalg.lstsq); output = residuals
    4. CBIG Lomb-Scargle interpolation + bandpass (batched, ~5000 voxels/batch)
       Bandpass 0.01–0.1 Hz via frequency-domain masking (no Butterworth)

    5-stage diagnostics printed to stdout for validation.

    Returns
    -------
    (nx, ny, nz, n_times) float32 — zeros outside mask, denoised inside
    """
    n_times  = bold_data.shape[3]
    bold_2d  = bold_data.reshape(-1, n_times).astype(np.float64)
    Y        = bold_2d[mask_flat]                                # (n_vox_mask, n_times)

    # ── Stage 1: raw BOLD ──────────────────────────────────────────────────────
    print(f"  [Stage 1] Raw BOLD std:                    {Y.std():.2f}")

    X = _build_regressors(confounds_df, spike_idx, include_gsr)
    X = _detrend(X)
    X = _standardize(X)

    Y_det = _detrend(Y.T).T

    # ── Stage 2: post-detrend ──────────────────────────────────────────────────
    print(f"  [Stage 2] Post-detrend std:                {Y_det.std():.2f}")

    betas, _, _, _ = np.linalg.lstsq(X, Y_det.T, rcond=None)
    residuals = Y_det - (X @ betas).T                           # (n_vox_mask, n_times)

    # ── Stage 3: post-GLM ──────────────────────────────────────────────────────
    print(f"  [Stage 3] Post-GLM residuals std:          {residuals.std():.2f}")

    good_idx = np.setdiff1d(np.arange(n_times), spike_idx)
    n_good   = len(good_idx)
    n_interp = len(spike_idx)
    n_vox    = residuals.shape[0]
    n_batches = -(-n_vox // LS_BATCH_SIZE)   # ceiling division
    print(f"  LS: {n_interp} frames censored, {n_good} good frames, "
          f"{n_vox} in-mask voxels → {n_batches} batch(es) of ≤{LS_BATCH_SIZE}")

    t          = np.arange(n_times, dtype=np.float64) * TR
    in_sample  = t[good_idx]                                    # (n_good,)
    out_sample = t                                              # (n_times,)
    outliers   = np.zeros(n_times, dtype=np.int32)
    outliers[spike_idx] = 1

    filtered = np.empty_like(residuals)
    stage4_std = None   # recorded from first batch

    for b_start in range(0, n_vox, LS_BATCH_SIZE):
        b_end   = min(b_start + LS_BATCH_SIZE, n_vox)
        in_ser  = residuals[b_start:b_end, :][:, good_idx].T   # (n_good, batch)

        out_bp, interm_out = _cbig_lomb_scargle_bandpass(
            in_series       = in_ser,
            in_sample_time  = in_sample,
            out_sample_time = out_sample,
            outliers        = outliers,
            low_f           = BP_LOW,
            high_f          = BP_HIGH,
            oversample_fac  = LS_OVERSAMPLE_FAC,
        )                                                       # each (n_times, batch)

        filtered[b_start:b_end, :] = out_bp.T

        if b_start == 0:
            stage4_std = float(interm_out.std())
            print(f"      Batch 1 sample — interm std: {stage4_std:.2f}  "
                  f"bp std: {float(out_bp.std()):.2f}")

    # ── Stage 4: post-LS full-spectrum (sample from batch 1) ──────────────────
    print(f"  [Stage 4] Post-LS full-spectrum std:       {stage4_std:.2f}  "
          f"(first-batch sample)")

    # ── Stage 5: post-LS + bandpass ───────────────────────────────────────────
    print(f"  [Stage 5] Post-LS+bandpass std:            {filtered.std():.2f}")

    out_2d = np.zeros_like(bold_2d)
    out_2d[mask_flat] = filtered
    return out_2d.reshape(bold_data.shape).astype(np.float32)


# ─── Entry point ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading BOLD:      {BOLD_PATH}")
    bold_img  = nib.load(BOLD_PATH)
    bold_data = bold_img.get_fdata(dtype=np.float32)
    affine    = bold_img.affine
    header    = bold_img.header.copy()
    header.set_data_dtype(np.float32)

    print(f"Loading mask:      {MASK_PATH}")
    mask_flat = nib.load(MASK_PATH).get_fdata().astype(bool).ravel()

    print(f"Loading confounds: {CONFOUNDS_PATH}")
    confounds_df = pd.read_csv(CONFOUNDS_PATH, sep="\t")

    sys.path.insert(0, str(REPO_ROOT))
    from utils.motion_qc import compute_custom_fd

    fd        = compute_custom_fd(confounds_df, tr=TR, backward_diff_n=1, verbose=False)
    spike_idx = np.where(~np.isnan(fd) & (fd > FD_THRESH))[0]

    n_frames   = bold_data.shape[3]
    n_censored = len(spike_idx)
    pct        = 100.0 * n_censored / n_frames

    gsr_path   = OUT_DIR / f"{SUBJECT}_{SESSION}_{TASK}_desc-denoisedGSR_bold.nii.gz"
    nogsr_path = OUT_DIR / f"{SUBJECT}_{SESSION}_{TASK}_desc-denoisedNoGSR_bold.nii.gz"

    print("\nRunning +GSR pipeline...")
    gsr_data = denoise(bold_data, mask_flat, confounds_df, spike_idx, include_gsr=True)
    nib.save(nib.Nifti1Image(gsr_data, affine, header), gsr_path)
    print(f"  Saved: {gsr_path}")

    print("Running -GSR pipeline...")
    nogsr_data = denoise(bold_data, mask_flat, confounds_df, spike_idx, include_gsr=False)
    nib.save(nib.Nifti1Image(nogsr_data, affine, header), nogsr_path)
    print(f"  Saved: {nogsr_path}")

    n_base  = 14
    summary = (
        f"\n─── Denoising summary: {SUBJECT}_{SESSION} ───\n"
        f"Total frames: {n_frames}\n"
        f"High-motion frames censored: {n_censored} ({pct:.1f}%)\n"
        f"Number of nuisance regressors: {n_base} (no GSR) or {n_base + 1} (with GSR)"
        f" + {n_censored} spike regressors\n"
        f"Bandpass: {BP_LOW}–{BP_HIGH} Hz via Lomb-Scargle frequency mask "
        f"(oversample_fac={LS_OVERSAMPLE_FAC}, no Butterworth)\n"
        f"Output: {gsr_path}\n"
        f"Output: {nogsr_path}\n"
    )
    print(summary)

    log_path = OUT_DIR / f"{SUBJECT}_{SESSION}_denoising_log.txt"
    log_path.write_text(summary)

    pre_std = float(np.std(bold_data.reshape(-1, n_frames).astype(np.float64)[mask_flat]))
    _sanity_check(f"{SUBJECT} {SESSION} -GSR", nogsr_data, pre_std, mask_flat)
    _sanity_check(f"{SUBJECT} {SESSION} +GSR", gsr_data,   pre_std, mask_flat)


if __name__ == "__main__":
    main()
