#!/usr/bin/env python3
"""
Single-subject denoising pipeline for DMT-MED fMRI data.
Subject: sub-01, ses-01

Implements Goldberg et al. (2024) strategy:
  - Single GLM with WM, CSF, 6 motion, 6 derivatives, spike regressors (+/- GSR)
  - Lomb-Scargle interpolation at censored frames: frequency grid from
    astropy LombScargle.autofrequency, capped at n_good//4 to keep the
    sinusoidal basis well-overdetermined (VanderPlas 2018)
  - Bandpass filter: 0.01-0.1 Hz, Butterworth order 2, zero-phase (filtfilt)

Two output versions (both include spike regressors + interpolation):
  +GSR +censor : desc-denoisedGSR_bold.nii.gz
  -GSR +censor : desc-denoisedNoGSR_bold.nii.gz
"""

import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from scipy import signal as sp_signal
from astropy.timeseries import LombScargle

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
BP_ORDER  = 2

SUBJECT = "sub-01"
SESSION = "ses-01"
TASK    = "task-rest"

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


def _lombscargle_interp(
    residuals: np.ndarray,
    good_idx: np.ndarray,
    bad_idx: np.ndarray,
    n_times: int,
    tr: float,
) -> np.ndarray:
    """
    Interpolate residuals at censored frames using a sinusoidal basis.

    Frequency grid from astropy LombScargle.autofrequency (VanderPlas 2018),
    subsampled to at most n_good//4 frequencies so the basis stays
    well-overdetermined (≥2x more data points than parameters).

    Parameters
    ----------
    residuals : (n_vox, n_times)
    good_idx  : non-censored frame indices
    bad_idx   : censored frame indices to fill
    n_times   : total frames
    tr        : TR in seconds

    Returns
    -------
    (n_vox, n_bad) interpolated values
    """
    t      = np.arange(n_times, dtype=np.float64) * tr
    t_good = t[good_idx]
    t_bad  = t[bad_idx]
    n_good = len(good_idx)

    # Full autofrequency grid, then cap to n_good//4 to keep the sinusoidal
    # basis overdetermined. Subsampling evenly preserves spectral coverage.
    all_freqs = LombScargle(t_good, np.ones(n_good)).autofrequency(
        samples_per_peak=5, nyquist_factor=1
    )
    n_cap  = max(1, n_good // 4)
    idx    = np.round(np.linspace(0, len(all_freqs) - 1, n_cap)).astype(int)
    freqs  = all_freqs[idx]   # (n_cap,) — evenly distributed across the grid

    def _basis(t_pts: np.ndarray) -> np.ndarray:
        phases = 2.0 * np.pi * t_pts[:, None] * freqs[None, :]
        return np.column_stack([np.ones(len(t_pts)), np.sin(phases), np.cos(phases)])

    X_good = _basis(t_good)   # (n_good, 1 + 2*n_cap)
    X_bad  = _basis(t_bad)    # (n_bad,  1 + 2*n_cap)

    # Vectorized fit: solve for all voxels simultaneously
    betas, _, _, _ = np.linalg.lstsq(X_good, residuals[:, good_idx].T, rcond=None)
    return (X_bad @ betas).T   # (n_vox, n_bad)


def _bandpass(
    data: np.ndarray,
    tr: float,
    low: float = 0.01,
    high: float = 0.1,
    order: int = 2,
) -> np.ndarray:
    """Zero-phase Butterworth bandpass filter along time axis (axis=1)."""
    nyq = 1.0 / (2.0 * tr)
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype="bandpass")
    return sp_signal.filtfilt(b, a, data, axis=1)


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

    # 1. NaN / Inf
    checks["nan_inf"] = bool(np.isnan(masked).any() or np.isinf(masked).any())

    # 2. Std magnitude
    checks["std_exploded"] = post_std > 10 * pre_std

    # 3. Zero-variance voxels
    checks["n_zero_var"] = int(np.sum(np.std(masked, axis=1) == 0))

    # 4. Mean centering
    checks["bad_mean"] = abs_mean > 100

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

    if checks["n_zero_var"] > 0:
        print(f"{'Voxel variance check:':<30} "
              f"{_fail(f'WARNING: {checks[\"n_zero_var\"]} voxels have zero variance after denoising')}")
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
    4. Lomb-Scargle interpolation at censored frames
    5. Bandpass filter 0.01-0.1 Hz (Butterworth order 2, filtfilt)

    Returns
    -------
    (nx, ny, nz, n_times) float32 — zeros outside mask, denoised inside
    """
    n_times  = bold_data.shape[3]
    bold_2d  = bold_data.reshape(-1, n_times).astype(np.float64)
    Y        = bold_2d[mask_flat]

    X = _build_regressors(confounds_df, spike_idx, include_gsr)
    X = _detrend(X)
    X = _standardize(X)

    Y = _detrend(Y.T).T

    betas, _, _, _ = np.linalg.lstsq(X, Y.T, rcond=None)
    residuals = Y - (X @ betas).T

    good_idx  = np.setdiff1d(np.arange(n_times), spike_idx)
    n_interp  = len(spike_idx)
    n_vox_mask = Y.shape[0]
    print(f"  LS interpolation: {n_interp} frames × {n_vox_mask} in-mask voxels"
          f"  (fallback: 0)")
    if n_interp > 0:
        residuals[:, spike_idx] = _lombscargle_interp(
            residuals, good_idx, spike_idx, n_times, TR
        )

    filtered = _bandpass(residuals, TR, low=BP_LOW, high=BP_HIGH, order=BP_ORDER)

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
        f"Bandpass: {BP_LOW}–{BP_HIGH} Hz, Butterworth order {BP_ORDER}\n"
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
