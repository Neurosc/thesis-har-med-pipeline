#!/usr/bin/env python3
"""
Single-subject denoising pipeline for DMT-MED fMRI data.
Subject: sub-01, ses-01

Implements Goldberg et al. (2024) strategy:
  - Single GLM with WM, CSF, 6 motion, 6 derivatives, spike regressors (+/- GSR)
  - Lomb-Scargle interpolation at censored frames via astropy.timeseries.LombScargle
    (frequency grid from autofrequency, VanderPlas 2018; falls back to linear
    interpolation if the basis is ill-conditioned)
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
    Interpolate residuals at censored frames using a sinusoidal basis whose
    frequency grid comes from astropy LombScargle.autofrequency (VanderPlas 2018).
    nyquist_factor=1 caps frequencies at the average Nyquist of the good frames.

    If the resulting basis is ill-conditioned (cond > 1e8), falls back to
    per-voxel linear interpolation between non-censored neighbours.

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
    n_vox  = residuals.shape[0]

    # Frequency grid: field-standard autofrequency (VanderPlas 2018).
    # samples_per_peak=5 keeps the grid well-resolved; nyquist_factor=1 prevents
    # extrapolation above the average Nyquist of the non-censored samples.
    freqs = LombScargle(t_good, np.ones(n_good)).autofrequency(
        samples_per_peak=5, nyquist_factor=1
    )

    def _basis(t_pts: np.ndarray) -> np.ndarray:
        phases = 2.0 * np.pi * t_pts[:, None] * freqs[None, :]
        return np.column_stack([np.ones(len(t_pts)), np.sin(phases), np.cos(phases)])

    X_good = _basis(t_good)   # (n_good, 1 + 2*n_freqs)
    X_bad  = _basis(t_bad)    # (n_bad,  1 + 2*n_freqs)

    # Condition check on the design matrix (same for all voxels).
    # A near-singular basis produces wildly wrong predictions; linear interpolation
    # is the safe fallback and sufficient for bandpass pre-conditioning.
    cond = np.linalg.cond(X_good)
    if cond > 1e8:
        result = np.empty((n_vox, len(bad_idx)), dtype=np.float64)
        for vi in range(n_vox):
            result[vi] = np.interp(t_bad, t_good, residuals[vi, good_idx])
        return result

    betas, _, _, _ = np.linalg.lstsq(X_good, residuals[:, good_idx].T, rcond=None)
    return (X_bad @ betas).T


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


# ─── Diagnostic helpers ───────────────────────────────────────────────────────

DIAG_DIR = OUT_DIR / "_diagnostic"


def _masked_stats(Y: np.ndarray) -> dict:
    flat = Y.ravel()
    return {
        "mean": float(np.nanmean(flat)),
        "std":  float(np.nanstd(flat)),
        "min":  float(np.nanmin(flat)),
        "max":  float(np.nanmax(flat)),
    }


def _save_snapshot(
    step: int,
    label: str,
    Y: np.ndarray,
    mask_flat: np.ndarray,
    vol_shape: tuple,
    affine: np.ndarray,
    header,
) -> Path:
    n_all = len(mask_flat)
    snap_flat = np.zeros(n_all, dtype=np.float32)
    snap_flat[mask_flat] = np.nanmean(Y, axis=1).astype(np.float32)
    out_path = DIAG_DIR / f"step{step:02d}_{label}_mean.nii.gz"
    nib.save(nib.Nifti1Image(snap_flat.reshape(vol_shape), affine, header), out_path)
    return out_path


def denoise_diagnostic(
    bold_data: np.ndarray,
    mask_flat: np.ndarray,
    confounds_df: pd.DataFrame,
    spike_idx: np.ndarray,
    affine: np.ndarray,
    header,
) -> None:
    """Run -GSR pipeline with stats + NIfTI snapshot after each of 5 stages."""
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    vol_shape = bold_data.shape[:3]
    n_times   = bold_data.shape[3]
    records   = []

    def _check(step, label, Y):
        s = _masked_stats(Y)
        records.append((step, label, s))
        snap = _save_snapshot(step, label, Y, mask_flat, vol_shape, affine, header)
        print(f"  [{step}] {label}")
        print(f"        mean={s['mean']:.6g}  std={s['std']:.6g}"
              f"  min={s['min']:.6g}  max={s['max']:.6g}")
        print(f"        snapshot → {snap.name}")

    print("\n" + "─" * 64)
    print("DIAGNOSTIC PIPELINE  (-GSR, sub-01 ses-01)")
    print("─" * 64)

    # Step 1: raw BOLD
    bold_2d = bold_data.reshape(-1, n_times).astype(np.float64)
    Y_raw   = bold_2d[mask_flat]
    _check(1, "raw_BOLD_input", Y_raw)

    # Step 2: after voxel detrending
    X = _build_regressors(confounds_df, spike_idx, include_gsr=False)
    X = _detrend(X)
    X = _standardize(X)
    Y_det = _detrend(Y_raw.T).T
    _check(2, "after_voxel_detrend", Y_det)

    # Step 3: after GLM residuals
    betas, _, _, _ = np.linalg.lstsq(X, Y_det.T, rcond=None)
    residuals = Y_det - (X @ betas).T
    _check(3, "after_GLM_residuals", residuals)

    # Step 4: after Lomb-Scargle interpolation
    good_idx   = np.setdiff1d(np.arange(n_times), spike_idx)
    res_interp = residuals.copy()
    if len(spike_idx) > 0:
        res_interp[:, spike_idx] = _lombscargle_interp(
            residuals, good_idx, spike_idx, n_times, TR
        )
    _check(4, "after_LS_interpolation", res_interp)

    # Step 5: after bandpass filter
    filtered = _bandpass(res_interp, TR, low=BP_LOW, high=BP_HIGH, order=BP_ORDER)
    _check(5, "after_bandpass_filter", filtered)

    # Summary table
    print("\n" + "─" * 72)
    print(f"{'Step':<6} {'Stage':<30} {'mean':>12} {'std':>12} {'min':>12} {'max':>12}")
    print("─" * 72)
    for step, label, s in records:
        print(
            f"{step:<6} {label:<30} "
            f"{s['mean']:>12.4g} {s['std']:>12.4g} "
            f"{s['min']:>12.4g} {s['max']:>12.4g}"
        )
    print("─" * 72)
    print(f"\nSnapshots saved to: {DIAG_DIR}")

    # Flag first explosion (std grows >1000x vs raw)
    raw_std = records[0][2]["std"]
    for step, label, s in records[1:]:
        if s["std"] > raw_std * 1000:
            print(f"\n*** EXPLOSION at step {step}: {label} ***")
            print(f"    std: {raw_std:.4g} (raw) → {s['std']:.4g}")
            break
    else:
        print("\nNo explosion >1000x raw std detected.")


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

    good_idx = np.setdiff1d(np.arange(n_times), spike_idx)
    if len(spike_idx) > 0:
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

    # Diagnostic mode: run -GSR only with per-stage instrumentation
    denoise_diagnostic(bold_data, mask_flat, confounds_df, spike_idx, affine, header)


if __name__ == "__main__":
    main()
