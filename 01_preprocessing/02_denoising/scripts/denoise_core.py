"""
Core denoising pipeline — shared between denoise_single_subject.py and denoise_batch.py.

One parameterized core drives three independent denoising pipelines (selected by the
`pipeline` argument) so the final results can be shown robust to denoising choice:

  - "detrend" (Keskin et al. 2025 style): polynomial detrend only. No nuisance GLM,
    no censoring, no interpolation, no bandpass.
  - "glm": full nuisance GLM (WM, CSF, 6 motion + 6 derivatives) on detrended BOLD.
    No censoring, no interpolation, no bandpass.
  - "maximal" (Goldberg et al. 2024): the "glm" regressors PLUS FD>0.3 frame censoring,
    CBIG Lomb-Scargle interpolation, and bandpass 0.01–0.1 Hz (frequency-domain mask).

All three are NoGSR by default. The step toggles for each pipeline live in
PIPELINE_PRESETS below.

NOTE: an earlier spec mentioned adding "FPC/FPCsq" regressors to the maximal pipeline.
Those turned out to be subject×session-level motion-quality covariates (pcf_diff,
pcf_sq_diff) for the statistics-stage residualization model — one scalar per run, NOT
per-timepoint nuisance regressors — so they are NOT added here. They belong to the
later statistics step.
"""

import gc
import sys
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from utils.motion_qc import compute_custom_fd

# ─── Pipeline parameters ─────────────────────────────────────────────────────

TR              = 1.8    # seconds (verified from BOLD JSON header, NOT 1.5)
FD_THRESH       = 0.3    # mm — Goldberg et al. 2024 recommendation for ACW
BP_LOW          = 0.01   # Hz
BP_HIGH         = 0.1    # Hz
LS_OVERSAMPLE_FAC = 8
LS_BATCH_SIZE   = 5000   # voxels per LS batch (memory guard)

TASK = "task-rest"
SPACE = "MNI152NLin2009cAsym"

# fMRIPrep confound column names
WM_CSF_COLS = ["white_matter", "csf"]
MOTION_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
DERIV_COLS  = [
    "trans_x_derivative1", "trans_y_derivative1", "trans_z_derivative1",
    "rot_x_derivative1",   "rot_y_derivative1",   "rot_z_derivative1",
]
GSR_COL = "global_signal"

# ─── Pipeline presets (one parameterized core, three pipelines) ──────────────
# Each preset toggles the steps in the denoising step-matrix. desc = filename tag.
PIPELINE_PRESETS = {
    "detrend": dict(desc="detrend", detrend_order=1,
                    do_nuisance=False, do_censor=False, do_ls=False, do_gsr=False),
    "glm":     dict(desc="glm",     detrend_order=1,
                    do_nuisance=True,  do_censor=False, do_ls=False, do_gsr=False),
    "maximal": dict(desc="maximal", detrend_order=1,
                    do_nuisance=True,  do_censor=True,  do_ls=True,  do_gsr=False),
}

_RED   = "\033[91m"
_RESET = "\033[0m"


# ─── Path helpers ─────────────────────────────────────────────────────────────

def build_fmriprep_paths(subject: str, session: str, fmriprep_root: Path) -> dict:
    """Return dict of fMRIPrep input paths for one subject-session."""
    func = fmriprep_root / subject / session / "func"
    tag  = f"{subject}_{session}_{TASK}"
    return {
        "bold":      func / f"{tag}_space-{SPACE}_desc-preproc_bold.nii.gz",
        "mask":      func / f"{tag}_space-{SPACE}_desc-brain_mask.nii.gz",
        "confounds": func / f"{tag}_desc-confounds_timeseries.tsv.gz",
    }


def build_output_path(subject: str, session: str, output_dir: Path, desc: str) -> Path:
    """Return the denoised NIfTI output path for one subject-session × pipeline.

    desc is the pipeline tag ("detrend" | "glm" | "maximal"), embedded in the
    filename so the file is self-identifying even if moved.
    """
    tag = f"{subject}_{session}_{TASK}"
    return output_dir / f"{tag}_desc-{desc}_bold.nii.gz"


# ─── GLM helpers ─────────────────────────────────────────────────────────────

def _poly_detrend(X: np.ndarray, order: int = 1) -> np.ndarray:
    """Remove a polynomial trend (intercept + up to `order`) from each column of X.

    order=1 → mean + linear (the previous _detrend behaviour); order=2 adds a
    quadratic term, etc. Time is centred and scaled to ~[-1, 1] for conditioning.
    """
    n = X.shape[0]
    t = np.arange(n, dtype=np.float64) - (n - 1) / 2.0
    scale = t.max() if t.max() > 0 else 1.0
    t = t / scale
    D = np.column_stack([t ** p for p in range(order + 1)])
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
    include_spikes: bool = True,
) -> np.ndarray:
    """Assemble (n_times, n_regressors) nuisance matrix; NaNs → 0.

    WM + CSF + 6 motion + 6 derivatives, optionally + GSR, optionally + spike
    regressors (one-hot per censored frame). Spike regressors are only added when
    `include_spikes` is True (i.e. the "maximal" pipeline that censors).
    """
    cols = WM_CSF_COLS + MOTION_COLS + DERIV_COLS
    if include_gsr:
        cols = cols + [GSR_COL]

    X = confounds_df[cols].to_numpy(dtype=np.float64)
    X = np.where(np.isnan(X), 0.0, X)

    if include_spikes and len(spike_idx) > 0:
        n_times = X.shape[0]
        spikes  = np.zeros((n_times, len(spike_idx)), dtype=np.float64)
        for k, frame in enumerate(spike_idx):
            spikes[frame, k] = 1.0
        X = np.hstack([X, spikes])

    return X


# ─── CBIG Lomb-Scargle + bandpass ────────────────────────────────────────────

def _cbig_lomb_scargle_bandpass(
    in_series: np.ndarray,
    in_sample_time: np.ndarray,
    out_sample_time: np.ndarray,
    outliers: np.ndarray,
    low_f: float = 0.01,
    high_f: float = 0.1,
    oversample_fac: int = 8,
) -> tuple:
    """
    Faithful Python port of CBIG_preproc_censor.m (Jingwei Li, Yeo Lab CBIG).

    Lomb-Scargle interpolation of censored frames with simultaneous bandpass
    filtering via frequency-domain masking (Power et al. 2014, Eq. 3-4 suppl.).

    Parameters
    ----------
    in_series       : (num_in_time, nChannel) — good-frame residuals
    in_sample_time  : (num_in_time,) — times of good frames
    out_sample_time : (num_out_time,) — times of all frames
    outliers        : (num_out_time,) int — 1=censored, 0=good

    Returns
    -------
    out_bp     : (num_out_time, nChannel) — bandpass-filtered reconstruction
    interm_out : (num_out_time, nChannel) — full-spectrum reconstruction
    """
    num_in_time, nChannel = in_series.shape

    input_mean = in_series.mean(axis=0)
    input_std  = in_series.std(axis=0, ddof=0)     # N-norm matches MATLAB std(x,1)
    data       = in_series - input_mean

    # Frequency grid: f = 1/(T*ofac) : 1/(T*ofac) : N_uncen/(2*T)
    T      = in_sample_time.max() - in_sample_time.min()
    df     = 1.0 / (T * oversample_fac)
    n_freq = int(num_in_time * oversample_fac / 2)
    f      = np.arange(1, n_freq + 1) * df
    w      = 2.0 * np.pi * f

    # Phase shift tau — Scargle 1982 (makes cos/sin orthogonal on irregular grid)
    phase_2w = 2.0 * w[:, None] * in_sample_time[None, :]
    tau = (np.arctan2(np.sin(phase_2w).sum(axis=1),
                      np.cos(phase_2w).sum(axis=1)) / (2.0 * w))

    # Sinusoidal basis at input times
    arg_in = w[:, None] * (in_sample_time[None, :] - tau[:, None])
    cterm  = np.cos(arg_in)
    sterm  = np.sin(arg_in)

    # Eq. 3 — diagonal projection coefficients
    cos_denom = (cterm ** 2).sum(axis=1, keepdims=True)
    sin_denom = (sterm ** 2).sum(axis=1, keepdims=True)
    cos_denom = np.where(cos_denom == 0.0, 1.0, cos_denom)
    sin_denom = np.where(sin_denom == 0.0, 1.0, sin_denom)

    cos_coeff = (cterm @ data) / cos_denom   # (num_freq_bin, nChannel)
    sin_coeff = (sterm @ data) / sin_denom
    del cterm, sterm, arg_in, phase_2w

    # Sinusoidal basis at output times
    arg_out   = w[:, None] * (out_sample_time[None, :] - tau[:, None])
    cterm_new = np.cos(arg_out)
    sterm_new = np.sin(arg_out)
    del arg_out

    # Eq. 4 — full-spectrum reconstruction
    interm_out = cterm_new.T @ cos_coeff + sterm_new.T @ sin_coeff

    # Std correction at good frames
    good_mask  = (outliers == 0)
    out_std    = interm_out[good_mask].std(axis=0, ddof=0)
    out_std    = np.where(out_std == 0.0, 1.0, out_std)
    interm_out = interm_out * (input_std / out_std)[None, :]

    # Bandpass via frequency mask (zero out-of-band coefficients)
    fpass  = ~((f < low_f) | (f > high_f))
    out_bp = (cterm_new[fpass].T @ cos_coeff[fpass]
              + sterm_new[fpass].T @ sin_coeff[fpass])

    out_std_bp = out_bp[good_mask].std(axis=0, ddof=0)
    out_std_bp = np.where(out_std_bp == 0.0, 1.0, out_std_bp)
    out_bp     = out_bp * (input_std / out_std_bp)[None, :]

    return out_bp, interm_out


# ─── Sanity check ─────────────────────────────────────────────────────────────

def sanity_check(label: str, post_data: np.ndarray,
                 pre_std: float, mask_flat: np.ndarray) -> None:
    """Print structured sanity-check block for one denoised output."""
    n_times  = post_data.shape[3]
    masked   = post_data.reshape(-1, n_times).astype(np.float64)[mask_flat]
    post_std = float(np.std(masked))
    abs_mean = float(np.abs(np.mean(masked)))
    nan_inf  = bool(np.isnan(masked).any() or np.isinf(masked).any())
    exploded = post_std > 10 * pre_std
    n_zero_var = int(np.sum(np.std(masked, axis=1) == 0))
    bad_mean = abs_mean > 100

    def _fail(msg):
        return f"{_RED}{msg}{_RESET}"

    print(f"\n─── Sanity check: {label} ───")
    print(f"{'Output shape:':<30} {post_data.shape}")

    if nan_inf:
        print(f"{'NaN/Inf check:':<30} {_fail('FAIL: output contains NaN/Inf values')}")
    else:
        print(f"{'NaN/Inf check:':<30} PASSED")

    if exploded:
        print(f"{'Std magnitude check:':<30} "
              f"{_fail(f'FAIL: post std ({post_std:.1f}) > 10x pre ({pre_std:.1f})')}")
    else:
        print(f"{'Std magnitude check:':<30} PASSED (post={post_std:.1f}, pre={pre_std:.1f})")

    if n_zero_var > 0:
        print(f"{'Voxel variance check:':<30} "
              f"{_fail(f'WARNING: {n_zero_var} voxels have zero variance')}")
    else:
        print(f"{'Voxel variance check:':<30} PASSED")

    if bad_mean:
        print(f"{'Mean centering check:':<30} "
              f"{_fail(f'WARNING: abs mean={abs_mean:.1f} — detrending may have failed')}")
    else:
        print(f"{'Mean centering check:':<30} PASSED (abs mean={abs_mean:.2f})")

    print("─" * 39)


# ─── Main denoising function ──────────────────────────────────────────────────

def _run_ls_interp_bandpass(
    residuals: np.ndarray,
    spike_idx: np.ndarray,
    n_times: int,
    verbose: bool = True,
) -> np.ndarray:
    """CBIG Lomb-Scargle interpolation + bandpass over all voxels (batched)."""
    good_idx  = np.setdiff1d(np.arange(n_times), spike_idx)
    n_good    = len(good_idx)
    n_interp  = len(spike_idx)
    n_vox     = residuals.shape[0]
    n_batches = -(-n_vox // LS_BATCH_SIZE)

    if verbose:
        print(f"  LS: {n_interp} censored, {n_good} good, "
              f"{n_vox} voxels → {n_batches} batch(es) of ≤{LS_BATCH_SIZE}")

    t          = np.arange(n_times, dtype=np.float64) * TR
    in_sample  = t[good_idx]
    out_sample = t
    outliers   = np.zeros(n_times, dtype=np.int32)
    outliers[spike_idx] = 1

    filtered   = np.empty_like(residuals)
    stage4_std = None

    for b_start in range(0, n_vox, LS_BATCH_SIZE):
        b_end  = min(b_start + LS_BATCH_SIZE, n_vox)
        in_ser = residuals[b_start:b_end, :][:, good_idx].T   # (n_good, batch)

        out_bp, interm_out = _cbig_lomb_scargle_bandpass(
            in_series       = in_ser,
            in_sample_time  = in_sample,
            out_sample_time = out_sample,
            outliers        = outliers,
            low_f           = BP_LOW,
            high_f          = BP_HIGH,
            oversample_fac  = LS_OVERSAMPLE_FAC,
        )

        filtered[b_start:b_end, :] = out_bp.T

        if b_start == 0:
            stage4_std = float(interm_out.std())
            if verbose:
                print(f"      Batch 1 — interm std: {stage4_std:.2f}  "
                      f"bp std: {float(out_bp.std()):.2f}")

    if verbose:
        print(f"  [Stage 4] Post-LS full-spectrum std:       {stage4_std:.2f}  (batch-1 sample)")
        print(f"  [Stage 5] Post-LS+bandpass std:            {filtered.std():.2f}")

    return filtered


def denoise(
    bold_data: np.ndarray,
    mask_flat: np.ndarray,
    confounds_df: pd.DataFrame,
    spike_idx: np.ndarray,
    config: dict,
    verbose: bool = True,
) -> np.ndarray:
    """
    Denoise one run inside the brain mask, following the given pipeline preset.

    `config` is one of PIPELINE_PRESETS values:
      detrend_order : polynomial detrend order applied to BOLD (and regressors)
      do_nuisance   : run the WM/CSF/motion GLM (False = detrend-only pipeline)
      do_censor     : add spike regressors for FD>thresh frames (maximal only)
      do_ls         : CBIG Lomb-Scargle interpolation + bandpass (maximal only)
      do_gsr        : add the global-signal regressor

    Returns (nx, ny, nz, n_times) float32; zeros outside mask.
    """
    detrend_order = config["detrend_order"]
    n_times = bold_data.shape[3]
    bold_2d = bold_data.reshape(-1, n_times).astype(np.float64)
    Y       = bold_2d[mask_flat]

    if verbose:
        print(f"  [Stage 1] Raw BOLD std:                    {Y.std():.2f}")

    # Polynomial detrend of BOLD (per voxel). Applied in every pipeline.
    Y_det = _poly_detrend(Y.T, detrend_order).T

    if verbose:
        print(f"  [Stage 2] Post-detrend std:                {Y_det.std():.2f}")

    if not config["do_nuisance"]:
        # "detrend" pipeline — detrend only, no GLM / censoring / interpolation.
        result = Y_det
        if verbose:
            print("  [detrend pipeline] no nuisance GLM applied")
    else:
        X = _build_regressors(
            confounds_df, spike_idx,
            include_gsr    = config["do_gsr"],
            include_spikes = config["do_censor"],
        )
        X = _poly_detrend(X, detrend_order)
        X = _standardize(X)

        betas, _, _, _ = np.linalg.lstsq(X, Y_det.T, rcond=None)
        result = Y_det - (X @ betas).T

        if verbose:
            print(f"  [Stage 3] Post-GLM residuals std:          {result.std():.2f}  "
                  f"({X.shape[1]} regressors)")

        if config["do_ls"]:
            result = _run_ls_interp_bandpass(result, spike_idx, n_times, verbose)

    out_2d = np.zeros_like(bold_2d)
    out_2d[mask_flat] = result
    return out_2d.reshape(bold_data.shape).astype(np.float32)


# ─── Full run orchestration ───────────────────────────────────────────────────

def denoise_run(
    subject: str,
    session: str,
    fmriprep_root: Path,
    output_dir: Path,
    pipeline: str,
    verbose: bool = True,
) -> dict:
    """
    Run one denoising pipeline for one subject-session.

    Loads data, applies the selected pipeline preset, saves the NIfTI output,
    frees memory.

    Parameters
    ----------
    subject      : e.g. "sub-01"
    session      : e.g. "ses-01"
    fmriprep_root: root of fMRIPrep derivatives directory
    output_dir   : directory to write the denoised NIfTI (one per pipeline)
    pipeline     : "detrend" | "glm" | "maximal"
    verbose      : print stage diagnostics (True for single-subject, False for batch)

    Returns
    -------
    dict with keys: pipeline, n_frames, n_censored, pct_censored, post_std, out_path
    """
    if pipeline not in PIPELINE_PRESETS:
        raise ValueError(f"Unknown pipeline '{pipeline}'. "
                         f"Expected one of {list(PIPELINE_PRESETS)}.")
    config = PIPELINE_PRESETS[pipeline]
    output_dir.mkdir(parents=True, exist_ok=True)

    paths    = build_fmriprep_paths(subject, session, fmriprep_root)
    out_path = build_output_path(subject, session, output_dir, config["desc"])

    if verbose:
        print(f"Loading BOLD:      {paths['bold']}")
    bold_img  = nib.load(paths["bold"])
    bold_data = bold_img.get_fdata(dtype=np.float32)
    affine    = bold_img.affine
    header    = bold_img.header.copy()
    header.set_data_dtype(np.float32)

    if verbose:
        print(f"Loading mask:      {paths['mask']}")
    mask_flat = nib.load(paths["mask"]).get_fdata().astype(bool).ravel()

    if verbose:
        print(f"Loading confounds: {paths['confounds']}")
    confounds_df = pd.read_csv(paths["confounds"], sep="\t")

    # FD / censoring is always computed for logging; only USED when config["do_censor"].
    fd        = compute_custom_fd(confounds_df, tr=TR, backward_diff_n=1, verbose=False)
    spike_idx = np.where(~np.isnan(fd) & (fd > FD_THRESH))[0]

    n_frames   = bold_data.shape[3]
    n_censored = int(len(spike_idx))
    pct        = 100.0 * n_censored / n_frames

    if verbose:
        print(f"\nRunning '{pipeline}' pipeline "
              f"(censoring {'ON' if config['do_censor'] else 'OFF'})...")
    data = denoise(bold_data, mask_flat, confounds_df, spike_idx,
                   config=config, verbose=verbose)
    nib.save(nib.Nifti1Image(data, affine, header), out_path)
    if verbose:
        print(f"  Saved: {out_path}")

    post_std = float(np.std(
        data.reshape(-1, n_frames).astype(np.float64)[mask_flat]
    ))

    # Free large arrays immediately
    del bold_data, data
    gc.collect()

    return {
        "pipeline":     pipeline,
        "n_frames":     n_frames,
        "n_censored":   n_censored,
        "pct_censored": pct,
        "post_std":     post_std,
        "out_path":     out_path,
    }
