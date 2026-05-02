import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import signal


MOTION_COLS = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]


def load_confounds(confounds_path):
    """Load confounds TSV and validate required columns exist."""
    df = pd.read_csv(confounds_path, sep="\t")
    print(f"  Shape: {df.shape[0]} frames x {df.shape[1]} columns")
    missing = [c for c in ("framewise_displacement", "std_dvars") if c not in df.columns]
    if missing:
        fd_cols = [c for c in df.columns if "fd" in c.lower() or "dvars" in c.lower()]
        raise ValueError(
            f"Required columns missing: {missing}\n"
            f"Columns containing 'fd' or 'dvars': {fd_cols}"
        )
    return df


def compute_custom_fd(
    confounds_df,
    tr=1.5,
    bandstop_low=0.2,
    bandstop_high=0.5,
    backward_diff_n=2,
    sphere_radius_mm=50.0,
):
    """
    Compute Goldberg/Lynch-style FD with respiratory bandstop filtering.

    Steps:
    1. Extract 6 motion parameters (trans x/y/z mm, rot x/y/z rad).
    2. Apply zero-phase Butterworth bandstop filter (order 4) at
       bandstop_low–bandstop_high Hz. If bandstop_high >= Nyquist,
       clips to 0.99 * Nyquist with a warning.
    3. Backward differences over backward_diff_n TRs; first
       backward_diff_n values are NaN.
    4. Rotational diffs converted to mm via sphere_radius_mm.
    5. FD = sum of absolute values across all 6 parameters.

    Returns 1D array of length n_timepoints (NaN at first backward_diff_n frames).
    """
    missing_motion = [c for c in MOTION_COLS if c not in confounds_df.columns]
    if missing_motion:
        raise ValueError(f"Motion parameter columns missing: {missing_motion}")

    params = confounds_df[MOTION_COLS].to_numpy(dtype=float)

    fs = 1.0 / tr
    nyq = fs / 2.0

    if bandstop_high >= nyq:
        clipped = nyq * 0.99
        print(
            f"  WARNING: bandstop_high ({bandstop_high} Hz) >= Nyquist "
            f"({nyq:.4f} Hz). Clipping to {clipped:.4f} Hz."
        )
        bandstop_high = clipped

    b, a = signal.butter(4, [bandstop_low / nyq, bandstop_high / nyq], btype="bandstop")

    filtered = np.empty_like(params)
    for i in range(params.shape[1]):
        filtered[:, i] = signal.filtfilt(b, a, params[:, i], padtype="odd")

    n = len(filtered)
    diffs = np.full((n, 6), np.nan)
    for t in range(backward_diff_n, n):
        diffs[t] = filtered[t] - filtered[t - backward_diff_n]

    # Rotational columns (indices 3-5) from radians to mm
    diffs[:, 3:] *= sphere_radius_mm

    fd = np.nansum(np.abs(diffs), axis=1)
    fd[:backward_diff_n] = np.nan

    return fd


def _fd_stats(fd_raw, fd_thresh, suffix):
    """
    Compute FD summary stats with explicit column-name suffix.

    suffix should be '_fmriprep' or '_custom' — this prevents any
    ambiguity between the two FD sources in returned dicts and TSV columns.
    """
    n_frames = len(fd_raw)
    n_censored = int(np.sum(fd_raw > fd_thresh, where=~np.isnan(fd_raw)))
    return {
        f"mean_fd{suffix}":             float(np.nanmean(fd_raw)),
        f"median_fd{suffix}":           float(np.nanmedian(fd_raw)),
        f"max_fd{suffix}":              float(np.nanmax(fd_raw)),
        f"n_censored{suffix}":          n_censored,
        f"pct_censored{suffix}":        100.0 * n_censored / n_frames,
        f"n_frames_remaining{suffix}":  n_frames - n_censored,
    }


def make_qc_figure(fd_raw, dvars_raw, censor_mask, subject_label, fd_thresh,
                   fig_path, fd_fmriprep=None, compare_fd=False):
    """
    Save 3-panel motion QC figure (FD / std_dvars / censored strip).

    If compare_fd=True and fd_fmriprep is provided, overlays fMRIPrep FD
    on panel 1 for visual comparison against the custom filtered FD.
    """
    n_frames = len(fd_raw)
    frames = np.arange(n_frames)
    cmap = matplotlib.colors.ListedColormap(["#d0d0d0", "#cc2222"])

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1]}
    )
    fig.suptitle(f"Motion QC — {subject_label}", fontsize=13, y=1.01)

    # Panel 1: FD
    axes[0].plot(frames, fd_raw, color="steelblue", linewidth=1.2,
                 label="Custom FD (filtered, 2-TR backward diff)")
    if compare_fd and fd_fmriprep is not None:
        axes[0].plot(frames, fd_fmriprep, color="gray", linewidth=0.9,
                     linestyle="--", alpha=0.7, label="fMRIPrep FD")
    axes[0].axhline(fd_thresh, color="black", linestyle="--", linewidth=1,
                    label=f"Threshold ({fd_thresh} mm)")
    axes[0].set_ylabel("FD (mm)")
    axes[0].set_title("Framewise Displacement — Custom FD (filtered, 2-TR backward diff)")
    axes[0].legend(fontsize=8, loc="upper right")

    # Panel 2: std_dvars
    axes[1].plot(frames, dvars_raw, color="darkorange", linewidth=1.2)
    axes[1].axhline(1.5, color="black", linestyle="--", linewidth=1,
                    label="Threshold (1.5, Power et al. 2014)")
    axes[1].set_ylim(0, 5)
    axes[1].set_ylabel("std_dvars")
    axes[1].set_title("Standardized DVARS (std_dvars)")
    axes[1].legend(fontsize=8, loc="upper right")

    # Panel 3: censored frames strip (based on active FD passed in)
    axes[2].imshow(censor_mask.astype(float).reshape(1, -1),
                   aspect="auto", cmap=cmap, vmin=0, vmax=1,
                   interpolation="nearest")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Frame index")
    axes[2].set_title(f"Censored frames (FD > {fd_thresh} mm)")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_motion_qc(confounds_path, subject_id, session_id, output_dirs,
                  fd_thresh=0.3, dvars_thresh=1.5,
                  use_custom_fd=True, compare_fd=False):
    """
    Run motion QC for one subject-session.

    Parameters
    ----------
    confounds_path : Path
    subject_id     : str   e.g. 'sub-01'
    session_id     : str   e.g. 'ses-01'
    output_dirs    : dict  keys 'figures' and 'results' (Path objects)
    fd_thresh      : float  FD censoring threshold in mm
    dvars_thresh   : float  stored in summary, not used for censoring
    use_custom_fd  : bool   use Goldberg/Lynch filtered FD for censoring (default True)
    compare_fd     : bool   overlay fMRIPrep FD on the figure for comparison

    Returns
    -------
    dict with fully-suffixed keys: *_fmriprep and *_custom for all FD stats.
    No ambiguous 'mean_fd' / 'n_censored' / 'pct_censored' keys exist.
    """
    label = f"{subject_id}_{session_id}"
    print(f"Loading confounds: {confounds_path}")

    df = load_confounds(confounds_path)
    n_frames = df.shape[0]
    fd_fmriprep = df["framewise_displacement"].to_numpy(dtype=float)
    dvars_raw = df["std_dvars"].to_numpy(dtype=float)

    fd_custom = compute_custom_fd(df)

    # Stats for both FDs independently — no shared/ambiguous keys
    stats_fmriprep = _fd_stats(fd_fmriprep, fd_thresh, "_fmriprep")
    stats_custom   = _fd_stats(fd_custom,   fd_thresh, "_custom")

    # Censoring mask uses whichever FD is active
    fd_active = fd_custom if use_custom_fd else fd_fmriprep
    fd_source = "custom (Goldberg/Lynch filtered)" if use_custom_fd else "fMRIPrep default"
    censor_mask = np.where(np.isnan(fd_active), False, fd_active > fd_thresh)
    censor_indices = np.where(censor_mask)[0].astype(int)

    print(f"\n── Motion summary: {label} ──")
    print(f"  FD used for censoring   : {fd_source}")
    print(f"  Total frames            : {n_frames}")
    print(f"  Mean FD  (fMRIPrep)     : {stats_fmriprep['mean_fd_fmriprep']:.4f} mm")
    print(f"  Mean FD  (custom)       : {stats_custom['mean_fd_custom']:.4f} mm")
    print(f"  Median FD (custom)      : {stats_custom['median_fd_custom']:.4f} mm")
    print(f"  Max FD   (custom)       : {stats_custom['max_fd_custom']:.4f} mm")
    print(f"  Frames FD > {fd_thresh} mm (custom) : {stats_custom['n_censored_custom']}")
    print(f"  Censored (custom)       : {stats_custom['pct_censored_custom']:.1f}%")
    print(f"  Mean std_dvars          : {float(np.nanmean(dvars_raw)):.4f}")

    npy_path = output_dirs["results"] / f"{label}_high_motion_indices.npy"
    np.save(npy_path, censor_indices)

    fig_path = output_dirs["figures"] / f"{label}_motion_qc.png"
    make_qc_figure(
        fd_raw=fd_active,
        dvars_raw=dvars_raw,
        censor_mask=censor_mask,
        subject_label=label,
        fd_thresh=fd_thresh,
        fig_path=fig_path,
        fd_fmriprep=fd_fmriprep,
        compare_fd=compare_fd,
    )

    print(f"  Figure saved : {fig_path}")
    print(f"  Indices saved: {npy_path}")

    return {
        "subject":         subject_id,
        "session":         session_id,
        "n_frames":        n_frames,
        **stats_fmriprep,
        **stats_custom,
        "mean_std_dvars":  float(np.nanmean(dvars_raw)),
        "max_std_dvars":   float(np.nanmax(dvars_raw)),
    }
