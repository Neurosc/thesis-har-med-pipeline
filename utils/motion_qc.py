import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


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


def compute_stats(fd_raw, dvars_raw, fd_thresh):
    """Return summary statistics dict; all stats exclude NaN (frame 0)."""
    n_frames = len(fd_raw)
    n_censored = int(np.sum(fd_raw > fd_thresh, where=~np.isnan(fd_raw)))
    return {
        "n_frames": n_frames,
        "mean_fd": float(np.nanmean(fd_raw)),
        "median_fd": float(np.nanmedian(fd_raw)),
        "max_fd": float(np.nanmax(fd_raw)),
        "n_censored": n_censored,
        "pct_censored": 100.0 * n_censored / n_frames,
        "mean_std_dvars": float(np.nanmean(dvars_raw)),
        "max_std_dvars": float(np.nanmax(dvars_raw)),
        "n_frames_remaining": n_frames - n_censored,
    }


def make_qc_figure(fd_raw, dvars_raw, censor_mask, subject_label, fd_thresh, fig_path):
    """Save 3-panel motion QC figure (FD / std_dvars / censored strip)."""
    n_frames = len(fd_raw)
    frames = np.arange(n_frames)
    cmap = matplotlib.colors.ListedColormap(["#d0d0d0", "#cc2222"])

    fig, axes = plt.subplots(
        3, 1, figsize=(10, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 3, 1]}
    )
    fig.suptitle(f"Motion QC — {subject_label}", fontsize=13, y=1.01)

    # Panel 1: FD
    axes[0].plot(frames, fd_raw, color="steelblue", linewidth=1.2)
    axes[0].axhline(fd_thresh, color="black", linestyle="--", linewidth=1,
                    label=f"Threshold ({fd_thresh} mm)")
    axes[0].set_ylabel("FD (mm)")
    axes[0].set_title("Framewise Displacement")
    axes[0].legend(fontsize=8, loc="upper right")

    # Panel 2: std_dvars
    axes[1].plot(frames, dvars_raw, color="darkorange", linewidth=1.2)
    axes[1].axhline(1.5, color="black", linestyle="--", linewidth=1,
                    label="Threshold (1.5, Power et al. 2014)")
    axes[1].set_ylim(0, 5)
    axes[1].set_ylabel("std_dvars")
    axes[1].set_title("Standardized DVARS (std_dvars)")
    axes[1].legend(fontsize=8, loc="upper right")

    # Panel 3: censored frames strip
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
                  fd_thresh=0.3, dvars_thresh=1.5):
    """
    Run motion QC for one subject-session.

    Parameters
    ----------
    confounds_path : Path
    subject_id     : str  e.g. 'sub-01'
    session_id     : str  e.g. 'ses-01'
    output_dirs    : dict with keys 'figures' and 'results' (Path objects)
    fd_thresh      : float
    dvars_thresh   : float  (stored in summary but not used for censoring)

    Returns
    -------
    dict of summary statistics
    """
    label = f"{subject_id}_{session_id}"
    print(f"Loading confounds: {confounds_path}")

    df = load_confounds(confounds_path)
    fd_raw = df["framewise_displacement"].to_numpy(dtype=float)
    dvars_raw = df["std_dvars"].to_numpy(dtype=float)

    stats = compute_stats(fd_raw, dvars_raw, fd_thresh)

    print(f"\n── Motion summary: {label} ──")
    print(f"  Total frames        : {stats['n_frames']}")
    print(f"  Mean FD             : {stats['mean_fd']:.4f} mm")
    print(f"  Median FD           : {stats['median_fd']:.4f} mm")
    print(f"  Max FD              : {stats['max_fd']:.4f} mm")
    print(f"  Frames FD > {fd_thresh} mm : {stats['n_censored']}")
    print(f"  Censored            : {stats['pct_censored']:.1f}%")
    print(f"  Mean std_dvars      : {stats['mean_std_dvars']:.4f}")

    censor_mask = np.where(np.isnan(fd_raw), False, fd_raw > fd_thresh)
    censor_indices = np.where(censor_mask)[0].astype(int)

    npy_path = output_dirs["results"] / f"{label}_high_motion_indices.npy"
    np.save(npy_path, censor_indices)

    fig_path = output_dirs["figures"] / f"{label}_motion_qc.png"
    make_qc_figure(fd_raw, dvars_raw, censor_mask, label, fd_thresh, fig_path)

    print(f"  Figure saved : {fig_path}")
    print(f"  Indices saved: {npy_path}")

    return {"subject": subject_id, "session": session_id, **stats}
