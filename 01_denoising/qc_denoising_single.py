#!/usr/bin/env python3
"""
Denoising QC for sub-01 ses-01.

Computes DVARS and FD-DVARS correlation pre- and post-denoising to verify
that nuisance regression reduced motion-coupled variance.
"""

import sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import pearsonr

# ─── Paths ────────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent

BOLD_PRE_PATH = Path(
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

DENOISING_DIR  = Path(__file__).resolve().parent
BOLD_GSR_PATH  = DENOISING_DIR / "results" / "sub-01_ses-01_task-rest_desc-denoisedGSR_bold.nii.gz"
BOLD_NOGSR_PATH = DENOISING_DIR / "results" / "sub-01_ses-01_task-rest_desc-denoisedNoGSR_bold.nii.gz"

FIG_DIR    = DENOISING_DIR / "figures"
RESULT_DIR = DENOISING_DIR / "results"

SUBJECT = "sub-01"
SESSION = "ses-01"
TR      = 1.8

# ─── Helpers ─────────────────────────────────────────────────────────────────

def compute_dvars(bold_data: np.ndarray, mask_flat: np.ndarray) -> np.ndarray:
    """RMS of voxel-wise temporal derivative within brain mask. Length = n_times."""
    n_times = bold_data.shape[3]
    data_2d = bold_data.reshape(-1, n_times).astype(np.float64)
    masked  = data_2d[mask_flat]              # (n_vox, n_times)
    diff    = np.diff(masked, axis=1)         # (n_vox, n_times-1)
    rms     = np.sqrt(np.mean(diff ** 2, axis=0))  # (n_times-1,)
    # Pad frame 0 with NaN to align with frame indices
    return np.concatenate([[np.nan], rms])


def fd_dvars_corr(fd: np.ndarray, dvars: np.ndarray) -> float:
    """Pearson r between FD and DVARS, excluding NaN frames from either."""
    valid = ~np.isnan(fd) & ~np.isnan(dvars)
    if valid.sum() < 3:
        return np.nan
    r, _ = pearsonr(fd[valid], dvars[valid])
    return float(r)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    sys.path.insert(0, str(REPO_ROOT))
    from utils.motion_qc import compute_custom_fd
    from utils.thesis_style import apply_thesis_style, PALETTE

    apply_thesis_style()

    # Load mask
    print(f"Loading mask: {MASK_PATH}")
    mask_flat = nib.load(MASK_PATH).get_fdata().astype(bool).ravel()

    # Load confounds and compute FD
    print(f"Loading confounds: {CONFOUNDS_PATH}")
    confounds_df = pd.read_csv(CONFOUNDS_PATH, sep="\t")
    fd = compute_custom_fd(confounds_df, tr=TR, backward_diff_n=1, verbose=False)

    # Load BOLD files and compute DVARS
    print(f"Loading pre-denoising BOLD: {BOLD_PRE_PATH}")
    dvars_pre = compute_dvars(nib.load(BOLD_PRE_PATH).get_fdata(dtype=np.float32), mask_flat)

    print(f"Loading +GSR BOLD: {BOLD_GSR_PATH}")
    dvars_gsr = compute_dvars(nib.load(BOLD_GSR_PATH).get_fdata(dtype=np.float32), mask_flat)

    print(f"Loading -GSR BOLD: {BOLD_NOGSR_PATH}")
    dvars_nogsr = compute_dvars(nib.load(BOLD_NOGSR_PATH).get_fdata(dtype=np.float32), mask_flat)

    n_frames = len(fd)

    # FD-DVARS correlations
    r_pre   = fd_dvars_corr(fd, dvars_pre)
    r_nogsr = fd_dvars_corr(fd, dvars_nogsr)
    r_gsr   = fd_dvars_corr(fd, dvars_gsr)

    # Mean DVARS (excluding NaN at frame 0)
    mean_pre   = float(np.nanmean(dvars_pre))
    mean_nogsr = float(np.nanmean(dvars_nogsr))
    mean_gsr   = float(np.nanmean(dvars_gsr))

    pct_nogsr = 100.0 * (mean_pre - mean_nogsr) / mean_pre
    pct_gsr   = 100.0 * (mean_pre - mean_gsr)   / mean_pre

    # ─── Figure ──────────────────────────────────────────────────────────────

    frames = np.arange(n_frames)

    dvars_all = np.concatenate([
        dvars_pre[~np.isnan(dvars_pre)],
        dvars_nogsr[~np.isnan(dvars_nogsr)],
        dvars_gsr[~np.isnan(dvars_gsr)],
    ])
    dvars_ymin = max(0.0, dvars_all.min() * 0.95)
    dvars_ymax = dvars_all.max() * 1.05

    fig, axes = plt.subplots(
        4, 1, figsize=(8, 9), sharex=True,
        gridspec_kw={"hspace": 0.45},
    )

    # Panel 1: FD
    axes[0].plot(frames, fd, color="#444444", linewidth=0.9)
    axes[0].axhline(0.3, color="black", linestyle="--", linewidth=0.8, label="0.3 mm")
    axes[0].set_ylabel("FD (mm)")
    axes[0].set_title("Framewise displacement")
    axes[0].legend(loc="upper right")

    # Panel 2: DVARS pre
    axes[1].plot(frames, dvars_pre, color=PALETTE["group_A_pre"], linewidth=0.9)
    axes[1].set_ylabel("DVARS")
    axes[1].set_title("DVARS — pre-denoising")
    axes[1].set_ylim(dvars_ymin, dvars_ymax)

    # Panel 3: DVARS -GSR
    axes[2].plot(frames, dvars_nogsr, color=PALETTE["group_A_post"], linewidth=0.9)
    axes[2].set_ylabel("DVARS")
    axes[2].set_title("DVARS — post-denoising (-GSR)")
    axes[2].set_ylim(dvars_ymin, dvars_ymax)

    # Panel 4: DVARS +GSR
    axes[3].plot(frames, dvars_gsr, color=PALETTE["group_B_pre"], linewidth=0.9)
    axes[3].set_ylabel("DVARS")
    axes[3].set_title("DVARS — post-denoising (+GSR)")
    axes[3].set_ylim(dvars_ymin, dvars_ymax)
    axes[3].set_xlabel("Frame index")

    fig.suptitle(f"Denoising QC — {SUBJECT} {SESSION}", fontsize=11, y=1.01)

    png_path = FIG_DIR / f"{SUBJECT}_{SESSION}_qc_denoising.png"
    pdf_path = FIG_DIR / f"{SUBJECT}_{SESSION}_qc_denoising.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"\nFigure saved: {png_path}")
    print(f"Figure saved: {pdf_path}")

    # ─── Console summary ─────────────────────────────────────────────────────

    summary = (
        f"\n─── Denoising QC: {SUBJECT} {SESSION} ───\n"
        f"\nMean DVARS:\n"
        f"  Pre-denoising:        {mean_pre:.3f}\n"
        f"  Post-denoising -GSR:  {mean_nogsr:.3f} ({pct_nogsr:.1f}% reduction)\n"
        f"  Post-denoising +GSR:  {mean_gsr:.3f} ({pct_gsr:.1f}% reduction)\n"
        f"\nFD-DVARS Pearson correlation:\n"
        f"  Pre-denoising:        r = {r_pre:.3f}\n"
        f"  Post-denoising -GSR:  r = {r_nogsr:.3f}\n"
        f"  Post-denoising +GSR:  r = {r_gsr:.3f}\n"
        f"\nInterpretation:\n"
        f"  - Lower DVARS post-denoising indicates noise reduction\n"
        f"  - Lower FD-DVARS coupling indicates motion-related variance was successfully removed\n"
    )
    print(summary)

    log_path = RESULT_DIR / f"{SUBJECT}_{SESSION}_qc_denoising_log.txt"
    log_path.write_text(summary)
    print(f"Log saved: {log_path}")


if __name__ == "__main__":
    main()
