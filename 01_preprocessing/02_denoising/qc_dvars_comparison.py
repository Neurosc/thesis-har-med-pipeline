#!/usr/bin/env python3
"""
DVARS before/after denoising comparison — sub-21 ses-01.

Panels (shared x-axis, frame index):
  1  FD                          (black, threshold at 0.3 mm)
  2  DVARS raw fMRIPrep          (#3D5A6C)
  3  DVARS post-denoising -GSR   (#7A95A6)
  4  DVARS post-denoising +GSR   (#8B7355)

Outputs
-------
  01_preprocessing/02_denoising/figures/sub-01_ses-01_dvars_comparison.png  (300 dpi)
  01_preprocessing/02_denoising/figures/sub-01_ses-01_dvars_comparison.pdf
  01_preprocessing/02_denoising/results/sub-01_ses-01_dvars_comparison.txt

Run on server (headless):
  conda activate fmri
  python 01_preprocessing/02_denoising/qc_dvars_comparison.py
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

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.motion_qc import compute_custom_fd
from utils.thesis_style import apply_thesis_style

# ─── Paths ────────────────────────────────────────────────────────────────────

FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
DENOISED_ROOT = Path(__file__).resolve().parent / "results"
OUT_DIR       = Path(__file__).resolve().parent / "figures"
LOG_DIR       = Path(__file__).resolve().parent / "results"

SUB = "sub-21"
SES = "ses-01"
TR  = 1.8

TAG      = f"{SUB}_{SES}_task-rest"
FUNC_DIR = FMRIPREP_ROOT / SUB / SES / "func"

BOLD_RAW   = FUNC_DIR / f"{TAG}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
MASK_PATH  = FUNC_DIR / f"{TAG}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
CONFOUNDS  = FUNC_DIR / f"{TAG}_desc-confounds_timeseries.tsv.gz"
BOLD_NOGSR = DENOISED_ROOT / f"{TAG}_desc-denoisedNoGSR_bold.nii.gz"
BOLD_GSR   = DENOISED_ROOT / f"{TAG}_desc-denoisedGSR_bold.nii.gz"

OUT_PNG  = OUT_DIR / f"{SUB}_{SES}_dvars_comparison.png"
OUT_PDF  = OUT_DIR / f"{SUB}_{SES}_dvars_comparison.pdf"
OUT_LOG  = LOG_DIR / f"{SUB}_{SES}_dvars_comparison.txt"


# ─── DVARS ────────────────────────────────────────────────────────────────────

def compute_dvars(Y_masked):
    """Y_masked: voxels × time. Returns DVARS length T-1."""
    diff = np.diff(Y_masked, axis=1)
    return np.sqrt(np.mean(diff ** 2, axis=0))


def load_masked_bold(bold_path, mask_path):
    """Return brain-masked BOLD as (n_vox, T) float32."""
    bold_data = nib.load(str(bold_path)).get_fdata(dtype=np.float32)
    mask_flat = nib.load(str(mask_path)).get_fdata().astype(bool).ravel()
    T = bold_data.shape[3]
    return bold_data.reshape(-1, T)[mask_flat]


# ─── Figure ───────────────────────────────────────────────────────────────────

def make_figure(fd, dvars_raw, dvars_nogsr, dvars_gsr):
    apply_thesis_style()

    fig, axes = plt.subplots(
        4, 1,
        figsize=(7, 8),
        sharex=True,
        gridspec_kw={"hspace": 0.35},
    )
    ax1, ax2, ax3, ax4 = axes

    T  = len(fd)
    xF = np.arange(T)         # frame index for FD (length T)
    xD = np.arange(1, T)      # frame index for DVARS (length T-1)

    # Shared y-range for all DVARS panels
    all_dvars = np.concatenate([dvars_raw, dvars_nogsr, dvars_gsr])
    dv_lo = all_dvars.min() * 0.95
    dv_hi = all_dvars.max() * 1.05

    # Panel 1 — FD
    ax1.plot(xF, fd, color="black", lw=1.2)
    ax1.axhline(0.3, color="#E41A1C", lw=1.0, ls="--")
    ax1.set_ylabel("FD (mm)")
    ax1.set_title("Framewise Displacement")

    # Panel 2 — Raw DVARS
    ax2.plot(xD, dvars_raw, color="#3D5A6C", lw=1.2)
    ax2.set_ylabel("DVARS")
    ax2.set_title("DVARS — pre-denoising (raw fMRIPrep)")
    ax2.set_ylim(dv_lo, dv_hi)

    # Panel 3 — -GSR DVARS
    ax3.plot(xD, dvars_nogsr, color="#7A95A6", lw=1.2)
    ax3.set_ylabel("DVARS")
    ax3.set_title("DVARS — post-denoising (-GSR)")
    ax3.set_ylim(dv_lo, dv_hi)

    # Panel 4 — +GSR DVARS
    ax4.plot(xD, dvars_gsr, color="#8B7355", lw=1.2)
    ax4.set_ylabel("DVARS")
    ax4.set_title("DVARS — post-denoising (+GSR)")
    ax4.set_ylim(dv_lo, dv_hi)
    ax4.set_xlabel("Frame index")

    for ax in axes:
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.spines['left'].set_position(('data', 0))
        ax.spines['bottom'].set_position(('data', 0))

    fig.patch.set_facecolor("white")
    return fig


# ─── Console summary ──────────────────────────────────────────────────────────

def print_summary(fd, dvars_raw, dvars_nogsr, dvars_gsr):
    mean_raw   = float(dvars_raw.mean())
    mean_nogsr = float(dvars_nogsr.mean())
    mean_gsr   = float(dvars_gsr.mean())

    pct_nogsr = (mean_raw - mean_nogsr) / mean_raw * 100
    pct_gsr   = (mean_raw - mean_gsr)   / mean_raw * 100

    fd_trim = fd[1:]   # align with DVARS (T-1 frames)
    r_raw,   _ = pearsonr(fd_trim, dvars_raw)
    r_nogsr, _ = pearsonr(fd_trim, dvars_nogsr)
    r_gsr,   _ = pearsonr(fd_trim, dvars_gsr)

    lines = [
        f"─── DVARS comparison: {SUB} {SES} ───",
        "",
        "Mean DVARS:",
        f"  Raw:                {mean_raw:.3f}",
        f"  Post -GSR:          {mean_nogsr:.3f} ({pct_nogsr:.1f}% reduction)",
        f"  Post +GSR:          {mean_gsr:.3f} ({pct_gsr:.1f}% reduction)",
        "",
        "FD-DVARS Pearson correlation:",
        f"  Raw:                r = {r_raw:.3f}",
        f"  Post -GSR:          r = {r_nogsr:.3f}",
        f"  Post +GSR:          r = {r_gsr:.3f}",
    ]
    summary = "\n".join(lines)
    print(summary)
    return summary


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    for path, label in [(BOLD_RAW, "raw"), (BOLD_NOGSR, "-GSR"), (BOLD_GSR, "+GSR"),
                        (MASK_PATH, "mask"), (CONFOUNDS, "confounds")]:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"Loading confounds ...")
    conf_df = pd.read_csv(str(CONFOUNDS), sep="\t")
    fd = compute_custom_fd(conf_df, tr=TR, backward_diff_n=1, verbose=False)

    print("Loading raw BOLD ...")
    Y_raw   = load_masked_bold(BOLD_RAW,   MASK_PATH)
    print("Loading -GSR denoised BOLD ...")
    Y_nogsr = load_masked_bold(BOLD_NOGSR, MASK_PATH)
    print("Loading +GSR denoised BOLD ...")
    Y_gsr   = load_masked_bold(BOLD_GSR,   MASK_PATH)

    print("Computing DVARS ...")
    dvars_raw   = compute_dvars(Y_raw.astype(np.float64))
    dvars_nogsr = compute_dvars(Y_nogsr.astype(np.float64))
    dvars_gsr   = compute_dvars(Y_gsr.astype(np.float64))

    summary = print_summary(fd, dvars_raw, dvars_nogsr, dvars_gsr)

    OUT_LOG.write_text(summary + "\n")
    print(f"\nLog saved: {OUT_LOG}")

    print("Generating figure ...")
    fig = make_figure(fd, dvars_raw, dvars_nogsr, dvars_gsr)
    for out in (OUT_PNG, OUT_PDF):
        fig.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
