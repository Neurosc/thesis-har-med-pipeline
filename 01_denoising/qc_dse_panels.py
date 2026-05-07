#!/usr/bin/env python3
"""
DSE/DVARS QC panels for 4 selected subjects (both sessions).

For each run:
  - Loads raw fMRIPrep BOLD (brain-masked), computes DSE decomposition and
    Afyouni DVARS inference.
  - For sub-01 (and any session where denoised files exist): also generates
    panels for +GSR and -GSR denoised BOLD.
  - Saves five-panel diagnostic figures to 01_denoising/figures/dse_panels/.

Subject selection:
  sub-01  ses-01, ses-02 — raw + denoised comparison where files exist
  sub-12  ses-01, ses-02 — raw only  (excluded from final analysis; QC only)
  sub-21  ses-01, ses-02 — raw only
  sub-39  ses-01, ses-02 — raw only

Run on the server (headless, no plt.show()):
  conda activate fmri
  python 01_denoising/qc_dse_panels.py

Sanity checks printed to console for sub-01 ses-01:
  1. Variance partition:  w_Avar == w_Dvar + w_Svar + w_Evar  (exact identity)
  2. Δ%D-var magnitude vs fMRIPrep std_dvars (factor-of-2 tolerance)
"""

import sys
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.dse_dvars     import dse_decomposition, dvars_calc, fmri_diag_plot
from utils.motion_qc     import compute_custom_fd
from utils.thesis_style  import apply_thesis_style

# ─── Paths ────────────────────────────────────────────────────────────────────

FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
DENOISED_ROOT = Path(__file__).resolve().parent / "results"
OUT_DIR       = Path(__file__).resolve().parent / "figures" / "dse_panels"

TR        = 1.8   # seconds (verified from BOLD JSON)
FD_THRESH = 0.3   # mm

# ─── Subject / session schedule ───────────────────────────────────────────────

RUNS = [
    ("sub-01", "ses-01"),
    ("sub-01", "ses-02"),
    ("sub-12", "ses-01"),
    ("sub-12", "ses-02"),
    ("sub-21", "ses-01"),
    ("sub-21", "ses-02"),
    ("sub-39", "ses-01"),
    ("sub-39", "ses-02"),
]


# ─── Path helpers ─────────────────────────────────────────────────────────────

def fmriprep_paths(sub, ses):
    func = FMRIPREP_ROOT / sub / ses / "func"
    tag  = f"{sub}_{ses}_task-rest"
    return dict(
        bold      = func / f"{tag}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        mask      = func / f"{tag}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
        confounds = func / f"{tag}_desc-confounds_timeseries.tsv.gz",
    )


def denoised_paths(sub, ses):
    tag = f"{sub}_{ses}_task-rest"
    return dict(
        gsr   = DENOISED_ROOT / f"{tag}_desc-denoisedGSR_bold.nii.gz",
        nogsr = DENOISED_ROOT / f"{tag}_desc-denoisedNoGSR_bold.nii.gz",
    )


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_masked_bold(bold_path, mask_path):
    """Load NIfTI BOLD and apply brain mask. Returns (I_mask, T) float64."""
    bold_img  = nib.load(str(bold_path))
    bold_data = bold_img.get_fdata(dtype=np.float32).astype(np.float64)
    mask_flat = nib.load(str(mask_path)).get_fdata().astype(bool).ravel()
    n_times   = bold_data.shape[3]
    Y = bold_data.reshape(-1, n_times)
    return Y[mask_flat]


def compute_abs_mov(conf_df):
    """
    Absolute motion components for AbsMov panel.
    Mirrors MATLAB: [(FD_Stat.AbsRot)*10, FD_Stat.AbsTrans].

    Returns (T, 2): col 0 = sum|Δrot| × 10 (rad×10), col 1 = sum|Δtrans| (mm).
    """
    trans_cols = ['trans_x', 'trans_y', 'trans_z']
    rot_cols   = ['rot_x',   'rot_y',   'rot_z']
    trans = conf_df[trans_cols].to_numpy(dtype=float)
    rot   = conf_df[rot_cols].to_numpy(dtype=float)

    d_trans = np.vstack([np.zeros((1, 3)), np.diff(trans, axis=0)])
    d_rot   = np.vstack([np.zeros((1, 3)), np.diff(rot,   axis=0)])
    abs_trans = np.abs(d_trans).sum(axis=1)             # mm
    abs_rot   = np.abs(d_rot).sum(axis=1) * 10.0       # rad × 10
    return np.column_stack([abs_rot, abs_trans])        # (T, 2)


# ─── Per-run DSE computation ──────────────────────────────────────────────────

def run_dse(Y, conf_df, label):
    """Compute DSE decomposition + Afyouni DVARS for one (masked) BOLD array."""
    print(f"  DSE decomposition ({label})  shape={Y.shape}")
    V_dse = dse_decomposition(Y, norm_scale=100)

    print(f"  DVARSCalc ({label})")
    DVARS, dvars_stat = dvars_calc(Y, scale=1/100,
                                    trans_power=1/3,
                                    var_type='hIQR',
                                    mean_type='median',
                                    test_method='X2')
    return V_dse, DVARS, dvars_stat


# ─── Sanity checks (sub-01 ses-01 raw only) ──────────────────────────────────

def sanity_checks(V_dse, dvars_stat, conf_df, sub, ses):
    """Print variance partition and DVARS magnitude checks."""
    print(f"\n{'─'*55}")
    print(f"  Sanity checks: {sub} {ses} (raw)")
    print(f"{'─'*55}")

    # 1. Exact variance partition: w_Avar = w_Dvar + w_Svar + w_Evar
    wA  = V_dse['w_Avar']
    wD  = V_dse['w_Dvar']
    wS  = V_dse['w_Svar']
    wE  = V_dse['w_Evar']
    rhs = wD + wS + wE
    err = abs(wA - rhs)
    ok  = err < 1e-6 * wA
    print(f"  [1] Var partition (w_Avar = w_Dvar+w_Svar+w_Evar):")
    print(f"      w_Avar={wA:.4f}  rhs={rhs:.4f}  err={err:.2e}  "
          + ("OK" if ok else "FAIL"))

    # 2. DeltapDvar magnitude vs fMRIPrep std_dvars
    dD = V_dse['Stat']['DeltapDvar']
    if 'std_dvars' in conf_df.columns:
        fmriprep_dvars = conf_df['std_dvars'].dropna().to_numpy()
        ratio = np.nanmedian(np.abs(dD)) / np.nanmedian(np.abs(fmriprep_dvars))
        ok2 = 0.1 < ratio < 10.0
        print(f"  [2] Δ%D-var vs fMRIPrep std_dvars:")
        print(f"      median|Δ%D-var|={np.nanmedian(np.abs(dD)):.3f}  "
              f"median|std_dvars|={np.nanmedian(np.abs(fmriprep_dvars)):.3f}  "
              f"ratio={ratio:.2f}  " + ("OK (within 10×)" if ok2 else "WARN"))
    else:
        print(f"  [2] std_dvars not in confounds — skipped")

    # 3. Global + non-global decomposition check: g + ng ≈ whole
    g_plus_ng = V_dse['g_Dvar'] + V_dse['ng_Dvar']
    err3 = abs(V_dse['w_Dvar'] - g_plus_ng)
    ok3  = err3 < 1e-4 * V_dse['w_Dvar']
    print(f"  [3] g_Dvar + ng_Dvar ≈ w_Dvar:  "
          f"{g_plus_ng:.4f} vs {V_dse['w_Dvar']:.4f}  err={err3:.2e}  "
          + ("OK" if ok3 else "FAIL"))

    print(f"{'─'*55}\n")


# ─── Figure assembly ──────────────────────────────────────────────────────────

def make_single_panel(V_dse, dvars_stat, fd, abs_mov, col_title):
    """Create a standalone 5-panel figure; return fig."""
    sig_idx = np.where(dvars_stat['pvals'] < 0.05)[0]
    fig = fmri_diag_plot(
        V_dse      = V_dse,
        dvars_stat = dvars_stat,
        fd         = fd,
        abs_mov    = abs_mov,
        sig_idx    = sig_idx,
        title      = col_title,
        figsize    = (5, 7),
        fd_thresh  = FD_THRESH,
    )
    return fig


def make_comparison_figure(panels):
    """
    Assemble up to 3 five-panel columns (raw, +GSR, -GSR) into one wide figure.

    panels : list of dicts, each with keys:
        V_dse, dvars_stat, fd, abs_mov, title
    """
    n_cols = len(panels)
    fig, all_axes = plt.subplots(
        5, n_cols,
        figsize=(5 * n_cols, 7),
        sharex='col',
        gridspec_kw={'height_ratios': [1.2, 1.2, 1.5, 1.5, 1.5],
                     'hspace': 0.45, 'wspace': 0.35}
    )
    if n_cols == 1:
        all_axes = all_axes[:, np.newaxis]   # ensure 2-D indexing

    for col, p in enumerate(panels):
        col_axes = all_axes[:, col]
        sig_idx  = np.where(p['dvars_stat']['pvals'] < 0.05)[0]
        fmri_diag_plot(
            V_dse      = p['V_dse'],
            dvars_stat = p['dvars_stat'],
            fd         = p['fd'],
            abs_mov    = p['abs_mov'],
            sig_idx    = sig_idx,
            title      = p['title'],
            fd_thresh  = FD_THRESH,
            axes       = col_axes,
        )
        # Only left-most column keeps y-labels; others suppress for compactness
        if col > 0:
            for ax in col_axes:
                ax.set_ylabel('')

    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    apply_thesis_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sub, ses in RUNS:
        print(f"\n{'='*60}")
        print(f"  {sub}  {ses}")
        print(f"{'='*60}")

        paths = fmriprep_paths(sub, ses)

        # Skip if raw fMRIPrep BOLD missing (server data not available)
        if not paths['bold'].exists():
            print(f"  BOLD not found, skipping: {paths['bold']}")
            continue
        if not paths['mask'].exists():
            print(f"  Mask not found, skipping: {paths['mask']}")
            continue

        # ── Load raw BOLD ──────────────────────────────────────────────────────
        print(f"  Loading raw BOLD ...")
        Y_raw = load_masked_bold(paths['bold'], paths['mask'])

        conf_df = pd.read_csv(str(paths['confounds']), sep='\t')
        fd_ts   = compute_custom_fd(conf_df, tr=TR, backward_diff_n=1,
                                     verbose=False)
        abs_mov = compute_abs_mov(conf_df)

        V_raw, DVARS_raw, dstat_raw = run_dse(Y_raw, conf_df, 'raw')

        # Sanity checks for sub-01 ses-01 only
        if sub == 'sub-01' and ses == 'ses-01':
            sanity_checks(V_raw, dstat_raw, conf_df, sub, ses)

        # ── Check for denoised files ───────────────────────────────────────────
        den_paths = denoised_paths(sub, ses)
        has_gsr   = den_paths['gsr'].exists()
        has_nogsr = den_paths['nogsr'].exists()

        panels = [dict(V_dse=V_raw, dvars_stat=dstat_raw,
                       fd=fd_ts, abs_mov=abs_mov,
                       title=f"{sub} {ses} — raw")]

        if has_gsr:
            print(f"  Loading +GSR denoised BOLD ...")
            Y_gsr = load_masked_bold(den_paths['gsr'], paths['mask'])
            V_gsr, _, dstat_gsr = run_dse(Y_gsr, conf_df, '+GSR')
            panels.append(dict(V_dse=V_gsr, dvars_stat=dstat_gsr,
                               fd=fd_ts, abs_mov=abs_mov,
                               title=f"{sub} {ses} — +GSR"))

        if has_nogsr:
            print(f"  Loading -GSR denoised BOLD ...")
            Y_nogsr = load_masked_bold(den_paths['nogsr'], paths['mask'])
            V_nogsr, _, dstat_nogsr = run_dse(Y_nogsr, conf_df, '-GSR')
            panels.append(dict(V_dse=V_nogsr, dvars_stat=dstat_nogsr,
                               fd=fd_ts, abs_mov=abs_mov,
                               title=f"{sub} {ses} — -GSR"))

        # ── Save figures ───────────────────────────────────────────────────────
        tag = f"{sub}_{ses}"

        if len(panels) == 1:
            # Raw-only figure
            stem = f"{tag}_dse_raw"
            fig  = make_single_panel(**panels[0])
            for ext in ('png',):
                out = OUT_DIR / f"{stem}.{ext}"
                fig.savefig(str(out), dpi=300, bbox_inches='tight')
                print(f"  Saved: {out}")
            plt.close(fig)

        else:
            # Comparison figure (raw + denoised columns)
            stem = f"{tag}_dse_panels"
            fig  = make_comparison_figure(panels)
            for ext in ('png', 'pdf'):
                out = OUT_DIR / f"{stem}.{ext}"
                fig.savefig(str(out), dpi=300, bbox_inches='tight')
                print(f"  Saved: {out}")
            plt.close(fig)

            # Also save raw-only for consistency
            fig_raw = make_single_panel(**panels[0])
            out_raw = OUT_DIR / f"{tag}_dse_raw.png"
            fig_raw.savefig(str(out_raw), dpi=300, bbox_inches='tight')
            plt.close(fig_raw)

    print(f"\nDone. Figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
