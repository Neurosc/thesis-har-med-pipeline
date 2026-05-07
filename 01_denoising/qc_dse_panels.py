#!/usr/bin/env python3
"""
DSE/DVARS QC panels — faithful replication of Afyouni & Nichols (2018) Fig. 5.

DO NOT call apply_thesis_style() here; fmri_diag_plot() handles its own
serif-font rcParams and restores them on exit.

Outputs (01_denoising/figures/dse_panels/)
------------------------------------------
  sub-01_ses-01_dse_raw.png / .pdf
  sub-01_ses-01_dse_GSR.png / .pdf
  sub-01_ses-01_dse_NoGSR.png / .pdf
  sub-12_ses-01_dse_raw.png / .pdf  ...  sub-39_ses-02_dse_raw.png / .pdf

Run on server (headless):
  conda activate fmri
  python 01_denoising/qc_dse_panels.py
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

from utils.dse_dvars import dse_decomposition, dvars_calc, fmri_diag_plot
from utils.motion_qc import compute_custom_fd

# ─── Paths ────────────────────────────────────────────────────────────────────

FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
DENOISED_ROOT = Path(__file__).resolve().parent / "results"
OUT_DIR       = Path(__file__).resolve().parent / "figures" / "dse_panels"

TR = 1.8   # seconds

# sub-01 ses-01 gets raw + GSR + NoGSR; all others get raw only
RUNS = [
    ("sub-01", "ses-01"),
    ("sub-12", "ses-01"),
    ("sub-12", "ses-02"),
    ("sub-21", "ses-01"),
    ("sub-21", "ses-02"),
    ("sub-39", "ses-01"),
    ("sub-39", "ses-02"),
]


# ─── Path helpers ─────────────────────────────────────────────────────────────

def _fmriprep_paths(sub, ses):
    func = FMRIPREP_ROOT / sub / ses / "func"
    tag  = f"{sub}_{ses}_task-rest"
    return {
        "bold":      func / f"{tag}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz",
        "mask":      func / f"{tag}_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz",
        "confounds": func / f"{tag}_desc-confounds_timeseries.tsv.gz",
    }


def _denoised_paths(sub, ses):
    tag = f"{sub}_{ses}_task-rest"
    return {
        "gsr":   DENOISED_ROOT / f"{tag}_desc-denoisedGSR_bold.nii.gz",
        "nogsr": DENOISED_ROOT / f"{tag}_desc-denoisedNoGSR_bold.nii.gz",
    }


# ─── Data loading ─────────────────────────────────────────────────────────────

def _load_masked_bold(bold_path, mask_path):
    """Return brain-masked BOLD as (n_vox, T) float32."""
    bold_data = nib.load(str(bold_path)).get_fdata(dtype=np.float32)
    mask_flat = nib.load(str(mask_path)).get_fdata().astype(bool).ravel()
    T = bold_data.shape[3]
    return bold_data.reshape(-1, T)[mask_flat].astype(np.float32)


def _compute_abs_mov(conf_df):
    """
    Per-frame absolute motion in mm for Panel 1.

    Returns (T, 2): col 0 = sum|Δrot|×50 mm, col 1 = sum|Δtrans| mm.
    """
    trans = conf_df[['trans_x', 'trans_y', 'trans_z']].to_numpy(dtype=float)
    rot   = conf_df[['rot_x',   'rot_y',   'rot_z'  ]].to_numpy(dtype=float)
    d_tr  = np.vstack([np.zeros((1, 3)), np.diff(trans, axis=0)])
    d_ro  = np.vstack([np.zeros((1, 3)), np.diff(rot,   axis=0)])
    return np.column_stack([np.abs(d_ro).sum(axis=1) * 50.0,
                             np.abs(d_tr).sum(axis=1)])


# ─── Fix 6: tissue-ordered carpet sort index ─────────────────────────────────

def _tissue_sort_idx(mask_path, sub):
    """
    Return a voxel sort index that orders the carpet plot rows:
      cortical GM (top) → subcortical GM → WM → CSF (bottom).

    Loads fMRIPrep MNI-space probseg files from sub/anat/, resamples to
    the BOLD mask space (nilearn.image.resample_to_img), then sorts by
    argmax tissue class, with within-GM ordering by descending GM probability
    (higher prob ≈ cortical, lower ≈ subcortical).

    Returns
    -------
    np.ndarray, shape (n_vox,) — sort indices for Y_carpet rows.
    None if probseg files are not found or nilearn is unavailable.
    """
    anat_dir = FMRIPREP_ROOT / sub / "anat"
    probsegs = {}
    for tissue in ('GM', 'WM', 'CSF'):
        # fMRIPrep 23.x: sub-XX_space-MNI152NLin2009cAsym[_res-N]_label-TISSUE_probseg.nii.gz
        matches = sorted(anat_dir.glob(
            f"{sub}_*_label-{tissue}_probseg.nii.gz"
        ))
        if not matches:
            print(f"  WARNING: No tissue probability maps found "
                  f"(missing {tissue}), carpet plot uses raw mask order.")
            return None
        # Prefer MNI152NLin2009cAsym space; take first match otherwise
        mni = [m for m in matches if 'MNI152NLin2009cAsym' in m.name]
        probsegs[tissue] = str(mni[0] if mni else matches[0])

    try:
        from nilearn.image import resample_to_img
    except ImportError:
        print("  WARNING: nilearn not available, carpet plot uses raw mask order.")
        return None

    mask_img  = nib.load(str(mask_path))
    mask_flat = mask_img.get_fdata().astype(bool).ravel()

    probs = {}
    for tissue in ('GM', 'WM', 'CSF'):
        img_r = resample_to_img(probsegs[tissue], mask_img,
                                interpolation='linear')
        probs[tissue] = np.clip(img_r.get_fdata().ravel()[mask_flat], 0, None)

    # Classify each masked voxel by argmax tissue probability
    prob_mat     = np.column_stack([probs['GM'], probs['WM'], probs['CSF']])
    tissue_label = np.argmax(prob_mat, axis=1)   # 0=GM, 1=WM, 2=CSF

    # Sort key: 0 (cortical GM, high prob) → 1 (subcortical GM) → 2 (WM) → 3 (CSF)
    # Within GM: descending GM probability  →  sort_key ∈ [-1, 0)
    sort_key = tissue_label.astype(float)
    gm_mask = tissue_label == 0
    if gm_mask.any():
        gm_max = float(probs['GM'][gm_mask].max())
        if gm_max > 0:
            sort_key[gm_mask] = -probs['GM'][gm_mask] / gm_max  # ∈ [-1, 0)

    return np.argsort(sort_key, kind='stable')


# ─── DSE / DVARS computation ──────────────────────────────────────────────────

def _run_dse_dvars(Y, label):
    print(f"    DSE decomposition ({label})  shape={Y.shape}")
    V_dse = dse_decomposition(Y.astype(np.float64), norm_scale=100)
    print(f"    DVARSCalc ({label})")
    _, dvars_stat = dvars_calc(Y.astype(np.float64), scale=1/100,
                               trans_power=1/3, var_type='hIQR',
                               mean_type='median', test_method='X2')
    return V_dse, dvars_stat


# ─── Figure generation ────────────────────────────────────────────────────────

def _save_figure(V_dse, dvars_stat, fd, abs_mov, Y_carpet, stem):
    """Build the 4-panel figure and write .png + .pdf at 300 dpi."""
    fig = fmri_diag_plot(
        V_dse           = V_dse,
        dvars_stat      = dvars_stat,
        fd              = fd,
        abs_mov         = abs_mov,
        Y_carpet        = Y_carpet,
        figsize         = (8, 10),
        fd_thresh       = 0.5,
        prac_sig_thresh = 5.0,
    )
    for ext in ('png', 'pdf'):
        out = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(str(out), dpi=300, bbox_inches='tight')
        print(f"    Saved: {out}")
    plt.close(fig)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for sub, ses in RUNS:
        print(f"\n{'='*60}")
        print(f"  {sub}  {ses}")
        print(f"{'='*60}")

        paths = _fmriprep_paths(sub, ses)

        if not paths['bold'].exists():
            print(f"  BOLD not found, skipping: {paths['bold']}")
            continue
        if not paths['mask'].exists():
            print(f"  Mask not found, skipping: {paths['mask']}")
            continue

        print(f"  Loading raw BOLD ...")
        Y_raw   = _load_masked_bold(paths['bold'], paths['mask'])
        conf_df = pd.read_csv(str(paths['confounds']), sep='\t')
        fd_ts   = compute_custom_fd(conf_df, tr=TR, backward_diff_n=1,
                                     verbose=False)
        abs_mov = _compute_abs_mov(conf_df)

        # Fix 6: tissue-sorted voxel order for carpet plot
        print(f"  Computing tissue sort order ...")
        sort_idx = _tissue_sort_idx(paths['mask'], sub)
        if sort_idx is not None:
            print(f"    Tissue sort: {(sort_idx < len(sort_idx)).sum()} voxels ordered")

        def _apply_sort(Y):
            return Y[sort_idx] if sort_idx is not None else Y

        V_raw, dstat_raw = _run_dse_dvars(Y_raw, 'raw')
        _save_figure(V_raw, dstat_raw, fd_ts, abs_mov, _apply_sort(Y_raw),
                     stem=f"{sub}_{ses}_dse_raw")

        # sub-01 ses-01: also generate +GSR and -GSR denoised panels
        if sub == 'sub-01' and ses == 'ses-01':
            den = _denoised_paths(sub, ses)

            if den['gsr'].exists():
                print(f"  Loading +GSR denoised BOLD ...")
                Y_gsr = _load_masked_bold(den['gsr'], paths['mask'])
                V_gsr, dstat_gsr = _run_dse_dvars(Y_gsr, '+GSR')
                _save_figure(V_gsr, dstat_gsr, fd_ts, abs_mov,
                             _apply_sort(Y_gsr),
                             stem=f"{sub}_{ses}_dse_GSR")
            else:
                print(f"  +GSR not found, skipping: {den['gsr']}")

            if den['nogsr'].exists():
                print(f"  Loading -GSR denoised BOLD ...")
                Y_nogsr = _load_masked_bold(den['nogsr'], paths['mask'])
                V_nogsr, dstat_nogsr = _run_dse_dvars(Y_nogsr, '-GSR')
                _save_figure(V_nogsr, dstat_nogsr, fd_ts, abs_mov,
                             _apply_sort(Y_nogsr),
                             stem=f"{sub}_{ses}_dse_NoGSR")
            else:
                print(f"  -GSR not found, skipping: {den['nogsr']}")

    print(f"\nDone. Figures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
