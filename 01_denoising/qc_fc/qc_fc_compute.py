#!/usr/bin/env python3
"""
QC-FC analysis (CONN methodology) — DV and DQ metrics.

Part A — DV (Data Validity): per-subject FC histogram shape.
  DV_i = 100 * exp(-k * |mode_i / IQR_i|)   [Nieto-Castanon 2020]
  k calibrated from pre-denoising data so median pre-DV = 5%.

Part B — DQ (Data Quality): QC-FC correlation vs permutation null.
  DQ = 100 * overlap(observed QC-FC KDE, permutation null KDE)  [Nieto-Castanon 2025]
  Computed for 3 QC measures; combined DQ = minimum across measures.

Both metrics use 5000 random voxel pairs sampled with seed 42 — same pairs
across all 70 runs (35 subjects × 2 sessions) and both pre/post versions.
Only the -GSR denoised BOLD is evaluated (Lynch standard).

Outputs in 01_denoising/qc_fc/:
  results/voxel_pairs_seed42.npy      (5000, 2) flat voxel index pairs
  results/fc_per_run_pre.npy          (70, 5000) FC values, pre-denoising
  results/fc_per_run_post.npy         (70, 5000) FC values, post-denoising (-GSR)
  results/qc_fc_observed.npy          (5000,) QC-FC correlations, post, QC_MeanMotion
  results/qc_fc_null.npy              (1000, 5000) permutation null, post, QC_MeanMotion
  results/dv_scores.tsv               per-subject DV scores (post-denoising)
  results/dq_scores.txt               DQ summary
  figures/fig_dv_distributions.png/.pdf
  figures/fig_dq_qcfc.png/.pdf

Run on server (headless):
  conda activate fmri
  python 01_denoising/qc_fc/qc_fc_compute.py
"""

import gc
import sys
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import gaussian_kde, iqr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from utils.subject_filter import get_included_subjects
from utils.motion_qc import compute_custom_fd
from utils.thesis_style import apply_thesis_style

# ─── Paths ────────────────────────────────────────────────────────────────────

FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
DENOISED_ROOT = REPO_ROOT / "01_denoising" / "results"
QC_DIR        = Path(__file__).resolve().parent
RESULTS_DIR   = QC_DIR / "results"
FIGURES_DIR   = QC_DIR / "figures"

PAIRS_NPY     = RESULTS_DIR / "voxel_pairs_seed42.npy"
FC_PRE_NPY    = RESULTS_DIR / "fc_per_run_pre.npy"
FC_POST_NPY   = RESULTS_DIR / "fc_per_run_post.npy"
QCFC_OBS_NPY  = RESULTS_DIR / "qc_fc_observed.npy"
QCFC_NULL_NPY = RESULTS_DIR / "qc_fc_null.npy"
DV_TSV        = RESULTS_DIR / "dv_scores.tsv"
DQ_TXT        = RESULTS_DIR / "dq_scores.txt"

DV_FIG_PNG    = FIGURES_DIR / "fig_dv_distributions.png"
DV_FIG_PDF    = FIGURES_DIR / "fig_dv_distributions.pdf"
DQ_FIG_PNG    = FIGURES_DIR / "fig_dq_qcfc.png"
DQ_FIG_PDF    = FIGURES_DIR / "fig_dq_qcfc.pdf"

# ─── Parameters ───────────────────────────────────────────────────────────────

TR          = 1.8    # seconds (verified from BOLD JSON; NOT 1.5)
FD_THRESH   = 0.3    # mm — Goldberg et al. 2024 for ACW
N_PAIRS     = 5000
SEED        = 42
SESSIONS    = ["ses-01", "ses-02"]
TASK        = "task-rest"
SPACE       = "MNI152NLin2009cAsym"
N_PERMS     = 1000


# ─── Path helpers ─────────────────────────────────────────────────────────────

def _func_dir(subject, session):
    return FMRIPREP_ROOT / subject / session / "func"


def _tag(subject, session):
    return f"{subject}_{session}_{TASK}"


def bold_pre_path(subject, session):
    return _func_dir(subject, session) / (
        f"{_tag(subject, session)}_space-{SPACE}_desc-preproc_bold.nii.gz"
    )


def bold_post_path(subject, session):
    return DENOISED_ROOT / f"{_tag(subject, session)}_desc-denoisedNoGSR_bold.nii.gz"


def mask_path(subject, session):
    return _func_dir(subject, session) / (
        f"{_tag(subject, session)}_space-{SPACE}_desc-brain_mask.nii.gz"
    )


def confounds_path(subject, session):
    return _func_dir(subject, session) / (
        f"{_tag(subject, session)}_desc-confounds_timeseries.tsv.gz"
    )


# ─── Voxel pair sampling ──────────────────────────────────────────────────────

def sample_voxel_pairs(ref_subject, ref_session):
    """
    Sample N_PAIRS unique voxel pairs from brain mask of the reference subject.
    Returns (N_PAIRS, 2) array of flat voxel indices into the BOLD volume.
    Fixed seed 42 — same pairs reused for all subjects and both pre/post.
    """
    mpath = mask_path(ref_subject, ref_session)
    mask_flat = nib.load(str(mpath)).get_fdata().astype(bool).ravel()
    brain_idx = np.where(mask_flat)[0]
    n_vox = len(brain_idx)

    rng = np.random.RandomState(SEED)
    idx_i = rng.randint(0, n_vox, N_PAIRS)
    idx_j = rng.randint(0, n_vox, N_PAIRS)

    # Guarantee no self-pairs
    bad = np.where(idx_i == idx_j)[0]
    while len(bad):
        idx_j[bad] = rng.randint(0, n_vox, len(bad))
        bad = np.where(idx_i == idx_j)[0]

    return np.stack([brain_idx[idx_i], brain_idx[idx_j]], axis=1)  # (N_PAIRS, 2)


# ─── FC computation ───────────────────────────────────────────────────────────

def compute_fc_pairs(bold_path, pairs):
    """
    Load BOLD, extract the two time series for each voxel pair, return
    vectorized Pearson r values — one per pair.  BOLD freed before return.
    Returns float32 array of shape (N_PAIRS,).
    """
    bold_4d  = nib.load(str(bold_path)).get_fdata(dtype=np.float32)
    n_vox    = int(np.prod(bold_4d.shape[:3]))
    bold_flat = bold_4d.reshape(n_vox, bold_4d.shape[3]).astype(np.float64)
    del bold_4d
    gc.collect()

    ts_i = bold_flat[pairs[:, 0], :]   # (N_PAIRS, T)
    ts_j = bold_flat[pairs[:, 1], :]
    del bold_flat
    gc.collect()

    ts_i -= ts_i.mean(1, keepdims=True)
    ts_j -= ts_j.mean(1, keepdims=True)

    cov   = (ts_i * ts_j).sum(1)
    std_i = np.sqrt((ts_i ** 2).sum(1))
    std_j = np.sqrt((ts_j ** 2).sum(1))

    return (cov / (std_i * std_j + 1e-12)).astype(np.float32)


# ─── QC measures ──────────────────────────────────────────────────────────────

def compute_qc_measures(conf_path):
    """
    Return (mean_fd, n_invalid, prop_valid) for one run.
    Uses custom FD (Power 2012, 1-TR backward diff, no respiratory filter).
    """
    df = pd.read_csv(str(conf_path), sep="\t")
    fd = compute_custom_fd(df, tr=TR, backward_diff_n=1, verbose=False)

    n_total   = len(fd)
    n_invalid = int(np.nansum(fd > FD_THRESH))
    mean_fd   = float(np.nanmean(fd))
    prop_valid = (n_total - n_invalid) / n_total

    return mean_fd, n_invalid, prop_valid


# ─── KDE mode ─────────────────────────────────────────────────────────────────

def kde_mode(values, n_grid=1000):
    r_grid = np.linspace(-1, 1, n_grid)
    return float(r_grid[np.argmax(gaussian_kde(values)(r_grid))])


# ─── DV (Data Validity) ───────────────────────────────────────────────────────

def compute_dv_scores(fc_pre, fc_post, subjects):
    """
    Compute per-subject DV scores (post-denoising only).

    DV_i = 100 * exp(-k * |mode_i / IQR_i|)

    k is calibrated from pre-denoising data: median |mode/IQR| across subjects
    maps to DV = 5% (log(20) / median_pre_displacement).  Post-denoising
    distributions centered at r≈0 will yield DV close to 100%.

    Returns
    -------
    dv_scores        : (n_subjects,) float64
    k                : float — calibration constant
    median_pre_disp  : float — median pre-denoising displacement
    """
    n_sub = len(subjects)
    disp_pre  = np.zeros(n_sub)
    disp_post = np.zeros(n_sub)

    for i in range(n_sub):
        run_a, run_b = i * 2, i * 2 + 1
        fc_sub_pre  = fc_pre[run_a:run_b + 1, :].ravel()
        fc_sub_post = fc_post[run_a:run_b + 1, :].ravel()

        disp_pre[i]  = abs(kde_mode(fc_sub_pre)  / (iqr(fc_sub_pre)  + 1e-10))
        disp_post[i] = abs(kde_mode(fc_sub_post) / (iqr(fc_sub_post) + 1e-10))

        if (i + 1) % 5 == 0:
            print(f"  DV: {i + 1}/{n_sub} subjects done")

    median_pre = float(np.median(disp_pre))
    k = np.log(20) / (median_pre + 1e-10)

    dv_scores = 100.0 * np.exp(-k * disp_post)
    return dv_scores, k, median_pre


# ─── QC-FC correlations ───────────────────────────────────────────────────────

def _pearson_fc_qc(fc_matrix, qc_vec):
    """
    Vectorized Pearson r between each column of fc_matrix and qc_vec.
    fc_matrix: (n_runs, N_PAIRS), qc_vec: (n_runs,) → (N_PAIRS,)
    """
    qc_z  = qc_vec - qc_vec.mean()
    fc_z  = fc_matrix - fc_matrix.mean(0, keepdims=True)
    qc_n  = np.sqrt((qc_z ** 2).sum()) + 1e-12
    fc_n  = np.sqrt((fc_z ** 2).sum(0)) + 1e-12
    return (fc_z * qc_z[:, np.newaxis]).sum(0) / (qc_n * fc_n)


def compute_qcfc_null(fc_matrix, qc_vec, n_perms, rng_seed):
    """
    Build permutation null: shuffle qc_vec n_perms times, recompute QC-FC.
    Returns (n_perms, N_PAIRS) float32.
    """
    rng  = np.random.RandomState(rng_seed)
    fc_z = fc_matrix - fc_matrix.mean(0, keepdims=True)
    fc_n = np.sqrt((fc_z ** 2).sum(0)) + 1e-12

    null = np.zeros((n_perms, N_PAIRS), dtype=np.float32)
    for p in range(n_perms):
        qc_p = qc_vec[rng.permutation(len(qc_vec))]
        qc_z = qc_p - qc_p.mean()
        qc_n = np.sqrt((qc_z ** 2).sum()) + 1e-12
        null[p] = (fc_z * qc_z[:, np.newaxis]).sum(0) / (qc_n * fc_n)

    return null


# ─── Overlap coefficient (DQ) ─────────────────────────────────────────────────

def overlap_coeff(observed, null_flat, n_grid=500):
    """Integral of min(f_observed, f_null) over r ∈ [-1, 1]."""
    r_grid = np.linspace(-1, 1, n_grid)
    dr     = r_grid[1] - r_grid[0]
    f_obs  = gaussian_kde(observed)(r_grid)
    f_null = gaussian_kde(null_flat)(r_grid)
    return float(np.sum(np.minimum(f_obs, f_null)) * dr)


# ─── Figures ──────────────────────────────────────────────────────────────────

def make_dv_figure(fc_pre, fc_post):
    apply_thesis_style()
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 6), gridspec_kw={"hspace": 0.50}
    )

    for run in range(fc_pre.shape[0]):
        ax_top.hist(
            fc_pre[run], bins=60, density=True,
            alpha=0.05, color="#3D5A6C",
            histtype="stepfilled", linewidth=0,
        )
    ax_top.axvline(0, color="black", lw=0.8, ls="--", alpha=0.8)
    ax_top.set_title("Before denoising")
    ax_top.set_xlabel("Functional Connectivity (r)")
    ax_top.set_ylabel("Density")
    ax_top.set_xlim(-1, 1)

    for run in range(fc_post.shape[0]):
        ax_bot.hist(
            fc_post[run], bins=60, density=True,
            alpha=0.05, color="#7A95A6",
            histtype="stepfilled", linewidth=0,
        )
    ax_bot.axvline(0, color="black", lw=0.8, ls="--", alpha=0.8)
    ax_bot.set_title("After denoising")
    ax_bot.set_xlabel("Functional Connectivity (r)")
    ax_bot.set_ylabel("Density")
    ax_bot.set_xlim(-1, 1)

    fig.patch.set_facecolor("white")
    return fig


def make_dq_figure(
    qcfc_pre, qcfc_post,
    null_pre, null_post,
    dq_pre, dq_post,
):
    apply_thesis_style()
    r_grid = np.linspace(-1, 1, 500)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(6, 6), gridspec_kw={"hspace": 0.50}
    )

    for ax, qcfc_obs, null_flat, dq_val, title in [
        (ax_top, qcfc_pre,  null_pre.ravel(),  dq_pre,  "Before denoising"),
        (ax_bot, qcfc_post, null_post.ravel(), dq_post, "After denoising"),
    ]:
        f_null = gaussian_kde(null_flat)(r_grid)

        ax.hist(
            qcfc_obs, bins=60, density=True,
            color="#9E9E9E", alpha=0.75,
            histtype="stepfilled", linewidth=0,
            label="Observed",
        )
        ax.plot(r_grid, f_null, color="#E41A1C", ls="--", lw=1.5, label="Null")
        ax.set_title(title)
        ax.set_xlabel("FC–QC association level (r)")
        ax.set_ylabel("Density")
        ax.set_xlim(-1, 1)
        ax.annotate(
            f"DQ = {dq_val:.1f}% (overlap coefficient)",
            xy=(0.03, 0.92), xycoords="axes fraction", fontsize=8,
        )
        ax.legend(loc="upper right")

    fig.patch.set_facecolor("white")
    return fig


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    subjects = get_included_subjects()   # 35 included subjects
    n_runs   = len(subjects) * len(SESSIONS)  # 70

    print(
        f"─── QC-FC Analysis ───\n"
        f"Subjects: {len(subjects)}\n"
        f"Runs: {n_runs}\n"
        f"Random voxel pairs: {N_PAIRS} (seed {SEED})\n"
        f"Version: -GSR denoised"
    )

    # ── Voxel pairs ────────────────────────────────────────────────────────────
    if PAIRS_NPY.exists():
        print(f"\nLoading existing voxel pairs: {PAIRS_NPY}")
        pairs = np.load(str(PAIRS_NPY))
    else:
        ref_sub = subjects[0]
        ref_ses = SESSIONS[0]
        print(f"\nSampling {N_PAIRS} voxel pairs from {ref_sub} {ref_ses} mask ...")
        pairs = sample_voxel_pairs(ref_sub, ref_ses)
        np.save(str(PAIRS_NPY), pairs)
        print(f"Saved: {PAIRS_NPY}")

    # ── FC per run (pre and post) ──────────────────────────────────────────────
    if FC_PRE_NPY.exists() and FC_POST_NPY.exists():
        print(f"\nLoading existing FC arrays ...")
        fc_pre  = np.load(str(FC_PRE_NPY))
        fc_post = np.load(str(FC_POST_NPY))
    else:
        fc_pre  = np.zeros((n_runs, N_PAIRS), dtype=np.float32)
        fc_post = np.zeros((n_runs, N_PAIRS), dtype=np.float32)

        run_idx = 0
        for sub in subjects:
            for ses in SESSIONS:
                label = f"{sub} {ses}"
                print(f"[{run_idx + 1:02d}/{n_runs}] {label}", end="  ", flush=True)

                print("pre...", end=" ", flush=True)
                fc_pre[run_idx] = compute_fc_pairs(bold_pre_path(sub, ses), pairs)

                print("post...", end=" ", flush=True)
                fc_post[run_idx] = compute_fc_pairs(bold_post_path(sub, ses), pairs)

                print("done")
                run_idx += 1

        np.save(str(FC_PRE_NPY),  fc_pre)
        np.save(str(FC_POST_NPY), fc_post)
        print(f"Saved: {FC_PRE_NPY}")
        print(f"Saved: {FC_POST_NPY}")

    # ── QC measures (70-vectors) ───────────────────────────────────────────────
    print(f"\nCollecting QC measures for {n_runs} runs ...")
    qc_mean_fd    = np.zeros(n_runs)
    qc_invalid    = np.zeros(n_runs)
    qc_prop_valid = np.zeros(n_runs)

    run_idx = 0
    for sub in subjects:
        for ses in SESSIONS:
            mfd, ninv, pv = compute_qc_measures(confounds_path(sub, ses))
            qc_mean_fd[run_idx]    = mfd
            qc_invalid[run_idx]    = ninv
            qc_prop_valid[run_idx] = pv
            run_idx += 1

    qc_measures = {
        "QC_MeanMotion":           qc_mean_fd,
        "QC_InvalidScans":         qc_invalid,
        "QC_ProportionValidScans": qc_prop_valid,
    }

    # ── Part A: DV scores ──────────────────────────────────────────────────────
    print("\nComputing DV scores (KDE mode + IQR per subject) ...")
    dv_scores, k_dv, median_pre_disp = compute_dv_scores(
        fc_pre.astype(np.float64), fc_post.astype(np.float64), subjects
    )

    low  = int(np.sum(dv_scores < 80))
    mid  = int(np.sum((dv_scores >= 80) & (dv_scores <= 95)))
    high = int(np.sum(dv_scores > 95))

    print(
        f"\nDV scores (n={len(subjects)}):\n"
        f"  Mean:   {dv_scores.mean():.1f}%\n"
        f"  Median: {np.median(dv_scores):.1f}%\n"
        f"  Range:  {dv_scores.min():.1f}–{dv_scores.max():.1f}%\n"
        f"  Subjects with DV < 80% (low): {low}\n"
        f"  Subjects with DV 80-95% (intermediate): {mid}\n"
        f"  Subjects with DV > 95% (high): {high}"
    )

    dv_df = pd.DataFrame({"subject": subjects, "dv_score_pct": dv_scores})
    dv_df.to_csv(str(DV_TSV), sep="\t", index=False, float_format="%.2f")
    print(f"\nSaved: {DV_TSV}")

    # ── Part B: DQ scores (post-denoising, all 3 QC measures) ─────────────────
    print("\nComputing DQ scores for post-denoising ...")
    fc_post_f64 = fc_post.astype(np.float64)

    dq_vals = {}
    for seed_off, (qc_name, qc_vec) in enumerate(qc_measures.items(), start=1):
        obs  = _pearson_fc_qc(fc_post_f64, qc_vec)
        null = compute_qcfc_null(fc_post_f64, qc_vec, N_PERMS, rng_seed=SEED + seed_off)
        dq_vals[qc_name] = 100.0 * overlap_coeff(obs, null.ravel())
        print(f"  {qc_name}: {dq_vals[qc_name]:.1f}%")

    dq_combined = float(min(dq_vals.values()))

    # Primary null (MeanMotion, post) — saved as reference
    qcfc_post_obs = _pearson_fc_qc(fc_post_f64, qc_mean_fd)
    null_post_mm  = compute_qcfc_null(fc_post_f64, qc_mean_fd, N_PERMS, rng_seed=SEED + 1)

    np.save(str(QCFC_OBS_NPY),  qcfc_post_obs)
    np.save(str(QCFC_NULL_NPY), null_post_mm)
    print(f"Saved: {QCFC_OBS_NPY}")
    print(f"Saved: {QCFC_NULL_NPY}")

    # Pre-denoising QC-FC (for figure top panel)
    print("\nComputing pre-denoising QC-FC for figure ...")
    fc_pre_f64   = fc_pre.astype(np.float64)
    qcfc_pre_obs = _pearson_fc_qc(fc_pre_f64, qc_mean_fd)
    null_pre_mm  = compute_qcfc_null(fc_pre_f64, qc_mean_fd, N_PERMS, rng_seed=SEED + 10)
    dq_pre_val   = 100.0 * overlap_coeff(qcfc_pre_obs, null_pre_mm.ravel())

    # ── DQ summary ────────────────────────────────────────────────────────────
    dq_lines = (
        f"QC_MeanMotion DQ:           {dq_vals['QC_MeanMotion']:.1f}%\n"
        f"QC_InvalidScans DQ:         {dq_vals['QC_InvalidScans']:.1f}%\n"
        f"QC_ProportionValidScans DQ: {dq_vals['QC_ProportionValidScans']:.1f}%\n"
        f"Combined DQ (min):          {dq_combined:.1f}%\n"
    )
    DQ_TXT.write_text(dq_lines)
    print(f"Saved: {DQ_TXT}")

    # ── Figures ────────────────────────────────────────────────────────────────
    print("\nGenerating DV figure ...")
    fig_dv = make_dv_figure(fc_pre, fc_post)
    for out in (DV_FIG_PNG, DV_FIG_PDF):
        fig_dv.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig_dv)

    print("\nGenerating DQ figure ...")
    fig_dq = make_dq_figure(
        qcfc_pre_obs, qcfc_post_obs,
        null_pre_mm, null_post_mm,
        dq_pre_val, dq_combined,
    )
    for out in (DQ_FIG_PNG, DQ_FIG_PDF):
        fig_dq.savefig(str(out), dpi=300, bbox_inches="tight")
        print(f"Saved: {out}")
    plt.close(fig_dq)

    # ── Final console output ───────────────────────────────────────────────────
    print(
        f"\n─── QC-FC Analysis ───\n"
        f"Subjects: {len(subjects)}\n"
        f"Runs: {n_runs}\n"
        f"Random voxel pairs: {N_PAIRS} (seed {SEED})\n"
        f"Version: -GSR denoised\n"
        f"\nDV (Data Validity) Scores\n"
        f"  Mean: {dv_scores.mean():.1f}%\n"
        f"  Median: {np.median(dv_scores):.1f}%\n"
        f"  Range: {dv_scores.min():.1f}–{dv_scores.max():.1f}%\n"
        f"  Distribution: low <80% ({low}), intermediate 80-95% ({mid}), high >95% ({high})\n"
        f"\nDQ (Data Quality) Scores\n"
        f"  QC_MeanMotion DQ:           {dq_vals['QC_MeanMotion']:.1f}%\n"
        f"  QC_InvalidScans DQ:         {dq_vals['QC_InvalidScans']:.1f}%\n"
        f"  QC_ProportionValidScans DQ: {dq_vals['QC_ProportionValidScans']:.1f}%\n"
        f"  Combined DQ (min):          {dq_combined:.1f}%\n"
        f"\nInterpretation:\n"
        f"  - DV measures center/width of FC distribution per subject\n"
        f"  - DQ measures FC-motion association (zero = no motion bias)\n"
        f"  - Both >95% indicates effective denoising"
    )


if __name__ == "__main__":
    main()
