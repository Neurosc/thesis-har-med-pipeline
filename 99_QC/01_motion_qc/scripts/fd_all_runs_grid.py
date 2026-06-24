"""
Per-run FD inspection for the whole DMT-MED dataset.

Computes custom FD (Power 2012, 1-TR backward difference, 50 mm sphere, no
respiratory filtering) for every subject-session and renders all runs as small
multiples in a single multi-page PDF so each run's FD trace can be inspected
individually. Excluded subjects (sub-06, 08, 12, 26, 36) are included and
flagged. A per-run FD summary TSV is written alongside.

Runs on the server (reads fMRIPrep confounds TSVs). Activate env: conda activate fmri.

    python 99_QC/01_motion_qc/scripts/fd_all_runs_grid.py
"""

import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.motion_qc import load_confounds, compute_custom_fd
from utils.subject_filter import _EXCLUDED

# ── Configuration ─────────────────────────────────────────────────────────────
FMRIPREP_ROOT   = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
_QC_DIR         = Path(__file__).resolve().parents[1]
FIGURES_DIR     = _QC_DIR / "figures"
RESULTS_DIR     = _QC_DIR / "results"
TASK            = "rest"
TR              = 1.8
BACKWARD_DIFF_N = 1
FD_THRESH       = 0.3      # mm — decided censoring threshold (Goldberg et al. 2024)
NCOLS           = 8        # grid columns per PDF page
NROWS           = 5        # grid rows per PDF page  → 40 cells/page
# ──────────────────────────────────────────────────────────────────────────────

for d in (FIGURES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(
    f"FD: Power 2012 — {BACKWARD_DIFF_N}-TR backward difference, 50 mm sphere, "
    f"TR={TR} s, threshold {FD_THRESH} mm. No respiratory filtering."
)

# ── Step 1: Discover all confounds files (all 40 subjects) ────────────────────
subject_dirs = sorted(FMRIPREP_ROOT.glob("sub-*"))
if not subject_dirs:
    raise RuntimeError(f"No sub-* directories found under {FMRIPREP_ROOT}")

runs = []
for sub_dir in subject_dirs:
    subject_id = sub_dir.name
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        session_id = ses_dir.name
        pattern = f"{subject_id}_{session_id}_task-{TASK}_desc-confounds_timeseries.tsv.gz"
        matches = list((ses_dir / "func").glob(pattern))
        if not matches:
            print(f"  WARNING: confounds missing for {subject_id}/{session_id} — skipping.")
            continue
        runs.append((subject_id, session_id, matches[0]))

print(f"Found {len(subject_dirs)} subject(s), {len(runs)} run(s) to process.\n")

# ── Step 2: Compute custom FD per run ─────────────────────────────────────────
fd_series = {}      # (subject_id, session_id) → fd array
run_rows  = []
failed    = []

for subject_id, session_id, confounds_path in runs:
    excluded = subject_id in _EXCLUDED
    try:
        df = load_confounds(confounds_path)
        fd = compute_custom_fd(df, tr=TR, backward_diff_n=BACKWARD_DIFF_N, verbose=False)
        fd_series[(subject_id, session_id)] = fd
        finite = fd[~np.isnan(fd)]
        n_frames = len(fd)
        n_above  = int(np.sum(finite > FD_THRESH))
        run_rows.append({
            "subject":        subject_id,
            "session":        session_id,
            "excluded":       excluded,
            "n_frames":       n_frames,
            "mean_fd":        float(np.nanmean(fd)),
            "median_fd":      float(np.nanmedian(fd)),
            "max_fd":         float(np.nanmax(fd)),
            "n_above_03":     n_above,
            "pct_above_03":   100.0 * n_above / n_frames,
        })
    except Exception as exc:
        print(f"  ERROR: {subject_id}/{session_id} — {exc}")
        failed.append(f"{subject_id}/{session_id}")

run_df = pd.DataFrame(run_rows).sort_values(["subject", "session"]).reset_index(drop=True)
run_tsv = RESULTS_DIR / "run_level_fd_all.tsv"
run_df.to_csv(run_tsv, sep="\t", index=False)
print(f"Run-level FD summary saved: {run_tsv}  ({len(run_df)} runs)")

# ── Step 3: Render small-multiple grid → single multi-page PDF ─────────────────
order = [(r.subject, r.session) for r in run_df.itertuples(index=False)]
per_page = NCOLS * NROWS
n_pages = int(np.ceil(len(order) / per_page))
pdf_path = FIGURES_DIR / "fd_all_runs_grid.pdf"

TRACE_COLOR    = "#4878CF"
CENSOR_COLOR   = "#cc2222"
EXCLUDED_TINT  = "#fbe4e4"

with PdfPages(pdf_path) as pdf:
    for page in range(n_pages):
        fig, axes = plt.subplots(NROWS, NCOLS, figsize=(16, 10))
        fig.suptitle(
            f"Framewise displacement per run — custom FD "
            f"({BACKWARD_DIFF_N}-TR backward diff, {FD_THRESH} mm threshold)   "
            f"page {page + 1}/{n_pages}",
            fontsize=12, y=0.995,
        )
        axes = np.atleast_1d(axes).ravel()
        chunk = order[page * per_page:(page + 1) * per_page]

        for ax, key in zip(axes, chunk):
            subject_id, session_id = key
            fd = fd_series[key]
            frames = np.arange(len(fd))
            mask = np.where(np.isnan(fd), False, fd > FD_THRESH)
            excluded = subject_id in _EXCLUDED

            row = run_df[(run_df.subject == subject_id) & (run_df.session == session_id)].iloc[0]
            ytop = max(0.45, float(row.max_fd) * 1.10)

            if excluded:
                ax.set_facecolor(EXCLUDED_TINT)
            ax.plot(frames, fd, color=TRACE_COLOR, linewidth=0.7)
            ax.axhline(FD_THRESH, color="black", linestyle="--", linewidth=0.6)
            ax.fill_between(frames, 0, ytop, where=mask, color=CENSOR_COLOR,
                            alpha=0.18, step="mid")
            ax.set_ylim(0, ytop)
            ax.set_xlim(0, len(fd) - 1)

            title = f"{subject_id} {session_id}"
            if excluded:
                title += "  [EXCL]"
            ax.set_title(title, fontsize=7,
                         color=CENSOR_COLOR if excluded else "black")
            ax.annotate(
                f"max {row.max_fd:.2f}  mean {row.mean_fd:.2f}\n{row.pct_above_03:.0f}% cens",
                xy=(0.97, 0.92), xycoords="axes fraction", ha="right", va="top",
                fontsize=5.5, color="#333333",
            )
            ax.tick_params(labelsize=5)

        # blank any unused cells on the last page
        for ax in axes[len(chunk):]:
            ax.axis("off")

        fig.supxlabel("Frame index", fontsize=9)
        fig.supylabel("FD (mm)", fontsize=9)
        plt.tight_layout(rect=[0.01, 0.01, 1, 0.98])
        pdf.savefig(fig, dpi=200)
        plt.close(fig)

print(f"Grid PDF saved: {pdf_path}  ({n_pages} page(s))")

# ── Step 4: Console summary ───────────────────────────────────────────────────
n_excl_runs = int(run_df["excluded"].sum())
print("\n── Summary ──")
print(f"  Runs processed : {len(run_df)}  ({n_excl_runs} from excluded subjects)")
if failed:
    print(f"  Failed         : {len(failed)} — {', '.join(failed)}")
print(f"  Mean FD across runs   : {run_df['mean_fd'].mean():.3f} mm")
print(f"  Median %censored      : {run_df['pct_above_03'].median():.1f}%")
print(f"  Runs >50% censored    : {(run_df['pct_above_03'] > 50).sum()}")
print(f"\n  Saved:\n    {run_tsv}\n    {pdf_path}")
