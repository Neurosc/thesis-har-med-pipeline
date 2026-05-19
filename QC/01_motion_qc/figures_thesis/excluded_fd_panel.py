import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from utils.motion_qc import load_confounds, compute_custom_fd

# ── Paths ──────────────────────────────────────────────────────────────────────
FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
QC_DIR        = Path(__file__).resolve().parents[1]
SUMMARY_TSV   = QC_DIR / "results" / "subject_level_fd_summary.tsv"
OUT_PATH      = QC_DIR / "thesis_figures" / "supplementary" / "fig_S_excluded_subjects_fd.png"
TASK          = "rest"
TR            = 1.8
FD_THRESH     = 0.3

EXCLUDED_RUNS = [
    ("sub-06", "ses-01"),
    ("sub-08", "ses-02"),
    ("sub-12", "ses-01"),
    ("sub-12", "ses-02"),
    ("sub-26", "ses-01"),
    ("sub-36", "ses-02"),
]
# ──────────────────────────────────────────────────────────────────────────────

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

summary_df = pd.read_csv(SUMMARY_TSV, sep="\t")


def _get_pct_censored(subject_id, session_id):
    ses_key = session_id.replace("-", "")   # "ses-01" → "ses01"
    col = f"pct_above_03_{ses_key}"
    row = summary_df[summary_df["subject"] == subject_id]
    if row.empty or col not in summary_df.columns:
        return float("nan")
    return float(row[col].iloc[0])


# ── Load FD for each excluded run ──────────────────────────────────────────────
fd_arrays = []
pct_values = []
for subject_id, session_id in EXCLUDED_RUNS:
    confounds_path = (
        FMRIPREP_ROOT / subject_id / session_id / "func"
        / f"{subject_id}_{session_id}_task-{TASK}_desc-confounds_timeseries.tsv.gz"
    )
    print(f"Loading {subject_id} {session_id}: {confounds_path}")
    df = load_confounds(confounds_path)
    fd = compute_custom_fd(df, tr=TR, verbose=False)
    fd_arrays.append(fd)
    pct = _get_pct_censored(subject_id, session_id)
    pct_values.append(pct)
    print(f"  {len(fd)} frames, {pct:.1f}% censored at FD > {FD_THRESH} mm")

# ── Shared Y axis: global max + 10% headroom ───────────────────────────────────
global_max = max(float(np.nanmax(fd)) for fd in fd_arrays)
y_upper = global_max * 1.10
print(f"\nShared Y-axis: 0 – {y_upper:.3f} mm (global max {global_max:.3f} mm + 10%)")

# ── 2×3 panel figure ───────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharey=True)

for idx, ((subject_id, session_id), fd, pct) in enumerate(
    zip(EXCLUDED_RUNS, fd_arrays, pct_values)
):
    ax = axes.flat[idx]
    ax.plot(np.arange(len(fd)), fd, linewidth=0.9)

    if idx == 0:
        ax.axhline(FD_THRESH, color="black", linestyle="--", linewidth=1.0,
                   label=f"FD = {FD_THRESH} mm")
        ax.legend(fontsize=7, loc="upper right")
        ax.set_ylim(0, y_upper)
    else:
        ax.axhline(FD_THRESH, color="black", linestyle="--", linewidth=1.0)

    pct_str = f"{pct:.1f}%" if not np.isnan(pct) else "?%"
    ax.set_title(f"{subject_id} {session_id} ({pct_str} censored)", fontsize=9)
    ax.set_xlim(0, 240)
    ax.set_xlabel("Frame index", fontsize=8)
    if idx % 3 == 0:
        ax.set_ylabel("FD (mm)", fontsize=8)
    ax.tick_params(labelsize=7)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
