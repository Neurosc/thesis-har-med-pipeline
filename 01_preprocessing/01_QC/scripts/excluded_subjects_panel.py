import sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from utils.motion_qc import load_confounds, compute_custom_fd

# ── Paths ──────────────────────────────────────────────────────────────────────
FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
QC_DIR        = Path(__file__).resolve().parent
OUT_PATH      = QC_DIR / "thesis_figures" / "supplementary" / "fig_S_excluded_subjects_panel.png"
TASK          = "rest"
TR            = 1.8
FD_THRESH     = 0.3
Y_MAX         = 5.0

# Ordered by % censored descending (sub-12 both sessions first, then sub-26/36/06/08)
EXCLUDED_RUNS = [
    ("sub-12", "ses-02"),
    ("sub-12", "ses-01"),
    ("sub-26", "ses-01"),
    ("sub-36", "ses-02"),
    ("sub-06", "ses-01"),
    ("sub-08", "ses-02"),
]
# ──────────────────────────────────────────────────────────────────────────────

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# ── Load confounds and compute FD ──────────────────────────────────────────────
fd_arrays = []
for subject_id, session_id in EXCLUDED_RUNS:
    confounds_path = (
        FMRIPREP_ROOT / subject_id / session_id / "func"
        / f"{subject_id}_{session_id}_task-{TASK}_desc-confounds_timeseries.tsv.gz"
    )
    print(f"Loading {subject_id} {session_id}: {confounds_path}")
    df = load_confounds(confounds_path)
    fd = compute_custom_fd(df, tr=TR, backward_diff_n=1, verbose=False)
    fd_arrays.append(fd)
    n_above = int(np.sum(fd[~np.isnan(fd)] > FD_THRESH))
    print(f"  {len(fd)} frames, {n_above} above threshold ({100.0 * n_above / len(fd):.1f}%)")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey=True, sharex=True)

fig.suptitle(
    "Runs excluded due to excessive head motion (>50% frames with FD > 0.3 mm)",
    fontsize=11,
)

for idx, ((subject_id, session_id), fd) in enumerate(zip(EXCLUDED_RUNS, fd_arrays)):
    ax = axes.flat[idx]
    frames = np.arange(len(fd))
    above = ~np.isnan(fd) & (fd > FD_THRESH)
    pct_censored = 100.0 * int(np.sum(above)) / len(fd)

    # Shaded vertical bands for censored frames
    ax.fill_between(frames, 0, Y_MAX, where=above,
                    color="red", alpha=0.25, linewidth=0, zorder=1)

    # FD time series
    ax.plot(frames, fd, linewidth=1.2, color="#333333", zorder=2)

    # Threshold line with right-edge label
    ax.axhline(FD_THRESH, color="black", linestyle="--", linewidth=0.8, zorder=3)
    ax.text(238, FD_THRESH + 0.10, "0.3 mm",
            fontsize=7, ha="right", va="bottom", color="black", zorder=4)

    # Two-line title: subject (black) + % censored (red)
    # annotate sits just above the axes; set_title sits above that
    ax.set_title(f"{subject_id} {session_id}", fontsize=10, color="black", pad=18)
    ax.annotate(
        f"{pct_censored:.1f}% censored",
        xy=(0.5, 1.0), xycoords="axes fraction",
        xytext=(0, 3), textcoords="offset points",
        fontsize=9, color="#C0392B", ha="center", va="bottom",
        clip_on=False, annotation_clip=False,
    )

    ax.set_xlim(0, 240)
    ax.set_ylim(0, Y_MAX)
    ax.tick_params(labelsize=9)

    if idx % 3 == 0:
        ax.set_ylabel("FD (mm)", fontsize=9)
    if idx >= 3:
        ax.set_xlabel("Frame", fontsize=9)

# Single figure-level legend between suptitle and top row
legend_elements = [
    Line2D([0], [0], color="#333333", linewidth=1.2, label="FD"),
    Patch(facecolor="red", alpha=0.25, edgecolor="none",
          label="Censored frames (FD > 0.3 mm)"),
    Line2D([0], [0], color="black", linestyle="--", linewidth=0.8,
           label="Threshold (0.3 mm)"),
]
fig.legend(handles=legend_elements, loc="upper center",
           bbox_to_anchor=(0.5, 0.95), ncol=3, fontsize=9, frameon=False)

plt.tight_layout(rect=[0, 0, 1, 0.85], h_pad=2.5)
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
