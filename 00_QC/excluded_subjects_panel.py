import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from utils.motion_qc import load_confounds, compute_custom_fd

# ── Paths ──────────────────────────────────────────────────────────────────────
FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
QC_DIR        = Path(__file__).resolve().parent
OUT_PATH      = QC_DIR / "thesis_figures" / "supplementary" / "fig_S_excluded_subjects_panel.png"
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

# ── Shared Y axis: 0 to global max rounded up ──────────────────────────────────
global_max = max(float(np.nanmax(fd)) for fd in fd_arrays)
y_upper = float(np.ceil(global_max))
print(f"\nShared Y-axis: 0 – {y_upper:.1f} mm (global max {global_max:.3f} mm)")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey=True, sharex=True)
fig.suptitle("Excluded runs (>50% frames censored at FD > 0.3 mm)", fontsize=11)

for idx, ((subject_id, session_id), fd) in enumerate(zip(EXCLUDED_RUNS, fd_arrays)):
    ax = axes.flat[idx]
    frames = np.arange(len(fd))

    ax.plot(frames, fd, linewidth=0.9, color="steelblue", zorder=2)

    above = ~np.isnan(fd) & (fd > FD_THRESH)
    ax.scatter(frames[above], fd[above], color="red", s=6, zorder=3, linewidths=0)

    ax.axhline(FD_THRESH, color="black", linestyle="--", linewidth=1.0)

    pct_censored = 100.0 * int(np.sum(above)) / len(fd)
    ax.set_title(f"{subject_id} {session_id} ({pct_censored:.1f}% censored)", fontsize=9)

    ax.set_xlim(0, 240)
    ax.set_ylim(0, y_upper)
    ax.tick_params(labelsize=7)

fig.supylabel("FD (mm)", fontsize=10)
fig.supxlabel("Frame", fontsize=10)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nSaved: {OUT_PATH}")
