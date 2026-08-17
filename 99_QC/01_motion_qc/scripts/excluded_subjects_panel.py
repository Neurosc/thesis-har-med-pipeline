import os
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
from utils.thesis_style import SET2, NEUTRAL, shade

# ── Paths ──────────────────────────────────────────────────────────────────────
FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
QC_DIR        = Path(__file__).resolve().parents[1]
OUT_DIR       = QC_DIR / "thesis_figures" / "supplementary"
OUT_PATH      = OUT_DIR / "fig_S_excluded_subjects_panel.png"
VARIANT_DIR   = OUT_DIR / "colour_variants"
TASK          = "rest"
TR            = 1.8
FD_THRESH     = 0.3

# Y_MAX is a DELIBERATE clip, not the data range. Real maxima are 0.79-4.26 mm, so a
# limit set by sub-08's spike squashes the other five runs into the bottom fifth of
# their panels. Anything above the limit is drawn to the top edge and annotated with
# its true value, so nothing is silently truncated.
Y_MAX         = 2.0
SHADE_ALPHA   = 0.30

# Censored-frame shading. One figure per colour; set FIGS03_COLOUR to also write the
# canonical fig_S_excluded_subjects_panel.png in that colour.
CANDIDATES = [("pink", SET2["pink"]), ("lime", SET2["lime"]), ("orange", SET2["orange"])]
CHOSEN     = os.environ.get("FIGS03_COLOUR", "").strip().lower()
TRACE      = "#333333"

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

OUT_DIR.mkdir(parents=True, exist_ok=True)
VARIANT_DIR.mkdir(parents=True, exist_ok=True)

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
def build_panel(colour):
    """Draw the six-run panel with `colour` shading the censored frames."""
    label_colour = shade(colour, 0.30)   # the pastels are illegible as text on white
    fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharey=True, sharex=True)

    # No suptitle: the description belongs in the LaTeX caption, not on the image.

    for idx, ((subject_id, session_id), fd) in enumerate(zip(EXCLUDED_RUNS, fd_arrays)):
        ax = axes.flat[idx]
        frames = np.arange(len(fd))
        above = ~np.isnan(fd) & (fd > FD_THRESH)
        pct_censored = 100.0 * int(np.sum(above)) / len(fd)

        # Full-height bands over censored frames
        ax.fill_between(frames, 0, Y_MAX, where=above,
                        color=colour, alpha=SHADE_ALPHA, linewidth=0, zorder=1)

        ax.axhline(FD_THRESH, color=NEUTRAL["ink"], linestyle="--", linewidth=0.8, zorder=2)

        # FD time series, on top of the shading so the signal stays readable
        ax.plot(frames, np.clip(fd, 0, Y_MAX), linewidth=1.2, color=TRACE, zorder=3)

        # Nothing is hidden by the clip: label any excursion with its true value
        run_max = float(np.nanmax(fd))
        if run_max > Y_MAX:
            pk = int(np.nanargmax(fd))
            ax.annotate(f"{run_max:.2f} mm",
                        xy=(pk, Y_MAX), xytext=(pk + 8, Y_MAX * 0.92),
                        fontsize=7.5, color=TRACE, ha="left", va="top", zorder=4,
                        arrowprops=dict(arrowstyle="-|>", color=TRACE, lw=0.7,
                                        shrinkA=0, shrinkB=1))

        # Two-line title: subject (black) + % censored (shading hue, darkened)
        ax.set_title(f"{subject_id} {session_id}", fontsize=10, color="black", pad=18)
        ax.annotate(
            f"{pct_censored:.1f}% censored",
            xy=(0.5, 1.0), xycoords="axes fraction",
            xytext=(0, 3), textcoords="offset points",
            fontsize=9, color=label_colour, ha="center", va="bottom",
            clip_on=False, annotation_clip=False,
        )

        ax.set_xlim(0, 240)
        ax.set_ylim(0, Y_MAX)
        ax.tick_params(labelsize=9)

        if idx % 3 == 0:
            ax.set_ylabel("FD (mm)", fontsize=9)
        if idx >= 3:
            ax.set_xlabel("Frame", fontsize=9)

    # The legend carries the threshold, so no in-axes "0.3 mm" text is needed.
    legend_elements = [
        Line2D([0], [0], color=TRACE, linewidth=1.2, label="FD"),
        Patch(facecolor=colour, alpha=SHADE_ALPHA, edgecolor="none",
              label="Censored frames (FD > 0.3 mm)"),
        Line2D([0], [0], color=NEUTRAL["ink"], linestyle="--", linewidth=0.8,
               label="Threshold (0.3 mm)"),
    ]
    fig.legend(handles=legend_elements, loc="upper center",
               bbox_to_anchor=(0.5, 0.97), ncol=3, fontsize=9, frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.90], h_pad=2.5)
    return fig


for name, colour in CANDIDATES:
    fig = build_panel(colour)
    for ext, kwargs in [("png", {"dpi": 300}), ("svg", {})]:
        path = VARIANT_DIR / f"fig_S_excluded_subjects_panel_{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        print(f"Saved variant: {path}")
    if name == CHOSEN:
        fig.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
        print(f"Saved CANONICAL ({name}): {OUT_PATH}")
    plt.close(fig)

if not CHOSEN:
    print("\nFIGS03_COLOUR not set - wrote the three variants only, canonical file "
          "untouched. Re-run with e.g. FIGS03_COLOUR=lime to write it.")
elif CHOSEN not in {n for n, _ in CANDIDATES}:
    print(f"\nWARNING: FIGS03_COLOUR={CHOSEN!r} is not one of "
          f"{[n for n, _ in CANDIDATES]} - canonical file NOT written.")
