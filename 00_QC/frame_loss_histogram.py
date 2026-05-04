import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["font.family"] = "sans-serif"
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
QC_DIR   = Path(__file__).resolve().parent
# Column name as written by threshold_comparison_qc.py
SUMMARY  = QC_DIR / "results" / "run_level_fd_summary.tsv"
OUT_PATH = QC_DIR / "thesis_figures" / "supplementary" / "fig_S_frame_loss_distribution.png"
PCT_COL  = "pct_frames_above_03"
# ──────────────────────────────────────────────────────────────────────────────

OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SUMMARY, sep="\t")
if PCT_COL not in df.columns:
    raise ValueError(
        f"Column '{PCT_COL}' not found. Available: {list(df.columns)}"
    )

pct = df[PCT_COL].to_numpy(dtype=float)
n_total    = len(pct)
n_excluded = int(np.sum(pct > 50))
print(f"Runs: {n_total}  |  Above 50% threshold: {n_excluded}")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

# Histogram: 20 bins of width 5 over [0, 100]
bins = np.arange(0, 101, 5)
ax.hist(pct, bins=bins, color="#4A6FA5", edgecolor="white", linewidth=0.5, zorder=2)

# Exclusion threshold: shaded region + dashed line + text label
ax.axvspan(50, 100, color="#C0392B", alpha=0.08, zorder=1)
ax.axvline(50, color="#C0392B", linewidth=1.5, linestyle="--", zorder=3)
ax.text(51, ax.get_ylim()[1] * 0.97, "50% exclusion\nthreshold",
        color="#C0392B", fontsize=9, va="top", ha="left", zorder=4)

# Excluded runs annotation (upper-right corner)
ax.text(0.97, 0.97, f"{n_excluded}/{n_total} runs excluded",
        transform=ax.transAxes, fontsize=9, color="#C0392B",
        ha="right", va="top")

# Axes formatting
ax.set_xlim(0, 100)
ax.set_xlabel("Frame loss (%)", fontsize=10)
ax.set_ylabel("Number of runs", fontsize=10)
ax.tick_params(labelsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(axis="y", alpha=0.15, zorder=0)

# Two-line title: main title (bold, 12pt) + subtitle (gray, 9pt)
ax.set_title(
    "Frame loss distribution under FD > 0.3 mm censoring",
    fontsize=12, fontweight="bold", pad=22,
)
ax.text(0.5, 1.0, "Red line: 50% threshold for run exclusion",
        transform=ax.transAxes, fontsize=9, color="#666666",
        ha="center", va="bottom", clip_on=False)

plt.tight_layout()
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT_PATH}")
