import sys
import matplotlib
matplotlib.use("Agg")
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.thesis_style import apply_thesis_style, PALETTE
apply_thesis_style()

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ── Paths ──────────────────────────────────────────────────────────────────────
QC_DIR   = Path(__file__).resolve().parent
SUMMARY  = QC_DIR / "results" / "run_level_fd_summary.tsv"
OUT_STEM = QC_DIR / "thesis_figures" / "supplementary" / "fig_S_frame_loss_distribution"
# Column names as written by threshold_comparison_qc.py
PCT_COL  = "pct_frames_above_03"
SES_COL  = "session"
# ──────────────────────────────────────────────────────────────────────────────

OUT_STEM.parent.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SUMMARY, sep="\t")
for col in (PCT_COL, SES_COL):
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Available: {list(df.columns)}")

pct_ses01 = df.loc[df[SES_COL] == "ses-01", PCT_COL].to_numpy(dtype=float)
pct_ses02 = df.loc[df[SES_COL] == "ses-02", PCT_COL].to_numpy(dtype=float)
print(f"ses-01 runs: {len(pct_ses01)}  |  ses-02 runs: {len(pct_ses02)}")

# ── Figure ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(3.35, 2.5))

bins = np.arange(0, 105, 5)
ax.hist(
    [pct_ses01, pct_ses02],
    bins=bins,
    stacked=True,
    color=[PALETTE["group_A_pre"], PALETTE["group_A_post"]],
    label=["Pre-retreat", "Post-retreat"],
    edgecolor="white",
    linewidth=0.3,
)

# Threshold marker only — no label, no shading
ax.axvline(50, color="#333333", linewidth=0.6, linestyle="--")

ax.set_xlim(0, 100)
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["", "25", "50", "75", "100"])
ax.set_xlabel("Frame loss (%)")
ax.set_ylabel("Number of runs")

counts_01, _ = np.histogram(pct_ses01, bins=bins)
counts_02, _ = np.histogram(pct_ses02, bins=bins)
stacked_max = int(np.max(counts_01 + counts_02))
max_y = int(np.ceil(stacked_max / 4) * 4)
yticks = np.arange(0, max_y + 4, 4)
ax.set_yticks(yticks)
ax.set_yticklabels(["" if t == 0 else str(t) for t in yticks])

ax.legend(loc="upper right", fontsize=7)

plt.tight_layout()

for suffix, kw in [(".png", {"dpi": 300}), (".pdf", {})]:
    path = OUT_STEM.with_suffix(suffix)
    fig.savefig(path, bbox_inches="tight", **kw)
    print(f"Saved: {path}")

plt.close(fig)
