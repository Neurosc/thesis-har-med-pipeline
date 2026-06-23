"""
fd_threshold_survey.py

Exploratory FD census for the DMT-MED dataset (fMRIPrep 23.0.2).
Reads framewise_displacement from fMRIPrep confounds TSVs.
Frame 0 is NaN by convention and is never counted as censored.

Prints discovered runs before computing anything so you can confirm
the match list. Outputs a per-run CSV, a subject rollup CSV, and
four distribution plots.  Does NOT exclude any runs by default.
"""

import matplotlib
matplotlib.use("Agg")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from itertools import groupby
from pathlib import Path

# ── Configurable thresholds ────────────────────────────────────────────────────
FD_THRESH        = 0.3   # mm — rerun at 0.5 to compare
MIN_RETAINED_MIN = None  # float | None — flag runs below this retained-minute cutoff
MAX_GAP_SEC      = None  # float | None — flag runs with a gap longer than this (seconds)
# ──────────────────────────────────────────────────────────────────────────────

TR             = 1.8
TASK           = "rest"
EXPECTED_FRAMES = 240

FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")

SCRIPT_DIR  = Path(__file__).resolve().parent
QC_DIR      = SCRIPT_DIR.parent
RESULTS_DIR = QC_DIR / "results"
FIGURES_DIR = QC_DIR / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

THRESH_TAG = str(FD_THRESH).replace(".", "")

# Palette: pre = darker slate, post = lighter slate (no group labels)
PRE_COLOR  = "#3D5A6C"
POST_COLOR = "#7A95A6"
SES_PALETTE = {"ses-01": PRE_COLOR, "ses-02": POST_COLOR}

# ── Step 1: Discover confounds files ──────────────────────────────────────────
FNAME_GZ = "{sub}_{ses}_task-{task}_desc-confounds_timeseries.tsv.gz"
FNAME    = "{sub}_{ses}_task-{task}_desc-confounds_timeseries.tsv"

print("=" * 70)
print(f"fMRIPrep root : {FMRIPREP_ROOT}")
print(f"Pattern (gz)  : {FNAME_GZ}")
print(f"FD threshold  : {FD_THRESH} mm")
print(f"MIN_RETAINED_MIN: {MIN_RETAINED_MIN}")
print(f"MAX_GAP_SEC     : {MAX_GAP_SEC}")
print("=" * 70)

if not FMRIPREP_ROOT.exists():
    sys.exit(f"\nERROR: fMRIPrep root not found: {FMRIPREP_ROOT}\n"
             f"Check the FMRIPREP_ROOT path at the top of this script.")

runs = []          # list of (sub, ses, path)
missing_runs = []  # (sub, ses) pairs where confounds were not found

for sub_dir in sorted(FMRIPREP_ROOT.glob("sub-*")):
    if not sub_dir.is_dir():
        continue
    sub = sub_dir.name
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        if not ses_dir.is_dir():
            continue
        ses    = ses_dir.name
        func   = ses_dir / "func"
        if not func.is_dir():
            missing_runs.append((sub, ses, "no func/ directory"))
            continue
        path_gz = func / FNAME_GZ.format(sub=sub, ses=ses, task=TASK)
        path_   = func / FNAME.format(sub=sub, ses=ses, task=TASK)
        if path_gz.exists():
            runs.append((sub, ses, path_gz))
        elif path_.exists():
            runs.append((sub, ses, path_))
        else:
            missing_runs.append((sub, ses, "confounds file absent"))

print(f"\nDiscovered {len(runs)} run(s):\n")
for sub, ses, path in runs:
    print(f"  {sub}  {ses}  {path.name}")

if missing_runs:
    print(f"\n  WARNING — {len(missing_runs)} subject/session(s) with no confounds file:")
    for sub, ses, reason in missing_runs:
        print(f"    {sub}  {ses}  ({reason})")

print(f"\n{'─' * 70}")
print("Confirm the run list above, then check results below.")
print(f"{'─' * 70}\n")


# ── Per-run computation ────────────────────────────────────────────────────────
def _run_stats(sub, ses, path, fd_thresh, tr, expected_frames):
    df = pd.read_csv(path, sep="\t")

    if "framewise_displacement" not in df.columns:
        fd_cols = [c for c in df.columns if "fd" in c.lower() or "disp" in c.lower()]
        raise KeyError(
            f"'framewise_displacement' not found. Candidates: {fd_cols}"
        )

    fd_raw  = df["framewise_displacement"].to_numpy(dtype=float)
    n_total = len(fd_raw)

    if n_total != expected_frames:
        print(f"  NOTE {sub}/{ses}: {n_total} frames (expected {expected_frames})")

    # Frame 0 is NaN by fMRIPrep convention; treat as non-censored
    censored = np.zeros(n_total, dtype=bool)
    for t in range(1, n_total):
        if not np.isnan(fd_raw[t]):
            censored[t] = fd_raw[t] > fd_thresh

    n_censored   = int(censored.sum())
    n_retained   = n_total - n_censored
    retained_min = n_retained * tr / 60.0

    # Contiguous-block analysis on the censored mask
    rle = [(flag, sum(1 for _ in grp)) for flag, grp in groupby(censored)]
    censored_block_lengths  = [length for flag, length in rle if flag]
    retained_block_lengths  = [length for flag, length in rle if not flag]

    n_blocks    = len(censored_block_lengths)
    max_gap_fr  = max(censored_block_lengths) if censored_block_lengths else 0
    max_gap_sec = max_gap_fr * tr
    mean_block_len   = float(np.mean(censored_block_lengths))   if censored_block_lengths else np.nan
    median_block_len = float(np.median(censored_block_lengths)) if censored_block_lengths else np.nan

    longest_ret_fr  = max(retained_block_lengths) if retained_block_lengths else 0
    longest_ret_sec = longest_ret_fr * tr

    # Mean / max FD over retained frames only (exclude frame 0 NaN and censored)
    retained_fd = fd_raw[~censored]
    retained_fd = retained_fd[~np.isnan(retained_fd)]
    mean_ret_fd = float(np.nanmean(retained_fd)) if len(retained_fd) else np.nan
    max_ret_fd  = float(np.nanmax(retained_fd))  if len(retained_fd) else np.nan

    return {
        "subject":               sub,
        "session":               ses,
        "n_total":               n_total,
        "n_censored":            n_censored,
        "pct_censored":          100.0 * n_censored / n_total,
        "n_retained":            n_retained,
        "retained_min":          round(retained_min, 3),
        "max_gap_frames":        max_gap_fr,
        "max_gap_sec":           round(max_gap_sec, 2),
        "n_blocks":              n_blocks,
        "mean_block_len_frames": round(mean_block_len,   2) if not np.isnan(mean_block_len)   else np.nan,
        "median_block_len_frames": round(median_block_len, 2) if not np.isnan(median_block_len) else np.nan,
        "longest_retained_frames": longest_ret_fr,
        "longest_retained_sec":  round(longest_ret_sec, 2),
        "mean_retained_fd":      round(mean_ret_fd, 4) if not np.isnan(mean_ret_fd) else np.nan,
        "max_retained_fd":       round(max_ret_fd,  4) if not np.isnan(max_ret_fd)  else np.nan,
    }


# ── Step 2: Process all runs ───────────────────────────────────────────────────
rows   = []
failed = []

for sub, ses, path in runs:
    try:
        row = _run_stats(sub, ses, path, FD_THRESH, TR, EXPECTED_FRAMES)
        rows.append(row)
    except Exception as exc:
        print(f"  ERROR {sub}/{ses}: {exc}")
        failed.append({"subject": sub, "session": ses, "error": str(exc)})

if not rows:
    sys.exit("No runs processed successfully — cannot continue.")

df_runs = (
    pd.DataFrame(rows)
    .sort_values("max_gap_frames", ascending=False)
    .reset_index(drop=True)
)


# ── Step 3: Subject rollup ─────────────────────────────────────────────────────
def _flag_run(subset, col, threshold, below):
    """Return True if the run should be flagged; False if threshold is None or run missing."""
    if threshold is None or len(subset) == 0:
        return False
    val = float(subset[col].iloc[0])
    return val < threshold if below else val > threshold


rollup_rows = []
for sub in sorted(df_runs["subject"].unique()):
    sub_df = df_runs[df_runs["subject"] == sub]
    pre    = sub_df[sub_df["session"] == "ses-01"]
    post   = sub_df[sub_df["session"] == "ses-02"]

    pre_min_flag  = _flag_run(pre,  "retained_min", MIN_RETAINED_MIN, below=True)
    post_min_flag = _flag_run(post, "retained_min", MIN_RETAINED_MIN, below=True)
    pre_gap_flag  = _flag_run(pre,  "max_gap_sec",  MAX_GAP_SEC,      below=False)
    post_gap_flag = _flag_run(post, "max_gap_sec",  MAX_GAP_SEC,      below=False)

    pre_flagged  = pre_min_flag  or pre_gap_flag
    post_flagged = post_min_flag or post_gap_flag
    unusable     = pre_flagged   or post_flagged   # within-subject design

    rollup_rows.append({
        "subject":           sub,
        "pre_flagged":       pre_flagged,
        "post_flagged":      post_flagged,
        "unusable":          unusable,
        "pre_retained_min":  pre["retained_min"].iloc[0]  if len(pre)  else np.nan,
        "post_retained_min": post["retained_min"].iloc[0] if len(post) else np.nan,
        "pre_max_gap_sec":   pre["max_gap_sec"].iloc[0]   if len(pre)  else np.nan,
        "post_max_gap_sec":  post["max_gap_sec"].iloc[0]  if len(post) else np.nan,
    })

df_rollup = pd.DataFrame(rollup_rows)


# ── Step 4: Save CSVs ─────────────────────────────────────────────────────────
run_csv    = RESULTS_DIR / f"fd_survey_runs_thresh{THRESH_TAG}.csv"
rollup_csv = RESULTS_DIR / f"fd_survey_subjects_thresh{THRESH_TAG}.csv"

df_runs.to_csv(run_csv,    index=False)
df_rollup.to_csv(rollup_csv, index=False)
print(f"Saved: {run_csv}")
print(f"Saved: {rollup_csv}\n")


# ── Step 5: Print tables ───────────────────────────────────────────────────────
pd.set_option("display.max_rows",    200)
pd.set_option("display.width",       160)
pd.set_option("display.float_format", lambda x: f"{x:.3f}" if isinstance(x, float) else str(x))

print("── Per-run table (sorted by max_gap_frames descending) " + "─" * 14)
print(df_runs.to_string(index=True))

print("\n── Subject rollup " + "─" * 51)
print(df_rollup.to_string(index=False))

n_flagged = int(df_rollup["unusable"].sum())
print(f"\nSubjects flagged unusable (with current thresholds): {n_flagged}")
if MIN_RETAINED_MIN is None and MAX_GAP_SEC is None:
    print("  (Both thresholds are None — no subjects flagged. Set thresholds to see exclusions.)")

if failed:
    print(f"\n── FAILED runs ({len(failed)}) " + "─" * 44)
    for f in failed:
        print(f"  {f['subject']}  {f['session']}  ERROR: {f['error']}")


# ── Step 6: Plots ─────────────────────────────────────────────────────────────
# Shared run labels: "01/s1", "02/s2", etc.
def _label(row):
    return row["subject"].replace("sub-", "") + "/" + row["session"].replace("ses-0", "s")

df_runs["_label"] = df_runs.apply(_label, axis=1)

pre_patch  = mpatches.Patch(color=PRE_COLOR,  label="ses-01 (pre)")
post_patch = mpatches.Patch(color=POST_COLOR, label="ses-02 (post)")
ses_legend = [pre_patch, post_patch]


# Plot 1 — Retained minutes sorted ascending
fig1, ax1 = plt.subplots(figsize=(11, 4))
bar_colors = [SES_PALETTE.get(s, "#888") for s in df_runs["session"]]
ax1.bar(range(len(df_runs)), df_runs["retained_min"],
        color=bar_colors, edgecolor="white", linewidth=0.4)
ax1.set_xticks(range(len(df_runs)))
ax1.set_xticklabels(df_runs["_label"], rotation=60, ha="right", fontsize=7)
ax1.set_ylabel("Retained minutes")
ax1.set_xlabel("Run (sorted ascending)")
ax1.set_title(f"Retained minutes per run — FD > {FD_THRESH} mm")
handles = ses_legend[:]
if MIN_RETAINED_MIN is not None:
    ax1.axhline(MIN_RETAINED_MIN, color="black", linestyle="--", linewidth=1.0)
    handles.append(mpatches.Patch(color="none",
                                  label=f"MIN_RETAINED_MIN = {MIN_RETAINED_MIN} min",
                                  linestyle="--"))
ax1.legend(handles=handles, fontsize=8)
plt.tight_layout()
p1 = FIGURES_DIR / f"fd_survey_retained_min_sorted_{THRESH_TAG}.png"
fig1.savefig(p1, dpi=300, bbox_inches="tight"); plt.close(fig1)
print(f"\nSaved: {p1}")


# Plot 2 — Retained minutes histogram
fig2, ax2 = plt.subplots(figsize=(6, 4))
max_min = df_runs["retained_min"].max()
bins = np.arange(0, max_min + 0.6, 0.5)
pre_vals  = df_runs[df_runs["session"] == "ses-01"]["retained_min"].values
post_vals = df_runs[df_runs["session"] == "ses-02"]["retained_min"].values
ax2.hist([pre_vals, post_vals], bins=bins, stacked=True,
         color=[PRE_COLOR, POST_COLOR], edgecolor="white", linewidth=0.4)
ax2.set_xlabel("Retained minutes")
ax2.set_ylabel("Runs")
ax2.set_title(f"Retained minutes — FD > {FD_THRESH} mm")
handles2 = ses_legend[:]
if MIN_RETAINED_MIN is not None:
    ax2.axvline(MIN_RETAINED_MIN, color="black", linestyle="--", linewidth=1.0)
    handles2.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                label=f"MIN_RETAINED_MIN = {MIN_RETAINED_MIN}"))
ax2.legend(handles=handles2, fontsize=8)
plt.tight_layout()
p2 = FIGURES_DIR / f"fd_survey_retained_min_hist_{THRESH_TAG}.png"
fig2.savefig(p2, dpi=300, bbox_inches="tight"); plt.close(fig2)
print(f"Saved: {p2}")


# Plot 3 — Longest contiguous gap (seconds) sorted descending
df_gap = df_runs.sort_values("max_gap_sec", ascending=False).reset_index(drop=True)
fig3, ax3 = plt.subplots(figsize=(11, 4))
bar_colors3 = [SES_PALETTE.get(s, "#888") for s in df_gap["session"]]
ax3.bar(range(len(df_gap)), df_gap["max_gap_sec"],
        color=bar_colors3, edgecolor="white", linewidth=0.4)
ax3.set_xticks(range(len(df_gap)))
ax3.set_xticklabels(df_gap["_label"], rotation=60, ha="right", fontsize=7)
ax3.set_ylabel("Longest contiguous gap (s)")
ax3.set_xlabel("Run (sorted descending)")
ax3.set_title(f"Longest contiguous censored gap — FD > {FD_THRESH} mm")
handles3 = ses_legend[:]
if MAX_GAP_SEC is not None:
    ax3.axhline(MAX_GAP_SEC, color="black", linestyle="--", linewidth=1.0)
    handles3.append(plt.Line2D([0], [0], color="black", linestyle="--",
                                label=f"MAX_GAP_SEC = {MAX_GAP_SEC}"))
ax3.legend(handles=handles3, fontsize=8)
plt.tight_layout()
p3 = FIGURES_DIR / f"fd_survey_max_gap_sorted_{THRESH_TAG}.png"
fig3.savefig(p3, dpi=300, bbox_inches="tight"); plt.close(fig3)
print(f"Saved: {p3}")


# Plot 4 — Censored fraction (%) sorted descending
df_pct = df_runs.sort_values("pct_censored", ascending=False).reset_index(drop=True)
fig4, ax4 = plt.subplots(figsize=(11, 4))
bar_colors4 = [SES_PALETTE.get(s, "#888") for s in df_pct["session"]]
ax4.bar(range(len(df_pct)), df_pct["pct_censored"],
        color=bar_colors4, edgecolor="white", linewidth=0.4)
ax4.set_xticks(range(len(df_pct)))
ax4.set_xticklabels(df_pct["_label"], rotation=60, ha="right", fontsize=7)
ax4.set_ylabel("Censored frames (%)")
ax4.set_xlabel("Run (sorted descending)")
ax4.set_title(f"Censored fraction per run — FD > {FD_THRESH} mm")
ax4.axhline(50, color="#aaaaaa", linestyle=":", linewidth=0.8)
ax4.legend(handles=ses_legend + [
    plt.Line2D([0], [0], color="#aaaaaa", linestyle=":", label="50% reference")
], fontsize=8)
plt.tight_layout()
p4 = FIGURES_DIR / f"fd_survey_pct_censored_sorted_{THRESH_TAG}.png"
fig4.savefig(p4, dpi=300, bbox_inches="tight"); plt.close(fig4)
print(f"Saved: {p4}")

# Plot 5 — Longest contiguous retained segment (seconds) sorted ascending
df_ret = df_runs.sort_values("longest_retained_sec", ascending=True).reset_index(drop=True)
fig5, ax5 = plt.subplots(figsize=(11, 4))
bar_colors5 = [SES_PALETTE.get(s, "#888") for s in df_ret["session"]]
ax5.bar(range(len(df_ret)), df_ret["longest_retained_sec"],
        color=bar_colors5, edgecolor="white", linewidth=0.4)
ax5.set_xticks(range(len(df_ret)))
ax5.set_xticklabels(df_ret["_label"], rotation=60, ha="right", fontsize=7)
ax5.set_ylabel("Longest retained segment (s)")
ax5.set_xlabel("Run (sorted ascending)")
ax5.set_title(f"Longest contiguous retained segment — FD > {FD_THRESH} mm")
ax5.legend(handles=ses_legend, fontsize=8)
plt.tight_layout()
p5 = FIGURES_DIR / f"fd_survey_longest_retained_sorted_{THRESH_TAG}.png"
fig5.savefig(p5, dpi=300, bbox_inches="tight"); plt.close(fig5)
print(f"Saved: {p5}")

print(f"\n── Run summary ──────────────────────────────────────────────────────")
print(f"  Runs processed : {len(rows)}")
print(f"  Runs failed    : {len(failed)}")
print(f"  Runs missing   : {len(missing_runs)}")
print(f"  FD threshold   : {FD_THRESH} mm")
print(f"  max_gap_frames : min={df_runs['max_gap_frames'].min()}  "
      f"max={df_runs['max_gap_frames'].max()}  "
      f"median={df_runs['max_gap_frames'].median():.0f}")
print(f"  longest_retained_sec : min={df_runs['longest_retained_sec'].min():.1f}  "
      f"max={df_runs['longest_retained_sec'].max():.1f}  "
      f"median={df_runs['longest_retained_sec'].median():.1f}")
print()

print("\nDone.\n")
