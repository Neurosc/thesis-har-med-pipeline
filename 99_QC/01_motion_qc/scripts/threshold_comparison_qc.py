import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.motion_qc import load_confounds, compute_custom_fd

# ── Configuration ─────────────────────────────────────────────────────────────
FMRIPREP_ROOT  = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
_QC_DIR        = Path(__file__).resolve().parents[1]
FIGURES_DIR    = _QC_DIR / "figures"
RESULTS_DIR    = _QC_DIR / "results"
FD_THRESHOLDS  = [0.3, 0.5]   # mm — comparison, no threshold decided yet
TASK           = "rest"
TR             = 1.8
BACKWARD_DIFF_N = 1
BAD_RUN_PCT    = 50.0          # % censored above which a run is flagged
# ──────────────────────────────────────────────────────────────────────────────

for d in (FIGURES_DIR, RESULTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

print(
    f"FD: Power 2012 method — {BACKWARD_DIFF_N}-TR backward difference, "
    f"50 mm sphere for rotations, TR={TR} s. No respiratory filtering."
)

# ── Step 1: Discover all confounds files ──────────────────────────────────────
subject_dirs = sorted(FMRIPREP_ROOT.glob("sub-*"))
if not subject_dirs:
    raise RuntimeError(f"No sub-* directories found under {FMRIPREP_ROOT}")
print(f"Found {len(subject_dirs)} subject(s).")

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

print(f"Runs to process: {len(runs)}\n")

# ── Step 2 & 3: Compute custom FD per run, build run-level table ──────────────
try:
    from tqdm import tqdm
    iterator = tqdm(runs, desc="Custom FD", unit="run")
except ImportError:
    iterator = runs

run_rows = []
fd_series = {}   # (subject_id, session_id) → fd array (for figure use)
failed = []

for subject_id, session_id, confounds_path in iterator:
    try:
        df = load_confounds(confounds_path)
        fd = compute_custom_fd(df, tr=TR, backward_diff_n=BACKWARD_DIFF_N, verbose=False)
        n_frames = len(fd)
        fd_series[(subject_id, session_id)] = fd
        row = {
            "subject":             subject_id,
            "session":             session_id,
            "n_frames":            n_frames,
            "mean_fd":             float(np.nanmean(fd)),
            "median_fd":           float(np.nanmedian(fd)),
            "max_fd":              float(np.nanmax(fd)),
        }
        for thresh in FD_THRESHOLDS:
            key = str(thresh).replace(".", "")   # "03" or "05"
            n_above = int(np.sum(fd[~np.isnan(fd)] > thresh))
            row[f"n_frames_above_{key}"]   = n_above
            row[f"pct_frames_above_{key}"] = 100.0 * n_above / n_frames
        run_rows.append(row)
    except Exception as exc:
        print(f"  ERROR: {subject_id}/{session_id} — {exc}")
        failed.append(f"{subject_id}/{session_id}")

run_df = pd.DataFrame(run_rows)
run_tsv = RESULTS_DIR / "run_level_fd_summary.tsv"
run_df.to_csv(run_tsv, sep="\t", index=False)
print(f"Run-level summary saved: {run_tsv}")

# ── Step 4: Subject-level summary ────────────────────────────────────────────
subject_rows = []
for subject_id, grp in run_df.groupby("subject"):
    ses01 = grp[grp["session"] == "ses-01"]
    ses02 = grp[grp["session"] == "ses-02"]

    def _val(subset, col):
        return float(subset[col].iloc[0]) if len(subset) == 1 else np.nan

    row = {"subject": subject_id}
    for ses_key, subset in [("ses01", ses01), ("ses02", ses02)]:
        row[f"mean_fd_{ses_key}"]          = _val(subset, "mean_fd")
        row[f"max_fd_{ses_key}"]           = _val(subset, "max_fd")
        row[f"pct_above_03_{ses_key}"]     = _val(subset, "pct_frames_above_03")
        row[f"pct_above_05_{ses_key}"]     = _val(subset, "pct_frames_above_05")

    for thresh_key in ["03", "05"]:
        col = f"pct_frames_above_{thresh_key}"
        vals = grp[col].values
        row[f"worst_session_pct_above_{thresh_key}"] = float(np.nanmax(vals))
        row[f"mean_pct_above_{thresh_key}"]          = float(np.nanmean(vals))

    subject_rows.append(row)

subj_df = pd.DataFrame(subject_rows)
subj_tsv = RESULTS_DIR / "subject_level_fd_summary.tsv"
subj_df.to_csv(subj_tsv, sep="\t", index=False)
print(f"Subject-level summary saved: {subj_tsv}")

# ── Step 5: Exclusion impact ──────────────────────────────────────────────────
exclusion_lines = []
exclusion_lines.append("Exclusion impact analysis")
exclusion_lines.append(f"Bad-run criterion: > {BAD_RUN_PCT}% frames censored")
exclusion_lines.append("")

for thresh in FD_THRESHOLDS:
    key = str(thresh).replace(".", "")
    col = f"pct_frames_above_{key}"
    bad_runs = run_df[run_df[col] > BAD_RUN_PCT]
    bad_subjects = bad_runs["subject"].unique()

    exclusion_lines.append(f"FD threshold = {thresh} mm")
    exclusion_lines.append(f"  Total runs flagged (>{BAD_RUN_PCT}% censored): {len(bad_runs)}")
    exclusion_lines.append(f"  Subjects with ≥1 bad run: {len(bad_subjects)}")
    if len(bad_runs):
        for _, r in bad_runs.iterrows():
            exclusion_lines.append(
                f"    {r['subject']} {r['session']}  "
                f"{r[col]:.1f}% censored  (mean FD {r['mean_fd']:.3f} mm)"
            )
    exclusion_lines.append("")

    pct_col = run_df[col]
    for boundary, label in [(10, "< 10%"), (25, "10–25%"), (50, "25–50%"), (None, "> 50%")]:
        if boundary is None:
            n = (pct_col > 50).sum()
        elif label == "< 10%":
            n = (pct_col < 10).sum()
        elif label == "10–25%":
            n = ((pct_col >= 10) & (pct_col < 25)).sum()
        else:
            n = ((pct_col >= 25) & (pct_col < 50)).sum()
        exclusion_lines.append(f"  {label}: {n} run(s)")
    exclusion_lines.append("")

exclusion_text = "\n".join(exclusion_lines)
print("\n── Exclusion impact ──")
print(exclusion_text)

exclusion_path = RESULTS_DIR / "exclusion_impact.txt"
exclusion_path.write_text(exclusion_text)
print(f"Exclusion impact saved: {exclusion_path}")

# ── Step 6a: Subject-level mean FD bar chart ──────────────────────────────────
subjects_sorted = sorted(run_df["subject"].unique())
n_subjects = len(subjects_sorted)

fig_bar, ax_bar = plt.subplots(figsize=(16, 5))
x = np.arange(n_subjects)
width = 0.35
ses_colors = {"ses-01": "#4878CF", "ses-02": "#D65F5F"}

for offset, ses_key, ses_label in [(-width / 2, "ses-01", "ses-01"), (width / 2, "ses-02", "ses-02")]:
    heights = []
    for sub in subjects_sorted:
        subset = run_df[(run_df["subject"] == sub) & (run_df["session"] == ses_key)]
        heights.append(float(subset["mean_fd"].iloc[0]) if len(subset) == 1 else 0.0)
    ax_bar.bar(x + offset, heights, width, label=ses_label,
               color=ses_colors[ses_key], alpha=0.85, edgecolor="white")

for thresh, ls, lw in [(0.3, "--", 1.2), (0.5, "-.", 1.0)]:
    ax_bar.axhline(thresh, color="black", linestyle=ls, linewidth=lw, label=f"{thresh} mm")

ax_bar.set_xticks(x)
ax_bar.set_xticklabels([s.replace("sub-", "") for s in subjects_sorted],
                        rotation=45, ha="right", fontsize=7)
ax_bar.set_ylabel("Mean FD — custom (mm)")
ax_bar.set_title("Mean custom FD per subject per session")
ax_bar.legend(fontsize=8, ncol=4, loc="upper right")
plt.tight_layout()
bar_path = FIGURES_DIR / "subject_level_fd_bars.png"
plt.savefig(bar_path, dpi=300, bbox_inches="tight")
plt.close(fig_bar)
print(f"Subject FD bar chart saved: {bar_path}")

# ── Step 6b: Threshold comparison figure ─────────────────────────────────────
fig_cmp, (ax_hist, ax_worst) = plt.subplots(1, 2, figsize=(14, 5))
fig_cmp.suptitle("Threshold comparison: FD > 0.3 mm vs FD > 0.5 mm", fontsize=13)

thresh_colors = {0.3: "#4878CF", 0.5: "#D65F5F"}

# Left: overlaid histograms of % censored
for thresh in FD_THRESHOLDS:
    key = str(thresh).replace(".", "")
    col = f"pct_frames_above_{key}"
    ax_hist.hist(run_df[col], bins=20, alpha=0.55,
                 color=thresh_colors[thresh], edgecolor="white",
                 label=f"FD > {thresh} mm")
for boundary, ls in [(10, "--"), (25, "-."), (50, ":")]:
    ax_hist.axvline(boundary, color="gray", linestyle=ls, linewidth=0.9, label=f"{boundary}%")
ax_hist.set_xlabel("% frames censored")
ax_hist.set_ylabel("Count (runs)")
ax_hist.set_title("Distribution of % censored frames")
ax_hist.legend(fontsize=8)

# Right: grouped bar of worst-session % censored per subject
x = np.arange(n_subjects)
width = 0.35
for offset, thresh in [(-width / 2, 0.3), (width / 2, 0.5)]:
    key = str(thresh).replace(".", "")
    heights = []
    for sub in subjects_sorted:
        row = subj_df[subj_df["subject"] == sub]
        heights.append(float(row[f"worst_session_pct_above_{key}"].iloc[0]) if len(row) else 0.0)
    ax_worst.bar(x + offset, heights, width, label=f"FD > {thresh} mm",
                 color=thresh_colors[thresh], alpha=0.85, edgecolor="white")

for boundary, ls in [(25, "--"), (50, "-.")]:
    ax_worst.axhline(boundary, color="black", linestyle=ls, linewidth=0.9, label=f"{boundary}%")
ax_worst.set_xticks(x)
ax_worst.set_xticklabels([s.replace("sub-", "") for s in subjects_sorted],
                          rotation=45, ha="right", fontsize=7)
ax_worst.set_ylabel("Worst-session % censored")
ax_worst.set_title("Worst-session % censored per subject")
ax_worst.legend(fontsize=8, ncol=2, loc="upper right")

plt.tight_layout()
cmp_path = FIGURES_DIR / "threshold_comparison.png"
plt.savefig(cmp_path, dpi=300, bbox_inches="tight")
plt.close(fig_cmp)
print(f"Threshold comparison figure saved: {cmp_path}")

# ── Step 7: Console summary ───────────────────────────────────────────────────
print(f"\n── Final summary ──")
print(f"  FD method: custom Goldberg/Lynch (bandstop 0.2–0.5 Hz, {BACKWARD_DIFF_N}-TR backward diff)")
print(f"  Runs processed: {len(run_rows)}  |  Failed: {len(failed)}")
if failed:
    print(f"  Failed: {', '.join(failed)}")
print()
for thresh in FD_THRESHOLDS:
    key = str(thresh).replace(".", "")
    col = f"pct_frames_above_{key}"
    print(f"  FD > {thresh} mm:")
    print(f"    Mean % censored : {run_df[col].mean():.1f}%")
    print(f"    Median          : {run_df[col].median():.1f}%")
    print(f"    Runs > {BAD_RUN_PCT}%    : {(run_df[col] > BAD_RUN_PCT).sum()}")
    print()
print(f"  Saved files:")
print(f"    {run_tsv}")
print(f"    {subj_tsv}")
print(f"    {exclusion_path}")
print(f"    {bar_path}")
print(f"    {cmp_path}")
