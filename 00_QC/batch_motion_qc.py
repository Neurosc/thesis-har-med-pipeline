import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.motion_qc import run_motion_qc

# ── Configurable paths and parameters ────────────────────────────────────────
FMRIPREP_ROOT = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep"
)
HAR_MED_ROOT = Path("/BICNAS2/group-northoff/jkokino/har_med_codes")
FIGURES_INDIV = HAR_MED_ROOT / "figures" / "individual"
RESULTS_INDIV = HAR_MED_ROOT / "results" / "individual"
FIGURES_GROUP = HAR_MED_ROOT / "figures"
RESULTS_GROUP = HAR_MED_ROOT / "results"
FD_THRESHOLD = 0.3    # mm (Goldberg et al. 2024)
DVARS_THRESHOLD = 1.5  # std_dvars (Power et al. 2014)
TASK = "rest"
USE_CUSTOM_FD = True   # Goldberg/Lynch filtered FD (recommended)
# ─────────────────────────────────────────────────────────────────────────────

for d in (FIGURES_INDIV, RESULTS_INDIV, FIGURES_GROUP, RESULTS_GROUP):
    d.mkdir(parents=True, exist_ok=True)

# Step 2: Auto-discover subjects and sessions
subject_dirs = sorted(FMRIPREP_ROOT.glob("sub-*"))
if not subject_dirs:
    raise RuntimeError(f"No sub-* directories found under {FMRIPREP_ROOT}")
print(f"Found {len(subject_dirs)} subject(s).")

runs = []  # list of (subject_id, session_id, confounds_path)
for sub_dir in subject_dirs:
    subject_id = sub_dir.name
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        session_id = ses_dir.name
        pattern = f"{subject_id}_{session_id}_task-{TASK}_desc-confounds_timeseries.tsv.gz"
        matches = list((ses_dir / "func").glob(pattern))
        if not matches:
            print(f"  WARNING: confounds file missing for {subject_id}/{session_id} — skipping.")
            continue
        runs.append((subject_id, session_id, matches[0]))

print(f"Runs to process: {len(runs)}\n")

# Step 3: Loop with optional tqdm
try:
    from tqdm import tqdm
    iterator = tqdm(runs, desc="Motion QC", unit="run")
except ImportError:
    iterator = runs

output_dirs = {"figures": FIGURES_INDIV, "results": RESULTS_INDIV}
summaries = []
failed = []

for subject_id, session_id, confounds_path in iterator:
    try:
        stats = run_motion_qc(
            confounds_path=confounds_path,
            subject_id=subject_id,
            session_id=session_id,
            output_dirs=output_dirs,
            fd_thresh=FD_THRESHOLD,
            dvars_thresh=DVARS_THRESHOLD,
            use_custom_fd=USE_CUSTOM_FD,
            compare_fd=False,  # keep individual figures clean in batch mode
        )
        summaries.append(stats)
    except Exception as exc:
        print(f"  ERROR: {subject_id}/{session_id} — {exc}")
        failed.append(f"{subject_id}/{session_id}")

# Step 4: Master summary table
df = pd.DataFrame(summaries, columns=[
    "subject", "session",
    "mean_fd_fmriprep", "mean_fd_custom",
    "n_frames", "mean_fd", "median_fd", "max_fd",
    "n_censored", "pct_censored", "mean_std_dvars", "max_std_dvars",
    "n_frames_remaining",
])
tsv_path = RESULTS_GROUP / "motion_summary_all.tsv"
df.to_csv(tsv_path, sep="\t", index=False)
print("\n── Master summary table ──")
print(df.to_string())
print(f"\nSaved: {tsv_path}")

# Step 5: Group-level figure
SESSION_COLORS = {"ses-01": "#4878CF", "ses-02": "#D65F5F"}

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle("Group Motion QC Summary", fontsize=14)

# A: histogram of mean FD
axes[0, 0].hist(df["mean_fd"], bins=20, color="steelblue", edgecolor="white")
for thresh, ls in [(0.2, "--"), (0.3, "-.")]:
    axes[0, 0].axvline(thresh, color="black", linestyle=ls, linewidth=1,
                       label=f"{thresh} mm")
axes[0, 0].set_xlabel("Mean FD (mm)")
axes[0, 0].set_ylabel("Count")
axes[0, 0].set_title("Distribution of mean FD")
axes[0, 0].legend(fontsize=8)

# B: histogram of % censored
axes[0, 1].hist(df["pct_censored"], bins=20, color="darkorange", edgecolor="white")
for thresh, ls in [(10, "--"), (25, "-."), (50, ":")]:
    axes[0, 1].axvline(thresh, color="black", linestyle=ls, linewidth=1,
                       label=f"{thresh}%")
axes[0, 1].set_xlabel("% censored frames")
axes[0, 1].set_ylabel("Count")
axes[0, 1].set_title("Distribution of % censored frames")
axes[0, 1].legend(fontsize=8)

# C: scatter mean FD vs % censored, coloured by session
for ses, grp in df.groupby("session"):
    axes[1, 0].scatter(grp["mean_fd"], grp["pct_censored"],
                       label=ses, color=SESSION_COLORS.get(ses, "gray"),
                       alpha=0.8, edgecolors="white", linewidths=0.5)
axes[1, 0].set_xlabel("Mean FD (mm)")
axes[1, 0].set_ylabel("% censored frames")
axes[1, 0].set_title("Mean FD vs % censored")
axes[1, 0].legend(fontsize=8)

# D: boxplot mean FD by session
sessions_ordered = sorted(df["session"].unique())
data_by_session = [df.loc[df["session"] == s, "mean_fd"].values for s in sessions_ordered]
bp = axes[1, 1].boxplot(data_by_session, labels=sessions_ordered, patch_artist=True)
for patch, ses in zip(bp["boxes"], sessions_ordered):
    patch.set_facecolor(SESSION_COLORS.get(ses, "gray"))
    patch.set_alpha(0.7)
axes[1, 1].set_ylabel("Mean FD (mm)")
axes[1, 1].set_title("Mean FD by session")

plt.tight_layout()
group_fig_path = FIGURES_GROUP / "group_motion_summary.png"
plt.savefig(group_fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Group figure saved: {group_fig_path}")

# FD comparison figure: fMRIPrep vs custom
fig_cmp, ax_cmp = plt.subplots(figsize=(7, 6))
ax_cmp.scatter(df["mean_fd_fmriprep"], df["mean_fd_custom"],
               alpha=0.8, edgecolors="white", linewidths=0.5, color="steelblue")
lim_max = max(df["mean_fd_fmriprep"].max(), df["mean_fd_custom"].max()) * 1.1
ax_cmp.plot([0, lim_max], [0, lim_max], color="black", linestyle="--",
            linewidth=1, label="y = x")
corr = df["mean_fd_fmriprep"].corr(df["mean_fd_custom"])
ratio = (df["mean_fd_custom"] / df["mean_fd_fmriprep"]).mean()
ax_cmp.set_xlabel("Mean FD — fMRIPrep (mm)")
ax_cmp.set_ylabel("Mean FD — Custom filtered (mm)")
ax_cmp.set_title("Effect of respiratory filtering on mean FD")
ax_cmp.legend(fontsize=8)
ax_cmp.set_xlim(0, lim_max)
ax_cmp.set_ylim(0, lim_max)
ax_cmp.text(0.05, 0.92, f"r = {corr:.3f}   ratio = {ratio:.3f}",
            transform=ax_cmp.transAxes, fontsize=9)
plt.tight_layout()
cmp_fig_path = FIGURES_GROUP / "fd_comparison.png"
plt.savefig(cmp_fig_path, dpi=300, bbox_inches="tight")
plt.close(fig_cmp)
print(f"FD comparison figure saved: {cmp_fig_path}")

print(f"\n── FD filtering effect ──")
print(f"  Pearson r (fMRIPrep vs custom) : {corr:.4f}")
print(f"  Mean ratio (custom / fMRIPrep) : {ratio:.4f}")
print(f"  Mean fMRIPrep FD               : {df['mean_fd_fmriprep'].mean():.4f} mm")
print(f"  Mean custom FD                 : {df['mean_fd_custom'].mean():.4f} mm")

# Step 6: Final console summary
total = len(summaries)
n_failed = len(failed)
print(f"\n── Final summary ──")
print(f"  Runs processed successfully : {total}")
print(f"  Runs failed                 : {n_failed}")
if failed:
    print(f"  Failed runs: {', '.join(failed)}")

pct = df["pct_censored"]
print(f"\n  Censored frame distribution:")
print(f"    < 10%     : {(pct < 10).sum()} runs")
print(f"    10–25%    : {((pct >= 10) & (pct < 25)).sum()} runs")
print(f"    25–50%    : {((pct >= 25) & (pct < 50)).sum()} runs")
print(f"    > 50%     : {(pct >= 50).sum()} runs")

mfd = df["mean_fd"]
print(f"\n  Mean FD distribution:")
print(f"    < 0.2 mm  : {(mfd < 0.2).sum()} runs")
print(f"    0.2–0.3   : {((mfd >= 0.2) & (mfd < 0.3)).sum()} runs")
print(f"    > 0.3 mm  : {(mfd >= 0.3).sum()} runs")

n_exclude_strict = (pct > 25).sum()
n_exclude_lenient = (pct > 50).sum()
print(f"\n  Exclusion suggestion (subject-level, by % censored):")
print(f"    Strict  (> 25% censored): {n_exclude_strict} run(s) at risk")
print(f"    Lenient (> 50% censored): {n_exclude_lenient} run(s) at risk")
