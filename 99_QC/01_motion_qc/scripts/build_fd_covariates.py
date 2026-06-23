"""
build_fd_covariates.py
======================
Builds a motion covariate table from fMRIPrep confounds files for use as
covariates in a later group-level regression.  Does NOT run any regression
and does NOT read group labels.

Outputs
-------
fd_covariates_long_thresh<TAG>.csv
    One row per subject × session.
    Columns: subject, session, pcf, pcf_sq, mean_fd_retained,
             n_retained, retained_min

fd_covariates_wide_thresh<TAG>.csv
    One row per subject.
    Per-session columns: <metric>_pre, <metric>_post  (ses-01 = pre, ses-02 = post)
    Difference columns:  <metric>_diff = post − pre

FD censoring
------------
    censored   : FD > FD_THRESH  (frame 0 NaN is NEVER censored)
    retained   : FD <= FD_THRESH (plus frame 0, which is non-censored but NaN)

mean_fd_retained is the mean over frames 1-239 where FD <= FD_THRESH only
(frame 0 is retained but NaN so it is excluded from the mean, not from n_retained).

Run
---
    conda activate fmri
    python 99_QC/01_motion_qc/scripts/build_fd_covariates.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────────────────
FD_THRESH       = 0.3    # mm  — change to 0.5 and rerun for comparison
TR              = 1.8    # seconds
EXPECTED_FRAMES = 240
TASK            = "rest"

FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")

SCRIPT_DIR  = Path(__file__).resolve().parent
QC_DIR      = SCRIPT_DIR.parent
RESULTS_DIR = QC_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

THRESH_TAG  = str(FD_THRESH).replace(".", "")
LONG_CSV    = RESULTS_DIR / f"fd_covariates_long_thresh{THRESH_TAG}.csv"
WIDE_CSV    = RESULTS_DIR / f"fd_covariates_wide_thresh{THRESH_TAG}.csv"

COVARIATES = ["pcf", "pcf_sq", "mean_fd_retained", "n_retained", "retained_min"]
# ──────────────────────────────────────────────────────────────────────────────


def _find_confounds(fmriprep_root, task):
    """Return sorted list of (subject, session, Path) for every confounds file found."""
    pattern_gz = "{sub}_{ses}_task-{task}_desc-confounds_timeseries.tsv.gz"
    pattern    = "{sub}_{ses}_task-{task}_desc-confounds_timeseries.tsv"

    runs    = []
    missing = []

    for sub_dir in sorted(fmriprep_root.glob("sub-*")):
        if not sub_dir.is_dir():
            continue
        sub = sub_dir.name
        for ses_dir in sorted(sub_dir.glob("ses-*")):
            if not ses_dir.is_dir():
                continue
            ses  = ses_dir.name
            func = ses_dir / "func"
            if not func.is_dir():
                missing.append((sub, ses, "no func/ directory"))
                continue
            p_gz = func / pattern_gz.format(sub=sub, ses=ses, task=task)
            p_   = func / pattern.format(sub=sub, ses=ses, task=task)
            if p_gz.exists():
                runs.append((sub, ses, p_gz))
            elif p_.exists():
                runs.append((sub, ses, p_))
            else:
                missing.append((sub, ses, "confounds file absent"))

    return runs, missing


def _compute_covariates(sub, ses, path, fd_thresh, tr, expected_frames):
    """
    Read one confounds file and return a dict of covariate values.

    Raises
    ------
    KeyError   if framewise_displacement column is absent
    """
    df = pd.read_csv(path, sep="\t")

    if "framewise_displacement" not in df.columns:
        candidates = [c for c in df.columns if "fd" in c.lower() or "disp" in c.lower()]
        raise KeyError(
            f"Column 'framewise_displacement' not found in {path.name}. "
            f"Candidate columns: {candidates}"
        )

    fd = df["framewise_displacement"].to_numpy(dtype=float)
    n_frames = len(fd)

    if n_frames != expected_frames:
        print(f"  WARNING  {sub} {ses}: {n_frames} frames (expected {expected_frames}) — proceeding")

    # ── Censoring mask ─────────────────────────────────────────────────────────
    # Frame 0 is NaN by fMRIPrep convention; treat as non-censored.
    censored = np.zeros(n_frames, dtype=bool)
    for t in range(1, n_frames):
        if not np.isnan(fd[t]):
            censored[t] = fd[t] > fd_thresh

    n_censored  = int(censored.sum())
    n_retained  = n_frames - n_censored          # includes frame 0

    # ── pcf ───────────────────────────────────────────────────────────────────
    pcf    = n_censored / n_frames * 100.0
    pcf_sq = pcf ** 2

    # ── mean_fd_retained ──────────────────────────────────────────────────────
    # Mean FD over frames 1-N that are retained (FD ≤ thresh).
    # Frame 0 is excluded from the mean because it is NaN (not a real FD value),
    # but it IS counted in n_retained.
    retained_fd = fd[1:][~censored[1:]]          # frames 1-N, not censored
    retained_fd = retained_fd[~np.isnan(retained_fd)]
    if len(retained_fd) == 0:
        mean_fd_retained = np.nan
        print(f"  WARNING  {sub} {ses}: no retained frames with finite FD — mean_fd_retained = NaN")
    else:
        mean_fd_retained = float(np.mean(retained_fd))

    # ── retained_min ──────────────────────────────────────────────────────────
    retained_min = n_retained * tr / 60.0

    return {
        "subject":          sub,
        "session":          ses,
        "pcf":              round(pcf,             4),
        "pcf_sq":           round(pcf_sq,          4),
        "mean_fd_retained": round(mean_fd_retained, 4) if not np.isnan(mean_fd_retained) else np.nan,
        "n_retained":       n_retained,
        "retained_min":     round(retained_min,    4),
    }


# ── Main ───────────────────────────────────────────────────────────────────────
print("=" * 70)
print("build_fd_covariates.py")
print(f"  fMRIPrep root : {FMRIPREP_ROOT}")
print(f"  FD threshold  : {FD_THRESH} mm  (censored = FD > {FD_THRESH})")
print(f"  Expected frames per run : {EXPECTED_FRAMES}")
print(f"  TR            : {TR} s")
print("=" * 70)

if not FMRIPREP_ROOT.exists():
    sys.exit(
        f"\nERROR: fMRIPrep root not found:\n  {FMRIPREP_ROOT}\n"
        "Check the FMRIPREP_ROOT variable at the top of this script."
    )

# ── Step 1: Discover runs ──────────────────────────────────────────────────────
runs, missing = _find_confounds(FMRIPREP_ROOT, TASK)

print(f"\nDiscovered {len(runs)} confounds file(s).")
if missing:
    print(f"  Missing ({len(missing)}):")
    for sub, ses, reason in missing:
        print(f"    {sub}  {ses}  — {reason}")

# ── Step 2: Compute covariates ─────────────────────────────────────────────────
rows   = []
errors = []

for sub, ses, path in runs:
    try:
        row = _compute_covariates(sub, ses, path, FD_THRESH, TR, EXPECTED_FRAMES)
        rows.append(row)
    except Exception as exc:
        errors.append((sub, ses, str(exc)))
        print(f"  ERROR  {sub} {ses}: {exc}")

if not rows:
    sys.exit("\nNo runs processed successfully — cannot continue.")

# ── Step 3: Long-format table ──────────────────────────────────────────────────
df_long = (
    pd.DataFrame(rows)
    .sort_values(["subject", "session"])
    .reset_index(drop=True)
)

# ── Step 4: Wide-format table ──────────────────────────────────────────────────
# ses-01 = pre, ses-02 = post
df_pre  = df_long[df_long["session"] == "ses-01"].set_index("subject")
df_post = df_long[df_long["session"] == "ses-02"].set_index("subject")

wide_rows = []
all_subjects = sorted(df_long["subject"].unique())

for sub in all_subjects:
    row = {"subject": sub}
    pre_data  = df_pre.loc[sub]  if sub in df_pre.index  else None
    post_data = df_post.loc[sub] if sub in df_post.index else None

    for col in COVARIATES:
        pre_val  = float(pre_data[col])  if pre_data  is not None else np.nan
        post_val = float(post_data[col]) if post_data is not None else np.nan
        row[f"{col}_pre"]  = pre_val
        row[f"{col}_post"] = post_val
        row[f"{col}_diff"] = (
            round(post_val - pre_val, 4)
            if not (np.isnan(pre_val) or np.isnan(post_val))
            else np.nan
        )

    wide_rows.append(row)

df_wide = pd.DataFrame(wide_rows)

# ── Step 5: Save ───────────────────────────────────────────────────────────────
df_long.to_csv(LONG_CSV, index=False)
df_wide.to_csv(WIDE_CSV, index=False)

# ── Step 6: Summary ────────────────────────────────────────────────────────────
print("\n── Summary " + "─" * 59)

subjects_found = sorted(df_long["subject"].unique())
print(f"  Subjects with ≥1 run processed : {len(subjects_found)}")

ses_counts = df_long.groupby("subject")["session"].count()
missing_session = []
for sub in subjects_found:
    n = ses_counts.get(sub, 0)
    if n < 2:
        present = list(df_long[df_long["subject"] == sub]["session"])
        missing_s = {"ses-01", "ses-02"} - set(present)
        missing_session.append((sub, sorted(missing_s)))

if missing_session:
    print(f"  Subjects missing a session ({len(missing_session)}):")
    for sub, sess in missing_session:
        print(f"    {sub}  missing: {', '.join(sess)}")
else:
    print("  All subjects have both ses-01 and ses-02.")

if errors:
    print(f"  Runs that errored ({len(errors)}):")
    for sub, ses, err in errors:
        print(f"    {sub} {ses}: {err}")
else:
    print("  No errors.")

print()
print("── Per-run table (long format) " + "─" * 39)
pd.set_option("display.max_rows",    200)
pd.set_option("display.width",       140)
pd.set_option("display.float_format", lambda x: f"{x:.4f}")
print(df_long.to_string(index=False))

print(f"\nSaved: {LONG_CSV}")
print(f"Saved: {WIDE_CSV}")
print("\nDone.\n")
