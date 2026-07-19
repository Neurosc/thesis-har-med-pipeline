#!/usr/bin/env python3
"""
Stage 2b — exclusion-threshold table over all 40 subjects. APPLIES NOTHING.

The maximal_nocensor sensitivity pipeline retains high-motion frames, so the
>50%-censoring subject-exclusion rule that produced the current n=35 does not
automatically carry over. This tabulates what each candidate subject-level
cutoff WOULD do, so the threshold can be chosen deliberately -- and, per the
task constraint, NOT chosen by which one makes the results look better.

Fixed frame criterion: FD > 0.3 mm, matching the pipeline's censoring criterion.
The only thing varied is the subject-level percentage cutoff.

Exclusion is at SUBJECT level: if EITHER session exceeds the cutoff, both
sessions are dropped. So the deciding quantity per subject is the WORSE of its
two sessions, which is what the "worst session" distribution below reports.

Note on sub-12: excluded from the denoising sample as a broken/missing run, a
different reason from the censoring rule. It has no maximal_nocensor timeseries
and cannot enter the analysis at any cutoff. It is shown in the table for
completeness and flagged, never silently dropped.

Input : 99_QC/01_motion_qc/results/fd_covariates_wide_thresh03.csv
        (pcf_pre / pcf_post = % frames FD > 0.3 mm, per session)
Output: 99_QC/01_motion_qc/results/exclusion_threshold_table/
            exclusion_threshold_table.csv     one row per cutoff
            exclusion_per_subject.csv         one row per subject
            exclusion_threshold_log.txt

Run locally (or on the server -- input is a committed CSV, no NIfTIs):
    python 99_QC/01_motion_qc/scripts/exclusion_threshold_table.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from utils.subject_filter import _EXCLUDED, _PIPELINE_EXCLUDED   # noqa: E402

FD_COV  = REPO_ROOT / "99_QC" / "01_motion_qc" / "results" / "fd_covariates_wide_thresh03.csv"
OUT_DIR = REPO_ROOT / "99_QC" / "01_motion_qc" / "results" / "exclusion_threshold_table"

CUTOFFS = [20, 25, 30, 35, 40, 50]     # percent of frames with FD > 0.3 mm
FD_THRESH_MM = 0.3                     # fixed; only the subject-level cutoff varies

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "exclusion_threshold_log.txt"


class _Tee:
    """Mirror stdout to a UTF-8 log.

    The console write is guarded: a Windows console is cp1252 and raises on the
    box-drawing characters, while the log file is always UTF-8. Without this the
    script dies on its own section headers when run locally.
    """

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "w", encoding="utf-8")

    def write(self, data):
        try:
            self.stream.write(data)
        except UnicodeEncodeError:
            self.stream.write(data.encode("ascii", "replace").decode("ascii"))
        self.fh.write(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


sys.stdout = _Tee(sys.__stdout__, LOG_PATH)

if not FD_COV.is_file():
    raise FileNotFoundError(f"Required input not found: {FD_COV}")

cov = pd.read_csv(FD_COV)
for col in ("subject", "pcf_pre", "pcf_post"):
    if col not in cov.columns:
        raise ValueError(
            f"{FD_COV.name} is missing column '{col}'. Present: {list(cov.columns)}"
        )
if cov[["pcf_pre", "pcf_post"]].isna().any().any():
    bad = cov.loc[cov[["pcf_pre", "pcf_post"]].isna().any(axis=1), "subject"].tolist()
    raise ValueError(f"Missing pcf value(s) for: {bad}. Refusing to guess.")

subj = cov[["subject", "pcf_pre", "pcf_post"]].copy()
subj["worst_session_pcf"] = subj[["pcf_pre", "pcf_post"]].max(axis=1)
subj["worse_session"] = np.where(subj.pcf_post > subj.pcf_pre, "ses-02", "ses-01")
subj["currently_dropped_n35"] = subj.subject.isin(_EXCLUDED)
subj["no_denoised_data"] = subj.subject.isin(_PIPELINE_EXCLUDED)
subj = subj.sort_values("worst_session_pcf").reset_index(drop=True)

print("=" * 78)
print("STAGE 2b — EXCLUSION THRESHOLD TABLE.  NO THRESHOLD IS APPLIED.")
print("=" * 78)
print(f"Frame criterion : FD > {FD_THRESH_MM} mm (fixed, matches the pipeline)")
print(f"Varying         : subject-level cutoff on % frames exceeding it")
print(f"Rule            : if EITHER session exceeds the cutoff, drop BOTH")
print(f"Subjects        : {len(subj)}")
print(f"Source          : {FD_COV.name}")
print()

# ── Distribution ──────────────────────────────────────────────────────────────
runs = pd.concat([
    subj[["subject"]].assign(session="ses-01", pcf=subj.pcf_pre),
    subj[["subject"]].assign(session="ses-02", pcf=subj.pcf_post),
])

print("── Distribution of % frames > 0.3 mm ──")
print(f"  {'':<22}{'min':>8}{'Q1':>8}{'median':>9}{'Q3':>8}{'max':>8}{'mean':>8}")
for label, x in (("per run (80 runs)", runs.pcf),
                 ("per subject (worst)", subj.worst_session_pcf)):
    q1, med, q3 = np.percentile(x, [25, 50, 75])
    print(f"  {label:<22}{x.min():>8.2f}{q1:>8.2f}{med:>9.2f}{q3:>8.2f}"
          f"{x.max():>8.2f}{x.mean():>8.2f}")
print("\n  'per subject (worst)' is the quantity the cutoff acts on, so that is")
print("  the row against which to read where each candidate cutoff falls.")
print()

# ── Threshold sweep ───────────────────────────────────────────────────────────
rows = []
print("── What each cutoff would do ──")
for cut in CUTOFFS:
    excl = subj[subj.worst_session_pcf > cut]
    names = sorted(excl.subject.tolist())
    # sub-12 has no denoised data at any cutoff, so it can never contribute.
    retained = sorted(set(subj.subject) - set(names) - set(_PIPELINE_EXCLUDED))
    rows.append({
        "cutoff_pct": cut,
        "n_excluded_by_cutoff": len(names),
        "excluded_by_cutoff": ";".join(names) if names else "",
        "n_also_unavailable": len(_PIPELINE_EXCLUDED - set(names)),
        "resulting_n": len(retained),
        "n_lost_vs_40": 40 - len(retained),
        "n_lost_vs_current35": 35 - len([s for s in retained if s not in _EXCLUDED]),
    })
    print(f"\n  cutoff > {cut}%   excluded by cutoff: {len(names)}   -> resulting n = {len(retained)}")
    if names:
        marks = []
        for s in names:
            tag = []
            if s in _EXCLUDED:
                tag.append("already dropped at n=35")
            if s in _PIPELINE_EXCLUDED:
                tag.append("no denoised data")
            w = subj.loc[subj.subject == s, "worst_session_pcf"].iloc[0]
            marks.append(f"{s} ({w:.1f}%{'; ' + ', '.join(tag) if tag else ''})")
        for m in marks:
            print(f"      {m}")
    else:
        print("      (none)")

sweep = pd.DataFrame(rows)

print("\n" + "=" * 78)
print("SUMMARY TABLE")
print("=" * 78)
print(f"  {'cutoff':>8}{'excluded':>10}{'resulting n':>13}{'lost vs 40':>12}{'lost vs n=35':>14}")
for r in rows:
    print(f"  {'>' + str(r['cutoff_pct']) + '%':>8}{r['n_excluded_by_cutoff']:>10}"
          f"{r['resulting_n']:>13}{r['n_lost_vs_40']:>12}{r['n_lost_vs_current35']:>14}")

print(f"\n  For reference, the 5 subjects currently dropped at n=35: "
      f"{sorted(_EXCLUDED)}")
print(f"  Of those, unavailable regardless of cutoff (no denoised data): "
      f"{sorted(_PIPELINE_EXCLUDED)}")

sweep_path = OUT_DIR / "exclusion_threshold_table.csv"
subj_path  = OUT_DIR / "exclusion_per_subject.csv"
sweep.to_csv(sweep_path, index=False)
subj.to_csv(subj_path, index=False)

print(f"\n  Saved: {sweep_path.name}")
print(f"  Saved: {subj_path.name}")
print("\n  NO THRESHOLD APPLIED. Choose one on motion grounds, not on which")
print("  cutoff produces the preferred result.")

sys.stdout.flush()
