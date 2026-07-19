#!/usr/bin/env python3
"""
Stage 1 preflight for the maximal_nocensor sensitivity analysis.

READ-ONLY. This script extracts nothing, denoises nothing and writes no
timeseries. It answers three questions that must be settled before any
extraction is launched, and stops.

  1. CONFIG DIFF. Print every key of PIPELINE_PRESETS["maximal"] against
     PIPELINE_PRESETS["maximal_nocensor"]. If anything other than do_censor
     differs (desc is the filename tag and is expected to differ), the script
     FAILS -- because then the two pipelines would not isolate censoring and
     the whole robustness argument would be confounded.

  2. WHAT INPUTS EXIST. For all 40 subjects x 2 sessions, check whether the
     fMRIPrep BOLD / mask / confounds actually exist. This settles empirically
     whether sub-12 can be processed at all: utils/subject_filter.py excludes
     sub-12 from the denoising sample as a "broken/missing run", which is a
     DIFFERENT reason from the >50%-censoring rule that dropped the other four.

  3. WHAT ALREADY EXISTS. Inventory the denoised NIfTIs per pipeline and the
     extracted timeseries per atlas, so it is unambiguous whether Stage 1 needs
     a denoising run first or only extraction.

Paths are not hardcoded here: build_fmriprep_paths() and build_output_path()
are imported from the denoising module, so this checks exactly the paths the
real pipeline uses.

Output (report only):
    02_timeseries_extraction/results/_preflight_maximal_nocensor/
        preflight_report.txt
        preflight_inventory.csv

Run on server:
    conda activate fmri
    python 02_timeseries_extraction/scripts/00_preflight_maximal_nocensor.py
"""

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "01_denoising" / "scripts"))

from denoise_pipelines import (          # noqa: E402
    PIPELINE_PRESETS, build_fmriprep_paths, build_output_path,
)

# ─── Configuration ────────────────────────────────────────────────────────────
FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
DENOISING_DIR = REPO_ROOT / "01_denoising" / "results"
TS_ROOT       = REPO_ROOT / "02_timeseries_extraction" / "results"
OUT_DIR       = TS_ROOT / "_preflight_maximal_nocensor"

REFERENCE  = "maximal"
CANDIDATE  = "maximal_nocensor"
EXPECTED_DIFF = {"do_censor"}       # desc handled separately (filename tag)

ALL_SUBJECTS = [f"sub-{i:02d}" for i in range(1, 41)]
SESSIONS     = ["ses-01", "ses-02"]
ATLASES      = ["qinspheres", "glasser360", "qinparcels"]
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = OUT_DIR / "preflight_report.txt"


class _Tee:
    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


sys.stdout = _Tee(sys.__stdout__, REPORT_PATH)

print("=" * 78)
print("STAGE 1 PREFLIGHT — maximal_nocensor.  READ-ONLY, extracts nothing.")
print("=" * 78)
print(f"repo      : {REPO_ROOT}")
print(f"fmriprep  : {FMRIPREP_ROOT}")
print(f"denoised  : {DENOISING_DIR}")
print()

# ═══ 1. CONFIG DIFF ═══════════════════════════════════════════════════════════
print("=" * 78)
print(f"1. CONFIG DIFF — PIPELINE_PRESETS['{REFERENCE}'] vs ['{CANDIDATE}']")
print("=" * 78)

for name in (REFERENCE, CANDIDATE):
    if name not in PIPELINE_PRESETS:
        raise KeyError(
            f"PIPELINE_PRESETS has no preset '{name}'. "
            f"Available: {sorted(PIPELINE_PRESETS)}"
        )

ref, cand = PIPELINE_PRESETS[REFERENCE], PIPELINE_PRESETS[CANDIDATE]
all_keys = sorted(set(ref) | set(cand))

print(f"  {'key':<16}{REFERENCE:>20}{CANDIDATE:>22}   same?")
differing = []
for k in all_keys:
    rv, cv = ref.get(k, "<absent>"), cand.get(k, "<absent>")
    same = rv == cv
    if not same:
        differing.append(k)
    print(f"  {k:<16}{str(rv):>20}{str(cv):>22}   {'yes' if same else 'NO'}")

print()
print(f"  Keys that differ: {sorted(differing)}")

unexpected = set(differing) - EXPECTED_DIFF - {"desc"}
if unexpected:
    raise SystemExit(
        f"\nSTOP: presets differ in {sorted(unexpected)}, not only in "
        f"{sorted(EXPECTED_DIFF)}.\n"
        f"maximal vs maximal_nocensor would then differ by more than censoring, "
        f"so the comparison would NOT isolate censoring+interpolation. "
        f"Resolve before extracting."
    )
if "do_censor" not in differing:
    raise SystemExit(
        "\nSTOP: do_censor is IDENTICAL in the two presets. The candidate is not "
        "a no-censor pipeline. Resolve before extracting."
    )
print(f"  OK: only do_censor differs (plus 'desc', the filename tag). "
      f"The comparison isolates censoring+interpolation.")
print()

# ═══ 2. fMRIPREP INPUT AVAILABILITY (all 40 subjects) ═════════════════════════
print("=" * 78)
print("2. fMRIPrep INPUT AVAILABILITY — all 40 subjects x 2 sessions")
print("=" * 78)

rows = []
for sub in ALL_SUBJECTS:
    for ses in SESSIONS:
        p = build_fmriprep_paths(sub, ses, FMRIPREP_ROOT)
        rows.append({
            "subject": sub, "session": ses,
            "bold": p["bold"].is_file(),
            "mask": p["mask"].is_file(),
            "confounds": p["confounds"].is_file(),
        })
inp = pd.DataFrame(rows)
inp["complete"] = inp[["bold", "mask", "confounds"]].all(axis=1)

incomplete = inp[~inp.complete]
print(f"  Runs with all three inputs present : {int(inp.complete.sum())} / {len(inp)}")
if len(incomplete):
    print(f"  Runs with something MISSING        : {len(incomplete)}")
    for r in incomplete.itertuples(index=False):
        miss = [c for c in ("bold", "mask", "confounds") if not getattr(r, c)]
        print(f"      {r.subject} {r.session}  missing: {', '.join(miss)}")
else:
    print("  Nothing missing — every one of the 80 runs has BOLD + mask + confounds.")

usable_subjects = sorted(
    inp.groupby("subject")["complete"].all().loc[lambda s: s].index
)
print(f"\n  Subjects with BOTH sessions complete: {len(usable_subjects)}")
unusable = sorted(set(ALL_SUBJECTS) - set(usable_subjects))
if unusable:
    print(f"  Subjects NOT fully usable          : {unusable}")

print("\n  sub-12 specifically (excluded from the n=39 denoising sample as a")
print("  'broken/missing run', a different reason from the >50% censoring rule):")
for r in inp[inp.subject == "sub-12"].itertuples(index=False):
    print(f"      {r.session}  bold={r.bold}  mask={r.mask}  confounds={r.confounds}")
print()

# ═══ 3. WHAT ALREADY EXISTS ═══════════════════════════════════════════════════
print("=" * 78)
print("3. EXISTING OUTPUTS")
print("=" * 78)

print("\n  Denoised NIfTIs in 01_denoising/results/:")
if not DENOISING_DIR.is_dir():
    raise FileNotFoundError(f"Denoising results dir not found: {DENOISING_DIR}")
pipeline_dirs = sorted(d for d in DENOISING_DIR.iterdir() if d.is_dir())
if not pipeline_dirs:
    print("      (no pipeline subdirectories at all)")
for d in pipeline_dirs:
    n = len(list(d.glob("*_bold.nii.gz")))
    print(f"      {d.name:<20} {n:>4} NIfTI(s)")

cand_dir = DENOISING_DIR / CANDIDATE
cand_present = []
if cand_dir.is_dir():
    for sub in ALL_SUBJECTS:
        for ses in SESSIONS:
            if build_output_path(sub, ses, cand_dir, PIPELINE_PRESETS[CANDIDATE]["desc"]).is_file():
                cand_present.append((sub, ses))
print(f"\n  {CANDIDATE} denoised runs present: {len(cand_present)}")

print("\n  Extracted timeseries (per atlas, per pipeline):")
ts_rows = []
for atlas in ATLASES:
    adir = TS_ROOT / atlas
    if not adir.is_dir():
        print(f"      {atlas:<12} (directory absent)")
        continue
    subdirs = sorted(d for d in adir.iterdir() if d.is_dir())
    if not subdirs:
        print(f"      {atlas:<12} (no pipeline subdirectories)")
    for d in subdirs:
        n = len(list(d.rglob("*_timeseries.csv")))
        print(f"      {atlas:<12} {d.name:<20} {n:>5} CSV(s)")
        ts_rows.append({"atlas": atlas, "pipeline": d.name, "n_csv": n})

# ═══ 4. WHAT STAGE 1 STILL NEEDS ══════════════════════════════════════════════
print()
print("=" * 78)
print("4. WHAT STAGE 1 STILL NEEDS")
print("=" * 78)

n_expected_runs = len(usable_subjects) * len(SESSIONS)
todo = []
if len(cand_present) < n_expected_runs:
    todo.append(
        f"DENOISE: {CANDIDATE} has {len(cand_present)} of {n_expected_runs} runs "
        f"-> 01_denoise_all.py --pipeline {CANDIDATE}"
    )
have_sphere = any(r["atlas"] == "qinspheres" and r["pipeline"] == CANDIDATE for r in ts_rows)
have_gl360 = any(r["atlas"] == "glasser360" and r["pipeline"] == CANDIDATE for r in ts_rows)
have_parcel = any(r["atlas"] == "qinparcels" and r["pipeline"] == CANDIDATE for r in ts_rows)
if not have_sphere:
    todo.append(f"EXTRACT qinspheres for {CANDIDATE}")
if not have_gl360:
    todo.append(f"EXTRACT glasser360 for {CANDIDATE} (prerequisite for qinparcels)")
if not have_parcel:
    todo.append(f"BUILD qinparcels for {CANDIDATE} (subsets glasser360)")

if todo:
    for i, t in enumerate(todo, 1):
        print(f"  {i}. {t}")
else:
    print("  Nothing — all maximal_nocensor outputs already exist.")

inv_path = OUT_DIR / "preflight_inventory.csv"
inp.to_csv(inv_path, index=False)
print(f"\n  Input inventory saved: {inv_path}")
print(f"  Report saved         : {REPORT_PATH}")
print("\n  READ-ONLY: nothing was denoised, extracted or modified.")

sys.stdout.flush()
