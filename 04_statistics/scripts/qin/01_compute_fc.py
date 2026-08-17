# 01_compute_fc.py — region-to-region functional connectivity from Qin-sphere timeseries.
#
# DATA PROCESSING step (rerun per robustness cell: pass [pipeline] [atlas]).
# For each subject x session:
#   - load the 5 region timeseries CSVs (Motor excluded), drop the Timepoint column
#   - drop the first 6 rows (dummy volumes) -> 234 timepoints
#   - average across each region's sphere ROIs -> one mean timeseries per region
#   - stack into [234 x 5] in FIXED column order, correlate the 10 region pairs (Pearson r)
#   - Fisher-z transform: z = arctanh(r)
# Then join arm (participants.tsv) + session motion covariates (pcf, pcf_sq, mean_fd)
# and write ONE long table for the model step (02_model_fc.R).
#
# Usage : python 04_statistics/scripts/qin/01_compute_fc.py [pipeline] [atlas] [arm]
#         defaults: maximal qinspheres all   (arm: all | placebo | verum)
# Output: 04_statistics/results/fc/{spheres|parcels/self_regions}/{pipeline}/tables/fc_long[_{arm}].csv
#         (columns: subject, session, arm, pair, z, mean_fd, pcf, pcf_sq)

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

# --- settings ---------------------------------------------------------------
PIPELINE = sys.argv[1] if len(sys.argv) > 1 else "maximal"
ATLAS    = sys.argv[2] if len(sys.argv) > 2 else "qinspheres"
ARM      = sys.argv[3] if len(sys.argv) > 3 else "all"  # all | placebo | verum
N_DUMMY  = 6                                            # dummy volumes to drop (matches ACW/SampEn)
assert ARM in ("all", "placebo", "verum"), f"arm must be all|placebo|verum, got {ARM!r}"
SUFFIX   = "" if ARM == "all" else f"_{ARM}"

REPO = Path(__file__).resolve().parents[3]

# The 5 analysed regions, in FIXED column order (Motor excluded — degraded broadband AUC).
# (folder name, display label) — order is identical across every subject.
REGIONS = [("visual",   "Visual"),
           ("intero",   "Interoception"),
           ("extero",   "Exteroception"),
           ("mental",   "Cognition"),
           ("auditory", "Auditory")]
LABELS  = [lab for _, lab in REGIONS]
PAIRS   = list(combinations(LABELS, 2))                 # 10 pairs, order-stable

# n=35 (get_included_subjects: drop sub-06/08/12/26/36) — same sample as the AUC stats.
SUBJECTS = [f"sub-{i:02d}" for i in range(1, 41) if i not in (6, 8, 12, 26, 36)]
SESSIONS = ["ses-01", "ses-02"]

# CLI atlas args stay `qinspheres|qinparcels`; both 02_timeseries_extraction/results
# and 04_statistics/results use the same `spheres | parcels/self_regions` dir names.
_ATLAS_DIR = Path("spheres") if ATLAS == "qinspheres" else Path("parcels") / "self_regions"

TS_DIR   = REPO / "02_timeseries_extraction" / "results" / _ATLAS_DIR / PIPELINE
COV_CSV  = REPO / "99_QC" / "01_motion_qc" / "results" / "fd_covariates_long_thresh03.csv"
PARTS    = REPO / "participants.tsv"
OUT_DIR  = REPO / "04_statistics" / "results" / "fc" / _ATLAS_DIR / PIPELINE / "tables"


def region_mean_timeseries(folder, subject, session):
    """Load one region CSV, drop Timepoint + first 6 rows, average over sphere ROIs -> (234,)."""
    csv = TS_DIR / folder / f"{subject}_{session}_{folder}_timeseries.csv"
    ts = pd.read_csv(csv).drop(columns="Timepoint").to_numpy()[N_DUMMY:]
    return ts.mean(axis=1)


def fisher_z_pairs(subject, session):
    """[234 x 5] region matrix -> Fisher-z of the 10 pairwise correlations. None if any CSV missing."""
    try:
        mat = np.column_stack([region_mean_timeseries(f, subject, session) for f, _ in REGIONS])
    except FileNotFoundError:
        return None
    r = np.corrcoef(mat, rowvar=False)                  # 5x5 Pearson r, same column order as LABELS
    return {(LABELS[i], LABELS[j]): np.arctanh(r[i, j])
            for i, j in combinations(range(len(LABELS)), 2)}


# --- arm map (also used to restrict the sample when arm != all) -------------
arm_map = (pd.read_csv(PARTS, sep="\t")
             .set_index("participant_id")["condition"].to_dict())
if ARM != "all":
    SUBJECTS = [s for s in SUBJECTS if arm_map.get(s) == ARM]

# --- compute FC per subject x session --------------------------------------
records = []
missing = 0
for subject in SUBJECTS:
    for session in SESSIONS:
        z = fisher_z_pairs(subject, session)
        if z is None:
            missing += 1
            continue
        for a, b in PAIRS:
            records.append({"subject": subject, "session": session,
                            "pair": f"{a}__{b}", "z": z[(a, b)]})
fc = pd.DataFrame.from_records(records)

# --- join arm + session motion covariates ----------------------------------
fc["arm"] = fc["subject"].map(arm_map)
cov = (pd.read_csv(COV_CSV)
         .rename(columns={"mean_fd_retained": "mean_fd"})[["subject", "session", "pcf", "pcf_sq", "mean_fd"]])

fc = (fc.merge(cov, on=["subject", "session"], how="left")
        [["subject", "session", "arm", "pair", "z", "mean_fd", "pcf", "pcf_sq"]])

# --- write ------------------------------------------------------------------
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_csv = OUT_DIR / f"fc_long{SUFFIX}.csv"
fc.to_csv(out_csv, index=False)

print(f"[{ATLAS}/{PIPELINE}/{ARM}] FC long table: {len(fc)} rows "
      f"({fc.subject.nunique()} subjects x {len(SESSIONS)} sessions x {len(PAIRS)} pairs), "
      f"{missing} subject-sessions missing")
print(f"  pairs: {', '.join(p.replace('__', '-') for p in fc.pair.unique())}")
print(f"  -> {out_csv}")
