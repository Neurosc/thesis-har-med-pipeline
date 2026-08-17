# 02_model_fc.py — per-pair PAIRED t-test of the session (post-pre) effect on region FC.
#
# STATS step (unchanged across robustness cells). Reads the long FC table written by
# 01_compute_fc.py and, for each of the 10 region pairs, pairs the Fisher-z FC by
# subject across session and tests the within-subject change:
#
#       d = z_post - z_pre   (per subject) ;  one-sample / paired t-test on d
#
# No arm factor, no motion covariates, no random effect: motion is balanced pre/post
# (checked), so the pooled post-pre contrast needs no adjustment. BH-FDR across the 10
# pairs. Default sample = pooled n=35 (arg arm=all); pass placebo/verum for one group.
#
# Usage : python 04_statistics/scripts/qin/02_model_fc.py [pipeline] [atlas] [arm]
#         defaults: maximal qinspheres all   (arm: all | placebo | verum)
# Output: 04_statistics/results/fc/{spheres|parcels/self_regions}/{pipeline}/tables/
#           fc_session_effects[_{arm}].csv        (pair, mean_d, 95% CI, t, p, q)
#           fc_session_zchange_matrix[_{arm}].csv (5x5 mean post-pre d, symmetric)

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

# --- settings & paths -------------------------------------------------------
PIPELINE = sys.argv[1] if len(sys.argv) > 1 else "maximal"
ATLAS    = sys.argv[2] if len(sys.argv) > 2 else "qinspheres"
ARM      = sys.argv[3] if len(sys.argv) > 3 else "all"     # all | placebo | verum
assert ARM in ("all", "placebo", "verum"), f"arm must be all|placebo|verum, got {ARM!r}"
SUFFIX   = "" if ARM == "all" else f"_{ARM}"

REPO      = Path(__file__).resolve().parents[3]
_ATLAS_DIR = Path("spheres") if ATLAS == "qinspheres" else Path("parcels") / "self_regions"
TABLE_DIR = REPO / "04_statistics" / "results" / "fc" / _ATLAS_DIR / PIPELINE / "tables"

# Fixed region order (matches 01_compute_fc.py), for the 5x5 matrix.
REGIONS = ["Visual", "Interoception", "Exteroception", "Cognition", "Auditory"]


def bh_fdr(p):
    """Benjamini-Hochberg adjusted p-values (same as R p.adjust(method='BH'))."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]        # enforce monotonicity
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


# --- read the long table ----------------------------------------------------
dat = pd.read_csv(TABLE_DIR / f"fc_long{SUFFIX}.csv")
dat["time"] = np.where(dat["session"] == "ses-01", "pre", "post")


def test_pair(pair_name):
    """Pair z by subject across session, paired t-test on d = post - pre."""
    wide = (dat[dat["pair"] == pair_name]
            .pivot(index="subject", columns="time", values="z")
            .dropna(subset=["pre", "post"]))
    d = (wide["post"] - wide["pre"]).to_numpy()
    n = d.size
    mean_d = d.mean()
    se = d.std(ddof=1) / np.sqrt(n)
    crit = stats.t.ppf(0.975, n - 1)
    t, p = stats.ttest_rel(wide["post"], wide["pre"])
    a, b = pair_name.split("__")
    return {"pair": f"{a}-{b}", "region_a": a, "region_b": b,
            "mean_d": mean_d, "ci_low": mean_d - crit * se, "ci_high": mean_d + crit * se,
            "t": t, "p": p, "n": n, "df": n - 1}


results = pd.DataFrame([test_pair(pr) for pr in dat["pair"].unique()])
results["q"] = bh_fdr(results["p"])
results["survives"] = np.where(results["q"] < 0.05, "yes", "no")

# --- write the effects table (rounded, sorted by p) ------------------------
num = ["mean_d", "ci_low", "ci_high", "t", "p", "q"]
out = results.copy()
out[num] = out[num].round(4)
out = (out.sort_values("p")
          [["pair", "region_a", "region_b", "mean_d", "ci_low", "ci_high",
            "t", "p", "q", "survives", "df", "n"]])
TABLE_DIR.mkdir(parents=True, exist_ok=True)
out.to_csv(TABLE_DIR / f"fc_session_effects{SUFFIX}.csv", index=False)

# --- 5x5 mean post-pre d matrix (symmetric, diagonal NaN) ------------------
mat = pd.DataFrame(np.nan, index=REGIONS, columns=REGIONS)
for _, r in results.iterrows():
    mat.loc[r.region_a, r.region_b] = mat.loc[r.region_b, r.region_a] = round(r.mean_d, 4)
mat.to_csv(TABLE_DIR / f"fc_session_zchange_matrix{SUFFIX}.csv")

# --- report -----------------------------------------------------------------
scope = f"{ARM} within-group" if ARM != "all" else "pooled"
print(f"\n######## FC SESSION EFFECT (post-pre, paired t) - {ATLAS}/{PIPELINE}/{ARM} "
      f"(n={results['n'].iloc[0]}, {scope}) ########")
print("mean_d = post-pre change in Fisher-z FC; q = BH-FDR across 10 pairs\n")
print(out.to_string(index=False))
print("\n---- 5x5 mean session z-change matrix ----")
print(mat.to_string())
print(f"\nWrote: fc_session_effects{SUFFIX}.csv, fc_session_zchange_matrix{SUFFIX}.csv -> {TABLE_DIR}")
