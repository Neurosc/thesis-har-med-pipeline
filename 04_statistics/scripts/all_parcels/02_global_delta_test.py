#!/usr/bin/env python3
"""
Whole-cortex drug-effect test: does verum blunt the retreat-induced cortical timescale increase?

Per subject x session: mean value over all 360 Glasser parcels (AUC > 0 for the AUC metric),
then delta = Post - Pre. Drug effect = mean(verum delta) - mean(placebo delta) (negative =
verum attenuates). 10k subject-label permutation (seed 42), two-tailed.

Whole-cortex ONLY — the per-network (12 Cole-Anticevic) breakdown was dropped from the
project, along with the G1 gradient that used to order it.

Metric-parameterized: pass `auc` (default) or `sampen` as argv[1]. AUC keeps the unsuffixed
filename; SampEn writes the *_sampen variant. Pipelines are taken from whatever is present
in the tidy df (SampEn is maximal-only by default).

In : 04_statistics/results/{acw|sampen}/parcels/all_parcels/tables/glasser360_{metric}_tidy.csv
Out: .../tables/global_delta_test{_metric}.csv
Run from repo root: python 04_statistics/scripts/all_parcels/02_global_delta_test.py [auc|sampen]
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

METRIC = sys.argv[1] if len(sys.argv) > 1 else "auc"
assert METRIC in ("auc", "sampen"), f"metric must be auc|sampen, got {METRIC!r}"
MSUF = "" if METRIC == "auc" else f"_{METRIC}"
MLAB = "SampEn" if METRIC == "sampen" else "AUC"

REPO = Path(__file__).resolve().parents[3]
TAB  = (REPO / "04_statistics" / "results" / ("acw" if METRIC == "auc" else METRIC)
        / "parcels" / "all_parcels" / "tables")
TAB.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(TAB / f"glasser360_{METRIC}_tidy.csv").rename(columns={METRIC: "value"})

# EntropyHub returns +/-inf for parcels with no template matches; drop non-finite parcels
# before any averaging, or a single inf poisons the whole subject mean.
_n0 = len(df)
df = df[np.isfinite(df["value"])]
if len(df) < _n0:
    print(f"[note] dropped {_n0 - len(df)} non-finite {METRIC} parcel values")

PIPELINES = sorted(df["pipeline"].unique())
NPERM = 10000


def delta_per_subject(d):
    """d has columns subject, session(Pre/Post), arm, value -> one delta per subject."""
    if METRIC == "auc":
        d = d[d["value"] > 0]      # AUC <= 0 = collapsed autocorrelation; SampEn has no such filter
    m = (d.groupby(["subject", "arm", "session"])["value"].mean()
         .unstack("session").dropna(subset=["Pre", "Post"]))
    m["delta"] = m["Post"] - m["Pre"]
    return m.reset_index()


def perm_test(v, p, nperm=NPERM, seed=42):
    v, p = np.asarray(v, float), np.asarray(p, float)
    obs = v.mean() - p.mean()
    alld = np.concatenate([v, p]); nv = len(v)
    rng = np.random.default_rng(seed)
    null = np.empty(nperm)
    for i in range(nperm):
        idx = rng.permutation(alld.size)
        null[i] = alld[idx[:nv]].mean() - alld[idx[nv:]].mean()
    return obs, (np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1)


rows = []
for pl in PIPELINES:
    g = delta_per_subject(df[df["pipeline"] == pl])
    v = g.loc[g["arm"] == "verum", "delta"]; p = g.loc[g["arm"] == "placebo", "delta"]
    diff, pv = perm_test(v, p)
    rows.append(dict(pipeline=pl, n_verum=len(v), n_placebo=len(p),
                     verum_mean_delta=v.mean(), placebo_mean_delta=p.mean(),
                     effect_verum_minus_placebo=diff, perm_p=pv))

out = pd.DataFrame(rows)
out.to_csv(TAB / f"global_delta_test{MSUF}.csv", index=False)
print(f"=== WHOLE-CORTEX d{MLAB} (verum vs placebo) ===")
print(out.round(4).to_string(index=False))
print(f"\nsaved -> {TAB / f'global_delta_test{MSUF}.csv'}")
