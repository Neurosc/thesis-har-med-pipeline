#!/usr/bin/env python3
"""
Whole-cortex Result-1 extension: does verum blunt the retreat-induced cortical timescale
increase, and where (which CA networks)?

Per subject x session: mean AUC over parcels (auc>0), then dAUC = Post - Pre. Drug effect =
mean(verum dAUC) - mean(placebo dAUC) (negative = verum attenuates, matching Result 1),
10k subject-label permutation (seed 42), two-tailed. Run globally (all 360) and within each
of the 12 CA networks (BH-FDR across networks). Networks ordered by mean G1 (low=unimodal).

In : 04_statistics/results/glasser_g1/tables/glasser_g1_auc_tidy.csv
Out: tables/global_delta_test.csv, tables/network_delta_test.csv,
     figures/glasser_g1_network_map_{pipeline}.png
Run from repo root: python 04_statistics/scripts/glasser_g1/03_network_breakdown.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[3]
OUT  = REPO / "04_statistics" / "results" / "glasser_g1"
TAB, FIG = OUT / "tables", OUT / "figures"
df = pd.read_csv(TAB / "glasser_g1_auc_tidy.csv")
PIPELINES = ["detrend", "glm", "maximal"]
NPERM = 10000


def delta_per_subject(d):
    """d has columns subject, session(Pre/Post), arm, auc -> one dAUC per subject."""
    d = d[d["auc"] > 0]
    m = (d.groupby(["subject", "arm", "session"])["auc"].mean()
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
    pval = (np.sum(np.abs(null) >= abs(obs)) + 1) / (nperm + 1)
    return obs, pval


def bh_fdr(pvals):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p)
    q = p[order] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(q, 0, 1)
    return out


global_rows, net_rows = [], []
for pl in PIPELINES:
    dpl = df[df["pipeline"] == pl]

    # ── global (all 360) ──
    g = delta_per_subject(dpl)
    v = g.loc[g["arm"] == "verum", "delta"]; p = g.loc[g["arm"] == "placebo", "delta"]
    diff, pv = perm_test(v, p)
    global_rows.append(dict(pipeline=pl, n_verum=len(v), n_placebo=len(p),
                            verum_mean_delta=v.mean(), placebo_mean_delta=p.mean(),
                            effect_verum_minus_placebo=diff, perm_p=pv))

    # ── per network ──
    g1_by_net = dpl.groupby("network")["G1"].mean()
    nets = g1_by_net.sort_values().index.tolist()        # low G1 (unimodal) first
    pvs = []
    tmp = []
    for net in nets:
        gn = delta_per_subject(dpl[dpl["network"] == net])
        vn = gn.loc[gn["arm"] == "verum", "delta"]; pn = gn.loc[gn["arm"] == "placebo", "delta"]
        d, pp = perm_test(vn, pn)
        pvs.append(pp)
        tmp.append(dict(pipeline=pl, network=net, mean_G1=g1_by_net[net],
                        verum_mean_delta=vn.mean(), placebo_mean_delta=pn.mean(),
                        gap_placebo_minus_verum=pn.mean() - vn.mean(),
                        effect_verum_minus_placebo=d, perm_p=pp))
    qs = bh_fdr(pvs)
    for r, q in zip(tmp, qs):
        r["perm_q"] = q
        net_rows.append(r)

gdf = pd.DataFrame(global_rows); ndf = pd.DataFrame(net_rows)
gdf.to_csv(TAB / "global_delta_test.csv", index=False)
ndf.to_csv(TAB / "network_delta_test.csv", index=False)

print("=== GLOBAL cortical dAUC (verum vs placebo) ===")
print(gdf.round(4).to_string(index=False))
print("\n=== PER-NETWORK (maximal), ordered low->high G1 ===")
mx = ndf[ndf["pipeline"] == "maximal"].copy()
print(mx[["network", "mean_G1", "placebo_mean_delta", "verum_mean_delta",
          "gap_placebo_minus_verum", "perm_p", "perm_q"]].round(4).to_string(index=False))

# ── network-map figure (per pipeline): gap (placebo - verum) ordered by G1 ──
for pl in PIPELINES:
    m = ndf[ndf["pipeline"] == pl].sort_values("mean_G1")
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2E8B8B" if q < 0.05 else ("#86b6b6" if pv < 0.05 else "0.7")
              for q, pv in zip(m["perm_q"], m["perm_p"])]
    ax.barh(range(len(m)), m["gap_placebo_minus_verum"], color=colors)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels([f"{n}  (G1={g:+.1f})" for n, g in zip(m["network"], m["mean_G1"])])
    ax.axvline(0, color="k", lw=0.8)
    ax.invert_yaxis()  # lowest G1 (unimodal) at top
    ax.set_xlabel("placebo − verum  ΔAUC gap   (positive = placebo lengthens more)")
    ax.set_title(f"Retreat ΔAUC: drug gap by network — {pl}\n"
                 "ordered unimodal (top) → transmodal (bottom);  teal = FDR<.05, light = raw p<.05")
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    fig.tight_layout()
    fig.savefig(FIG / f"glasser_g1_network_map_{pl}.png", dpi=200, facecolor="white")
    plt.close(fig)
print("\nfigures: glasser_g1_network_map_{detrend,glm,maximal}.png")
