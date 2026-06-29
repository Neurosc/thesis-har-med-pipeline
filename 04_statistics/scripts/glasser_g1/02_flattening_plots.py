#!/usr/bin/env python3
"""
G1 flattening plots — whole-cortex Glasser AUC vs the Margulies principal gradient.

Builds the tidy dataframe (parcel, network, G1, AUC, subject, session[Pre/Post], arm,
pipeline) and the two per-arm flattening plots: parcels as points, x = G1, y = group-mean
AUC per parcel, with OLS fit lines for Pre and Post and the slopes annotated (the Pre->Post
slope change is the flattening readout).

Inputs:
  03_intrinsic_neural_metrics/results/glasser360_auc/{pipeline}/glasser360_auc_{pipeline}.csv
  02_timeseries_extraction/atlases/CortexSubcortex_..._LabelKey.txt   (NETWORK + GLASSERLABELNAME per INDEX)
  _archive/reference_data/g1_parcellated.pscalar.nii                  (G1 = row 0; name-mapped, ordering-safe)
  participants.tsv                                                    (arm = condition)

Outputs (04_statistics/results/glasser_g1/):
  tables/glasser_g1_auc_tidy.csv
  tables/glasser_g1_slopes.csv
  figures/glasser_g1_flattening_{pipeline}_{placebo,verum}.png

Primary pipeline = maximal (consistent with the qinspheres results); detrend/glm generated too.
Run from repo root:  python 04_statistics/scripts/glasser_g1/02_flattening_plots.py
"""

from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

REPO = Path(__file__).resolve().parents[3]
AUC_BASE = REPO / "03_intrinsic_neural_metrics" / "results" / "glasser360_auc"
KEY      = REPO / "02_timeseries_extraction" / "atlases" / \
           "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"
PSCALAR  = REPO / "_archive" / "reference_data" / "g1_parcellated.pscalar.nii"
PARTS    = REPO / "participants.tsv"
OUT      = REPO / "04_statistics" / "results" / "glasser_g1"
FIG      = OUT / "figures"; TAB = OUT / "tables"
FIG.mkdir(parents=True, exist_ok=True); TAB.mkdir(parents=True, exist_ok=True)

PIPELINES = ["detrend", "glm", "maximal"]
# per-arm session colors (task spec)
COLORS = {("placebo", "Pre"): "#CD5C5C", ("placebo", "Post"): "#E8963E",
          ("verum",   "Pre"): "#4682B4", ("verum",   "Post"): "#2E8B8B"}

# ── parcel -> (network, G1) via INDEX -> GLASSERLABELNAME -> pscalar name (name-based, ordering-safe)
key = pd.read_csv(KEY, sep="\t")
cort = key[(key["INDEX"] <= 360) & (key["GLASSERLABELNAME"] != "NA")][
    ["INDEX", "NETWORK", "GLASSERLABELNAME"]].copy()
g1img = nib.load(str(PSCALAR))
pnames = list(g1img.header.get_axis(1).name)
g1row  = np.asarray(g1img.get_fdata())[0]            # row 0 = gradient 1
name2g1 = dict(zip(pnames, g1row))
cort["G1"] = cort["GLASSERLABELNAME"].map(name2g1)
assert len(cort) == 360 and cort["G1"].notna().all(), "G1 mapping incomplete (expected 360)"
idx2net = dict(zip(cort["INDEX"], cort["NETWORK"]))
idx2g1  = dict(zip(cort["INDEX"], cort["G1"]))

part = pd.read_csv(PARTS, sep="\t")
arm  = dict(zip(part["participant_id"], part["condition"]))

# ── tidy dataframe (all pipelines) ──────────────────────────────────────────
frames = []
for pl in PIPELINES:
    a = pd.read_csv(AUC_BASE / pl / f"glasser360_auc_{pl}.csv")
    a["pipeline"] = pl
    a["network"]  = a["parcel_id"].map(idx2net)
    a["G1"]       = a["parcel_id"].map(idx2g1)
    a["arm"]      = a["subject"].map(arm)
    a["session"]  = a["session"].map({"ses-01": "Pre", "ses-02": "Post"})
    frames.append(a)
df = pd.concat(frames, ignore_index=True)
df = df[["parcel_id", "network", "G1", "auc", "subject", "session", "arm", "pipeline"]]
df.to_csv(TAB / "glasser_g1_auc_tidy.csv", index=False)
print(f"tidy df: {len(df)} rows -> {TAB/'glasser_g1_auc_tidy.csv'}")

# ── flattening plots + slopes table ─────────────────────────────────────────
slopes = []

def plot_arm(dpl, arm_name, pipeline):
    d = dpl[(dpl["arm"] == arm_name) & (dpl["auc"] > 0)]      # drop non-positive AUC (invalid timescale)
    n_drop = ((dpl["arm"] == arm_name) & (dpl["auc"] <= 0)).sum()
    fig, ax = plt.subplots(figsize=(7, 6))
    notes = []
    for ses in ["Pre", "Post"]:
        g = (d[d["session"] == ses].groupby("parcel_id")
             .agg(G1=("G1", "first"), AUC=("auc", "mean")).dropna())
        x, y = g["G1"].values, g["AUC"].values
        c = COLORS[(arm_name, ses)]
        ax.scatter(x, y, s=14, alpha=0.6, color=c, edgecolors="none", label=ses)
        lr = linregress(x, y)
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, lr.intercept + lr.slope * xs, color=c, lw=2.2)
        notes.append(f"{ses}: slope = {lr.slope:+.3f}  (p={lr.pvalue:.3f})")
        slopes.append(dict(pipeline=pipeline, arm=arm_name, session=ses,
                           slope=lr.slope, intercept=lr.intercept, r=lr.rvalue,
                           p=lr.pvalue, n_parcels=len(g)))
    # Pre->Post slope change = flattening readout
    sp = {s["session"]: s["slope"] for s in slopes if s["pipeline"] == pipeline and s["arm"] == arm_name}
    dslope = sp["Post"] - sp["Pre"]
    notes.append(f"Δslope (Post−Pre) = {dslope:+.3f}")
    ax.set_xlabel("G1 (Margulies principal gradient)")
    ax.set_ylabel("AUC")
    ax.set_title(f"{arm_name.capitalize()} — {pipeline}"
                 + ("" if n_drop == 0 else f"  (dropped {n_drop} non-positive AUC)"))
    ax.legend(frameon=False, loc="upper right")
    ax.annotate("\n".join(notes), xy=(0.03, 0.97), xycoords="axes fraction", va="top",
                fontsize=10, family="monospace",
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    fig.tight_layout()
    out = FIG / f"glasser_g1_flattening_{pipeline}_{arm_name}.png"
    fig.savefig(out, dpi=200, facecolor="white")
    plt.close(fig)
    print(f"  {out.name}")

for pl in PIPELINES:
    print(f"plots [{pl}]:")
    dpl = df[df["pipeline"] == pl]
    plot_arm(dpl, "placebo", pl)
    plot_arm(dpl, "verum", pl)

pd.DataFrame(slopes).to_csv(TAB / "glasser_g1_slopes.csv", index=False)
print(f"\nslopes -> {TAB/'glasser_g1_slopes.csv'}")
print(pd.DataFrame(slopes).round(4).to_string(index=False))
