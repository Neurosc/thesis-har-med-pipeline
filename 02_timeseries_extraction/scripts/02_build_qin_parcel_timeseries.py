#!/usr/bin/env python3
"""
Build the Glasser-PARCEL version of the 6 Qin regions (parallel to the 4mm spheres).

Each region becomes whole Glasser parcels (mean over all parcel voxels) rather than 4mm
spheres. Self layers use the Keskin focus-point -> Glasser-parcel mapping
(glasser_self_metadata.tsv); nonself layers use the CA networks (Visual1+Visual2 / Somatomotor
/ Auditory) minus the self parcels — the SAME parcel selection as the sphere extractor's nonself.

Source: the already-extracted whole-cortex Glasser-360 parcel-mean timeseries
(02_timeseries_extraction/results/glasser360/, n=35) — we just SUBSET each run to the 6
regions' parcel columns. No re-extraction.

Out: 02_timeseries_extraction/results/qinparcels/{pipeline}/{layer}/{sub}_{ses}_{layer}_timeseries.csv
     (rows=Timepoint incl. dummies, cols=parcel INDEX) — same format as the sphere CSVs, so the
     ACW (01_compute_acw.jl) and SampEn (02_compute_sampen.py) scripts read it unchanged.

Run from repo root: python 02_timeseries_extraction/scripts/02_build_qin_parcel_timeseries.py
"""
import re
from pathlib import Path
import pandas as pd

REPO     = Path(__file__).resolve().parents[2]
GL360    = REPO / "02_timeseries_extraction" / "results" / "glasser360"
SELFMETA = REPO / "02_timeseries_extraction" / "atlases" / "glasser_self_metadata.tsv"
KEY      = REPO / "02_timeseries_extraction" / "atlases" / \
           "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"
OUT      = REPO / "02_timeseries_extraction" / "results" / "qinparcels"
PIPELINES = ["detrend", "glm", "maximal"]

# ── self layers from Keskin parcel mapping (a parcel may belong to 2 layers) ──
sm = pd.read_csv(SELFMETA, sep="\t")
CATMAP = {"Interoceptive": "intero", "Exteroceptive": "extero", "Mental Self": "mental"}
LAYERS = {"intero": [], "extero": [], "mental": [], "visual": [], "motor": [], "auditory": []}
for _, r in sm.iterrows():
    for c in str(r["categories"]).split(","):
        c = c.strip()
        if c in CATMAP:
            LAYERS[CATMAP[c]].append(int(r["parcel_id"]))
self_ids = set(int(p) for p in sm["parcel_id"])

# ── nonself layers from CA networks minus self (same as sphere extractor) ──
key  = pd.read_csv(KEY, sep="\t")
cort = key[(key["INDEX"] <= 360) & (key["GLASSERLABELNAME"] != "NA")]
NETMAP = {"visual": {"Visual1", "Visual2"}, "motor": {"Somatomotor"}, "auditory": {"Auditory"}}
for layer, nets in NETMAP.items():
    LAYERS[layer] = [int(r["INDEX"]) for _, r in cort.iterrows()
                     if r["NETWORK"] in nets and int(r["INDEX"]) not in self_ids]

print("Parcels per layer:")
for l in ["intero", "extero", "mental", "visual", "motor", "auditory"]:
    print(f"  {l:9s}: {len(LAYERS[l])}  {sorted(LAYERS[l])}")

FN = re.compile(r"^(sub-\d+)_(ses-\d+)_glasser360_timeseries\.csv$")
n_written = 0
for pl in PIPELINES:
    files = sorted((GL360 / pl).glob("*_glasser360_timeseries.csv"))
    for f in files:
        m = FN.match(f.name); sub, ses = m.group(1), m.group(2)
        df = pd.read_csv(f, index_col=0)            # Timepoint index, cols "1".."360"
        for layer, ids in LAYERS.items():
            cols = [str(i) for i in ids if str(i) in df.columns]
            od = OUT / pl / layer
            od.mkdir(parents=True, exist_ok=True)
            df[cols].to_csv(od / f"{sub}_{ses}_{layer}_timeseries.csv")
            n_written += 1
    print(f"{pl}: {len(files)} runs -> {len(files)*6} layer CSVs")
print(f"Done. {n_written} CSVs written.")
