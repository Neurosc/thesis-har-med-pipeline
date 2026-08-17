#!/usr/bin/env python3
"""
Build the tidy whole-cortex Glasser SampEn dataframe — the SampEn twin of 01_build_tidy.py.

Same whole-cortex treatment (all 360 parcels as one cortex), same participants join, same
Pre/Post naming. No `network` and no `G1` column: the per-network breakdown and the gradient
were both dropped from the project.

Columns: parcel_id, sampen, subject, session(Pre/Post), arm, pipeline.

In : 03_intrinsic_neural_metrics/results/sampen/parcels/all_parcels/{pipeline}/glasser360_sampen_{pipeline}.csv
Out: 04_statistics/results/sampen/parcels/all_parcels/tables/glasser360_sampen_tidy.csv
Run from repo root: python 04_statistics/scripts/all_parcels/01b_build_tidy_sampen.py
Pipelines default to whatever SampEn was computed for (maximal); override with env PIPELINES.
"""
import os
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
SE_BASE = REPO / "03_intrinsic_neural_metrics" / "results" / "sampen" / "parcels" / "all_parcels"
PARTS   = REPO / "participants.tsv"
TAB     = REPO / "04_statistics" / "results" / "sampen" / "parcels" / "all_parcels" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
PIPELINES = os.environ.get("PIPELINES", "maximal").split(",")

arm = dict(zip(*[pd.read_csv(PARTS, sep="\t")[c] for c in ("participant_id", "condition")]))

frames = []
for pl in PIPELINES:
    src = SE_BASE / pl / f"glasser360_sampen_{pl}.csv"
    if not src.is_file():
        print(f"[SKIP] {src} not found — run sampen/03_parcels_all_parcels.py first")
        continue
    d = pd.read_csv(src)
    d["pipeline"] = pl
    d["arm"]      = d["subject"].map(arm)
    d["session"]  = d["session"].map({"ses-01": "Pre", "ses-02": "Post"})
    frames.append(d)

if not frames:
    raise SystemExit("no SampEn inputs found")

df = pd.concat(frames, ignore_index=True)[
    ["parcel_id", "sampen", "subject", "session", "arm", "pipeline"]]
out = TAB / "glasser360_sampen_tidy.csv"
df.to_csv(out, index=False)
print(f"tidy df: {len(df)} rows, {df['subject'].nunique()} subjects -> {out}")
