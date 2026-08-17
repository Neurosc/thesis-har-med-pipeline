#!/usr/bin/env python3
"""
Build the tidy whole-cortex Glasser AUC dataframe (input for the global whole-cortex test).

Whole-cortex = all 360 Glasser parcels treated as ONE cortex. The per-network (12
Cole-Anticevic) breakdown and the G1 gradient were both dropped from the project, so this
df carries neither a `network` nor a `G1` column — and needs neither the CAB-NP LabelKey
nor g1_parcellated.pscalar.nii.

Columns: parcel_id, auc, subject, session(Pre/Post), arm, pipeline.

In : 03_intrinsic_neural_metrics/results/acw/parcels/all_parcels/{pipeline}/glasser360_auc_{pipeline}.csv
Out: 04_statistics/results/acw/parcels/all_parcels/tables/glasser360_auc_tidy.csv
Run from repo root: python 04_statistics/scripts/all_parcels/01_build_tidy.py
"""
from pathlib import Path
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
AUC_BASE = REPO / "03_intrinsic_neural_metrics" / "results" / "acw" / "parcels" / "all_parcels"
PARTS    = REPO / "participants.tsv"
TAB      = REPO / "04_statistics" / "results" / "acw" / "parcels" / "all_parcels" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
PIPELINES = ["detrend", "maximal"]

arm = dict(zip(*[pd.read_csv(PARTS, sep="\t")[c] for c in ("participant_id", "condition")]))

frames = []
for pl in PIPELINES:
    src = AUC_BASE / pl / f"glasser360_auc_{pl}.csv"
    if not src.exists():
        print(f"[SKIP] {src} not found")
        continue
    a = pd.read_csv(src)
    a["pipeline"] = pl
    a["arm"]      = a["subject"].map(arm)
    a["session"]  = a["session"].map({"ses-01": "Pre", "ses-02": "Post"})
    frames.append(a)
df = pd.concat(frames, ignore_index=True)[
    ["parcel_id", "auc", "subject", "session", "arm", "pipeline"]]
out = TAB / "glasser360_auc_tidy.csv"
df.to_csv(out, index=False)
print(f"tidy df: {len(df)} rows ({df['pipeline'].nunique()} pipelines) -> {out}")
