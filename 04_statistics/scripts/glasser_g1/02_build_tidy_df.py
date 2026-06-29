#!/usr/bin/env python3
"""
Build the tidy whole-cortex Glasser AUC dataframe (shared input for the network analyses).

(The earlier G1-slope "flattening" plots were removed — we no longer use the gradient slope;
the per-parcel G1 column is kept in the df only as optional metadata.)

Columns: parcel_id, network (12 CA networks), G1, auc, subject, session(Pre/Post), arm, pipeline.
G1 = row 0 of g1_parcellated.pscalar.nii, name-mapped (INDEX -> GLASSERLABELNAME -> pscalar name;
ordering-safe — glasser360 INDEX is CAB-NP/L-first vs pscalar Glasser/R-first).

In : 03_intrinsic_neural_metrics/results/glasser360_auc/{pipeline}/glasser360_auc_{pipeline}.csv
Out: 04_statistics/results/glasser_g1/tables/glasser_g1_auc_tidy.csv
Run from repo root: python 04_statistics/scripts/glasser_g1/02_build_tidy_df.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
import nibabel as nib

REPO = Path(__file__).resolve().parents[3]
AUC_BASE = REPO / "03_intrinsic_neural_metrics" / "results" / "glasser360_auc"
KEY      = REPO / "02_timeseries_extraction" / "atlases" / \
           "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"
PSCALAR  = REPO / "_archive" / "reference_data" / "g1_parcellated.pscalar.nii"
PARTS    = REPO / "participants.tsv"
TAB      = REPO / "04_statistics" / "results" / "glasser_g1" / "tables"
TAB.mkdir(parents=True, exist_ok=True)
PIPELINES = ["detrend", "glm", "maximal"]

key = pd.read_csv(KEY, sep="\t")
cort = key[(key["INDEX"] <= 360) & (key["GLASSERLABELNAME"] != "NA")][
    ["INDEX", "NETWORK", "GLASSERLABELNAME"]].copy()
g1img = nib.load(str(PSCALAR))
name2g1 = dict(zip(list(g1img.header.get_axis(1).name), np.asarray(g1img.get_fdata())[0]))
cort["G1"] = cort["GLASSERLABELNAME"].map(name2g1)
assert len(cort) == 360 and cort["G1"].notna().all(), "G1 mapping incomplete (expected 360)"
idx2net = dict(zip(cort["INDEX"], cort["NETWORK"]))
idx2g1  = dict(zip(cort["INDEX"], cort["G1"]))

arm = dict(zip(*[pd.read_csv(PARTS, sep="\t")[c] for c in ("participant_id", "condition")]))

frames = []
for pl in PIPELINES:
    a = pd.read_csv(AUC_BASE / pl / f"glasser360_auc_{pl}.csv")
    a["pipeline"] = pl
    a["network"]  = a["parcel_id"].map(idx2net)
    a["G1"]       = a["parcel_id"].map(idx2g1)
    a["arm"]      = a["subject"].map(arm)
    a["session"]  = a["session"].map({"ses-01": "Pre", "ses-02": "Post"})
    frames.append(a)
df = pd.concat(frames, ignore_index=True)[
    ["parcel_id", "network", "G1", "auc", "subject", "session", "arm", "pipeline"]]
df.to_csv(TAB / "glasser_g1_auc_tidy.csv", index=False)
print(f"tidy df: {len(df)} rows -> {TAB/'glasser_g1_auc_tidy.csv'}")
