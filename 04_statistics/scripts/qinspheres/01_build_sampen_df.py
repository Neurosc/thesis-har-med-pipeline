#!/usr/bin/env python3
"""
Build the qinspheres SampEn dataframe (parallel to 01_build_df.jl for ACW).

Reads per-run SampEn CSVs (roi_id, sampen) for the 6 sphere layers × 3 denoising pipelines
and writes one flat CSV per pipeline in the SAME format the qinspheres R stats scripts expect
(columns: subject, session, drug_group, category, roi_id, auc) — the value column is named
`auc` (carryover) so the metric-parameterized scripts read it generically; here it holds SampEn.

Sample = n=39 (all sub dirs present minus sub-12), matching the ACW stats.

In : 03_intrinsic_neural_metrics/results/sampen/{pipeline}/{layer}/{sub}_{ses}_{layer}_sampen.csv
Out: 04_statistics/results/qinspheres/sampen/{pipeline}/tables/qinspheres_sampen.csv
Run from repo root: python 04_statistics/scripts/qinspheres/01_build_sampen_df.py
"""
import re
import sys
from pathlib import Path
import pandas as pd

REPO    = Path(__file__).resolve().parents[3]
ATLAS   = sys.argv[1] if len(sys.argv) > 1 else "qinspheres"   # qinspheres | qinparcels
SAMPEN  = REPO / "03_intrinsic_neural_metrics" / "results" / ("sampen" if ATLAS == "qinspheres" else "sampen_parcels")
PARTS   = REPO / "participants.tsv"
OUTBASE = REPO / "04_statistics" / "results" / ATLAS / "sampen"
PIPELINES = ["detrend", "glm", "maximal"]
LAYERS    = ["intero", "extero", "mental", "visual", "motor", "auditory"]

arm = dict(zip(*[pd.read_csv(PARTS, sep="\t")[c] for c in ("participant_id", "condition")]))
FN  = re.compile(r"^(sub-\d+)_(ses-\d+)_(\w+)_sampen\.csv$")

for pl in PIPELINES:
    rows = []
    for layer in LAYERS:
        ld = SAMPEN / pl / layer
        for f in sorted(ld.glob(f"*_{layer}_sampen.csv")):
            m = FN.match(f.name)
            sub, ses = m.group(1), m.group(2)
            d = pd.read_csv(f)
            for _, r in d.iterrows():
                rows.append(dict(subject=sub, session=ses, drug_group=arm.get(sub, "unknown"),
                                 category=layer, roi_id=int(r["roi_id"]), auc=float(r["sampen"])))
    out = OUTBASE / pl / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "qinspheres_sampen.csv", index=False)
    print(f"{pl:8s}  {len(df):6d} rows  "
          f"({df['subject'].nunique()} subj, {df['category'].nunique()} layers, "
          f"finite={df['auc'].apply(lambda x: x == x and abs(x) != float('inf')).sum()})")
print("Done.")
