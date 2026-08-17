#!/usr/bin/env python3
"""
Build the SampEn dataframe (parallel to 01_build_auc.jl for ACW), for either atlas.

Reads per-run SampEn CSVs (roi_id, sampen) for the 6 layers × the denoising pipelines in
PIPELINES and writes one flat CSV per pipeline in the SAME format the R stats scripts expect
(columns: subject, session, drug_group, category, roi_id, auc) — the value column is named
`auc` (carryover) so the metric-parameterized scripts read it generically; here it holds SampEn.

Sample = n=35 (get_included_subjects: minus sub-06/08/12/26/36 — see EXCL below), matching
01_build_auc.jl. The upstream compute stage runs at n=39; the inclusion filter is applied here.

In : 03_intrinsic_neural_metrics/results/sampen/{spheres|parcels/self_regions}/{pipeline}/{layer}/{sub}_{ses}_{layer}_sampen.csv
Out: 04_statistics/results/sampen/{spheres|parcels/self_regions}/{pipeline}/tables/sampen.csv
Run from repo root: python 04_statistics/scripts/qin/02_build_sampen.py
"""
import re
import sys
from pathlib import Path
import pandas as pd

REPO    = Path(__file__).resolve().parents[3]
ATLAS   = sys.argv[1] if len(sys.argv) > 1 else "qinspheres"   # qinspheres | qinparcels
_SE_ROOT = REPO / "03_intrinsic_neural_metrics" / "results" / "sampen"
SAMPEN  = _SE_ROOT / "spheres" if ATLAS == "qinspheres" else _SE_ROOT / "parcels" / "self_regions"
PARTS   = REPO / "participants.tsv"
_ATLAS_DIR = Path("spheres") if ATLAS == "qinspheres" else Path("parcels") / "self_regions"
OUTBASE = REPO / "04_statistics" / "results" / "sampen" / _ATLAS_DIR
PIPELINES = ["detrend", "maximal"]
LAYERS    = ["intero", "extero", "mental", "visual", "motor", "auditory"]
EXCL      = {"sub-06", "sub-08", "sub-12", "sub-26", "sub-36"}   # get_included_subjects (n=35)

arm = dict(zip(*[pd.read_csv(PARTS, sep="\t")[c] for c in ("participant_id", "condition")]))
FN  = re.compile(r"^(sub-\d+)_(ses-\d+)_(\w+)_sampen\.csv$")

for pl in PIPELINES:
    rows = []
    for layer in LAYERS:
        ld = SAMPEN / pl / layer
        for f in sorted(ld.glob(f"*_{layer}_sampen.csv")):
            m = FN.match(f.name)
            sub, ses = m.group(1), m.group(2)
            if sub in EXCL:
                continue
            d = pd.read_csv(f)
            for _, r in d.iterrows():
                rows.append(dict(subject=sub, session=ses, drug_group=arm.get(sub, "unknown"),
                                 category=layer, roi_id=int(r["roi_id"]), auc=float(r["sampen"])))
    out = OUTBASE / pl / "tables"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "sampen.csv", index=False)
    print(f"{pl:8s}  {len(df):6d} rows  "
          f"({df['subject'].nunique()} subj, {df['category'].nunique()} layers, "
          f"finite={df['auc'].apply(lambda x: x == x and abs(x) != float('inf')).sum()})")
print("Done.")
