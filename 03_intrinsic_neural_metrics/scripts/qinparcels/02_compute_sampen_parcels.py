#!/usr/bin/env python3
"""
SampEn for the Glasser-PARCEL version of the 6 Qin regions — identical EntropyHub config
to 03_intrinsic_neural_metrics/scripts/02_compute_sampen.py
(m=1, tau=1, r=0.3 absolute, log base 2; drop 6 dummy; linear detrend each ROI).

In : 02_timeseries_extraction/results/qinparcels/{pipeline}/{layer}/{sub}_{ses}_{layer}_timeseries.csv
Out: 03_intrinsic_neural_metrics/results/sampen_parcels/{pipeline}/{layer}/{sub}_{ses}_{layer}_sampen.csv
     (roi_id, sampen). Subjects derived from files (n=35).
Run from repo root: python 03_intrinsic_neural_metrics/scripts/qinparcels/02_compute_sampen_parcels.py
"""
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import detrend
import EntropyHub as EH

REPO = Path(__file__).resolve().parents[3]
M, TAU, R, LOGBASE, DUMMY = 1, 1, 0.3, 2, 6
# Override with env PIPELINES="maximal_nocensor" (same convention as the sphere scripts).
PIPELINES = os.environ.get("PIPELINES", "detrend,maximal").split(",")
LAYERS    = ["intero", "extero", "mental", "visual", "motor", "auditory"]
SESSIONS  = ["ses-01", "ses-02"]
TS_BASE   = REPO / "02_timeseries_extraction" / "results" / "qinparcels"
OUT_BASE  = REPO / "03_intrinsic_neural_metrics" / "results" / "sampen_parcels"

FN   = re.compile(r"^(sub-\d+)_ses-01_intero_timeseries\.csv$")
SUBS = sorted({FN.match(f.name).group(1)
               for f in (TS_BASE / "detrend" / "intero").glob("*_ses-01_intero_timeseries.csv")})
print(f"Subjects: {len(SUBS)}")


def sampen_col(col):
    if not np.all(np.isfinite(col)):
        return np.nan
    try:
        Samp, _, _ = EH.SampEn(col, m=M, tau=TAU, r=R, Logx=LOGBASE)
        return float(Samp[M])
    except Exception:
        return np.nan


n = 0
for pl in PIPELINES:
    for layer in LAYERS:
        od = OUT_BASE / pl / layer
        od.mkdir(parents=True, exist_ok=True)
        for sub in SUBS:
            for ses in SESSIONS:
                csv = TS_BASE / pl / layer / f"{sub}_{ses}_{layer}_timeseries.csv"
                if not csv.is_file():
                    continue
                df = pd.read_csv(csv)
                roi_ids = [str(c) for c in df.columns[1:]]
                ts = df.iloc[:, 1:].to_numpy(dtype=float)[DUMMY:]
                ts = detrend(ts, axis=0)
                vals = [sampen_col(ts[:, i]) for i in range(ts.shape[1])]
                pd.DataFrame({"roi_id": roi_ids, "sampen": vals}).to_csv(
                    od / f"{sub}_{ses}_{layer}_sampen.csv", index=False)
                n += 1
    print(f"{pl} done")
print(f"Done. {n} CSVs.")
