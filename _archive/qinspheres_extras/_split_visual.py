"""
Split visual timeseries CSVs into vis1 and vis2 based on CAB-NP network.

Visual1 parcels (6):  primary visual cortex
Visual2 parcels (49): extrastriate / association visual cortex

Reads  : qinspheres/visual/sub-XX_ses-YY_visual_timeseries.csv
Writes : qinspheres/vis1/sub-XX_ses-YY_vis1_timeseries.csv
         qinspheres/vis2/sub-XX_ses-YY_vis2_timeseries.csv
"""

import csv
import os
from pathlib import Path

BASE   = Path(__file__).parent
VIS    = BASE / "visual"
VIS1   = BASE / "vis1"
VIS2   = BASE / "vis2"
LABEL_KEY = Path(__file__).parents[2] / "data" / \
            "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"

VIS1.mkdir(exist_ok=True)
VIS2.mkdir(exist_ok=True)

# Build Visual1 / Visual2 parcel ID sets from CAB-NP label key
vis1_ids = set()
vis2_ids = set()
with open(LABEL_KEY, newline="") as f:
    for row in csv.DictReader(f, delimiter="\t"):
        net = row["NETWORK"]
        idx = row["INDEX"]
        if net == "Visual1":
            vis1_ids.add(idx)
        elif net == "Visual2":
            vis2_ids.add(idx)

print(f"Visual1 parcel IDs ({len(vis1_ids)}): {sorted(vis1_ids, key=int)}")
print(f"Visual2 parcel IDs ({len(vis2_ids)}): {sorted(vis2_ids, key=int)}")

csv_files = sorted(VIS.glob("*_visual_timeseries.csv"))
print(f"\nProcessing {len(csv_files)} CSVs...")

ok = skipped = 0
for src in csv_files:
    stem  = src.stem.replace("_visual_timeseries", "")   # sub-XX_ses-YY
    dst1  = VIS1 / f"{stem}_vis1_timeseries.csv"
    dst2  = VIS2 / f"{stem}_vis2_timeseries.csv"

    with open(src, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows   = list(reader)

    # Identify column indices for each network (col 0 = Timepoint)
    idx1 = [0] + [i for i, h in enumerate(header) if h in vis1_ids]
    idx2 = [0] + [i for i, h in enumerate(header) if h in vis2_ids]

    # Verify all visual columns are accounted for
    all_data_cols = set(header[1:])
    covered = set(header[i] for i in idx1[1:]) | set(header[i] for i in idx2[1:])
    missing = all_data_cols - covered
    if missing:
        print(f"  WARNING {src.name}: {len(missing)} columns not assigned: {missing}")
        skipped += 1
        continue

    for dst, idx in [(dst1, idx1), (dst2, idx2)]:
        with open(dst, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([header[i] for i in idx])
            for row in rows:
                writer.writerow([row[i] for i in idx])

    ok += 1

print(f"\nDone: {ok} split, {skipped} skipped.")
print(f"vis1: {len(list(VIS1.glob('*.csv')))} files  ({len(vis1_ids)} cols each)")
print(f"vis2: {len(list(VIS2.glob('*.csv')))} files  ({len(vis2_ids)} cols each)")
