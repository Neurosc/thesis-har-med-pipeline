#!/usr/bin/env python3
"""
06_parse_keskin_coordinates.py
Parse Self2Glasser.csv and produce a coordinate file for the Keskin
self-parcel sphere atlas.

Column mapping in Self2Glasser.csv (0-indexed after ignoring header row 0):
  col 0:  # (row counter within section)
  col 1:  Hemi
  col 2:  Label (anatomical name)
  col 3:  Brodmann Area
  col 4:  Volume (mm^3)
  col 5:  Peak Z Value
  col 6:  x (MNI)
  col 7:  y (MNI)
  col 8:  z (MNI)
  col 9:  Focus Point (Glasser parcel name(s))
  col 10: parcel_index (Glasser standard ID 1-360; empty = subcortical, skip)

Output:
  02_timeseries_extraction/results/atlases/keskin_self_coordinates.txt
  Tab-separated: ROI_Number  ROI_Name  X  Y  Z  Layer

Run from repo root:
  python 02_timeseries_extraction/scripts/06_parse_keskin_coordinates.py
"""

import sys
from pathlib import Path

REPO_ROOT   = Path(__file__).resolve().parents[2]
INPUT_CSV   = REPO_ROOT / "Self2Glasser.csv"
ATLAS_DIR   = REPO_ROOT / "02_timeseries_extraction" / "results" / "atlases"
OUTPUT_FILE = ATLAS_DIR / "keskin_self_coordinates.txt"

LAYER_MAP = {
    "Interoceptive-processing": "Interoception",
    "Exteroceptive-processing": "Exteroception",
    "Mental-self-processing":   "Cognition",
}

SEP = "=" * 70
print(SEP)
print("06_parse_keskin_coordinates.py")
print(SEP, "\n")

if not INPUT_CSV.exists():
    sys.exit(f"ERROR: {INPUT_CSV} not found.")

# ── Parse Self2Glasser.csv ─────────────────────────────────────────────────────
rows = []           # list of dicts: layer, glasser_id, roi_name, x, y, z
current_layer = None
n_subcortical = 0
n_parsed      = 0

with open(INPUT_CSV, "r", encoding="utf-8") as fh:
    for line_no, raw in enumerate(fh, start=1):
        line = raw.rstrip("\n").rstrip("\r")
        cols = line.split(",")

        # Skip first empty row and column-header row
        if line_no <= 2:
            continue

        # Section headers: first non-empty cell matches a known section name,
        # all remaining cells are empty
        first = cols[0].strip()
        if first in LAYER_MAP and all(c.strip() == "" for c in cols[1:]):
            current_layer = LAYER_MAP[first]
            continue

        # Skip completely empty rows
        if all(c.strip() == "" for c in cols):
            continue

        # Data row — needs at least 11 columns
        if len(cols) < 11:
            continue

        if current_layer is None:
            continue

        # Parcel index: column 10 (0-indexed)
        parcel_raw = cols[10].strip()
        if parcel_raw == "":
            n_subcortical += 1
            print(f"  [SKIP subcortical] {current_layer} row: "
                  f"'{cols[2].strip()}' (no Glasser parcel)")
            continue

        try:
            glasser_id = int(parcel_raw)
        except ValueError:
            print(f"  [SKIP bad index] {current_layer} row: '{parcel_raw}'")
            continue

        # Coordinates
        try:
            x = float(cols[6].strip().replace("−", "-").replace("−", "-"))
            y = float(cols[7].strip().replace("−", "-").replace("−", "-"))
            z = float(cols[8].strip().replace("−", "-").replace("−", "-"))
        except ValueError as e:
            print(f"  [SKIP bad coords] {current_layer} row: {e}")
            continue

        # ROI name: first token of Focus Point (col 9)
        focus = cols[9].strip()
        roi_name = focus.split()[0] if focus else f"Glasser_{glasser_id}"

        rows.append({
            "layer":      current_layer,
            "glasser_id": glasser_id,
            "roi_name":   roi_name,
            "x": x, "y": y, "z": z,
        })
        n_parsed += 1

print(f"\nParsed {n_parsed} data rows, skipped {n_subcortical} subcortical.\n")

# ── Report per layer ───────────────────────────────────────────────────────────
from collections import defaultdict, Counter

by_layer = defaultdict(list)
for r in rows:
    by_layer[r["layer"]].append(r["glasser_id"])

print("Parcels per layer (before deduplication):")
for lyr, ids in by_layer.items():
    print(f"  {lyr:<18}: {len(ids)} parcels  {ids}")

# ── Find duplicates ────────────────────────────────────────────────────────────
all_ids    = [r["glasser_id"] for r in rows]
id_counts  = Counter(all_ids)
duplicates = {gid: cnt for gid, cnt in id_counts.items() if cnt > 1}

if duplicates:
    print(f"\nGlasser IDs appearing in multiple layers ({len(duplicates)} found):")
    for gid, cnt in sorted(duplicates.items()):
        layers_with = [r["layer"] for r in rows if r["glasser_id"] == gid]
        print(f"  ID {gid:>3}: {', '.join(layers_with)}")
else:
    print("\nNo duplicate parcel IDs across layers.")

# ── Build unique-parcel coordinate list ───────────────────────────────────────
# For the ATLAS: use each Glasser parcel once (first occurrence for coordinates).
# For LABELING: keep all layer assignments (handled in downstream analysis).
seen      = {}    # glasser_id → first row
layer_map = defaultdict(list)  # glasser_id → [layers]

for r in rows:
    gid = r["glasser_id"]
    layer_map[gid].append(r["layer"])
    if gid not in seen:
        seen[gid] = r

unique_parcels = sorted(seen.values(), key=lambda r: r["glasser_id"])
print(f"\nTotal unique Glasser parcel IDs: {len(unique_parcels)}")

# ── Save coordinate file ───────────────────────────────────────────────────────
ATLAS_DIR.mkdir(parents=True, exist_ok=True)

with open(OUTPUT_FILE, "w") as fh:
    fh.write("ROI_Number\tROI_Name\tX\tY\tZ\tLayer\n")
    for r in unique_parcels:
        gid       = r["glasser_id"]
        all_lyrs  = " + ".join(sorted(set(layer_map[gid])))
        fh.write(f"{gid}\t{r['roi_name']}\t{r['x']:.4f}\t{r['y']:.4f}\t{r['z']:.4f}\t{all_lyrs}\n")

print(f"\nSaved: {OUTPUT_FILE}")
print(f"  ({len(unique_parcels)} unique parcels, tab-separated)")
print("\nSample rows:")
for r in unique_parcels[:5]:
    gid = r["glasser_id"]
    print(f"  {gid:>3}  {r['roi_name']:<30}  {r['x']:>7.2f}  {r['y']:>7.2f}  {r['z']:>7.2f}"
          f"  {' + '.join(set(layer_map[gid]))}")

print("\nDone.")
