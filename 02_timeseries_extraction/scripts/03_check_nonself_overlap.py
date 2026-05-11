#!/usr/bin/env python3
# 03_check_nonself_overlap.py
#
# Purpose: Check which nonself Glasser ROIs spatially overlap with the self atlas
#          and save a clean list of non-overlapping ROIs.
#
# Inputs:  02_timeseries_extraction/results/atlases/self_atlas_1mm.nii.gz
#          glasser_coordinates_nonself_327_original.txt (server path)
# Outputs: 02_timeseries_extraction/results/atlases/glasser_coordinates_nonself_clean_1mm.txt
#          02_timeseries_extraction/results/atlases/glasser_overlap_report.txt
#
# Run (after 01_create_self_atlas.sh):
#   conda activate fmri
#   python 02_timeseries_extraction/scripts/03_check_nonself_overlap.py

import sys
import numpy as np
import nibabel as nib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ATLAS_DIR = REPO_ROOT / "02_timeseries_extraction" / "results" / "atlases"

self_atlas_path = ATLAS_DIR / "self_atlas_1mm.nii.gz"
nonself_coords  = "/home/jkokino/meditation_project/templates/nonself_roi/glasser_coordinates_nonself_327_original.txt"
out_clean       = ATLAS_DIR / "glasser_coordinates_nonself_clean_1mm.txt"
out_report      = ATLAS_DIR / "glasser_overlap_report.txt"

RADIUS_MM = 4.0

# ── Load self atlas ────────────────────────────────────────────────────────────
img       = nib.load(str(self_atlas_path))
self_mask = (img.get_fdata() > 0).astype(np.int8)
affine    = img.affine
vox_size  = img.header.get_zooms()[0]
shape     = self_mask.shape
print(f"Self atlas loaded — {self_mask.sum()} voxels")

# ── Helpers ────────────────────────────────────────────────────────────────────
def mni_to_vox(x, y, z):
    v = np.linalg.inv(affine) @ [x, y, z, 1]
    return tuple(np.round(v[:3]).astype(int))

def make_sphere(center, radius_mm):
    cx, cy, cz = center
    r = int(np.ceil(radius_mm / vox_size)) + 1
    sphere = np.zeros(shape, dtype=np.int8)
    for dx in range(-r, r + 1):
        for dy in range(-r, r + 1):
            for dz in range(-r, r + 1):
                if dx**2 + dy**2 + dz**2 <= (radius_mm / vox_size)**2:
                    nx, ny, nz = cx + dx, cy + dy, cz + dz
                    if 0 <= nx < shape[0] and 0 <= ny < shape[1] and 0 <= nz < shape[2]:
                        sphere[nx, ny, nz] = 1
    return sphere

# ── Read coordinates ───────────────────────────────────────────────────────────
rois = []
with open(nonself_coords) as f:
    next(f)  # skip header
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) < 5:
            continue
        rois.append((int(parts[0]), parts[1],
                     float(parts[2]), float(parts[3]), float(parts[4])))
print(f"Loaded {len(rois)} nonself ROIs")

# ── Check overlap ──────────────────────────────────────────────────────────────
clean, removed = [], []

for roi_num, name, x, y, z in rois:
    vox = mni_to_vox(x, y, z)
    if not all(0 <= vox[i] < shape[i] for i in range(3)):
        print(f"  WARNING: ROI {roi_num} {name} outside image bounds, skipping")
        continue
    overlap = int((make_sphere(vox, RADIUS_MM) * self_mask).sum())
    if overlap > 0:
        removed.append((roi_num, name, x, y, z, overlap))
    else:
        clean.append((roi_num, name, x, y, z))

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\nTotal checked : {len(rois)}")
print(f"Removed       : {len(removed)}")
print(f"Clean         : {len(clean)}")

if removed:
    print("\nRemoved ROIs:")
    for roi_num, name, x, y, z, n in removed:
        print(f"  {roi_num:03d}  {name:<35}  {n} voxels overlap")

# ── Save outputs ───────────────────────────────────────────────────────────────
with open(str(out_clean), 'w') as f:
    f.write("ROI_Number\tROI_Name\tX\tY\tZ\n")
    for roi_num, name, x, y, z in clean:
        f.write(f"{roi_num}\t{name}\t{x:.4f}\t{y:.4f}\t{z:.4f}\n")

with open(str(out_report), 'w') as f:
    f.write(f"Spatial overlap report\n")
    f.write(f"Self atlas : {self_atlas_path}\n")
    f.write(f"Radius     : {RADIUS_MM}mm\n")
    f.write(f"Checked    : {len(rois)}\n")
    f.write(f"Removed    : {len(removed)}\n")
    f.write(f"Clean      : {len(clean)}\n\n")
    f.write("REMOVED:\n")
    for roi_num, name, x, y, z, n in removed:
        f.write(f"{roi_num}\t{name}\t{x:.2f}\t{y:.2f}\t{z:.2f}\t{n} voxels\n")
    f.write("\nKEPT:\n")
    for roi_num, name, x, y, z in clean:
        f.write(f"{roi_num}\t{name}\t{x:.2f}\t{y:.2f}\t{z:.2f}\n")

print(f"\nSaved: {out_clean}")
print(f"Saved: {out_report}")
