#!/usr/bin/env python3
"""
04_create_glasser_self_nonself.py
Build Glasser MMP1.0-based self and nonself atlases following Keskin et al. 2025.

Self parcels  = Glasser 1-360 parcels whose name appears in the Keskin self-
                referential table (focus_point column).
Nonself       = all remaining Glasser parcels (1-360).
Per-category  = one NIfTI per Keskin category (Interoceptive, Exteroceptive,
                Mental Self).  Parcels that span categories appear in all
                relevant per-category atlases.

Inputs  (02_timeseries_extraction/data/):
  glasser360MNI.nii.gz
  CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt
  keskin_self_coordinates.tsv
  sample_bold_for_resampling.nii.gz

Outputs (02_timeseries_extraction/results/atlases/):
  glasser_self_atlas_{1mm,native}.nii.gz
  glasser_self_{interoceptive,exteroceptive,mental}_{1mm,native}.nii.gz
  glasser_nonself_atlas_{1mm,native}.nii.gz
  glasser_self_metadata.tsv
  glasser_nonself_metadata.tsv
"""

import sys
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import image
from pathlib import Path
from collections import defaultdict

# ─── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR  = REPO_ROOT / "02_timeseries_extraction" / "data"
ATLAS_DIR = REPO_ROOT / "02_timeseries_extraction" / "results" / "atlases"

GLASSER_NIFTI     = DATA_DIR / "glasser360MNI.nii.gz"
GLASSER_LABEL_KEY = DATA_DIR / "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"
KESKIN_TABLE      = DATA_DIR / "keskin_self_coordinates.tsv"
SAMPLE_BOLD       = DATA_DIR / "sample_bold_for_resampling.nii.gz"

CATEGORY_DISPLAY = {
    "Interoceptive-processing": "Interoceptive",
    "Exteroceptive-processing": "Exteroceptive",
    "Mental-self-processing":   "Mental Self",
}
CATEGORY_SLUGS = {
    "Interoceptive-processing": "interoceptive",
    "Exteroceptive-processing": "exteroceptive",
    "Mental-self-processing":   "mental",
}

SEP = "─" * 60

# ─── Check inputs ──────────────────────────────────────────────────────────────
DATA_DIR.mkdir(parents=True, exist_ok=True)
ATLAS_DIR.mkdir(parents=True, exist_ok=True)

for p in (GLASSER_NIFTI, GLASSER_LABEL_KEY, KESKIN_TABLE):
    if not p.exists():
        sys.exit(
            f"ERROR: Missing input file:\n  {p}\n"
            f"Place it in {DATA_DIR}/ and rerun."
        )

if not SAMPLE_BOLD.exists():
    sys.exit(
        f"ERROR: {SAMPLE_BOLD} not found.\n"
        f"Copy any single denoised BOLD file (any subject/session) to:\n"
        f"  {SAMPLE_BOLD}\nthen rerun."
    )

print(SEP)
print("04_create_glasser_self_nonself.py")
print(SEP)

# ─── Step 1: Load Glasser parcel name → ID mapping ────────────────────────────
print("\n[Step 1] Loading Glasser label key...")

df_key = pd.read_csv(GLASSER_LABEL_KEY, sep="\t")

if "INDEX" not in df_key.columns or "GLASSERLABELNAME" not in df_key.columns:
    sys.exit(
        f"ERROR: Label key missing expected columns.\n"
        f"  Found:    {list(df_key.columns)}\n"
        f"  Expected: INDEX, GLASSERLABELNAME"
    )

parcel_name_to_id = {}  # stripped canonical name → Glasser ID (1-360)
id_to_name        = {}  # Glasser ID → stripped canonical name

for _, row in df_key.iterrows():
    idx = int(row["INDEX"])
    if 1 <= idx <= 360:
        raw  = str(row["GLASSERLABELNAME"]).strip()
        name = raw[:-4] if raw.endswith("_ROI") else raw
        parcel_name_to_id[name] = idx
        id_to_name[idx]         = name

print(f"  Loaded N={len(parcel_name_to_id)} Glasser parcels from label key")

# ─── Step 2: Parse Keskin self coordinate table ────────────────────────────────
print("\n[Step 2] Parsing Keskin self coordinate table...")

df_keskin = pd.read_csv(KESKIN_TABLE, sep="\t")

required_cols = {"category", "x", "y", "z", "focus_point"}
missing_cols  = required_cols - set(df_keskin.columns)
if missing_cols:
    sys.exit(
        f"ERROR: keskin_self_coordinates.tsv missing columns: {missing_cols}\n"
        f"  Found: {list(df_keskin.columns)}"
    )

n_rows_total            = len(df_keskin)
n_subcortical           = 0
all_parcel_names_tried  = 0
unmatched: list[dict]   = []

category_parcel_ids: dict[str, set[int]] = defaultdict(set)
parcel_categories:   dict[int, set[str]] = defaultdict(set)
parcel_coords:       dict[int, tuple]    = {}

for _, row in df_keskin.iterrows():
    category  = str(row["category"]).strip()
    focus_raw = row.get("focus_point", "")

    if pd.isna(focus_raw) or str(focus_raw).strip() == "":
        n_subcortical += 1
        continue

    tokens = str(focus_raw).strip().split()

    try:
        x, y, z = float(row["x"]), float(row["y"]), float(row["z"])
    except (ValueError, KeyError):
        x, y, z = float("nan"), float("nan"), float("nan")

    for token in tokens:
        name = token[:-4] if token.endswith("_ROI") else token
        all_parcel_names_tried += 1

        if name not in parcel_name_to_id:
            unmatched.append({
                "category":         category,
                "focus_point_name": name,
                "raw_token":        token,
            })
            continue

        pid = parcel_name_to_id[name]
        category_parcel_ids[category].add(pid)
        parcel_categories[pid].add(category)
        if pid not in parcel_coords:
            parcel_coords[pid] = (x, y, z)

self_parcel_ids = set()
for ids in category_parcel_ids.values():
    self_parcel_ids |= ids

print(f"  Keskin coordinates loaded: N={n_rows_total}")
print(f"  Subcortical entries dropped (no focus_point): N={n_subcortical}")
print(f"  Parcel names total (after multi-expand): N={all_parcel_names_tried}")
print(f"  Parcel names not found in Glasser label key: N={len(unmatched)}")
for u in unmatched:
    print(f"    WARNING: '{u['focus_point_name']}' (category={u['category']})")
print(f"  Unique self parcels resolved: N={len(self_parcel_ids)}")

if unmatched:
    unmatched_path = ATLAS_DIR / "keskin_unmatched_parcels.tsv"
    pd.DataFrame(unmatched).to_csv(unmatched_path, sep="\t", index=False)
    print(f"\nWARNING: {len(unmatched)} unmatched parcel name(s).")
    print(f"  See {unmatched_path.name} for details.")
    print("  Edit keskin_self_coordinates.tsv to fix and rerun.")
    sys.exit(1)

# ─── Helper: build masked integer NIfTI ───────────────────────────────────────
def build_atlas_nifti(
    glasser_data: np.ndarray,
    include_ids:  set,
    affine:       np.ndarray,
    header,
) -> nib.Nifti1Image:
    out  = np.zeros_like(glasser_data, dtype=np.int32)
    mask = np.isin(glasser_data, sorted(include_ids))
    out[mask] = glasser_data[mask]
    return nib.Nifti1Image(out, affine, header)

# ─── Step 3: Build all 1mm atlases ────────────────────────────────────────────
print("\n[Step 3] Building atlases at 1mm MNI resolution...")

glasser_img  = nib.load(GLASSER_NIFTI)
glasser_data = np.asarray(glasser_img.dataobj, dtype=np.int32)
affine       = glasser_img.affine
header       = glasser_img.header

# Combined self
self_img_1mm = build_atlas_nifti(glasser_data, self_parcel_ids, affine, header)
nib.save(self_img_1mm, ATLAS_DIR / "glasser_self_atlas_1mm.nii.gz")
print(f"  glasser_self_atlas_1mm.nii.gz  ({len(self_parcel_ids)} parcels)")

# Nonself
all_glasser_ids    = set(range(1, 361))
nonself_parcel_ids = all_glasser_ids - self_parcel_ids
nonself_img_1mm    = build_atlas_nifti(glasser_data, nonself_parcel_ids, affine, header)
nib.save(nonself_img_1mm, ATLAS_DIR / "glasser_nonself_atlas_1mm.nii.gz")
print(f"  glasser_nonself_atlas_1mm.nii.gz  ({len(nonself_parcel_ids)} parcels)")

# Per-category
cat_imgs_1mm: dict[str, nib.Nifti1Image] = {}
for cat, slug in CATEGORY_SLUGS.items():
    ids = category_parcel_ids.get(cat, set())
    img = build_atlas_nifti(glasser_data, ids, affine, header)
    fname = f"glasser_self_{slug}_1mm.nii.gz"
    nib.save(img, ATLAS_DIR / fname)
    cat_imgs_1mm[cat] = img
    print(f"  {fname}  ({len(ids)} parcels)")

# ─── Step 4 / Step 5: Resample all atlases to native BOLD grid ───────────────
print("\n[Step 4/5] Resampling to native BOLD grid...")

bold_ref_raw = nib.load(SAMPLE_BOLD)
bold_ref = image.index_img(bold_ref_raw, 0) if bold_ref_raw.ndim == 4 else bold_ref_raw
print(f"  BOLD reference shape: {bold_ref.shape}  voxel size: "
      f"{tuple(np.diag(bold_ref.affine)[:3].round(4))}")

def resample_nearest(src: nib.Nifti1Image, ref: nib.Nifti1Image) -> nib.Nifti1Image:
    out = image.resample_to_img(src, ref, interpolation="nearest")
    # Cast back to int32 to guarantee integer parcel IDs
    data = np.asarray(out.dataobj, dtype=np.int32)
    return nib.Nifti1Image(data, out.affine, out.header)

self_native    = resample_nearest(self_img_1mm, bold_ref)
nonself_native = resample_nearest(nonself_img_1mm, bold_ref)
nib.save(self_native,    ATLAS_DIR / "glasser_self_atlas_native.nii.gz")
nib.save(nonself_native, ATLAS_DIR / "glasser_nonself_atlas_native.nii.gz")
print(f"  glasser_self_atlas_native.nii.gz  → {self_native.shape}")
print(f"  glasser_nonself_atlas_native.nii.gz  → {nonself_native.shape}")

for cat, slug in CATEGORY_SLUGS.items():
    native = resample_nearest(cat_imgs_1mm[cat], bold_ref)
    fname  = f"glasser_self_{slug}_native.nii.gz"
    nib.save(native, ATLAS_DIR / fname)
    print(f"  {fname}  → {native.shape}")

# ─── Step 6: Save metadata TSVs ───────────────────────────────────────────────
print("\n[Step 6] Writing metadata TSVs...")

# Self metadata
self_rows = []
for pid in sorted(self_parcel_ids):
    name  = id_to_name.get(pid, f"Glasser_{pid}")
    hemi  = name[0] if name and name[0] in ("L", "R") else "?"
    cats  = sorted(parcel_categories[pid])
    cats_display = ",".join(CATEGORY_DISPLAY.get(c, c) for c in cats)
    cx, cy, cz   = parcel_coords.get(pid, (float("nan"), float("nan"), float("nan")))
    self_rows.append({
        "parcel_id":      pid,
        "parcel_name":    name,
        "hemisphere":     hemi,
        "categories":     cats_display,
        "keskin_coord_x": cx,
        "keskin_coord_y": cy,
        "keskin_coord_z": cz,
    })

pd.DataFrame(self_rows).to_csv(ATLAS_DIR / "glasser_self_metadata.tsv", sep="\t", index=False)
print(f"  glasser_self_metadata.tsv  ({len(self_rows)} rows)")

# Nonself metadata
nonself_rows = [
    {
        "parcel_id":   pid,
        "parcel_name": id_to_name.get(pid, f"Glasser_{pid}"),
        "hemisphere":  (id_to_name.get(pid, "?")[0]
                        if id_to_name.get(pid, "?")[0] in ("L", "R") else "?"),
    }
    for pid in sorted(nonself_parcel_ids)
]
pd.DataFrame(nonself_rows).to_csv(ATLAS_DIR / "glasser_nonself_metadata.tsv", sep="\t", index=False)
print(f"  glasser_nonself_metadata.tsv  ({len(nonself_rows)} rows)")

# ─── Step 7: Summary ──────────────────────────────────────────────────────────
overlapping = {pid: cats for pid, cats in parcel_categories.items() if len(cats) > 1}

print(f"\n{SEP}")
print("─── Glasser self/nonself atlas generation ───")
print(f"Total Glasser parcels: 360")
print(f"Self parcels:    {len(self_parcel_ids)}")
print(f"Nonself parcels: {len(nonself_parcel_ids)} (should = {360 - len(self_parcel_ids)})")

print("\nSelf breakdown by category:")
cat_sum = 0
for cat, slug in CATEGORY_SLUGS.items():
    n = len(category_parcel_ids.get(cat, set()))
    cat_sum += n
    print(f"  {CATEGORY_DISPLAY[cat]:<14}: {n} parcels")
overlap_note = " — overlap exists" if cat_sum > len(self_parcel_ids) else ""
print(f"  Combined unique:  {len(self_parcel_ids)} parcels "
      f"(sum={cat_sum}{overlap_note})")

if overlapping:
    print("\nCross-category overlaps:")
    for pid, cats in sorted(overlapping.items()):
        name = id_to_name.get(pid, f"Glasser_{pid}")
        disp = [CATEGORY_DISPLAY.get(c, c) for c in sorted(cats)]
        print(f"  {name} in [{', '.join(disp)}]")
else:
    print("\nCross-category overlaps: none")

print("\nPer-category atlas voxel counts (1mm):")
for cat, slug in CATEGORY_SLUGS.items():
    n_vox = int(np.sum(np.asarray(cat_imgs_1mm[cat].dataobj) > 0))
    print(f"  {CATEGORY_DISPLAY[cat]:<14}: {n_vox} voxels")
n_self_vox = int(np.sum(np.asarray(self_img_1mm.dataobj) > 0))
print(f"  Combined self : {n_self_vox} voxels")

print("\nFiles written:")
print("  glasser_self_atlas_1mm.nii.gz")
print("  glasser_self_atlas_native.nii.gz")
for slug in CATEGORY_SLUGS.values():
    print(f"  glasser_self_{slug}_1mm.nii.gz")
    print(f"  glasser_self_{slug}_native.nii.gz")
print("  glasser_nonself_atlas_1mm.nii.gz")
print("  glasser_nonself_atlas_native.nii.gz")
print("  glasser_self_metadata.tsv")
print("  glasser_nonself_metadata.tsv")
print(SEP)
