#!/usr/bin/env python3
"""
05_roi_placement_viewer.py — Interactive marker viewer + PDF montage for atlas QC

Outputs per atlas (self: 37 ROIs, nonself: ~316 ROIs):
  *_markers.html   — interactive 3-D viewer; hover/click each sphere for ROI name
  *_montage.pdf    — one page per ROI, ortho slice centred on the sphere
  *_atlas_summary.tsv — centroid MNI coords and voxel counts (placement sanity)

RUN ON SERVER:
  conda activate fmri
  cd /BICNAS2/group-northoff/jkokino/codes/har_med_codes
  python 99_QC/03_acw_qc/scripts/05_roi_placement_viewer.py

Transfer outputs to local machine (scp from repo root on server):
  scp jkokino@10.156.156.21:.../99_QC/03_acw_qc/figures/self_atlas_markers.html    <local>/
  scp jkokino@10.156.156.21:.../99_QC/03_acw_qc/figures/nonself_atlas_markers.html <local>/
  scp jkokino@10.156.156.21:.../99_QC/03_acw_qc/figures/self_atlas_montage.pdf     <local>/
  scp jkokino@10.156.156.21:.../99_QC/03_acw_qc/figures/nonself_atlas_montage.pdf  <local>/
  scp jkokino@10.156.156.21:.../99_QC/03_acw_qc/results/self_atlas_summary.tsv     <local>/
  scp jkokino@10.156.156.21:.../99_QC/03_acw_qc/results/nonself_atlas_summary.tsv  <local>/
"""

import sys

# Must be set before any matplotlib/nilearn import — server is headless
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf

import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from nilearn import plotting, datasets

# ── Repo root (script at 99_QC/03_acw_qc/scripts/) ──────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[3]

# ── Input paths ───────────────────────────────────────────────────────────────
ATLAS_DIR = REPO_ROOT / "02_timeseries_extraction" / "results" / "atlases"

SELF_ATLAS     = ATLAS_DIR / "self_atlas_1mm.nii.gz"
NONSELF_ATLAS  = ATLAS_DIR / "nonself_atlas_1mm.nii.gz"
# self_coordinates.txt: TSV, columns ROI_Number ROI_Name X Y Z Layer
SELF_COORDS    = ATLAS_DIR / "self_coordinates.txt"
# glasser_coordinates_nonself_clean.txt: TSV, columns ROI_Number ROI_Name X Y Z
NONSELF_COORDS = ATLAS_DIR / "glasser_coordinates_nonself_clean.txt"

# MNI 1mm template: server path preferred; nilearn built-in as fallback
MNI_SERVER = Path("/home/jkokino/meditation_project/templates/MNI/mni_icbm152_1mm.nii")

# ── Output paths ──────────────────────────────────────────────────────────────
FIG_DIR     = REPO_ROOT / "99_QC" / "03_acw_qc" / "figures"
RESULTS_DIR = REPO_ROOT / "99_QC" / "03_acw_qc" / "results"

SELF_HTML    = FIG_DIR     / "self_atlas_markers.html"
NONSELF_HTML = FIG_DIR     / "nonself_atlas_markers.html"
SELF_PDF     = FIG_DIR     / "self_atlas_montage.pdf"
NONSELF_PDF  = FIG_DIR     / "nonself_atlas_montage.pdf"
SELF_TSV     = RESULTS_DIR / "self_atlas_summary.tsv"
NONSELF_TSV  = RESULTS_DIR / "nonself_atlas_summary.tsv"

# ── Sanity thresholds (4mm-radius sphere at 1mm resolution) ──────────────────
# Perfect sphere ≈ 268 voxels; after discretisation typical count is ~257.
# Boundary ROIs (V1, temporal poles) can reach ~150. Below 100 → likely
# out-of-brain; above 300 → impossible for this sphere size.
VOXEL_MIN = 100
VOXEL_MAX = 300

# ── Check required inputs ─────────────────────────────────────────────────────
missing = [f for f in [SELF_ATLAS, NONSELF_ATLAS, SELF_COORDS, NONSELF_COORDS]
           if not f.exists()]
if missing:
    for m in missing:
        print(f"ERROR: Missing required input: {m}", file=sys.stderr)
    sys.exit(1)

FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── MNI template ──────────────────────────────────────────────────────────────
if MNI_SERVER.exists():
    mni_img = nib.load(str(MNI_SERVER))
    print(f"MNI template: {MNI_SERVER}")
else:
    mni_img = datasets.load_mni152_template(resolution=1)
    print("MNI template: nilearn built-in (1mm)")


# ── Coordinate loaders ────────────────────────────────────────────────────────
def load_self_coords():
    """
    self_coordinates.txt: tab-separated with header ROI_Number/ROI_Name/X/Y/Z/Layer.
    Created by 01_create_self_atlas.sh.
    """
    df = pd.read_csv(SELF_COORDS, sep="\t")
    return df.rename(columns={"ROI_Number": "roi_id", "ROI_Name": "roi_name"})


def load_nonself_coords():
    """
    glasser_coordinates_nonself_clean.txt: tab-separated with header
    ROI_Number/ROI_Name/X/Y/Z.  ROI_Number values are original Glasser parcel
    IDs (used as voxel values in nonself_atlas_1mm.nii.gz by 3dUndump).
    """
    df = pd.read_csv(NONSELF_COORDS, sep="\t")
    return df.rename(columns={"ROI_Number": "roi_id", "ROI_Name": "roi_name"})


# ── TSV summary ───────────────────────────────────────────────────────────────
def compute_roi_summary(atlas_img, coords_df):
    """
    For each ROI in coords_df, compute atlas centroid (MNI mm) and voxel count.
    Returns DataFrame: roi_id, roi_name, x, y, z, n_voxels.
    """
    data   = atlas_img.get_fdata()
    affine = atlas_img.affine
    rows   = []

    for _, row in coords_df.iterrows():
        roi_id = int(row["roi_id"])
        name   = str(row["roi_name"])
        mask   = data == roi_id
        n_vox  = int(mask.sum())

        if n_vox == 0:
            print(f"  WARNING: ROI {roi_id} ({name}) — 0 voxels in atlas image")
            rows.append({"roi_id": roi_id, "roi_name": name,
                         "x": np.nan, "y": np.nan, "z": np.nan, "n_voxels": 0})
            continue

        vox_ijk  = np.array(np.where(mask), dtype=float)
        centroid = affine @ np.append(vox_ijk.mean(axis=1), 1.0)
        rows.append({
            "roi_id":   roi_id,
            "roi_name": name,
            "x":        round(float(centroid[0]), 1),
            "y":        round(float(centroid[1]), 1),
            "z":        round(float(centroid[2]), 1),
            "n_voxels": n_vox,
        })

    return pd.DataFrame(rows, columns=["roi_id", "roi_name", "x", "y", "z", "n_voxels"])


def flag_anomalous(summary_df, atlas_name):
    flagged = summary_df[
        (summary_df["n_voxels"] < VOXEL_MIN) | (summary_df["n_voxels"] > VOXEL_MAX)
    ]
    if flagged.empty:
        print(f"  All {len(summary_df)} {atlas_name} ROIs within expected range "
              f"[{VOXEL_MIN}, {VOXEL_MAX}] voxels.")
    else:
        print(f"  !!! {atlas_name}: {len(flagged)} ROIs with anomalous voxel counts:")
        for _, r in flagged.iterrows():
            tag = "TOO FEW" if r["n_voxels"] < VOXEL_MIN else "TOO MANY"
            print(f"    [{tag}]  ROI {int(r['roi_id']):4d}  "
                  f"{r['roi_name']:<40}  {int(r['n_voxels'])} voxels")


# ── Interactive marker HTML ───────────────────────────────────────────────────
def make_marker_html(coords_df, html_path, title):
    """
    Build a nilearn view_markers HTML — each sphere is labelled 'ROI N: name'
    and shows on hover/click.  Coordinates come from the atlas construction
    files (sphere centres fed to 3dUndump), not computed centroids.
    """
    coords = coords_df[["X", "Y", "Z"]].values.tolist()
    labels = [f"ROI {int(r.roi_id)}: {r.roi_name}"
              for _, r in coords_df.iterrows()]
    view = plotting.view_markers(
        marker_coords=coords,
        marker_labels=labels,
        marker_size=6,
        title=title,
    )
    view.save_as_html(str(html_path))
    print(f"  Marker HTML → {html_path}")


# ── PDF montage ───────────────────────────────────────────────────────────────
def make_montage_pdf(atlas_img, coords_df, summary_df, pdf_path, atlas_label):
    """
    One page per ROI: ortho slice centred on the sphere, title with ROI number,
    name, MNI coords, and voxel count.  Prints progress every 50 ROIs.
    """
    atlas_data = atlas_img.get_fdata()
    merged     = coords_df.merge(
        summary_df[["roi_id", "n_voxels"]], on="roi_id", how="left"
    )
    n_rois = len(merged)

    print(f"  Generating {atlas_label} PDF montage ({n_rois} pages)...")

    with mpdf.PdfPages(str(pdf_path)) as pdf_out:
        for i, row in enumerate(merged.itertuples(index=False), 1):
            roi_id = int(row.roi_id)
            mask   = (atlas_data == roi_id).astype(np.float32)
            roi_nii = nib.Nifti1Image(mask, atlas_img.affine)

            n_vox      = int(row.n_voxels) if not np.isnan(row.n_voxels) else 0
            cut_coords = (float(row.X), float(row.Y), float(row.Z))
            title = (f"ROI {roi_id}: {row.roi_name}  |  "
                     f"MNI ({row.X:.0f}, {row.Y:.0f}, {row.Z:.0f})  |  "
                     f"{n_vox} voxels")

            display = plotting.plot_roi(
                roi_nii,
                bg_img=mni_img,
                cut_coords=cut_coords,
                display_mode="ortho",
                title=title,
                colorbar=False,
            )
            fig = plt.gcf()
            pdf_out.savefig(fig)
            display.close()

            if i % 50 == 0 or i == n_rois:
                print(f"  [{i}/{n_rois}] {row.roi_name}")

    print(f"  Montage PDF → {pdf_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Self atlas
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Self atlas ===")
self_coords = load_self_coords()
self_img    = nib.load(str(SELF_ATLAS))
print(f"  Atlas:  {SELF_ATLAS.name}  shape={self_img.shape}")
print(f"  Coords: {len(self_coords)} ROIs")

# TSV summary
if SELF_TSV.exists():
    self_summary = pd.read_csv(SELF_TSV, sep="\t")
    print(f"  [skip] TSV already exists: {SELF_TSV.name}")
else:
    self_summary = compute_roi_summary(self_img, self_coords)
    flag_anomalous(self_summary, "self")
    self_summary.to_csv(SELF_TSV, sep="\t", index=False, float_format="%.1f")
    print(f"  Summary TSV → {SELF_TSV}")

# Marker HTML
if SELF_HTML.exists():
    print(f"  [skip] HTML already exists: {SELF_HTML.name}")
else:
    make_marker_html(self_coords, SELF_HTML,
                     "Self atlas — 37 ROIs (Qin et al. 2020)")

# PDF montage
if SELF_PDF.exists():
    print(f"  [skip] PDF already exists: {SELF_PDF.name}")
else:
    make_montage_pdf(self_img, self_coords, self_summary, SELF_PDF, "Self")


# ═══════════════════════════════════════════════════════════════════════════════
# Nonself atlas
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== Nonself atlas ===")
nonself_coords = load_nonself_coords()
nonself_img    = nib.load(str(NONSELF_ATLAS))
print(f"  Atlas:  {NONSELF_ATLAS.name}  shape={nonself_img.shape}")
print(f"  Coords: {len(nonself_coords)} ROIs")

# TSV summary
if NONSELF_TSV.exists():
    nonself_summary = pd.read_csv(NONSELF_TSV, sep="\t")
    print(f"  [skip] TSV already exists: {NONSELF_TSV.name}")
else:
    nonself_summary = compute_roi_summary(nonself_img, nonself_coords)
    flag_anomalous(nonself_summary, "nonself")
    nonself_summary.to_csv(NONSELF_TSV, sep="\t", index=False, float_format="%.1f")
    print(f"  Summary TSV → {NONSELF_TSV}")

# Marker HTML
if NONSELF_HTML.exists():
    print(f"  [skip] HTML already exists: {NONSELF_HTML.name}")
else:
    make_marker_html(nonself_coords, NONSELF_HTML,
                     f"Nonself atlas — {len(nonself_coords)} ROIs (Glasser parcellation)")

# PDF montage  (~5-10 min for 316 ROIs)
if NONSELF_PDF.exists():
    print(f"  [skip] PDF already exists: {NONSELF_PDF.name}")
else:
    make_montage_pdf(nonself_img, nonself_coords, nonself_summary,
                     NONSELF_PDF, "Nonself")


# ── Final summary ─────────────────────────────────────────────────────────────
print("\n=== Done ===")
for label, df in [("Self   ", self_summary), ("Nonself", nonself_summary)]:
    valid = df[df["n_voxels"] > 0]
    print(f"  {label}: {len(df)} ROIs — "
          f"mean={valid['n_voxels'].mean():.1f}, "
          f"min={valid['n_voxels'].min()}, "
          f"max={valid['n_voxels'].max()} voxels/ROI")

print()
print("Outputs:")
for p in [SELF_HTML, NONSELF_HTML, SELF_PDF, NONSELF_PDF, SELF_TSV, NONSELF_TSV]:
    print(f"  {p}")
