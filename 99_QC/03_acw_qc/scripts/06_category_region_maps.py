#!/usr/bin/env python3
"""
06_category_region_maps.py — Brain maps of the Glasser parcels used per category.

Renders, for each analysis category (Interoception, Exteroception, Cognition, and
the Somatomotor / non-self reference), the parcels that were USED in the analysis,
in THREE views.

Region membership is read from `category_parcels.tsv` (roi_pos_id, roi_name,
category), which is derived from the analysis dataframe so the figure provably
reflects the parcels that entered the LMMs (Interoception 14, Exteroception 16,
Cognition 12, Somatomotor 105). Parcel IDs index `glasser360MNI.nii.gz` (1-360).

Two render modes (set MODE below):
  "glass"   — nilearn plot_glass_brain, display_mode="ortho" → sagittal + coronal +
              axial maximum-intensity projections (all parcels visible at once).
              Robust, volumetric, no surface projection. DEFAULT.
  "surface" — parcels projected onto the inflated fsaverage surface, lateral +
              medial per hemisphere (prettier; needs vol->surf projection).

Outputs (99_QC/03_acw_qc/figures/):
  category_<name>_<mode>.png      one per category (3 views each)
  categories_combined_<mode>.png  all four categories color-coded on one brain
  category_region_maps_<mode>.pdf all of the above, one per page

RUNS LOCALLY — all inputs (glasser360MNI.nii.gz, category_parcels.tsv) are local; only
nilearn is needed (no raw fMRIPrep NIfTIs, so no server required). One-off setup:
  py -3.13 -m venv C:\\Users\\JLU-MBB\\nilearn-venv
  C:\\Users\\JLU-MBB\\nilearn-venv\\Scripts\\python -m pip install nilearn nibabel matplotlib
Then:
  C:\\Users\\JLU-MBB\\nilearn-venv\\Scripts\\python 99_QC/03_acw_qc/scripts/06_category_region_maps.py

(Or on the server: conda activate fmri && python 99_QC/03_acw_qc/scripts/06_category_region_maps.py)
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless server — never plt.show()
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.backends.backend_pdf as mpdf

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn import plotting

# Windows consoles default to cp1252 and choke on the arrows/em-dashes in our
# progress prints; force UTF-8 so the same script runs identically on Windows + server.
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# ── Config ──────────────────────────────────────────────────────────────────────
MODE = "surface"        # "glass" | "surface"

# Okabe-Ito colourblind-safe palette; somatomotor = grey (the reference layer)
CATEGORIES = ["Interoception", "Exteroception", "Cognition", "Somatomotor"]
COLORS = {
    "Interoception": "#E69F00",
    "Exteroception": "#56B4E9",
    "Cognition":     "#009E73",
    "Somatomotor":   "#9970AB",   # muted purple — visible on both white (glass) and grey (surface)
}

# ── Paths (script at 99_QC/03_acw_qc/scripts/) ─────────────────────────────────
REPO_ROOT  = Path(__file__).resolve().parents[3]
GLASSER360 = REPO_ROOT / "02_timeseries_extraction" / "data" / "glasser360MNI.nii.gz"
PARCELS_TSV = REPO_ROOT / "02_timeseries_extraction" / "results" / "atlases" / "category_parcels.tsv"
FIG_DIR    = REPO_ROOT / "99_QC" / "03_acw_qc" / "figures"

missing = [p for p in (GLASSER360, PARCELS_TSV) if not p.exists()]
if missing:
    for m in missing:
        print(f"ERROR: missing required input: {m}", file=sys.stderr)
    sys.exit(1)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def load_category_ids():
    """category -> sorted list of Glasser parcel IDs used in that category."""
    df = pd.read_csv(PARCELS_TSV, sep="\t")
    out = {}
    for cat in CATEGORIES:
        out[cat] = sorted(int(i) for i in df.loc[df["category"] == cat, "roi_pos_id"])
    return out


def category_mask_img(atlas_data, affine, header, ids):
    """Binary NIfTI mask of the voxels belonging to `ids` in the Glasser atlas."""
    mask = np.isin(atlas_data, ids).astype(np.int8)
    return nib.Nifti1Image(mask, affine, header), int(mask.sum())


# ── Glass-brain rendering (volumetric, 3 MIP views) ─────────────────────────────
def render_glass(cat_ids, atlas_img):
    data, affine, header = atlas_img.get_fdata(), atlas_img.affine, atlas_img.header
    pdf_path = FIG_DIR / "category_region_maps_glass.pdf"
    figs = []

    with mpdf.PdfPages(str(pdf_path)) as pdf_out:
        # one page per category
        for cat in CATEGORIES:
            ids = cat_ids[cat]
            mask_img, n_vox = category_mask_img(data, affine, header, ids)
            disp = plotting.plot_glass_brain(
                None, display_mode="ortho",
                title=f"{cat} — {len(ids)} parcels ({n_vox} voxels)")
            disp.add_contours(mask_img, levels=[0.5], filled=True,
                              colors=[COLORS[cat]], alpha=0.85)
            png = FIG_DIR / f"category_{cat.lower()}_glass.png"
            disp.savefig(str(png), dpi=300)
            pdf_out.savefig(plt.gcf())
            disp.close()
            print(f"  [{cat}] {len(ids)} parcels, {n_vox} voxels → {png.name}")

        # combined: all four categories color-coded on one brain.
        # Draw the grey somatomotor reference first so the coloured self
        # categories sit on top of it in the maximum-intensity projection.
        disp = plotting.plot_glass_brain(None, display_mode="ortho",
                                         title="Categories used (combined)")
        draw_order = ["Somatomotor"] + [c for c in CATEGORIES if c != "Somatomotor"]
        for cat in draw_order:
            mask_img, _ = category_mask_img(data, affine, header, cat_ids[cat])
            disp.add_contours(mask_img, levels=[0.5], filled=True,
                              colors=[COLORS[cat]], alpha=0.65)
        # manual legend on the underlying figure
        handles = [mpatches.Patch(color=COLORS[c], label=c) for c in CATEGORIES]
        plt.gcf().legend(handles=handles, loc="lower center", ncol=4,
                         frameon=False, fontsize=9)
        combined = FIG_DIR / "categories_combined_glass.png"
        disp.savefig(str(combined), dpi=300)
        pdf_out.savefig(plt.gcf())
        disp.close()
        print(f"  [combined] 4 categories → {combined.name}")

    print(f"  PDF → {pdf_path}")


# ── Surface rendering (inflated fsaverage, lateral + medial per hemisphere) ─────
def render_surface(cat_ids, atlas_img):
    from nilearn import datasets, surface
    from matplotlib.colors import ListedColormap
    data, affine, header = atlas_img.get_fdata(), atlas_img.affine, atlas_img.header
    fsavg = datasets.fetch_surf_fsaverage("fsaverage")
    pdf_path = FIG_DIR / "category_region_maps_surface.pdf"
    hemis = [("left", fsavg.infl_left, fsavg.pial_left, fsavg.white_left, fsavg.sulc_left),
             ("right", fsavg.infl_right, fsavg.pial_right, fsavg.white_right, fsavg.sulc_right)]
    views = ["lateral", "medial"]

    with mpdf.PdfPages(str(pdf_path)) as pdf_out:
        for cat in CATEGORIES:
            mask_img, n_vox = category_mask_img(data, affine, header, cat_ids[cat])
            fig, axes = plt.subplots(len(hemis), len(views), figsize=(10, 9),
                                     subplot_kw={"projection": "3d"})
            for r, (hname, infl, pial, white, sulc) in enumerate(hemis):
                # sample the full cortical ribbon (pial outer + white inner) so a
                # volumetric parcel projects to a filled patch, not a thin sliver
                tex = surface.vol_to_surf(mask_img, pial, inner_mesh=white,
                                          interpolation="nearest_most_frequent")
                roi = (np.nan_to_num(tex) > 0.5).astype(float)
                cmap = ListedColormap([COLORS[cat]])
                for c, view in enumerate(views):
                    if roi.sum() == 0:                # no parcels of this category here
                        plotting.plot_surf(infl, hemi=hname, view=view,
                                           bg_map=sulc, axes=axes[r, c])
                    else:
                        plotting.plot_surf_roi(
                            infl, roi_map=roi, hemi=hname, view=view,
                            bg_map=sulc, bg_on_data=True, threshold=0.5,
                            cmap=cmap, axes=axes[r, c], colorbar=False)
                    axes[r, c].set_title(f"{hname} {view}", fontsize=9)
            fig.suptitle(f"{cat} — {len(cat_ids[cat])} parcels", fontsize=12)
            png = FIG_DIR / f"category_{cat.lower()}_surface.png"
            fig.savefig(str(png), dpi=300, bbox_inches="tight")
            pdf_out.savefig(fig)
            plt.close(fig)
            print(f"  [{cat}] {len(cat_ids[cat])} parcels → {png.name}")
    print(f"  PDF → {pdf_path}")


# ── Main ────────────────────────────────────────────────────────────────────────
def main():
    print("=" * 70)
    print(f"06_category_region_maps.py — mode={MODE}")
    print("=" * 70)
    cat_ids = load_category_ids()
    total = sum(len(v) for v in cat_ids.values())
    print(f"Atlas:   {GLASSER360.name}")
    print(f"Parcels: {total} category-parcel pairs from {PARCELS_TSV.name}")
    for c in CATEGORIES:
        print(f"  {c:15s} {len(cat_ids[c])} parcels")
    print()

    atlas_img = nib.load(str(GLASSER360))
    if MODE == "glass":
        render_glass(cat_ids, atlas_img)
    elif MODE == "surface":
        render_surface(cat_ids, atlas_img)
    else:
        sys.exit(f"Unknown MODE={MODE!r} (use 'glass' or 'surface')")
    print("\nDone.")


if __name__ == "__main__":
    main()
