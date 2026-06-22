#!/usr/bin/env python3
"""
Sphere-based timeseries extraction — DMT-MED dataset.

Extracts mean BOLD timeseries for all six self/nonself layers using 4mm-radius
sphere ROIs in native BOLD space (NoGSR only).

Self layers  (Qin et al. 2020 MNI coordinates, 37 ROIs, 11/14/12 per layer):
  Interoception
  Exteroception
  Cognition

Nonself layers (CAB-NP Glasser network parcels; centroids computed from
glasser360MNI.nii.gz; self-overlapping parcels excluded via category_parcels.tsv;
tSNR exclusion NOT applied — to be recalculated for sphere-based extraction):
  Visual    Visual1 + Visual2 network parcels
  Motor     Somatomotor network parcels
  Auditory  Auditory network parcels

35 subjects × 2 sessions × 6 layers = 420 CSVs total.

Outputs
-------
  02_timeseries_extraction/results/timeseries_self/{Layer}/
      sub-XX_ses-YY_{Layer}_timeseries.csv   (rows=timepoints, cols=ROI_Number)
  02_timeseries_extraction/results/timeseries_nonself/{Layer}/
      sub-XX_ses-YY_{Layer}_timeseries.csv   (rows=timepoints, cols=parcel_id)
  02_timeseries_extraction/results/_sphere_extraction_log.tsv

Run on server:
  conda activate fmri
  python 02_timeseries_extraction/scripts/spheres/01_extract_sphere_timeseries.py
"""

import sys
import csv
import time
import datetime
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from utils.subject_filter import get_included_subjects

# ─── Paths ────────────────────────────────────────────────────────────────────

FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
DENOISING_DIR = REPO_ROOT / "01_preprocessing" / "02_denoising" / "results"
ATLAS_DIR     = REPO_ROOT / "02_timeseries_extraction" / "results" / "atlases"
DATA_DIR      = REPO_ROOT / "02_timeseries_extraction" / "data"

GLASSER_NII  = DATA_DIR / "glasser360MNI.nii.gz"
CABNP_KEY    = DATA_DIR / "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"
CATEGORY_TSV = ATLAS_DIR / "category_parcels.tsv"

OUT_SELF    = REPO_ROOT / "02_timeseries_extraction" / "results" / "timeseries_self"
OUT_NONSELF = REPO_ROOT / "02_timeseries_extraction" / "results" / "timeseries_nonself"
LOG_PATH    = REPO_ROOT / "02_timeseries_extraction" / "results" / "_sphere_extraction_log.tsv"

SUBJECTS  = get_included_subjects()
SESSIONS  = ["ses-01", "ses-02"]
RADIUS_MM = 4.0

LOG_COLS = ["subject", "session", "layer", "n_rois", "n_timepoints",
            "status", "elapsed_sec", "timestamp"]

# ─── Self ROI definitions (Qin et al. 2020, coordinates from 01_create_self_atlas.sh)

SELF_ROIS = [
    # (roi_id, name, X_mni, Y_mni, Z_mni, layer)
    (1,  "Interoception_R_Insula",         34,   14,  12, "Interoception"),
    (2,  "Interoception_L_DACC",            0,    4,  48, "Interoception"),
    (3,  "Interoception_R_Thalamus",       12,  -14,   4, "Interoception"),
    (4,  "Interoception_R_Parahippocampal", 30,   -4, -24, "Interoception"),
    (5,  "Interoception_L_Parahippocampal",-20,   -4, -20, "Interoception"),
    (6,  "Interoception_L_Insula_1",       -40,   -2,   2, "Interoception"),
    (7,  "Interoception_L_Insula_2",       -36,   24,   4, "Interoception"),
    (8,  "Interoception_R_IPL",             56,  -26,  26, "Interoception"),
    (9,  "Interoception_R_SFG",              4,   24,  48, "Interoception"),
    (10, "Interoception_L_STG",            -56,    6,   6, "Interoception"),
    (11, "Interoception_L_Postcentral",    -48,  -16,  32, "Interoception"),
    (12, "Exteroception_R_Fusiform",        48,  -58, -12, "Exteroception"),
    (13, "Exteroception_R_IFG",             48,   40,   8, "Exteroception"),
    (14, "Exteroception_R_Premotor",        50,    8,  26, "Exteroception"),
    (15, "Exteroception_R_Insula",          40,    8,   0, "Exteroception"),
    (16, "Exteroception_L_Fusiform",       -44,  -68,  -6, "Exteroception"),
    (17, "Exteroception_R_SPL",             26,  -72,  44, "Exteroception"),
    (18, "Exteroception_R_Postcentral",     58,  -22,  38, "Exteroception"),
    (19, "Exteroception_R_IPL",             36,  -50,  56, "Exteroception"),
    (20, "Exteroception_L_Insula",         -36,   18,  -4, "Exteroception"),
    (21, "Exteroception_R_IOG",             38,  -80,  -2, "Exteroception"),
    (22, "Exteroception_L_IPL",            -46,  -34,  40, "Exteroception"),
    (23, "Exteroception_L_SPL",            -22,  -64,  50, "Exteroception"),
    (24, "Exteroception_R_Cingulate",        4,    8,  38, "Exteroception"),
    (25, "Exteroception_L_mPFC",            -6,   60,  22, "Exteroception"),
    (26, "Cognition_L_ACC",                 -6,   48,   0, "Cognition"),
    (27, "Cognition_L_PCC",                 -4,  -54,  28, "Cognition"),
    (28, "Cognition_L_Insula",             -36,   22,  -2, "Cognition"),
    (29, "Cognition_L_MTG",               -48,  -66,  28, "Cognition"),
    (30, "Cognition_L_Thalamus",            -8,    2,   8, "Cognition"),
    (31, "Cognition_L_SFG_1",             -20,   36,  46, "Cognition"),
    (32, "Cognition_Cingulate",              0,  -18,  40, "Cognition"),
    (33, "Cognition_L_ITG",               -62,   -6, -18, "Cognition"),
    (34, "Cognition_R_MTG",                54,  -60,  24, "Cognition"),
    (35, "Cognition_R_Insula",             52,   10,  -6, "Cognition"),
    (36, "Cognition_L_SFG_2",            -24,   50,  22, "Cognition"),
    (37, "Cognition_R_Premotor",           46,    6,  24, "Cognition"),
]

SELF_LAYERS = ["Interoception", "Exteroception", "Cognition"]

# CAB-NP NETWORK column strings for each nonself layer
NONSELF_NETWORKS = {
    "Visual":   {"Visual1", "Visual2"},
    "Motor":    {"Somatomotor"},
    "Auditory": {"Auditory"},
}

# ─── Sphere utilities ─────────────────────────────────────────────────────────

def sphere_offsets(radius_mm, vox_sizes):
    """Voxel index offsets (di, dj, dk) forming a sphere of radius_mm in anisotropic space."""
    r = [int(np.ceil(radius_mm / s)) + 1 for s in vox_sizes]
    r2 = radius_mm ** 2
    offsets = []
    for di in range(-r[0], r[0] + 1):
        for dj in range(-r[1], r[1] + 1):
            for dk in range(-r[2], r[2] + 1):
                if (di * vox_sizes[0])**2 + (dj * vox_sizes[1])**2 + (dk * vox_sizes[2])**2 <= r2:
                    offsets.append((di, dj, dk))
    return offsets


def mni_to_ijk(mni_xyz, affine):
    """Convert MNI mm coordinates to nearest voxel indices."""
    v = np.linalg.inv(affine) @ np.array([*mni_xyz, 1.0])
    return tuple(np.round(v[:3]).astype(int))


def extract_sphere_ts(bold_data, center_ijk, offsets, shape):
    """
    Mean BOLD timeseries for a 4mm sphere at center_ijk.
    Returns 1-D array (length T) or None if no voxels fall within bounds.
    """
    ci, cj, ck = center_ijk
    rows = [
        (ci + di, cj + dj, ck + dk)
        for di, dj, dk in offsets
        if 0 <= ci + di < shape[0]
        and 0 <= cj + dj < shape[1]
        and 0 <= ck + dk < shape[2]
    ]
    if not rows:
        return None
    idx = np.array(rows)
    return bold_data[idx[:, 0], idx[:, 1], idx[:, 2]].mean(axis=0)

# ─── Nonself layer definitions ────────────────────────────────────────────────

def load_nonself_layers():
    """
    Returns dict {layer_name: [(parcel_id, glasser_label), ...]}.

    Source: CAB-NP label key (NETWORK column, cortical parcels only, INDEX 1-360).
    Exclusion: parcels whose INDEX appears in category_parcels.tsv with
    category Interoception / Exteroception / Cognition (Keskin self mapping).
    """
    cat_df = pd.read_csv(CATEGORY_TSV, sep="\t")
    self_ids = set(
        cat_df.loc[
            cat_df["category"].isin({"Interoception", "Exteroception", "Cognition"}),
            "roi_pos_id",
        ]
    )

    key_df = pd.read_csv(CABNP_KEY, sep="\t")
    cortical = key_df[
        (key_df["INDEX"] <= 360) & (key_df["GLASSERLABELNAME"] != "NA")
    ].copy()

    layers = {}
    for layer_name, networks in NONSELF_NETWORKS.items():
        subset = cortical[cortical["NETWORK"].isin(networks)]
        parcels = [
            (int(row["INDEX"]), str(row["GLASSERLABELNAME"]))
            for _, row in subset.iterrows()
            if int(row["INDEX"]) not in self_ids
        ]
        layers[layer_name] = parcels
        print(f"  {layer_name}: {len(parcels)} parcels after self-exclusion")
    return layers


def compute_centroids(glasser_data, glasser_affine, parcel_ids):
    """MNI centroid (mm) for each parcel ID in glasser360MNI.nii.gz."""
    centroids = {}
    for pid in parcel_ids:
        vox = np.argwhere(glasser_data == pid)
        if len(vox) == 0:
            print(f"    Warning: parcel {pid} not found in Glasser atlas")
            continue
        c_vox = vox.mean(axis=0)
        c_mni = (glasser_affine @ np.append(c_vox, 1.0))[:3]
        centroids[pid] = c_mni
    return centroids

# ─── Log helper ───────────────────────────────────────────────────────────────

def _append_log(row):
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LOG_COLS, delimiter="\t")
        if write_header:
            writer.writeheader()
        writer.writerow(row)

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    for layer in SELF_LAYERS:
        (OUT_SELF / layer).mkdir(parents=True, exist_ok=True)
    for layer in NONSELF_NETWORKS:
        (OUT_NONSELF / layer).mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Loading nonself layer definitions...")
    nonself_layers = load_nonself_layers()

    print("\nLoading Glasser atlas for centroid computation...")
    glasser_img    = nib.load(str(GLASSER_NII))
    glasser_data   = glasser_img.get_fdata()
    glasser_affine = glasser_img.affine
    print(f"  Atlas shape: {glasser_data.shape}")

    all_nonself_ids = {pid for parcels in nonself_layers.values() for pid, _ in parcels}
    print(f"  Computing centroids for {len(all_nonself_ids)} parcels...")
    all_centroids = compute_centroids(glasser_data, glasser_affine, all_nonself_ids)
    del glasser_data

    n_runs = len(SUBJECTS) * len(SESSIONS)
    print(f"\nExtraction: {len(SUBJECTS)} subjects × {len(SESSIONS)} sessions = {n_runs} runs")
    print(f"Layers: {SELF_LAYERS} (self) + {list(NONSELF_NETWORKS)} (nonself)\n")

    n_completed = n_skipped = n_failed = 0
    failed_runs = []

    for run_idx, sub in enumerate(SUBJECTS, start=1):
        for ses in SESSIONS:
            prefix = f"[{run_idx:03d}/{len(SUBJECTS)}] {sub} {ses}"

            # Collect expected output paths
            self_csvs = {
                lay: OUT_SELF / lay / f"{sub}_{ses}_{lay}_timeseries.csv"
                for lay in SELF_LAYERS
            }
            nonself_csvs = {
                lay: OUT_NONSELF / lay / f"{sub}_{ses}_{lay}_timeseries.csv"
                for lay in NONSELF_NETWORKS
            }
            pending_self    = [lay for lay, p in self_csvs.items()    if not p.exists()]
            pending_nonself = [lay for lay, p in nonself_csvs.items() if not p.exists()]

            if not pending_self and not pending_nonself:
                print(f"{prefix} ... SKIPPED (all 6 layers exist)")
                n_skipped += 1
                continue

            bold = DENOISING_DIR / f"{sub}_{ses}_task-rest_desc-denoisedNoGSR_bold.nii.gz"
            t0 = time.time()
            try:
                if not bold.exists():
                    raise FileNotFoundError(f"BOLD not found: {bold}")

                print(f"{prefix} loading BOLD...")
                bold_img    = nib.load(str(bold))
                bold_data   = bold_img.get_fdata()
                bold_affine = bold_img.affine
                shape       = bold_data.shape[:3]
                n_tp        = bold_data.shape[3]
                vox_sizes   = tuple(float(v) for v in bold_img.header.get_zooms()[:3])
                print(f"  shape={bold_data.shape}  vox={vox_sizes}")

                offsets = sphere_offsets(RADIUS_MM, vox_sizes)

                # ── Self layers ────────────────────────────────────────────
                for layer in pending_self:
                    csv_out = self_csvs[layer]
                    rois = [(roi_id, name, x, y, z)
                            for roi_id, name, x, y, z, lyr in SELF_ROIS
                            if lyr == layer]

                    ts_dict = {}
                    for roi_id, name, x, y, z in rois:
                        center = mni_to_ijk((x, y, z), bold_affine)
                        ts = extract_sphere_ts(bold_data, center, offsets, shape)
                        if ts is None:
                            print(f"    Warning: {name} (ROI {roi_id}) no in-bounds voxels")
                        else:
                            ts_dict[roi_id] = ts

                    df = pd.DataFrame(ts_dict)
                    df.index.name = "Timepoint"
                    df.to_csv(str(csv_out))
                    print(f"  {layer}: {len(ts_dict)} ROIs → {csv_out.name}")

                    _append_log({
                        "subject": sub, "session": ses, "layer": layer,
                        "n_rois": len(ts_dict), "n_timepoints": n_tp,
                        "status": "DONE",
                        "elapsed_sec": f"{time.time() - t0:.1f}",
                        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    })

                # ── Nonself layers ─────────────────────────────────────────
                for layer in pending_nonself:
                    csv_out = nonself_csvs[layer]
                    parcels = nonself_layers[layer]

                    ts_dict = {}
                    for pid, pname in parcels:
                        if pid not in all_centroids:
                            continue
                        center = mni_to_ijk(all_centroids[pid], bold_affine)
                        ts = extract_sphere_ts(bold_data, center, offsets, shape)
                        if ts is None:
                            print(f"    Warning: {pname} (parcel {pid}) no in-bounds voxels")
                        else:
                            ts_dict[pid] = ts

                    df = pd.DataFrame(ts_dict)
                    df.index.name = "Timepoint"
                    df.to_csv(str(csv_out))
                    print(f"  {layer}: {len(ts_dict)} parcels → {csv_out.name}")

                    _append_log({
                        "subject": sub, "session": ses, "layer": layer,
                        "n_rois": len(ts_dict), "n_timepoints": n_tp,
                        "status": "DONE",
                        "elapsed_sec": f"{time.time() - t0:.1f}",
                        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                    })

                elapsed = time.time() - t0
                print(f"  Done in {elapsed:.0f}s")
                n_completed += 1

            except Exception as exc:
                elapsed = time.time() - t0
                print(f"  ERROR: {exc}")
                _append_log({
                    "subject": sub, "session": ses, "layer": "ALL",
                    "n_rois": "", "n_timepoints": "",
                    "status": f"FAILED: {exc}",
                    "elapsed_sec": f"{elapsed:.1f}",
                    "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
                })
                n_failed += 1
                failed_runs.append(f"{sub}_{ses}: {exc}")

    print(f"\n─── Extraction summary ───")
    print(f"Total runs:  {n_runs}")
    print(f"Completed:   {n_completed}")
    print(f"Skipped:     {n_skipped}")
    print(f"Failed:      {n_failed}")
    if failed_runs:
        print("\nFailed runs:")
        for r in failed_runs:
            print(f"  {r}")


if __name__ == "__main__":
    main()
