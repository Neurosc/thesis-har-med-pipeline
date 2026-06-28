#!/usr/bin/env python3
"""
Glasser whole-cortex parcel-mean timeseries extraction (G1-flattening analysis).

Extracts mean denoised BOLD for ALL 360 cortical Cole-Anticevic / Glasser parcels
(cortex only; CA subcortex is out of scope — no clean cortical gradient value), one
file per subject x session, from EACH of the three NoGSR denoising pipelines
(detrend / glm / maximal). Reuses the already-denoised NIfTIs in 01_denoising/results/
(NO re-denoising; NoGSR throughout — `wSubcorGSR` is only the atlas's build name).

Sample: utils.subject_filter.get_included_subjects() -> n=35 (FD>0.3 exclusions:
sub-06/08/12/26/36). Reported per-arm at runtime.

Extractor: nilearn NiftiLabelsMasker over glasser360MNI.nii.gz (strategy='mean',
NoGSR, no standardize/detrend — denoising is already done). Columns = parcel INDEX
(1..360, the glasser360MNI voxel-label = CAB-NP LabelKey INDEX), sorted ascending.

VERIFICATION ASSERTS (guard against the prior sphere-substitution bug):
  * output is exactly (n_TR, 360) per run
  * the 360 columns are precisely the 360 cortical INDEX values from the masker over the
    parcellation — no value can come from a sphere or any other source (there is no sphere
    code path in this script)
  * the atlas label set is a subset of 1..360 (cortex only)
Fails loudly (SystemExit) on any violation.

GRID CHECK: if glasser360MNI's grid/affine != the BOLD's, the script STOPS and reports the
resampling it would use (atlas -> BOLD, NEAREST-NEIGHBOR — the only correct choice for a
label image). Re-run with --allow-resample to proceed. It will NOT silently resample.

Outputs:
  02_timeseries_extraction/results/glasser360/{pipeline}/{sub}_{ses}_glasser360_timeseries.csv
      (rows = Timepoint [full length; 6 dummy vols dropped later at the AUC stage, as in
       01_compute_acw.jl], cols = parcel INDEX 1..360)
  02_timeseries_extraction/results/glasser360/manifest.tsv
      (pipeline, subject, session, arm, n_TR, pct_censored_fd03)

Run on server (DO NOT silently resample):
  conda activate fmri
  python 02_timeseries_extraction/scripts/glasser_g1/01_extract_glasser_cortex.py [--allow-resample]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.maskers import NiftiLabelsMasker
from nilearn import image as nimg

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
from utils.subject_filter import get_included_subjects   # noqa: E402

# ─── Paths ────────────────────────────────────────────────────────────────────
DENOISING_DIR = REPO_ROOT / "01_denoising" / "results"
ATLAS_DIR     = REPO_ROOT / "02_timeseries_extraction" / "atlases"
GLASSER_NII   = ATLAS_DIR / "glasser360MNI.nii.gz"
CABNP_KEY     = ATLAS_DIR / "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt"
PARTICIPANTS  = REPO_ROOT / "participants.tsv"
FD_COV        = REPO_ROOT / "99_QC" / "01_motion_qc" / "results" / "fd_covariates_wide_thresh03.csv"
OUT_ROOT      = REPO_ROOT / "02_timeseries_extraction" / "results" / "glasser360"
MANIFEST      = OUT_ROOT / "manifest.tsv"

PIPELINES = ["detrend", "glm", "maximal"]
SESSIONS  = ["ses-01", "ses-02"]
N_CORTICAL = 360


def cortical_labels_from_key():
    """The 360 cortical INDEX values (1..360) from the CA LabelKey."""
    key = pd.read_csv(CABNP_KEY, sep="\t")
    cort = key[(key["INDEX"] <= N_CORTICAL) & (key["GLASSERLABELNAME"] != "NA")]
    labels = sorted(int(i) for i in cort["INDEX"].tolist())
    if len(labels) != N_CORTICAL:
        sys.exit(f"FATAL: expected {N_CORTICAL} cortical parcels in LabelKey, got {len(labels)}")
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-resample", action="store_true",
                    help="permit nearest-neighbor resampling of the atlas to the BOLD grid")
    args = ap.parse_args()

    subjects = get_included_subjects()
    part = pd.read_csv(PARTICIPANTS, sep="\t")
    arm = dict(zip(part["participant_id"], part["condition"]))
    n_verum   = sum(arm.get(s) == "verum"   for s in subjects)
    n_placebo = sum(arm.get(s) == "placebo" for s in subjects)
    print(f"Subjects (get_included_subjects, FD>0.3): n={len(subjects)} "
          f"({n_verum} verum + {n_placebo} placebo)")
    print("  " + ", ".join(subjects))

    # percent-censored (FD>0.3) per subject x session, for the manifest
    pct_cens = {}
    if FD_COV.exists():
        cov = pd.read_csv(FD_COV)
        for _, r in cov.iterrows():
            pct_cens[(r["subject"], "ses-01")] = r.get("pcf_pre", "")
            pct_cens[(r["subject"], "ses-02")] = r.get("pcf_post", "")
    else:
        print(f"  (note: {FD_COV.name} not found — pct_censored_fd03 left blank)")

    labels_to_use = cortical_labels_from_key()
    atlas_img = nib.load(str(GLASSER_NII))
    atlas_present = sorted(int(v) for v in np.unique(np.asarray(atlas_img.dataobj)) if v != 0)
    if not set(atlas_present) <= set(range(1, N_CORTICAL + 1)):
        extra = sorted(set(atlas_present) - set(range(1, N_CORTICAL + 1)))
        sys.exit(f"FATAL: glasser360MNI has labels outside 1..360 (e.g. {extra[:5]}). "
                 "This script is cortex-only; restrict the atlas before proceeding.")
    if set(atlas_present) != set(labels_to_use):
        sys.exit(f"FATAL: atlas label set ({len(atlas_present)}) != LabelKey cortical set "
                 f"({len(labels_to_use)}). Mismatch — stop and inspect.")
    print(f"Atlas: glasser360MNI shape={atlas_img.shape}, {len(atlas_present)} cortical labels (1..360)")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows = []
    n_done = n_skip = n_miss = 0

    for pipeline in PIPELINES:
        (OUT_ROOT / pipeline).mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}\nPIPELINE: {pipeline}\n{'='*60}")
        for sub in subjects:
            for ses in SESSIONS:
                out_csv = OUT_ROOT / pipeline / f"{sub}_{ses}_glasser360_timeseries.csv"
                bold = DENOISING_DIR / pipeline / f"{sub}_{ses}_task-rest_desc-{pipeline}_bold.nii.gz"
                tag = f"[{pipeline}] {sub} {ses}"

                if not bold.exists():
                    print(f"{tag} ... MISSING BOLD ({bold.name})")
                    n_miss += 1
                    continue

                bold_img = nib.load(str(bold))
                n_tr = bold_img.shape[3]

                if out_csv.exists():
                    print(f"{tag} ... skip (exists)")
                    n_skip += 1
                    rows.append(dict(pipeline=pipeline, subject=sub, session=ses,
                                     arm=arm.get(sub, "unknown"), n_TR=n_tr,
                                     pct_censored_fd03=pct_cens.get((sub, ses), "")))
                    continue

                # ── grid check (no silent resample) ──
                same_grid = (atlas_img.shape[:3] == bold_img.shape[:3] and
                             np.allclose(atlas_img.affine, bold_img.affine, atol=1e-3))
                if not same_grid:
                    if not args.allow_resample:
                        print("\nGRID MISMATCH — atlas vs BOLD differ in shape/affine.")
                        print(f"  atlas: shape={atlas_img.shape[:3]} affine0={atlas_img.affine[:3,3]}")
                        print(f"  BOLD : shape={bold_img.shape[:3]} affine0={bold_img.affine[:3,3]}")
                        print("  Would resample ATLAS -> BOLD grid with NEAREST-NEIGHBOR "
                              "(only valid choice for a label image).")
                        print("  Re-run with --allow-resample to proceed.")
                        sys.exit(2)
                    atlas_use = nimg.resample_to_img(atlas_img, bold_img, interpolation="nearest")
                else:
                    atlas_use = atlas_img

                t0 = time.time()
                masker = NiftiLabelsMasker(labels_img=atlas_use, strategy="mean",
                                           resampling_target=None, standardize=False,
                                           detrend=False, verbose=0)
                ts = masker.fit_transform(bold_img)   # (n_TR, n_regions)

                # ── verification asserts ──
                masker_labels = sorted(int(x) for x in np.atleast_1d(masker.labels_))
                if ts.shape != (n_tr, N_CORTICAL):
                    sys.exit(f"FATAL {tag}: timeseries shape {ts.shape} != ({n_tr}, {N_CORTICAL})")
                if masker_labels != labels_to_use:
                    sys.exit(f"FATAL {tag}: masker labels != 360 cortical INDEX set")

                df = pd.DataFrame(ts, columns=masker_labels)
                df.index.name = "Timepoint"
                df.to_csv(str(out_csv))
                print(f"{tag} -> {ts.shape[1]} parcels x {ts.shape[0]} TR  ({time.time()-t0:.0f}s)")
                n_done += 1
                rows.append(dict(pipeline=pipeline, subject=sub, session=ses,
                                 arm=arm.get(sub, "unknown"), n_TR=n_tr,
                                 pct_censored_fd03=pct_cens.get((sub, ses), "")))

    pd.DataFrame(rows).to_csv(MANIFEST, sep="\t", index=False)
    print(f"\n─── summary ───\nextracted={n_done}  skipped={n_skip}  missing={n_miss}")
    print(f"manifest -> {MANIFEST}")


if __name__ == "__main__":
    main()
