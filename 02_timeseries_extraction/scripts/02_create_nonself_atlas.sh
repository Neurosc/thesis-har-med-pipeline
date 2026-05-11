#!/bin/bash
# Script: 02_create_nonself_atlas.sh
# Purpose: Create a NIfTI atlas of Glasser nonself ROIs as 4mm-radius spheres
#          in MNI space, then resample to native BOLD grid.
#          Coordinates file excludes ROIs that overlap with the self atlas
#          (see 03_check_nonself_overlap.py).
# Inputs:  MNI 1mm template (server path)
#          Glasser coordinates file (overlap-cleaned, 316 ROIs)
#          sub-01 ses-01 BOLD (native grid reference)
# Outputs: nonself_atlas_1mm.nii.gz      (1mm MNI, sphere radius = 4mm)
#          nonself_atlas_native.nii.gz   (native BOLD grid, 1.72x1.72x2.00mm)
# Adapted for har-med dataset from task_med_codes pipeline.
# Date: May 2026

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
OUTPUT_DIR="$REPO_ROOT/02_timeseries_extraction/results/atlases"

TEMPLATE_1MM=/home/jkokino/meditation_project/templates/MNI/mni_icbm152_1mm.nii
COORDS_FILE=/BICNAS2/group-northoff/jkokino/codes/har_med_codes/02_timeseries_extraction/results/atlases/glasser_coordinates_nonself_clean.txt
BOLD_REF=/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep/sub-01/ses-01/func/sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz

echo "Using coordinates: $COORDS_FILE"
echo "Number of ROIs in coords file: $(tail -n +2 $COORDS_FILE | wc -l)"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

tail -n +2 "$COORDS_FILE" | awk '{print $3, $4, $5, $1}' > glasser_coords_undump.txt

echo "Building 1mm nonself atlas with 3dUndump..."
3dUndump \
    -prefix nonself_atlas_1mm \
    -master "$TEMPLATE_1MM" \
    -srad 4 \
    -xyz glasser_coords_undump.txt

3dAFNItoNIFTI -prefix nonself_atlas_1mm.nii.gz nonself_atlas_1mm+tlrc

echo "Resampling to native BOLD grid..."
3dresample \
    -input nonself_atlas_1mm.nii.gz \
    -master "$BOLD_REF" \
    -prefix nonself_atlas_native.nii.gz \
    -rmode NN

rm -f nonself_atlas_1mm+tlrc.* glasser_coords_undump.txt

echo "Done."
echo "  nonself_atlas_1mm.nii.gz    -> $OUTPUT_DIR"
echo "  nonself_atlas_native.nii.gz -> $OUTPUT_DIR"
