#!/bin/bash
# Script: 07_create_keskin_atlas.sh
# Purpose: Build Keskin self-parcel sphere atlas (4mm radius) in MNI space,
#          then resample to native BOLD grid.
#          Uses the coordinate file produced by 06_parse_keskin_coordinates.py.
# Run on SERVER (requires AFNI + conda fmri):
#   cd /BICNAS2/group-northoff/jkokino/codes/har_med_codes
#   bash 02_timeseries_extraction/scripts/07_create_keskin_atlas.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
OUTPUT_DIR="$REPO_ROOT/02_timeseries_extraction/results/atlases"
COORDS_FILE="$OUTPUT_DIR/keskin_self_coordinates.txt"

TEMPLATE_1MM=/home/jkokino/meditation_project/templates/MNI/mni_icbm152_1mm.nii
BOLD_REF=/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep/sub-01/ses-01/func/sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz

if [ ! -f "$COORDS_FILE" ]; then
    echo "ERROR: coordinate file not found at $COORDS_FILE"
    echo "Run 06_parse_keskin_coordinates.py locally first and push to server."
    exit 1
fi

echo "Using coordinates: $COORDS_FILE"
echo "Number of Keskin parcels: $(tail -n +2 $COORDS_FILE | wc -l)"

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

# Build undump input: x y z roi_number (columns 3,4,5,1 from the TSV)
# Skip header row; use tab separator
tail -n +2 "$COORDS_FILE" | awk -F'\t' '{print $3, $4, $5, $1}' > keskin_coords_undump.txt

echo ""
echo "Undump input (first 5 lines):"
head -5 keskin_coords_undump.txt

echo ""
echo "Building 1mm Keskin atlas with 3dUndump (radius 4mm)..."
3dUndump \
    -prefix keskin_self_atlas_1mm \
    -master "$TEMPLATE_1MM" \
    -srad 4 \
    -xyz keskin_coords_undump.txt

3dAFNItoNIFTI -prefix keskin_self_atlas_1mm.nii.gz keskin_self_atlas_1mm+tlrc

echo ""
echo "Checking voxel counts per ROI..."
3dBrickStat -count -non-zero keskin_self_atlas_1mm.nii.gz

echo ""
echo "Resampling to native BOLD grid..."
3dresample \
    -input keskin_self_atlas_1mm.nii.gz \
    -master "$BOLD_REF" \
    -prefix keskin_self_atlas_native.nii.gz \
    -rmode NN

rm -f keskin_self_atlas_1mm+tlrc.* keskin_coords_undump.txt

echo ""
echo "Done."
echo "  keskin_self_atlas_1mm.nii.gz    -> $OUTPUT_DIR"
echo "  keskin_self_atlas_native.nii.gz -> $OUTPUT_DIR"
