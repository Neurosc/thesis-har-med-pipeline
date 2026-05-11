#!/bin/bash
# Script: 01_create_self_atlas.sh
# Purpose: Create a NIfTI atlas of 37 self-referential ROIs (Qin et al. 2020)
#          as 4mm-radius spheres in MNI space, then resample to native BOLD grid.
#          Covers three layers of self: Interoception, Exteroception, Cognition.
# Inputs:  MNI 1mm template (server path)
#          sub-01 ses-01 BOLD (native grid reference)
# Outputs: self_atlas_1mm.nii.gz      (1mm MNI, sphere radius = 4mm)
#          self_atlas_native.nii.gz   (native BOLD grid, 1.72x1.72x2.00mm)
#          self_labels.txt            (ROI number -> ROI name mapping)
#          self_coordinates.txt       (full coordinate table with layer labels)
# Adapted for har-med dataset from task_med_codes pipeline.
# Date: May 2026

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
OUTPUT_DIR="$REPO_ROOT/02_timeseries_extraction/results/atlases"

TEMPLATE_1MM=/home/jkokino/meditation_project/templates/MNI/mni_icbm152_1mm.nii
BOLD_REF=/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep/sub-01/ses-01/func/sub-01_ses-01_task-rest_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz

mkdir -p "$OUTPUT_DIR"
cd "$OUTPUT_DIR"

echo "Creating self-referential coordinates file..."

cat > self_coordinates.txt << EOF
ROI_Number	ROI_Name	X	Y	Z	Layer
1	Interoception_R_Insula	34	14	12	Interoception
2	Interoception_L_DACC	0	4	48	Interoception
3	Interoception_R_Thalamus	12	-14	4	Interoception
4	Interoception_R_Parahippocampal	30	-4	-24	Interoception
5	Interoception_L_Parahippocampal	-20	-4	-20	Interoception
6	Interoception_L_Insula_1	-40	-2	2	Interoception
7	Interoception_L_Insula_2	-36	24	4	Interoception
8	Interoception_R_IPL	56	-26	26	Interoception
9	Interoception_R_SFG	4	24	48	Interoception
10	Interoception_L_STG	-56	6	6	Interoception
11	Interoception_L_Postcentral	-48	-16	32	Interoception
12	Exteroception_R_Fusiform	48	-58	-12	Exteroception
13	Exteroception_R_IFG	48	40	8	Exteroception
14	Exteroception_R_Premotor	50	8	26	Exteroception
15	Exteroception_R_Insula	40	8	0	Exteroception
16	Exteroception_L_Fusiform	-44	-68	-6	Exteroception
17	Exteroception_R_SPL	26	-72	44	Exteroception
18	Exteroception_R_Postcentral	58	-22	38	Exteroception
19	Exteroception_R_IPL	36	-50	56	Exteroception
20	Exteroception_L_Insula	-36	18	-4	Exteroception
21	Exteroception_R_IOG	38	-80	-2	Exteroception
22	Exteroception_L_IPL	-46	-34	40	Exteroception
23	Exteroception_L_SPL	-22	-64	50	Exteroception
24	Exteroception_R_Cingulate	4	8	38	Exteroception
25	Exteroception_L_mPFC	-6	60	22	Exteroception
26	Cognition_L_ACC	-6	48	0	Cognition
27	Cognition_L_PCC	-4	-54	28	Cognition
28	Cognition_L_Insula	-36	22	-2	Cognition
29	Cognition_L_MTG	-48	-66	28	Cognition
30	Cognition_L_Thalamus	-8	2	8	Cognition
31	Cognition_L_SFG_1	-20	36	46	Cognition
32	Cognition_Cingulate	0	-18	40	Cognition
33	Cognition_L_ITG	-62	-6	-18	Cognition
34	Cognition_R_MTG	54	-60	24	Cognition
35	Cognition_R_Insula	52	10	-6	Cognition
36	Cognition_L_SFG_2	-24	50	22	Cognition
37	Cognition_R_Premotor	46	6	24	Cognition
EOF

tail -n +2 self_coordinates.txt | awk '{print $3, $4, $5, $1}' > self_coords_undump.txt

echo "Building 1mm atlas with 3dUndump..."
3dUndump \
    -prefix self_atlas_1mm \
    -master "$TEMPLATE_1MM" \
    -srad 4 \
    -xyz self_coords_undump.txt

3dAFNItoNIFTI -prefix self_atlas_1mm.nii.gz self_atlas_1mm+tlrc

echo "Resampling to native BOLD grid..."
3dresample \
    -input self_atlas_1mm.nii.gz \
    -master "$BOLD_REF" \
    -prefix self_atlas_native.nii.gz \
    -rmode NN

tail -n +2 self_coordinates.txt | awk '{print $1, $2}' > self_labels.txt

rm -f self_atlas_1mm+tlrc.* self_coords_undump.txt

echo "Done."
echo "  self_atlas_1mm.nii.gz    -> $OUTPUT_DIR"
echo "  self_atlas_native.nii.gz -> $OUTPUT_DIR"
echo "  self_labels.txt          -> $OUTPUT_DIR"
echo "  self_coordinates.txt     -> $OUTPUT_DIR"
