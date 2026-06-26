# _archive — retired material (nothing deleted, only moved)

Everything here was moved out of the active pipeline when it was narrowed to:
**NoGSR + Qin 4mm spheres (6 layers) + ACW/SampEn, 35 subjects.**
See the repo-root `CLAUDE.md` for the active design. To restore anything, move it back.

| Folder | What it is | Why retired |
|--------|------------|-------------|
| `parcels_approach/` | Glasser/Keskin **parcels** atlas: extraction scripts, `timeseries_parcels/`, parcel atlases, parcels ACW results, old keskin ACW scripts | Replaced by Qin 4mm spheres |
| `sampen_parcels_stale/` | Old parcels-based SampEn results (5 layers) | Replaced by 6-layer sphere SampEn (`03_acw_analysis/results/sampen_qinspheres/`) |
| `excluded_subjects_runs/` | Extracted CSVs + ACW jld2 for sub-06/08/26/36 | The 4 FD-censoring-excluded subjects; on-disk sample is now exactly 35 |
| `literature/` | Reference PDFs (`ayahuasca/`, `int_meditation_nonduality/`) | Not code |
| `slides/` | `.pptx` decks | Not code |
| `reference_data/` | HCP_MMP1 surfaces, MNI template, g1 files, Self2Glasser, old tSNR exclusion list | Inputs for retired atlases / unused |
| `old_preexisting/` | The former top-level `_old/` | Pre-existing archive |
| `misc/` | Empty `conda` stray file, the mistaken `figures_thesis/` dir | Clutter |

**Note:** git-ignored binaries (`*.nii.gz`, the MNI template, `HCP_MMP1_atlas/`) are not
carried by git, so on the server they stay in their original locations — move them by hand
there if you want a full match.
