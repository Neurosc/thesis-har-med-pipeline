# Project: fMRI Analysis Pipeline (DMT-MED Dataset)

## Goal
Resting-state fMRI denoising and analysis pipeline. The pipeline implements
post-processing of fMRIPrep outputs following Goldberg et al. (2024):
nuisance regression (WM, CSF, 6 motion + derivatives), optional GSR,
high-motion frame censoring (FD > 0.3 mm) with Lomb-Scargle interpolation,
and bandpass filtering (0.01-0.1 Hz).

**Final analysis:** Autocorrelation Window (ACW) calculation in self vs non-self
brain regions. (Analysis details to be specified later.)

## Configuration system

All analysis scripts read the active analysis configuration from `config.toml` at
repo root. Edit the two values there to switch which (denoising × atlas) combination
is being analyzed:

```toml
[active]
denoising_method = "NoGSR"   # NoGSR | GSR
atlas_method = "parcels"     # spheres | parcels
```

Each script imports a small loader (`utils/config_loader.{py,jl,R}`), reads the
active config, and writes outputs into folders tagged with the combination (e.g.,
`parcels_NoGSR/`). This keeps outputs from different runs separated.

- Python loader: `utils/config_loader.py` — `from utils.config_loader import load_config, tag`
- Julia loader:  `utils/config_loader.jl` — `include(...)`, then call `tag()`
- R loader:      `utils/config_loader.R`  — `source(...)`, then call `tag()`

Scripts updated to use config tags: `03_acw_analysis/scripts/01_compute_acw.jl`,
`03_acw_analysis/scripts/02_boxplot_session_self_nonself.jl`.

Scripts not yet updated (pending confirmation of scope): timeseries extraction
(`scripts/spheres/04_extract_nonself_timeseries.py`,
`scripts/spheres/05_extract_self_timeseries.py`) and statistics R scripts
(`04_statistics/scripts/05a_per_category_glasser.R`, `05b_per_category_sphere.R`).

## Data Locations

### Remote research server (where fMRIPrep outputs live)
- Host: `10.156.156.21`
- Username: `jkokino`
- VPN required: Global Protect
- SSH access: `ssh jkokino@10.156.156.21`
- Project root on server: `/BICNAS2/group-northoff/jkokino/`

### Data paths on server

**Repository structure:**
har_med_codes/
├── CLAUDE.md
├── README.md
├── utils/                              # subject_filter, motion_qc, thesis_style
├── 99_QC/                                 # all quality-control work (see "QC organization" section)
│   ├── 01_motion_qc/                   # motion QC, FD, subject exclusion, thesis figures
│   │   ├── scripts/
│   │   ├── results/
│   │   ├── figures/
│   │   └── thesis_figures/
│   ├── 02_denoising_qc/                # DVARS-based denoising validation
│   │   ├── scripts/
│   │   ├── results/
│   │   └── figures/
│   ├── 03_acw_qc/                      # tSNR checks, tau distribution, spike investigation
│   │   ├── scripts/
│   │   ├── results/
│   │   └── figures/
│   └── troubleshooting/                # non-thesis experimental scripts and outputs
├── 01_preprocessing/
│   └── 02_denoising/
│       ├── scripts/                    # denoise_core.py, denoise_batch.py, denoise_single_subject.py
│       ├── results/                    # denoised NIfTIs + batch log
│       └── figures/
├── 02_timeseries_extraction/
├── 03_acw_analysis/
├── 04_statistics/
└── _old/                               # archived material

## Software Environment

- Python: 3.x (via Miniconda)
- Conda env name on server: `fmri`
- Activate env on server: `conda activate fmri`
- Allowed packages: pandas, numpy, scipy, matplotlib, nibabel, nilearn

## Dataset Specifics

- Dataset name: DMT-MED
- Number of subjects: 40
- Sessions per subject: 2 (ses-01, ses-02)
- Task name: `rest`
- TR: 1.8 s
- Number of volumes per run: 240
- Total scan duration: 432 s (7.2 minutes)
- BOLD voxel grid (MNI space): (113, 134, 97)
- Output space: MNI152NLin2009cAsym
- fMRIPrep version: 23.0.2
- BIDS version: 1.4.0

## Subject Naming Convention

`sub-XX_ses-YY` (zero-padded, two digits)

Examples:
- `sub-01_ses-01`
- `sub-23_ses-02`

## Pipeline Reference

Goldberg, A., Rosario, I., Power, J., Horga, G., & Wengler, K. (2024).
Strategies for motion- and respiration-robust estimation of fMRI intrinsic
neural timescales. *Imaging Neuroscience*, 2.

## Decisions Made So Far

### Dataset specifics (corrected)
- TR is **1.8 s** (NOT 1.5 s — original CLAUDE.md was wrong, corrected after reading BOLD JSON header)
- Sampling frequency: 0.556 Hz, Nyquist: 0.278 Hz
- Phase-encoding direction: j (= Y axis, so trans_y is the phase-encode parameter)
- Volumes per run: 240 → total scan duration: 432 s = 7.2 min

### FD computation
- Standard Power et al. 2012 FD as computed by fMRIPrep
- 1-TR backward differences
- 50 mm sphere for rotation conversion
- **No respiratory bandstop filtering** — empirically tested via Power 2019 PSD inspection (see `99_QC/01_motion_qc/scripts/respiratory_spectrum_check.py`); no respiratory peaks observed in trans_y (phase-encode parameter), so filter is unnecessary
- Rationale for not using Lynch/Goldberg 4-TR window: no published guidance exists for non-HCP TRs; all major pipelines use 1-TR

### Frame censoring
- Threshold: **FD > 0.3 mm** (following Goldberg et al. 2024 recommendation for ACW)
- Rationale: ACW is particularly sensitive to residual tissue-specific effects of motion spikes; supervisor (Ben) recommended sticking to 0.3 mm despite data loss

### Subject exclusion
- Criterion: any run with >50% frames censored at FD > 0.3 mm → entire subject excluded (within-subject design)
- **Excluded subjects: sub-06, sub-08, sub-12, sub-26, sub-36** (5 subjects)
- **Final sample: 35 subjects × 2 sessions = 70 runs**
- See `99_QC/01_motion_qc/results/excluded_subjects.tsv` for details
- Use `utils/subject_filter.py:get_included_subjects()` to filter subject lists in downstream scripts

### Denoising pipeline (implemented)
- Volumetric only (MNI152NLin2009cAsym)
- Single GLM regression with: WM mean, CSF mean, 6 motion + 6 derivatives, spike regressors for FD>0.3 mm frames
- Two versions: +GSR+censor and -GSR+censor (frame censoring always on)
- Lomb-Scargle interpolation of censored frames after regression; faithful port of CBIG_preproc_censor.m (Jingwei Li, Yeo Lab)
- Bandpass 0.01–0.1 Hz via frequency-domain masking inside LS (no Butterworth)
- Output: `01_preprocessing/02_denoising/results/sub-XX_ses-YY_task-rest_desc-denoisedGSR_bold.nii.gz` and `desc-denoisedNoGSR_bold.nii.gz`
- Shared core logic: `01_preprocessing/02_denoising/scripts/denoise_core.py` (imported by both scripts below)
- Single-subject test script: `01_preprocessing/02_denoising/scripts/denoise_single_subject.py` (sub-01 ses-01)
- Batch script: `01_preprocessing/02_denoising/scripts/denoise_batch.py` — processes all 35 included subjects × 2 sessions = 70 runs; skips existing outputs; appends per-run log to `01_preprocessing/02_denoising/results/_batch_log.tsv`

### Denoising QC
- Method: DVARS pre/post comparison + FD-DVARS coupling
- Script: `99_QC/02_denoising_qc/scripts/qc_dvars_comparison.py`
- Figures: `99_QC/02_denoising_qc/figures/sub-XX_ses-YY_dvars_comparison.png/.pdf`
- Logs: `99_QC/02_denoising_qc/results/sub-XX_ses-YY_dvars_comparison.txt`
- Both sub-01 ses-01 (low motion) and sub-21 ses-01 (high motion) figures retained as evidence

### ROI Atlas Generation (OLD — 4mm sphere approach, superseded)
- Self-referential atlas: 37 ROIs (Qin et al. 2020, Interoception/Exteroception/Cognition), 4mm-radius spheres
  - Built at 1mm MNI resolution, resampled to native BOLD grid (1.72×1.72×2.00 mm, MNI152NLin2009cAsym)
- Nonself atlas: Glasser parcellation (~327 ROIs), 4mm-radius spheres
  - Original coords at `/home/jkokino/meditation_project/templates/nonself_roi/glasser_coordinates_nonself_327_original.txt`
  - After overlap check, clean coords saved to `02_timeseries_extraction/results/atlases/glasser_coordinates_nonself_clean.txt`
- Scripts in `02_timeseries_extraction/scripts/spheres/`:
  - `01_create_self_atlas.sh` — builds self atlas at 1mm + native BOLD resolution
  - `02_create_nonself_atlas.sh` — builds nonself atlas at 1mm + native BOLD resolution
  - `03_check_nonself_overlap.py` — removes nonself ROIs overlapping with self atlas (radius 4mm)
- Atlas outputs: `02_timeseries_extraction/results/atlases/`
- Requires AFNI (`3dUndump`, `3dAFNItoNIFTI`, `3dresample`) + conda env `fmri` for Python script
- MNI 1mm template: `/home/jkokino/meditation_project/templates/MNI/mni_icbm152_1mm.nii`
- **NOTE: Replaced by Glasser-based atlas below. Old outputs and all derived timeseries/ACW are obsolete.**

### Glasser-based atlas (methodological correction, replaces 4mm sphere approach)
- Self atlas: Glasser MMP1.0 parcels whose name appears in Keskin et al. 2025 self-referential table
- 4 self NIfTI files (combined + per-category):
  - Combined self (all 3 categories merged): `glasser_self_atlas_{1mm,native}.nii.gz`
  - Interoceptive only: `glasser_self_interoceptive_{1mm,native}.nii.gz`
  - Exteroceptive only: `glasser_self_exteroceptive_{1mm,native}.nii.gz`
  - Mental Self only: `glasser_self_mental_{1mm,native}.nii.gz`
- Each category has its own atlas file at both 1mm MNI and native BOLD grid
- Cross-category overlapping parcels (rare) are included in all relevant per-category atlases
- Nonself atlas: all remaining Glasser parcels (1 file at 1mm + native): `glasser_nonself_atlas_{1mm,native}.nii.gz`
- Subcortical coordinates (thalamus, etc.) dropped — Glasser is cortical-only
- Multi-parcel Keskin focus_point entries treated as multiple separate parcels
- Script: `02_timeseries_extraction/scripts/parcels/01_create_glasser_self_nonself.py`
- Input atlas: `glasser360MNI.nii.gz` (1mm MNI, 1-360 indexing, matches Cole-Anticevic label key)
- Input data dir: `02_timeseries_extraction/data/` (user places files manually)
- Metadata: `glasser_self_metadata.tsv` (parcel_id, parcel_name, hemisphere, categories, keskin coords)
- NOTE: Old 4mm sphere atlases (`self_atlas_*.nii.gz`, `nonself_atlas_*.nii.gz`) and all derived timeseries/ACW outputs are now obsolete and must be regenerated.

### Timeseries extraction (two methods, kept in parallel for now)
- **Sphere method** (legacy, Qin 2020 coordinates + 4mm spheres):
  - Scripts: `02_timeseries_extraction/scripts/spheres/`
  - Atlas outputs: `02_timeseries_extraction/results/atlases/self_atlas_*.nii.gz`, `nonself_atlas_*.nii.gz`
- **Parcel method** (Keskin et al. 2025, direct Glasser MMP1.0 parcels):
  - Scripts: `02_timeseries_extraction/scripts/parcels/`
  - Atlas outputs: `02_timeseries_extraction/results/atlases/glasser_*_atlas_*.nii.gz`
- Extraction outputs separated via `config.toml` tag (`spheres_NoGSR`, `parcels_NoGSR`, etc.)

### Parcel-based timeseries extraction (Keskin method)
- Script: `02_timeseries_extraction/scripts/parcels/02_extract_parcel_timeseries.py`
- Runs on server (uses NIfTI files only available there)
- 5 atlases × 3 BOLD versions × 70 subject-sessions = 1,050 CSVs
- Output: `02_timeseries_extraction/results/timeseries_parcels/{atlas}/{version}/{subject}_{session}_{atlas}_parcel_timeseries.csv`
- Per-parcel mean BOLD per timepoint, 240 timepoints (or 234 after dummy removal if applied downstream)
- Optimization: BOLD loaded once per (sub, ses, version), all 5 atlases extracted from same in-memory array → 210 BOLD loads instead of 1,050
- Log: `02_timeseries_extraction/results/timeseries_parcels/_extraction_log.tsv`

### ACW computation (multi-method)
- Script: `03_acw_analysis/scripts/01_compute_acw.jl`
- Reads `config.toml` → switches between sphere and parcel input directories automatically
- TR = 1.8 s, n_lags = 100, dummy volumes discarded = 6; ACW types: tau, auc
- Output: `03_acw_analysis/results/{atlas_method}_{denoising_method}/{atlas_name}/{subject}_{session}.jld2`
- Per atlas: self + nonself (spheres) OR self + nonself + interoceptive + exteroceptive + mental (parcels)
- JLD2 variables: `acw_results` (ACWResults struct), `parcel_ids` (column names from CSV)

### ACW Boxplot Analysis
- Script: `03_acw_analysis/scripts/02_boxplot_session_self_nonself.jl`
- Compares: ses-01 Self vs ses-01 Nonself, ses-02 Self vs ses-02 Nonself
- Test: paired Wilcoxon signed-rank
- Metric: tau (intrinsic timescale)
- Versions: raw + denoisedNoGSR (separate figures)
- Aggregation: median ACW across ROIs per (subject, session, atlas)

### τ spike investigation (τ ≈ 1 s)
- Hypothesis: 46 nonself Glasser ROIs in temporal poles, OFC, and frontal poles show τ ≈ 1 s due to susceptibility dropout (sinuses, petrous bone)
- Diagnostic: per-ROI tSNR across 70 runs (raw fMRIPrep BOLD)
- Script: `99_QC/03_acw_qc/scripts/04_tsnr_check.py` (runs on server)
- Output: `99_QC/03_acw_qc/results/tsnr/tsnr_per_roi.tsv`
- Spike ROI list (Glasser ROI numbers): `99_QC/03_acw_qc/results/tsnr/spike_rois.tsv`
- Action depending on results: if spike ROIs have median tSNR < 30, exclude them from downstream analysis with anatomical justification

### Final analysis (planned)
- Autocorrelation Window (ACW) calculation in self vs non-self regions
- Comparison: pre-retreat (ses-01) vs post-retreat (ses-02)
- DMT vs non-DMT subgroup comparison

## Thesis Figure Output

- All thesis-bound figures go to: `99_QC/01_motion_qc/thesis_figures/supplementary/`
- Do NOT use `99_QC/01_motion_qc/figures_thesis/` — that directory was created by mistake
- Active excluded-subjects panel script: `99_QC/01_motion_qc/scripts/excluded_subjects_panel.py`
- See `99_QC/01_motion_qc/thesis_figures/README.md` for the full figure index

### QC organization
- `99_QC/01_motion_qc/` — motion QC, framewise displacement, subject exclusion, thesis figures
- `99_QC/02_denoising_qc/` — DVARS-based denoising validation
- `99_QC/03_acw_qc/` — tau distribution checks, tSNR-based ROI filtering, dropout investigation
  - `99_QC/03_acw_qc/scripts/05_roi_placement_viewer.py` — atlas placement QC: interactive marker HTML (hover = ROI name) + per-ROI ortho PDF montage
  - Outputs: `*_markers.html` (view_markers, open in browser), `*_montage.pdf` (37/316 pages), `*_atlas_summary.tsv` (centroid + voxel counts)
- `99_QC/troubleshooting/` — non-thesis experimental scripts and outputs

## Things NOT to redo
- Do not reapply respiratory filter (empirically tested, not needed)
- Do not switch to 4-TR backward difference (no justification for our TR)
- Do not include excluded subjects in any downstream analysis
- Do not change FD threshold without consulting supervisor

## Conventions

- Figures saved as 300 dpi PNG
- Numerical outputs saved as `.npy` or `.tsv`
- Subject-level outputs include `sub-XX_ses-YY` in filename
- fMRIPrep outputs are READ-ONLY — never modify
- Print summary statistics to console for every script

## What NOT to do

- Do NOT install large pipeline libraries (XCP-D, fMRIPost-AROMA)
- Do NOT use any GUI tools or interactive plots — server is headless
  - Always save plots to files with `plt.savefig()`, never use `plt.show()`
- Do NOT push to feature branches without explicit request
- Do NOT modify fMRIPrep outputs
- Do NOT load real data on Windows during development
- Do NOT use the AROMA-denoised BOLD file (`desc-smoothAROMAnonaggr_bold.nii.gz`)
