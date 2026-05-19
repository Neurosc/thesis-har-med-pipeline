# Project: fMRI Analysis Pipeline (DMT-MED Dataset)

## Goal
Resting-state fMRI denoising and analysis pipeline. The pipeline implements
post-processing of fMRIPrep outputs following Goldberg et al. (2024):
nuisance regression (WM, CSF, 6 motion + derivatives), optional GSR,
high-motion frame censoring (FD > 0.3 mm) with Lomb-Scargle interpolation,
and bandpass filtering (0.01-0.1 Hz).

**Final analysis:** Autocorrelation Window (ACW) calculation in self vs non-self
brain regions. (Analysis details to be specified later.)

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
├── QC/                                 # all quality-control work (see "QC organization" section)
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
- **No respiratory bandstop filtering** — empirically tested via Power 2019 PSD inspection (see `QC/01_motion_qc/scripts/respiratory_spectrum_check.py`); no respiratory peaks observed in trans_y (phase-encode parameter), so filter is unnecessary
- Rationale for not using Lynch/Goldberg 4-TR window: no published guidance exists for non-HCP TRs; all major pipelines use 1-TR

### Frame censoring
- Threshold: **FD > 0.3 mm** (following Goldberg et al. 2024 recommendation for ACW)
- Rationale: ACW is particularly sensitive to residual tissue-specific effects of motion spikes; supervisor (Ben) recommended sticking to 0.3 mm despite data loss

### Subject exclusion
- Criterion: any run with >50% frames censored at FD > 0.3 mm → entire subject excluded (within-subject design)
- **Excluded subjects: sub-06, sub-08, sub-12, sub-26, sub-36** (5 subjects)
- **Final sample: 35 subjects × 2 sessions = 70 runs**
- See `QC/01_motion_qc/results/excluded_subjects.tsv` for details
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
- Script: `QC/02_denoising_qc/scripts/qc_dvars_comparison.py`
- Figures: `QC/02_denoising_qc/figures/sub-XX_ses-YY_dvars_comparison.png/.pdf`
- Logs: `QC/02_denoising_qc/results/sub-XX_ses-YY_dvars_comparison.txt`
- Both sub-01 ses-01 (low motion) and sub-21 ses-01 (high motion) figures retained as evidence

### ROI Atlas Generation
- Self-referential atlas: 37 ROIs (Qin et al. 2020, Interoception/Exteroception/Cognition), 4mm-radius spheres
  - Built at 1mm MNI resolution, resampled to native BOLD grid (1.72×1.72×2.00 mm, MNI152NLin2009cAsym)
- Nonself atlas: Glasser parcellation (~327 ROIs), 4mm-radius spheres
  - Original coords at `/home/jkokino/meditation_project/templates/nonself_roi/glasser_coordinates_nonself_327_original.txt`
  - After overlap check, clean coords saved to `02_timeseries_extraction/results/atlases/glasser_coordinates_nonself_clean.txt`
- Scripts in `02_timeseries_extraction/scripts/`:
  - `01_create_self_atlas.sh` — builds self atlas at 1mm + native BOLD resolution
  - `02_create_nonself_atlas.sh` — builds nonself atlas at 1mm + native BOLD resolution
  - `03_check_nonself_overlap.py` — removes nonself ROIs overlapping with self atlas (radius 4mm)
- Atlas outputs: `02_timeseries_extraction/results/atlases/`
- Requires AFNI (`3dUndump`, `3dAFNItoNIFTI`, `3dresample`) + conda env `fmri` for Python script
- MNI 1mm template: `/home/jkokino/meditation_project/templates/MNI/mni_icbm152_1mm.nii`

### Self timeseries extraction
- Script: `02_timeseries_extraction/scripts/05_extract_self_timeseries.py`
- Same scope as nonself: 35 subjects × 2 sessions × 3 versions = 210 runs
- Atlas: `self_atlas_native.nii.gz` (37 ROIs, Qin et al. 2020)
- Output: `02_timeseries_extraction/results/timeseries_self/{version}/sub-XX_ses-YY_self_timeseries.csv`
- Log: `02_timeseries_extraction/results/timeseries_self/_extraction_log.tsv`
- Idempotent: skips runs where output CSV already exists

### Nonself timeseries extraction
- Script: `02_timeseries_extraction/scripts/04_extract_nonself_timeseries.py`
- Subjects: 35 included × 2 sessions = 70 runs
- 3 BOLD versions extracted per run: raw fMRIPrep, denoisedNoGSR, denoisedGSR (210 total)
- Atlas: `nonself_atlas_native.nii.gz` (Glasser-derived, ~290-310 ROIs after self-overlap removal)
- Output: CSV per (subject, session, version) — rows=timepoints, cols=ROI numbers
- Output directory: `02_timeseries_extraction/results/timeseries_nonself/{version}/`
- Log: `02_timeseries_extraction/results/timeseries_nonself/_extraction_log.tsv`
- Idempotent: skips runs where output CSV already exists

### ACW Computation
- Script: `03_acw_analysis/scripts/01_compute_acw.jl`
- TR = 1.8 s, n_lags = 100, dummy volumes discarded = 6
- ACW types: tau (exponential decay), auc (area under curve)
- Processes self + nonself × raw + denoisedNoGSR = 4 combinations × 70 subject-sessions = 280 JLD2 outputs
- Output: `03_acw_analysis/results/acw/{atlas}/{version}/{subject}_{session}.jld2`

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
- Script: `QC/03_acw_qc/scripts/04_tsnr_check.py` (runs on server)
- Output: `QC/03_acw_qc/results/tsnr/tsnr_per_roi.tsv`
- Spike ROI list (Glasser ROI numbers): `QC/03_acw_qc/results/tsnr/spike_rois.tsv`
- Action depending on results: if spike ROIs have median tSNR < 30, exclude them from downstream analysis with anatomical justification

### Final analysis (planned)
- Autocorrelation Window (ACW) calculation in self vs non-self regions
- Comparison: pre-retreat (ses-01) vs post-retreat (ses-02)
- DMT vs non-DMT subgroup comparison

## Thesis Figure Output

- All thesis-bound figures go to: `QC/01_motion_qc/thesis_figures/supplementary/`
- Do NOT use `QC/01_motion_qc/figures_thesis/` — that directory was created by mistake
- Active excluded-subjects panel script: `QC/01_motion_qc/scripts/excluded_subjects_panel.py`
- See `QC/01_motion_qc/thesis_figures/README.md` for the full figure index

### QC organization
- `QC/01_motion_qc/` — motion QC, framewise displacement, subject exclusion, thesis figures
- `QC/02_denoising_qc/` — DVARS-based denoising validation
- `QC/03_acw_qc/` — tau distribution checks, tSNR-based ROI filtering, dropout investigation
- `QC/troubleshooting/` — non-thesis experimental scripts and outputs

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
