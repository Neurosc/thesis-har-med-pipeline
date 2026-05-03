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

**Raw BIDS data:**
har_med_codes/
├── 00_QC/              # Quality control (motion, DVARS visualization)
├── 01_denoising/       # Nuisance regression, censoring, bandpass
├── 02_acw/             # Autocorrelation window calculation
├── 03_analysis/        # Self vs non-self comparisons
├── utils/              # Shared helper functions
├── figures/            # Output figures (gitignored)
├── results/            # Output data (gitignored)
└── CLAUDE.md           # This file

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
