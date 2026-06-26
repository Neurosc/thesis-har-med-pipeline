# Project: fMRI ACW / SampEn Pipeline (DMT-MED Dataset)

## What this pipeline computes

Two intrinsic-dynamics metrics from resting-state fMRI, compared across
**six brain layers** (Qin et al. 2020 self + sensory/motor nonself), in a
within-subject pre/post design:

- **ACW** — Autocorrelation Window (AUC and τ), the intrinsic neural timescale.
- **SampEn** — Sample Entropy, signal irregularity.

**Dataset:** DMT-MED, **35 subjects × 2 sessions = 70 runs**, NoGSR denoising only.

This document covers the **calculation part only** (denoise → extract → ACW + SampEn).
The statistics part is **deferred** and will be specified later (see `04_statistics`,
which still holds the *previous* parcels analysis and is not part of the current flow).

---

## Scope of the current design (read first)

The pipeline was narrowed to a single, fixed configuration. Everything outside it
was moved to **`_archive/`** (nothing was deleted — see "Archive" below).

| Dimension      | Active                                   | Retired → `_archive/`             |
|----------------|------------------------------------------|-----------------------------------|
| Denoising      | **NoGSR only**                           | GSR                               |
| Atlas / ROIs   | **Qin et al. 2020, 4mm spheres**         | Glasser MMP1.0 parcels (Keskin)   |
| Layers         | **6** (see below)                        | self/nonself parcel layers        |
| Metrics        | **ACW + SampEn**                         | —                                 |
| Sample         | **35 subjects** (`utils/subject_filter`) | sub-06/08/12/26/36 runs           |

`config.toml` is pinned to `denoising_method = "NoGSR"`, `atlas_method = "qinspheres"`.
Do not change it.

### The six layers

Self layers — **Qin et al. 2020** MNI coordinates, 4mm-radius spheres (37 ROIs total):

| Layer    | Folder name | ROIs |
|----------|-------------|------|
| Interoception | `intero` | 11 |
| Exteroception | `extero` | 14 |
| Cognition     | `mental` | 12 |

Nonself layers — **CAB-NP / Glasser** network parcels, centroids → 4mm spheres,
self-overlapping parcels excluded:

| Layer    | Folder name | Source networks      |
|----------|-------------|----------------------|
| Visual   | `visual`    | Visual1 + Visual2    |
| Motor    | `motor`     | Somatomotor          |
| Auditory | `auditory`  | Auditory             |

ROI definitions and the nonself-exclusion list are hardcoded in the extraction
script (`02_timeseries_extraction/scripts/01_extract_sphere_timeseries.py`).

---

## Pipeline flow (server vs. local)

```
                        ┌──────────────── SERVER (needs NIfTIs) ───────────────┐   ┌──── LOCAL (CSVs onward) ────┐
fMRIPrep BOLD ──▶ 01 denoise (NoGSR) ──▶ 02 extract 6-layer sphere timeseries ──▶ 03 ACW (Julia) + SampEn (Python)
```

- **Server steps** (raw/denoised NIfTIs only exist there):
  1. `01_denoising/` — three NoGSR pipelines → `results/{detrend,glm,maximal}/…desc-<pipeline>_bold.nii.gz`.
  2. `02_timeseries_extraction/scripts/01_extract_sphere_timeseries.py` — writes the
     6-layer CSVs to `02_timeseries_extraction/results/qinspheres/{layer}/`.
- **Local steps** (timeseries CSVs are committed to the repo, so these run on Windows):
  3a. `03_acw_analysis/scripts/01_compute_acw.jl` — ACW → `results/qinspheres_NoGSR/`.
  3b. `03_acw_analysis/scripts/02_compute_sampen.py` — SampEn → `results/sampen_qinspheres/`.

### Run order
```
# Server (conda env: fmri)
python 02_timeseries_extraction/scripts/01_extract_sphere_timeseries.py
# Local (or server)
julia  03_acw_analysis/scripts/01_compute_acw.jl          # config.toml = NoGSR + qinspheres
python 03_acw_analysis/scripts/02_compute_sampen.py
```

---

## Repository structure

```
thesis-har-med-pipeline/
├── CLAUDE.md  README.md  config.toml  participants.tsv  .gitignore
├── utils/                         subject_filter, motion_qc, thesis_style, config_loader.{py,jl,R}
├── 01_denoising/                 denoise_pipelines.py + 00_test_one_subject/01_denoise_all (3 NoGSR pipelines)
├── 02_timeseries_extraction/
│   ├── data/                      glasser360MNI.nii.gz, CAB-NP label key  (extraction inputs)
│   ├── scripts/01_extract_sphere_timeseries.py
│   └── results/qinspheres/{intero,extero,mental,visual,motor,auditory}/   70 CSVs each
├── 03_acw_analysis/
│   ├── scripts/01_compute_acw.jl          (ACW)
│   ├── scripts/02_compute_sampen.py       (SampEn)
│   └── results/qinspheres_NoGSR/{layer}/  + results/sampen_qinspheres/
├── 99_QC/                         01_motion_qc, 02_denoising_qc, 03_acw_qc, troubleshooting
├── 04_statistics/                 DEFERRED — holds the previous parcels analysis, not the current flow
└── _archive/                      everything retired (see below)
```

### Archive (`_archive/` — nothing deleted, only moved)
- `parcels_approach/` — Glasser/Keskin parcels: extraction scripts, `timeseries_parcels/`,
  parcel atlases, parcels ACW results, old keskin ACW scripts.
- `sampen_parcels_stale/` — the old parcels-based SampEn results (5 layers; superseded by
  the new 6-layer sphere SampEn).
- `excluded_subjects_runs/` — extracted CSVs + ACW jld2 for sub-06/08/26/36 (the 4
  censoring-excluded subjects). Removed so the on-disk sample is exactly 35.
- `literature/` — `ayahuasca/`, `int_meditation_nonduality/` reference PDFs.
- `slides/` — `.pptx` decks.   `reference_data/` — HCP_MMP1 surfaces, MNI template, g1 files,
  Self2Glasser, old tSNR list.   `old_preexisting/` — the former `_old/`.

`_archive/` is committed to git so the same relocation reproduces on the server when you
pull. **Caveat:** git-ignored binaries (`*.nii.gz` atlases, the MNI template,
`HCP_MMP1_atlas/`) are *not* carried by git — move those by hand on the server if needed.

---

## Dataset specifics

- Dataset: DMT-MED · Task: `rest` · Sessions: ses-01 (pre), ses-02 (post)
- Sample: 35 subjects (excluded by FD>0.3mm censoring: sub-06, sub-08, sub-12, sub-26, sub-36)
- **TR = 1.8 s** (corrected from an earlier wrong 1.5 s) · 240 volumes · 432 s (7.2 min)
- Sampling 0.556 Hz, Nyquist 0.278 Hz · Phase-encode dir j (trans_y)
- Output space MNI152NLin2009cAsym · BOLD grid (113,134,97) · fMRIPrep 23.0.2 · BIDS 1.4.0
- Naming: `sub-XX_ses-YY` (zero-padded)
- Drug groups (verum/placebo) in `participants.tsv` (`condition` column)

---

## Methodology decisions (still in force)

### Denoising — three parallel pipelines (robustness design)
To show the final results are robust to denoising choice, the same data is denoised
three independent ways (all volumetric MNI, all **NoGSR**). One parameterized core
(`denoise_pipelines.PIPELINE_PRESETS`) drives all three; pick with `--pipeline`:

| Pipeline | Detrend | WM+CSF | Motion 6+6 | FD>0.3 censor | LS interp | Bandpass |
|----------|:------:|:------:|:----------:|:-------------:|:---------:|:--------:|
| `detrend` (Keskin 2025) | polort 1 | – | – | – | – | – |
| `glm` | polort 1 | ✓ | ✓ | – | – | – |
| `maximal` (Goldberg 2024) | polort 1 | ✓ | ✓ | ✓ | ✓ | 0.01–0.1 Hz |

- **Common sample = n=39** (all subjects minus `sub-12`): keeps the 4 high-motion
  subjects (06/08/26/36) so only denoising varies. Uses
  `utils/subject_filter.py:get_pipeline_subjects()` — the legacy `get_included_subjects()`
  (n=35) is **untouched** and still drives ACW/SampEn/stats. Motion for the 4 kept
  subjects is a **stats-stage** concern (carry mean FD / percent-censored as covariates,
  and/or a drop-those-4 sensitivity analysis) — NOT handled at denoising.
- `maximal` is identical denoising to the retired flat `desc-denoisedNoGSR` outputs;
  re-running only extends the sample to n=39 and writes the new folder/naming.
- LS interpolation = port of CBIG_preproc_censor.m; bandpass via frequency-domain mask
  inside LS (no Butterworth).
- Outputs: `results/{detrend,glm,maximal}/sub-XX_ses-YY_task-rest_desc-<pipeline>_bold.nii.gz`
  + per-pipeline `_batch_log.tsv`. Run: `01_denoise_all.py --pipeline {detrend|glm|maximal}`
  (server); test one run with `00_test_one_subject.py --pipeline <p>`.
- **FPC/FPCsq are NOT denoising regressors.** An earlier spec listed "FPC+FPCsq" for
  the maximal pipeline; they are actually subject×session motion-quality covariates
  (`pcf_diff`, `pcf_sq_diff` from percent-censored-frames) for the statistics-stage
  residualization model — one scalar per run, so they cannot be per-timepoint GLM
  regressors. Deferred to the (later) statistics step.
- Legacy flat `results/*_desc-denoisedGSR/NoGSR_bold.nii.gz`: obsolete; flagged for
  archival to `_archive/` later (not deleted in the denoising task).

### FD & censoring
- Standard Power 2012 FD (fMRIPrep), 1-TR backward differences, 50mm rotation sphere.
- **No respiratory bandstop** — empirically tested (no respiratory peak in trans_y).
- Censoring threshold **FD > 0.3 mm** (Goldberg 2024; supervisor-confirmed).

### Subject exclusion
- Criterion: any run with >50% frames censored at FD>0.3mm → whole subject excluded.
- Excluded: **sub-06, sub-08, sub-12, sub-26, sub-36** → final **35 × 2 = 70**.
- Single source of truth: `utils/subject_filter.py:get_included_subjects()`.

### ACW (`01_compute_acw.jl`, Julia + IntrinsicTimescales.jl)
- TR 1.8 s, n_lags 100, dummy volumes discarded = 6, types `[:auc, :tau]`.
- Reads `config.toml`; for qinspheres loops the 6 layer folders.
- Output: `results/qinspheres_NoGSR/{layer}/{sub}_{ses}.jld2`
  (vars: `acw_results` [1]=AUC [2]=τ, `parcel_ids`).

### SampEn (`02_compute_sampen.py`, Python + EntropyHub)
- Per run × layer: drop 6 dummy volumes → linear detrend → SampEn (m=1, τ=1, r=0.3
  absolute, log base 2; Keskin/Northoff conventions).
- Output: per-run `results/sampen_qinspheres/{sub}_{ses}_{layer}_sampen.csv`
  + `results/sampen_qinspheres/sampen_long_qinspheres.csv` (self-contained;
  no coupling to `04_statistics`).

---

## Software environment

- **Server:** Python 3.x (Miniconda env `fmri`: pandas, numpy, scipy, matplotlib,
  nibabel, nilearn, EntropyHub). `conda activate fmri`. Headless — always `plt.savefig()`.
- **Local (Windows):** Julia 1.12 (IntrinsicTimescales.jl), R 4.6.0
  (`C:\Program Files\R\R-4.6.0\bin\Rscript.exe`, not on PATH), Python + EntropyHub.
  The timeseries CSVs, ACW, and SampEn all run locally.
- **Server access:** `ssh jkokino@10.156.156.21` (GlobalProtect VPN required).
  Project root `/BICNAS2/group-northoff/jkokino/`; code repo
  `/BICNAS2/group-northoff/jkokino/codes/har_med_codes/`. fMRIPrep at
  `…/data/dmt_med/derivatives/fmriprep`.
- **Sync model:** local ⇄ server via GitHub (`origin = github.com/Neurosc/thesis-har-med-pipeline`).
  Changes are pushed to `master`; pull on the server. fMRIPrep outputs are READ-ONLY.

---

## QC (`99_QC/`)
- `01_motion_qc/` — FD, motion, subject exclusion, thesis figures
  (`thesis_figures/supplementary/` is the only valid thesis-figure dir).
- `02_denoising_qc/` — DVARS pre/post + FD-DVARS coupling.
- `03_acw_qc/` — τ distribution, tSNR / dropout checks, ROI-placement viewer.
- `troubleshooting/` — non-thesis experiments.
- **tSNR note:** the old 58-parcel tSNR exclusion list belonged to the *parcels* atlas
  (now in `_archive/`). A sphere-based tSNR check must be recomputed if ROI exclusion is
  wanted for this design.

---

## Statistics (`04_statistics/`) — DEFERRED
Not part of the current calculation flow. The folder still contains the **previous
parcels analysis** (parcels_NoGSR / parcels_GSR LMMs, figures) and a partial
`qinspheres/` attempt. When we resume statistics it will be rebuilt for the 6-layer
sphere design on top of `results/qinspheres_NoGSR/` + `results/sampen_qinspheres/`.
Do not assume any script in here matches the current pipeline yet.

---

## Things NOT to redo / change
- Do not reintroduce GSR or the Glasser-parcels atlas into the active flow.
- Do not reapply a respiratory filter (empirically unnecessary).
- Do not switch to a 4-TR backward difference (no justification for TR=1.8s).
- Do not include excluded subjects (06/08/12/26/36) in calculation or analysis.
- Do not change the FD threshold (0.3mm) without consulting the supervisor.
- Do not modify fMRIPrep outputs. Do not use the AROMA BOLD (`desc-smoothAROMAnonaggr`).
- Do not install heavy pipeline libraries (XCP-D, fMRIPost-AROMA) or use GUI/interactive plots.

## Conventions
- Figures 300 dpi PNG (`plt.savefig`, never `plt.show`). Numerical outputs `.tsv`/`.csv`.
- Subject-level filenames include `sub-XX_ses-YY`. Print summary stats per script.
