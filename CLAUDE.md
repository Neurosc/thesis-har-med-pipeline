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
  2. `02_timeseries_extraction/scripts/01_extract_sphere_timeseries.py` — extracts the
     6-layer CSVs from **all three** denoising pipelines (n=39 subjects) to
     `results/qinspheres/{detrend,glm,maximal}/{layer}/`. CSVs are committed to git.
- **Local steps** (timeseries CSVs are committed to the repo, so these run on Windows):
  3a. `03_intrinsic_neural_metrics/scripts/01_compute_acw.jl` — ACW (3 pipelines) → `results/acw/{pipeline}/{layer}/`.
  3b. `03_intrinsic_neural_metrics/scripts/02_compute_sampen.py` — SampEn (3 pipelines) → `results/sampen/{pipeline}/{layer}/`.

### Run order
```
# Server (conda env: fmri)
python 02_timeseries_extraction/scripts/01_extract_sphere_timeseries.py
# Local (or server)
julia  03_intrinsic_neural_metrics/scripts/01_compute_acw.jl          # 3 pipelines × 6 layers × 39 subj
python 03_intrinsic_neural_metrics/scripts/02_compute_sampen.py      # 3 pipelines × 6 layers × 39 subj
```

---

## Repository structure

```
thesis-har-med-pipeline/
├── CLAUDE.md  README.md  config.toml  participants.tsv  .gitignore
├── utils/                         subject_filter, motion_qc, thesis_style, config_loader.{py,jl,R}
├── 01_denoising/                 denoise_pipelines.py + 00_test_one_subject/01_denoise_all (3 NoGSR pipelines)
├── 02_timeseries_extraction/
│   ├── atlases/                   glasser360MNI + CAB-NP key (inputs) + self/nonself coords + metadata
│   ├── scripts/01_extract_sphere_timeseries.py   (extracts from all 3 denoising pipelines)
│   └── results/qinspheres/{detrend,glm,maximal}/{layer}/   39×2 CSVs each (committed)
├── 03_intrinsic_neural_metrics/
│   ├── scripts/01_compute_acw.jl          (ACW, 3 pipelines)
│   ├── scripts/02_compute_sampen.py       (SampEn, 3 pipelines)
│   └── results/acw/{pipeline}/{layer}/  + results/sampen/{pipeline}/{layer}/
├── 99_QC/                         01_motion_qc, 02_denoising_qc, 03_acw_qc, troubleshooting
├── 04_statistics/scripts/qinspheres/   ACW drug-effect analysis (maximal, n=39) + results/qinspheres/
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

Both ACW and SampEn loop the **three pipelines × six layers** at **n=39**
(matches the extraction; the 35-subject filter is applied later at stats). Input:
`02_timeseries_extraction/results/qinspheres/{pipeline}/{layer}/`.

### ACW (`01_compute_acw.jl`, Julia + IntrinsicTimescales.jl)
- TR 1.8 s, n_lags 100, dummy volumes discarded = 6, types `[:auc, :acw50]`.
- Self-contained (no longer reads `config.toml`); 39-subject list hardcoded to match
  `get_pipeline_subjects()`. Idempotent (skips existing JLD2).
- Output: `results/acw/{pipeline}/{layer}/{sub}_{ses}.jld2`
  (vars: `acw_results` [1]=AUC [2]=ACW-50, `parcel_ids`).

### SampEn (`02_compute_sampen.py`, Python + EntropyHub)
- Per run × layer: drop 6 dummy volumes → linear detrend → SampEn (m=1, τ=1, r=0.3
  absolute, log base 2; Keskin/Northoff conventions). Uses `get_pipeline_subjects()`.
- Output: per-run `results/sampen/{pipeline}/{layer}/{sub}_{ses}_{layer}_sampen.csv`
  + per-pipeline `results/sampen/{pipeline}/sampen_long_{pipeline}.csv` (self-contained;
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

## Statistics (`04_statistics/`) — ACW drug-effect analysis (qinspheres, maximal)
The active analysis is `scripts/qinspheres/`, on the **maximal** pipeline ACW-AUC at
**n=39** (the parcels_NoGSR / GSR / _old analyses are archived under
`_archive/statistics_parcels_and_old/`).

**Design:** per-layer (6 layers, kept separate) subject-level drug-effect test —
ΔAUC = mean post − pre per subject×layer; statistic = mean(verum Δ) − mean(placebo Δ);
Shapiro→Welch/Mann-Whitney + 10k permutation; BH-FDR across the 6 layers; ±2.5SD / 1.5×IQR
outlier-sensitivity variants. Both raw ΔAUC and **motion-quality-residualized** ΔAUC
(partialled for `pcf_diff`/`pcf_sq_diff`/`mean_fd_retained_diff` from
`99_QC/01_motion_qc/results/fd_covariates_wide_thresh03.csv`) are run for **all three
pipelines** — the residualization removes the **session-wise FD/motion** confound, which
matters most for detrend/glm (they don't censor, so motion stays in the data). Run
`qc_residualize_auc.R` + `statistics_resid.R` per pipeline.

Scripts are **pipeline × metric-parameterized**: R scripts take `<pipeline> <metric>`
trailing args (metric ∈ `auc` | `acw50`, default `maximal auc`). AUC stays at the
top-level `results/qinspheres/{pipeline}/`; other metrics nest at
`results/qinspheres/{metric}/{pipeline}/` (so AUC is undisturbed). `01_build_df.jl` emits
both metrics (acw_results[1]=AUC, [2]=ACW-50; value column named `auc` for both — a
carryover; figure labels are metric-correct). `safe_sw()` guards Shapiro-Wilk against the
degenerate (all-identical) groups that the discrete ACW-50 metric can produce.

**Non-positive AUC dropped:** `01_build_df.jl` removes ROIs with `auc ≤ 0` (AUC metric only).
Without bandpass, the broadband signal's autocorrelation collapses to/below zero, giving 0 or
negative "area" — not a valid timescale. This is a **detrend/glm** problem (≈1042 / 1665 ROIs
dropped, concentrated in motor); **maximal** (bandpassed) has **none** (0 dropped) and is
unaffected. Underlying ACW values are still in the JLD2; only the analysis CSV is filtered.

**Run order** (from repo root; Rscript at `C:\Program Files\R\R-4.6.0\bin\Rscript.exe`):
```
julia   04_statistics/scripts/qinspheres/01_build_df.jl                       # both metrics × pipelines
# then per (pipeline, metric), e.g. detrend acw50:
Rscript 04_statistics/scripts/qinspheres/qc_residualize_auc.R detrend acw50
Rscript 04_statistics/scripts/qinspheres/statistics.R        detrend acw50
Rscript 04_statistics/scripts/qinspheres/statistics_resid.R  detrend acw50
Rscript 04_statistics/scripts/qinspheres/04_figures.R        detrend acw50
Rscript 04_statistics/scripts/qinspheres/05_pipeline_comparison.R acw50       # per-metric comparison
```
`04_figures.R` ports the parcels raincloud/serif/bracket aesthetic to 6 layers.
`02_scatter.R` / `03_drug_effect_figures.R` are earlier simpler-style figures (AUC only,
superseded by `04_figures.R`).

**Results:**
- **AUC:** exteroception ΔAUC reduction under verum (after dropping non-positive AUC — see
  the build note below). Motion-residualized perm p = 0.007 / 0.155 / 0.025, FDR q =
  **0.040** / 0.932 / **0.151** (detrend / glm / maximal). The **primary report is
  maximal-residualized: q=0.151 — marginal, does NOT survive FDR**. In the robustness set it's
  FDR-significant in detrend (q=0.040) but null in glm; i.e. inconsistent across pipelines, and
  the lighter detrend/glm pipelines also suffer the AUC degeneracy. The visual effect is a
  maximal-only denoising artifact. (Maximal is held back partly by one placebo outlier, sub-37;
  Tukey-IQR → maximal raw q=0.116 — see `pipeline_comparison.csv`.)
- **Category × Drug interaction (result 2 — do the 6 layers change *differently*?):**
  `06_interaction.R` — `lmer(ΔAUC ~ category*drug + (1|subject))`, Type III F (Satterthwaite)
  + subject-level permutation of drug labels, raw + residualized, all 3 pipelines.
  **Significant only in maximal:** raw F(5,185)=2.39, p=0.040 (perm 0.050); resid p=0.037
  (perm 0.057). The raw interaction is partly the **visual** artifact (drops to p=0.105 without
  visual), but the **residualized interaction survives removing visual (p=0.051)** — a genuine
  **exteroception↓ vs cognition↑ dissociation** under verum (cognition/intero timescales rise,
  extero/visual fall; auditory/motor flat). Absent in detrend/glm. Borderline and maximal-only →
  suggestive. Fig `qin_interaction_category_by_drug.png`, table `interaction_category_by_drug.csv`.
- **ACW-50:** values are **computed and kept** (`acw_results[2]` in the JLD2 +
  `results/qinspheres/acw50/{pipeline}/tables/qinspheres_acw50.csv`) but the statistical
  analysis was **removed** — ACW-50 showed no robust effect and did not reproduce the
  AUC-exteroception finding (the effect is AUC-specific; see the earlier ACW-50/tau checks).
  To regenerate ACW-50 stats, re-run the chain with the `acw50` metric arg. Only **AUC** is
  the reported analysis.

**SampEn (`01_build_sampen_df.py` + the metric-parameterized R pipeline, metric=`sampen`):**
same 6 sphere layers × 3 pipelines, n=39; per-layer drug-effect test (permutation + BH-FDR),
identical methodology to AUC. SampEn df built from
`03_intrinsic_neural_metrics/results/sampen/{pipeline}/{layer}/…_sampen.csv` →
`results/qinspheres/sampen/{pipeline}/tables/qinspheres_sampen.csv` (value column named `auc`).
**Reported RAW** (per user; SampEn residualized was equally null, residual outputs removed —
the always-residualize rule still applies to all *other* analyses).
**Result: comprehensive null** — nothing significant in any layer/pipeline (smallest perm p = 0.14
[intero/glm], smallest FDR q = 0.61). Exteroception SampEn p=0.32–0.97, so the AUC-exteroception
effect is **timescale-specific, not reflected in signal irregularity (SampEn)**.

---

## Whole-cortex Glasser network analysis (`scripts/glasser_g1/`)
Whole-cortex **Glasser 360** cortical parcels (NoGSR, 3 pipelines, **n=35** =
`get_included_subjects`, FD>0.3), AUC aggregated into **Cole-Anticevic networks**; drug effect
(ΔAUC = Post−Pre) compared across networks. Arm = `participants.condition` (17 verum + 18 placebo).
**The G1-slope "flattening" approach was dropped** (G1↔AUC coupling weak, |r|≤0.26, no
drug-specific flattening) — the per-parcel G1 column is kept only as optional metadata.

- **Extract (server):** `02_timeseries_extraction/scripts/glasser_g1/01_extract_glasser_cortex.py`
  — `NiftiLabelsMasker` over `glasser360MNI.nii.gz` from the existing denoised NIfTIs (no
  re-denoising). Atlas is **MNI152NLin6 1mm**, BOLD is **2009cAsym** — no native-2009c Glasser exists
  (TemplateFlow has Schaefer/HO/DiFuMo, not Glasser), so atlas is **nearest-neighbor resampled
  NLin6→2009c** (`--allow-resample`), the same approximation as the qinsphere pipeline.
- **AUC (local):** `03_intrinsic_neural_metrics/scripts/glasser_g1/01_compute_glasser_auc.jl` —
  exact `acw()` config (TR 1.8, n_lags 100, 6 dummy, AUC=`acw_results[1]`).
- **Tidy df (local):** `02_build_tidy_df.py` → `glasser_g1_auc_tidy.csv` (parcel, network, G1, auc,
  subject, session, arm, pipeline). G1 from `g1_parcellated.pscalar.nii` row 0, name-mapped
  (INDEX→`GLASSERLABELNAME`→pscalar name; all 360 join — glasser360 INDEX is CAB-NP/L-first vs
  pscalar Glasser/R-first, so the old direct-index map would MIS-MAP). Networks = LabelKey `NETWORK`.
- **Network ΔAUC (local):** `03_network_breakdown.py` (global + 12-network verum-vs-placebo perm
  test, BH-FDR) and `04_network_drugdiff_plot.R` (first-glance jitter+median plot, **5 focus
  networks**: Sensorimotor=Somatomotor, Salience=Cingulo-Opercular, Dorsal-Attention,
  CentralExecutive=Frontoparietal, **DefaultMode=Default**).
- **Result:** placebo lengthens timescales more than verum in **every** network (consistent with
  Result 1) but underpowered. Per-network drug test (residualized, `05_…test.R`): all 5 verum<placebo;
  Dorsal-Attention p=0.033 / Sensorimotor p=0.064 nominal, **none survive FDR** (q≥0.16); DMN mid-pack.
  Network×drug interaction and **DMN-vs-others** both **null** (`06_…interaction.R`; maximal p=0.53 / 0.59).
  → the drug effect is **global/uniform, not network-specific; DMN is not special** (hypothesis tested,
  not supported). All residualized (mandatory). Detrend agrees in direction (broadband-inflated), glm null.

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
