# Project: fMRI ACW / SampEn Pipeline (DMT-MED Dataset)

## What this pipeline computes

Two intrinsic-dynamics metrics from resting-state fMRI, compared across
**six brain layers** (Qin et al. 2020 self + sensory/motor nonself), in a
within-subject pre/post design:

- **ACW** — Autocorrelation Window (AUC), the intrinsic neural timescale.
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
  1. `01_denoising/` — two NoGSR pipelines → `results/{detrend,maximal}/…desc-<pipeline>_bold.nii.gz`.
  2. `02_timeseries_extraction/scripts/01_extract_sphere_timeseries.py` — extracts the
     6-layer CSVs from **both** denoising pipelines (n=39 subjects) to
     `results/qinspheres/{detrend,maximal}/{layer}/`. CSVs are committed to git.
- **Local steps** (timeseries CSVs are committed to the repo, so these run on Windows):
  3a. `03_intrinsic_neural_metrics/scripts/acw/01_spheres.jl` — ACW (2 pipelines) → `results/acw/spheres/{pipeline}/{layer}/`.
  3b. `03_intrinsic_neural_metrics/scripts/sampen/01_spheres.py` — SampEn (2 pipelines) → `results/sampen/spheres/{pipeline}/{layer}/`.

### Run order
```
# Server (conda env: fmri)
python 02_timeseries_extraction/scripts/01_extract_sphere_timeseries.py
# Local (or server)
julia  03_intrinsic_neural_metrics/scripts/acw/01_spheres.jl          # 2 pipelines × 6 layers × 39 subj
python 03_intrinsic_neural_metrics/scripts/sampen/01_spheres.py      # 2 pipelines × 6 layers × 39 subj
```

---

## Repository structure

```
thesis-har-med-pipeline/
├── CLAUDE.md  README.md  config.toml  participants.tsv  .gitignore
├── utils/                         subject_filter, motion_qc, thesis_style, config_loader.{py,jl,R}
├── 01_denoising/                 denoise_pipelines.py + 00_test_one_subject/01_denoise_all (2 NoGSR pipelines)
├── 02_timeseries_extraction/
│   ├── atlases/                   glasser360MNI + CAB-NP key (inputs) + self/nonself coords + metadata
│   ├── scripts/01_extract_sphere_timeseries.py   (extracts from both denoising pipelines)
│   └── results/qinspheres/{detrend,maximal}/{layer}/   39×2 CSVs each (committed)
├── 03_intrinsic_neural_metrics/   scripts/ mirrors results/ — same {metric}/{atlas} path, different root
│   ├── scripts/acw/     01_spheres.jl  02_parcels_self_regions.jl  03_parcels_all_parcels.jl
│   ├── scripts/sampen/  01_spheres.py  02_parcels_self_regions.py  03_parcels_all_parcels.py
│   └── results/{acw,sampen}/{spheres,parcels/self_regions,parcels/all_parcels}/{pipeline}/…
│         spheres + parcels/self_regions → {layer}/ per-run files;  parcels/all_parcels → one CSV
│         (the six scripts are mutually independent — the numbering is grouping, not a dependency chain.
│          Script filenames are the results path with `/` → `_`: 02_parcels_self_regions ↔
│          parcels/self_regions/. The mirror stops at the ROI level because one script emits every
│          pipeline and every layer, so there is nothing to put in those directories.)
├── 99_QC/                         01_motion_qc, 02_denoising_qc, 03_acw_qc, troubleshooting
├── 04_statistics/               results/ mirrors 03_ — {metric}/{atlas}/{pipeline}/{figures,tables}
│   ├── scripts/qin/             ACW+SampEn+FC stats — atlas-parameterized (n=35)
│   ├── scripts/all_parcels/     whole-cortex GLOBAL stats (writes into parcels/all_parcels)
│   ├── scripts/demographics/    arm balance on the baseline participant variables
│   └── results/{acw,sampen,fc}/{spheres,parcels/self_regions,parcels/all_parcels}/{pipeline}/{figures,tables}
│         EVERY branch carries the {pipeline} level, all_parcels included — the pipeline is
│         a directory, never a filename infix. Cells = metric × atlas × {maximal,
│         maximal_nocensor}; fc has no all_parcels (FC needs ≥2 regions).
│         + mixed_models/{per_config,supplementary}   (flat by design: a cross-config index)
│         + demographics/        (flat: not a metric × atlas × pipeline cell)
│         CLI atlas args stay `qinspheres|qinparcels`; scripts map them to the dir names
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

### Denoising — two parallel pipelines (robustness design)
To show the final results are robust to denoising choice, the same data is denoised
two independent ways (all volumetric MNI, all **NoGSR**). One parameterized core
(`denoise_pipelines.PIPELINE_PRESETS`) drives both; pick with `--pipeline`:

| Pipeline | Detrend | WM+CSF | Motion 6+6 | FD>0.3 censor | LS interp | Bandpass |
|----------|:------:|:------:|:----------:|:-------------:|:---------:|:--------:|
| `detrend` (Keskin 2025) | polort 1 | – | – | – | – | – |
| `maximal` (Goldberg 2024) | polort 1 | ✓ | ✓ | ✓ | ✓ | 0.01–0.1 Hz |

- **`glm` was removed from the project entirely** (preset, results, and all analyses).
- `maximal_nocensor` is an optional **control** preset (identical to `maximal` but with
  censoring/LS-interpolation OFF — the same band-pass applied to the full continuous
  series). Not part of the main flow; used only by the interpolation robustness check
  (`04_statistics/scripts/qin/07_interp_artifact_checks.R`, Check B). Downstream scripts
  take an env override `PIPELINES=maximal_nocensor` to build it in isolation.

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
- Outputs: `results/{detrend,maximal}/sub-XX_ses-YY_task-rest_desc-<pipeline>_bold.nii.gz`
  + per-pipeline `_batch_log.tsv`. Run: `01_denoise_all.py --pipeline {detrend|maximal}`
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

Both ACW and SampEn loop the **two pipelines × six layers** at **n=39**
(matches the extraction; the 35-subject filter is applied later at stats). Input:
`02_timeseries_extraction/results/qinspheres/{pipeline}/{layer}/`.

### ACW (`acw/01_spheres.jl`, Julia + IntrinsicTimescales.jl)
- TR 1.8 s, n_lags 100, dummy volumes discarded = 6, type `[:auc]`.
- Self-contained (no longer reads `config.toml`); 39-subject list hardcoded to match
  `get_pipeline_subjects()`. Idempotent (skips existing JLD2).
- Output: `results/acw/spheres/{pipeline}/{layer}/{sub}_{ses}.jld2`
  (vars: `acw_results` [1]=AUC, `parcel_ids`).

### SampEn (`sampen/01_spheres.py`, Python + EntropyHub)
- Per run × layer: drop 6 dummy volumes → linear detrend → SampEn (m=1, τ=1, r=0.3
  absolute, log base 2; Keskin/Northoff conventions). Uses `get_pipeline_subjects()`.
- Output: per-run `results/sampen/spheres/{pipeline}/{layer}/{sub}_{ses}_{layer}_sampen.csv`
  + per-pipeline `results/sampen/spheres/{pipeline}/sampen_long_{pipeline}.csv` (self-contained;
  no coupling to `04_statistics`).

---

## Software environment

- **Server:** Python 3.x (Miniconda env `fmri`: pandas, numpy, scipy, matplotlib,
  nibabel, nilearn, EntropyHub). `conda activate fmri`. Headless — always `plt.savefig()`.
- **Local (Windows):** Julia 1.12 (IntrinsicTimescales.jl), R 4.6.0
  (`C:\Program Files\R\R-4.6.0\bin\Rscript.exe`, not on PATH). Python **3.13** at
  `C:\Users\JLU-MBB\AppData\Local\Programs\Python\Python313\python.exe` (NOT on PATH — `python`
  resolves to 3.14, which has no scientific stack; always call 3.13 by full path). 3.13 has
  numpy/scipy/pandas/EntropyHub. The timeseries CSVs, ACW, and SampEn all run locally.
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
- `03_acw_qc/` — AUC distribution, tSNR / dropout checks, ROI-placement viewer.
- `troubleshooting/` — non-thesis experiments.
- **tSNR note:** the old 58-parcel tSNR exclusion list belonged to the *parcels* atlas
  (now in `_archive/`). A sphere-based tSNR check must be recomputed if ROI exclusion is
  wanted for this design.

---

## Statistics (`04_statistics/`) — ACW/SampEn analysis (qinspheres, maximal)
**The reported effect is the TIME effect (post−pre); the drug contrast is null.** The section is
still organised around the DiD because that was the original design, but do not describe this as a
drug-effect analysis.
The active analysis is `scripts/qin/` — one **atlas-parameterized** codebase (`[pipeline] [atlas]`,
default `maximal qinspheres`) that drives **both** `qinspheres` and `qinparcels`, both metrics
(`auc`, `sampen`). (The former `scripts/glasser_parcellations/` was a stale duplicate of this — it
still had the retired retreat model — and was archived to `_archive/glasser_parcellations_stats_dupe/`.)
The primary result is on the **maximal** pipeline ACW-AUC at
**n=35** (the parcels_NoGSR / GSR / _old analyses are archived under
`_archive/statistics_parcels_and_old/`). **Sample = n=35** = `get_included_subjects` (drop
sub-06/08/12/26/36): the ACW/SampEn *compute* loops n=39 (keeps high-motion for denoising
robustness), but `01_build_auc.jl` / `02_build_sampen.py` filter to **n=35** so all stats
(spheres AND parcels) are on the same FD-included sample.

**Design (current — single mixed model).** Stats are `03_mixed_models.R` — per region, one REML LMM
with **Kenward-Roger df** (lmerTest + pbkrtest) on the long table (subject × session × region), with
**session-specific motion covariates** `pcf`, `pcf_sq = pcf²`, `mean_fd` (from
`99_QC/01_motion_qc/results/fd_covariates_wide_thresh03.csv`):
`AUC ~ session*arm + pcf + pcf_sq + mean_fd + (1|subject)`. Three effects are pulled from it:
- **DiD / drug effect** (`_did.csv`): `sessionpost:armverum` (treatment coding) → drug modulation of post−pre.
- **Overall main effects** (`_overall.csv`, sum contrasts so each effect is averaged over the other
  factor): `session` = overall post−pre time effect, `arm` = verum−placebo baseline; per region, BH-FDR each.
- **Global** (`_overall_global.csv`): one pooled model over all 5 regions (region as fixed factor) →
  overall session / arm / DiD.
BH-FDR across the **5 regions** (**motor excluded** — the region worst-hit by the broadband-AUC
degeneration); singular DiD fit → label-permutation fallback. Parameterized `[pipeline] [atlas]`
(default `maximal qinspheres`). Out: `results/mixed_models/per_config/{atlas}_{pipeline}{_metric}_{longtable,did,overall,overall_global}.csv`
+ `_model_summaries.txt` (full lme4/lmerTest `summary()` per region + pooled global — Random effects, Fixed effects table, etc.).
**`03b_consolidate.R`** then collapses the per-config files into one tidy table per atlas×metric —
`results/mixed_models/{atlas}_{metric}_summary.csv` (long: pipeline × analysis[did|overall|global] ×
region × term) — the single place to read results. It carries **both `maximal` and
`maximal_nocensor`** (36 rows), so the censoring robustness check is readable in one table;
filter on the `pipeline` column for the primary analysis alone. **detrend is dropped from the
stats stage** (broadband-degraded) and **glm was removed from the project entirely**.
`04_figures.R` reads the DiD bracket from `per_config/{atlas}_{pipeline}{_metric}_did.csv`.

**Pipeline lists are DISCOVERED, never hardcoded.** `01_build_auc.jl`, `02_build_sampen.py` and
`all_parcels/01_build_tidy.py` each used to carry a literal `["detrend", "maximal"]` — a retired
preset plus the live one, silently skipping `maximal_nocensor` (and `02_build_sampen.py` had no
env override at all, so nocensor was unreachable). All three now list the subdirectories of their
03 input tree, guarded against picking up `tables`/`figures`. Env `PIPELINES=…` still overrides.
- The earlier separate placebo-only **retreat** model (`AUC ~ session + …`) was **dropped** — the
  placebo post−pre it captured is recoverable from the overall session effect. Its `_retreat.csv`
  outputs and the placebo-only retreat figures (`qin_retreat_per_layer*.png`) were removed; the
  both-arm descriptive `qin_retreat_paired_prepost.png` (paired-t, no LMM) is kept.

**Reorg note:** the earlier **per-layer permutation/residualization pipeline** (`statistics.R`,
`statistics_resid.R`, `qc_residualize_auc.R`, `05_pipeline_comparison.R`, `06_interaction.R`) plus
the summary/global scripts (`07_summary_table.R`, `08_global_coupling_motion_tests.R`) were
**archived to `_archive/qinspheres_permutation_stats/`**. Their committed result tables under
`results/{metric}/{atlas}/…` (`layer_drug_effect*.csv`, `layer_outlier_comparison*.csv`,
`interaction_category_by_drug.csv`, `auc_diff_quality_*`, `pipeline_comparison.csv`) were **deleted**
in the switch to the single mixed model (git-recoverable) — the current `04_figures.R` reads **only**
the mixed-model `_did.csv` (drug bracket) plus its input `{metric}.csv`, none of these.
Each `results/{metric}/{atlas}/{pipeline}/tables/` now holds just `descriptive_pvalues.csv` + the input df.
**ACW-50 and τ were removed** as defective operationalizations. Builds are `01_build_auc.jl` /
`02_build_sampen.py` (both filter to n=35); plots are `04_figures.R`.

**Non-positive AUC dropped:** `01_build_auc.jl` removes ROIs with `auc ≤ 0` (AUC metric only).
Without bandpass, the broadband signal's autocorrelation collapses to/below zero, giving 0 or
negative "area" — not a valid timescale. This is a **detrend** problem (≈1042 ROIs
dropped, concentrated in motor); **maximal** (bandpassed) has **none** (0 dropped) and is
unaffected. Underlying ACW values are still in the JLD2; only the analysis CSV is filtered.

**Run order** (from repo root; Rscript at `C:\Program Files\R\R-4.6.0\bin\Rscript.exe`):
```
# --- ACW / SampEn chain --------------------------------------------------------------
julia   04_statistics/scripts/qin/01_build_auc.jl [atlas]                     # build AUC df (n=35), all pipelines found in 03
python  04_statistics/scripts/qin/02_build_sampen.py [atlas]                  # build SampEn df (n=35), same
Rscript 04_statistics/scripts/qin/03_mixed_models.R [pipeline] [atlas] [metric]  # DiD LMM + overall/global -> mixed_models/per_config/
Rscript 04_statistics/scripts/qin/03b_consolidate.R                           # -> mixed_models/{atlas}_{metric}_summary.csv (maximal + maximal_nocensor)
Rscript 04_statistics/scripts/qin/04_figures.R [pipeline] [metric] [atlas]    # fast figures (all configs)
Rscript 04_statistics/scripts/qin/04b_bootstrap_ridges.R [pipeline] [metric] [atlas]  # HEAVY: bootstrap ridgeline (primary config only)
Rscript 04_statistics/scripts/qin/04c_placebo_forest.R [pipeline] [metric] [atlas]  # placebo-only forest (the retreat effect, no drug)
Rscript 04_statistics/scripts/qin/04d_acf_panel.R [pipeline] [atlas]          # pre/post group-mean ACF with the AUC lobe shaded
Rscript 04_statistics/scripts/qin/05_supplementary_tables.R [pipeline] [atlas]  # 3 suppl. tables (AUC) across pipeline × atlas configs
Rscript 04_statistics/scripts/qin/06_auc_robustness_motion_interp.R [pipeline] [atlas]  # 2 descriptive robustness checks on extero AUC
Rscript 04_statistics/scripts/qin/07_interp_artifact_checks.R [pipeline] [atlas]  # Check B needs maximal_nocensor to exist for that atlas

# --- FC chain (independent of the above; the 01_/02_ numbers are a SEPARATE chain) ----
python  04_statistics/scripts/qin/01_compute_fc.py [pipeline] [atlas] [arm]   # between-region, 10 pairs -> fc_long[_arm].csv
python  04_statistics/scripts/qin/01b_compute_fc_within.py [pipeline] [atlas] # within-region scalar, 5 regions -> fc_within_long.csv
python  04_statistics/scripts/qin/02_model_fc.py [pipeline] [atlas] [arm]     # paired t + 5x5 z-change matrix
python  04_statistics/scripts/qin/02b_fc_contrasts.py [pipeline] [atlas] [between|within]   # all 5 contrasts, t-tests
Rscript 04_statistics/scripts/qin/05_fc_figures.R [pipeline] [atlas]          # Panel A (within) + Panel B (4x 5x5 matrices)

# --- demographics --------------------------------------------------------------------
python  04_statistics/scripts/demographics/01_arm_balance.py                  # arm balance: meditation hours, age, sex, retreat
```
`04_figures.R` ports the parcels raincloud/serif/bracket aesthetic to the **5 layers**
(motor excluded; flexible `wrap_plots` layout). The **main figure** `qin_time_effect_forest.png`
(per-region overall post−pre time effect: estimate + 95% CI, ordered by magnitude, FDR-survivors
marked) reads the session main effect + CIs from `_overall.csv` (`03_mixed_models.R` emits
`session_ci_low/high`, `arm_ci_low/high`). `qin_time_effect_boxplots.png` shows per-region **Pre vs Post** boxplots (both arms, n=35) with the
LMM session p/q bracket (the time effect). It writes the canonical set —
`qin_time_effect_forest`, `qin_time_effect_boxplots`, `qin_prepost_boxplots_placebo` (the same
per-region Pre vs Post panels for the **placebo arm only**, n=18, bracket = LMM within-arm
`_simple.csv` p/q),
`qin_delta_by_region` (one-panel ΔAUC across
the 5 regions, ordered by median, FDR stars), `qin_overview_raincloud_4conditions`,
`qin_QC_baseline_balance_{pooled,per_layer}`, `qin_retreat_paired_prepost`, `qin_overall_auc_histogram`,
`qin_acf_single_subject` — plus `descriptive_pvalues.csv`.
(The per-region drug-effect DiD raincloud `qin_drug_effect_significance.png` was **removed** — the
DiD lives in `_did.csv`; the drug effect is null and the reported effect is the time effect.)
**`04b_bootstrap_ridges.R`** (separate, HEAVY — B×5 `lmer` refits, run for the **primary config only**,
not the sweep) makes `qin_time_effect_bootstrap_ridges.png`: subject-level **cluster-bootstrap** (resample
the 35 subjects w/ replacement, both sessions kept together, B=2000) of the per-region **standardized**
post−pre coefficient → ggridges density ridges ordered by median (visual→auditory), red line at 0,
FDR-survivors starred. Draws saved to `…/tables/qin_time_effect_bootstrap_draws.csv`. B via env `BOOT_B`.
(The legacy `02_scatter.R` / `03_drug_effect_figures.R` figure scripts were **removed** — superseded
by `04_figures.R`; their `*_res.png` / scatter / density static outputs were deleted in the full clean.)

**Results (AUC, n=35) — the current mixed models (`03b_consolidate.R` summary tables), NOT the
archived permutation analysis:**
- **The headline is the SESSION (time) effect, not the drug.** AUC lengthens post−pre, and
  **visual survives BH-FDR in all four cells** (atlas × pipeline): spheres **+0.209 s (q=0.006)**,
  spheres-nocensor **+0.311 (q=0.002)**, parcels **+0.277 (q=0.0022)**, parcels-nocensor
  **+0.462 (q=0.003)**. The uncensored estimate is **larger in every cell**, so censoring
  attenuates this effect rather than creating it.
- **Interoception** survives in 3 of 4 cells (spheres q=0.044, parcels q=0.004,
  parcels-nocensor q=0.005; not spheres-nocensor, q=0.177). **Exteroception** survives in both
  parcel cells (q=0.0022 / 0.003), is borderline in spheres-nocensor (q=0.053) and ns in spheres
  (q=0.152). **Cognition** is ns everywhere (best q=0.084). **Auditory is null in all four
  cells** — the sensory-specificity control.
- **Pooled global** (the 5 regions in one model): session **+0.105 to +0.221 s, p<0.0001 in every
  cell**. The lengthening is broad, not confined to one region.
- **Drug (DiD) is null everywhere — nothing survives FDR in any region, atlas or pipeline.** The
  largest is exteroception in spheres/maximal, −0.208 (p=0.019, **q=0.094**), weakening to p=0.11
  uncensored and p=0.34 in parcels. The drug contrast is secondary/exploratory.
- **Retired, do not cite:** the earlier permutation drug-effect result (extero resid p=0.022,
  FDR-significant only in detrend) and the **category × drug interaction** (`06_interaction.R`).
  Both belonged to the archived permutation pipeline; `interaction_category_by_drug.csv` and its
  figure were deleted. Not reproducible from this repo — recoverable from git history.

**SampEn (`02_build_sampen.py` + the metric-parameterized R pipeline, metric=`sampen`):**
same 5 regions × 2 pipelines, n=35, run through the **same mixed model as AUC** (motion enters as
the `pcf` / `pcf_sq` / `mean_fd` covariates inside the LMM, so there is no separate residualized
pass). SampEn df built from
`03_intrinsic_neural_metrics/results/sampen/spheres/{pipeline}/{layer}/…_sampen.csv` →
`results/sampen/spheres/{pipeline}/tables/sampen.csv` (value column named `auc`).
**Result: null per region** — **nothing survives BH-FDR in any region, atlas or pipeline**; the
smallest q is 0.124 (visual, spheres/maximal: +0.111, p=0.025). The **pooled global** session
effect is nominally positive in all four cells (+0.048 to +0.073; p = 0.026 / 0.0007 / 0.007 /
0.0003), and one nominal global DiD appears in parcels/maximal (+0.087, p=0.036) that does not
hold uncensored (p=0.114). Per region the picture is a null, so the AUC lengthening is
**timescale-specific, not reflected in signal irregularity (SampEn)**.

**Parcel version (`qinparcels`):** the same 6 regions as **whole Glasser parcels** (self from the
Keskin focus-point→parcel mapping in `glasser_self_metadata.tsv` — intero 14 / extero 16 / mental 12;
nonself from CA networks minus self — visual 55 / motor 36 / auditory 14), built by subsetting the
existing glasser360 timeseries (`02_build_qin_parcel_timeseries.py`, **n=35**) → `qinparcels` TS.
Parallel ACW (`03_…/scripts/acw/02_parcels_self_regions.jl` → `results/acw/parcels/self_regions/`) + SampEn
(`…/sampen/02_parcels_self_regions.py` → `results/sampen/parcels/self_regions/`), identical configs. The
stats/figures reuse the same scripts via an **`atlas` arg** (`qinspheres|qinparcels`): the 4 R
pipeline-metric scripts take it as the **3rd** trailing arg, `05_supplementary_tables.R` as the **2nd**, `01_build_auc.jl`
as `ARGS[1]`, `02_build_sampen.py` as `argv[1]`. Outputs at `results/acw/parcels/self_regions/{pipeline}/`
(AUC) + `qinparcels/sampen/{pipeline}/` (SampEn). **Result: the parcel version REPLICATES the time
effect and is null for the drug.** AUC session effects survive FDR in **visual (q=0.0022),
exteroception (q=0.0022) and interoception (q=0.004)** in both parcel pipelines — parcels are the
*stronger* cell for extero/intero, spheres the weaker. SampEn is null past FDR. The **drug** DiD is
null in parcels exactly as in spheres, so the earlier exteroception *drug* effect remains
sphere-only and unconfirmed. → final layout = 4 categories: `acw{qin,parcel}` × `sampen{qin,parcel}`, each 2 pipelines.

---

## Whole-cortex Glasser analysis (`scripts/all_parcels/`)
Whole-cortex **Glasser 360** cortical parcels (NoGSR, **n=35** = `get_included_subjects`, FD>0.3),
treated as **ONE cortex**: AUC/SampEn averaged over all 360 parcels → one value per subject ×
session. Arm = `participants.condition` (17 verum + 18 placebo).

**Both the G1 gradient and the 12-network Cole-Anticevic breakdown were REMOVED from the project.**
The G1-slope "flattening" approach had already been dropped (G1↔AUC coupling weak, |r|≤0.26); the
per-parcel G1 column, the pscalar dependency, and the whole per-network layer went with it. The
analysis is now global-only. Retired per-network findings (per-network placebo effects, the
network×drug interaction, the DMN-vs-others test) are **no longer reproducible from this repo** —
recoverable from git history (`glasser_network_*` tables, `07_network_mixed_models.R`) if ever needed.

- **Extract (server):** `02_timeseries_extraction/scripts/all_parcels/01_extract_glasser_cortex.py`
  — `NiftiLabelsMasker` over `glasser360MNI.nii.gz` from the existing denoised NIfTIs (no
  re-denoising). Atlas is **MNI152NLin6 1mm**, BOLD is **2009cAsym** — no native-2009c Glasser exists,
  so the atlas is **nearest-neighbor resampled NLin6→2009c** (`--allow-resample`), the same
  approximation as the qinsphere pipeline.
- **AUC / SampEn (local):** `03_intrinsic_neural_metrics/scripts/{acw,sampen}/03_parcels_all_parcels.{jl,py}`
  — exact same configs as the sphere scripts.
- **Layout:** every output sits under `{metric}/parcels/all_parcels/{pipeline}/{tables,figures}/`,
  exactly like the qin branch. The pipeline is the **directory**; it is NOT part of any filename
  (so `glasser360_lmm_overall.csv`, not `glasser360_lmm_maximal_overall.csv`). One tidy df per
  pipeline, mirroring `{atlas}/{pipeline}/tables/auc.csv`.
- **Tidy df (local):** `01_build_tidy.py` → `{pipeline}/tables/glasser360_auc_tidy.csv` (parcel_id,
  auc, subject, session, arm, pipeline — **no network, no G1**; needs neither the CAB-NP LabelKey
  nor the pscalar). `01b_build_tidy_sampen.py` is the SampEn twin. Both discover pipelines from
  the 03 tree rather than hardcoding them.
- **Global permutation test:** `02_global_delta_test.py [auc|sampen]` — mean over 360 parcels per
  subject × session, Δ = Post−Pre, verum vs placebo, 10k subject-label permutation (seed 42).
  Loops every pipeline directory, one result file each.
- **Global mixed model:** `03_global_mixed_model.R [pipeline] [arms] [metric]` — the **same LMM as
  the qin analysis**, `AUC ~ session*arm + pcf + pcf_sq + mean_fd + (1|subject)`, REML, KR df, on
  **70 rows** (one whole-cortex value per run). No FDR needed (one test per effect).
  **This replaces the former pooled-global row**, which fitted 840 correlated rows (12 networks × 70)
  under one subject intercept and was anticonservative — that caveat is now resolved by construction.
- **Figures:** `04_figures.R [pipeline] [metric]` → `{pipeline}/figures/glasser360_prepost.png`
  (A: four boxes arm × session; B: per-subject paired lines split by arm). Asterisks key to raw
  p here — with one test per effect there is no family to correct over, so corrected and
  uncorrected are the same number and the thesis convention `* = survives correction` still holds.

**Results (maximal, n=35):**
- **AUC — whole cortex lengthens post−pre:** session effect **+0.141 s, 95% CI [0.040, 0.243],
  KR df 32.3, p=0.008** (dz=0.52). Within **placebo**: **+0.214 s, p=0.004** (n=18, dz=0.64).
  **DiD (drug) null: −0.145, p=0.148.** Global permutation test agrees: verum−placebo −0.115, p=0.273.
  → the retreat effect is **cortex-wide and sits in the placebo arm**; the drug does not modulate it.
- **SampEn — null:** session +0.060, p=0.163; placebo +0.040, p=0.499; DiD +0.041, p=0.622.
  Permutation verum−placebo +0.061, p=0.430.
  → **timescale-specific, not an irregularity effect** — the whole-cortex confirmation of the same
  AUC-vs-SampEn dissociation found in the qin spheres.

Both pipelines are built. `maximal_nocensor` agrees with `maximal` on every effect — AUC DiD
−0.166 (p=0.274) vs −0.145 (p=0.148), SampEn DiD +0.027 (p=0.739) vs +0.041 (p=0.622) — so the
drug null holds under the no-censoring control for both metrics.

```
python  04_statistics/scripts/all_parcels/01_build_tidy.py
python  04_statistics/scripts/all_parcels/01b_build_tidy_sampen.py          # SampEn only
python  04_statistics/scripts/all_parcels/02_global_delta_test.py [auc|sampen]
Rscript 04_statistics/scripts/all_parcels/03_global_mixed_model.R [pipeline] placebo [auc|sampen]
Rscript 04_statistics/scripts/all_parcels/04_figures.R [pipeline] [auc|sampen]
```

---

## Functional connectivity (`scripts/qin/`, metric dir `fc/`)
Same five regions, same n=35, both atlases, both pipelines — 4 cells, all built.

- **Between-region** (`01_compute_fc.py`): each region's ROIs are averaged into one mean
  timeseries, the 10 region pairs correlated, Fisher-z. → `fc_long[_{arm}].csv`.
- **Within-region** (`01b_compute_fc_within.py`): the mean of all C(k,2) pairwise Fisher-z
  correlations among the ROIs *inside* a region — one scalar per subject × session × region.
  This is what makes a per-region boxplot panel possible (Egger Fig. 3B analogue); the
  between-region script destroys this structure by averaging first. → `fc_within_long.csv`.
- **Tests** (`02b_fc_contrasts.py [pipeline] [atlas] [between|within]`): five contrasts —
  `time`, `within_placebo`, `within_verum`, `did`, `post_diff` — all ordinary t-tests, BH-FDR
  within each contrast family. `02_model_fc.py` is the older paired-t script; it still writes
  the 5×5 z-change matrices and agrees exactly with `02b`'s `time` contrast.
- **A per-pair LMM was tried and REJECTED.** `z ~ session*arm + pcf + pcf_sq + mean_fd +
  (1|subject)` returns **singular fits for 3 of the 10 pairs** (Visual–Auditory,
  Visual–Cognition, Visual–Exteroception) — between-subject variance in FC change collapses to
  zero — and one of the three is the pair the chapter discusses. Do not reintroduce it. Because
  each subject contributes exactly one number per contrast, the t-tests are on independent
  observations and singularity cannot arise; for a balanced 2×2 the two-sample t on change
  scores IS the interaction test.

**Results (n=35):**
- **Within-region visual FC rises post−pre and survives FDR in ALL FOUR cells** — parcels
  +0.108 (q=0.018), parcels-nocensor +0.105 (q=0.040), spheres +0.080 (q=0.011),
  spheres-nocensor +0.080 (q=0.032). Robust to both ROI definition and censoring, which neither
  the exteroception AUC effect nor the between-region result manages. Visual is also the region
  with the largest timescale lengthening.
- **Between-region: only Visual–Cognition moves**, and only in spheres (−0.189, q=0.011;
  nocensor −0.160, q=0.027). In parcels it does not survive (−0.146, q=0.32). Degradation is
  monotone: spheres > parcels, censored > uncensored — the ROI definition decides, not censoring.
- **DiD null everywhere**, both units, all cells.
- **Descriptive only:** visual connectivity separates by target — falls with cognition, rises
  slightly with extero/interoception — same direction under both ROI definitions, 8/10 pairs
  sign-concordant. Nine non-significant tests; reported as a pattern, not a result.

```
python  04_statistics/scripts/qin/01_compute_fc.py [pipeline] [atlas] [all|placebo|verum]
python  04_statistics/scripts/qin/01b_compute_fc_within.py [pipeline] [atlas]
python  04_statistics/scripts/qin/02_model_fc.py [pipeline] [atlas] [all|placebo|verum]
python  04_statistics/scripts/qin/02b_fc_contrasts.py [pipeline] [atlas] [between|within]
Rscript 04_statistics/scripts/qin/05_fc_figures.R [pipeline] [atlas]
```

## Things NOT to redo / change
- Do not put the FC data back into a mixed model — the per-pair LMM gives singular fits for 3
  of 10 pairs and was rejected for that reason (see the Functional connectivity section).
- Do not hardcode pipeline lists in any builder. They are discovered from the 03 output tree;
  the old literal `["detrend", "maximal"]` named a retired preset and hid `maximal_nocensor`.
- Do not key figure asterisks to uncorrected p. Thesis-wide, `*` means **survives BH-FDR**
  (`sig_star()` in `04_figures.R` takes q); uncorrected trends use `#`/`##`. Egger et al. use
  the opposite convention — matching them would make one glyph mean two things.
- `scripts/qin/` holds TWO independent chains sharing one numbering: `01_build_auc` →
  `02_build_sampen` → `03_mixed_models` → `04_figures` (ACW/SampEn), and `01_compute_fc` →
  `02_model_fc` → `02b_fc_contrasts` → `05_fc_figures` (FC). `01`/`02` mean different things
  depending on which chain you are reading.
- Do not reintroduce GSR or the Glasser-parcels atlas into the active flow.
- Do not reintroduce the `glm` denoising pipeline — it was removed from the project
  entirely (preset, all results, and every analysis). Do not add band-pass to any
  existing preset either; `maximal_nocensor` already covers "GLM regressors + band-pass"
  as a control.
- Do not reapply a respiratory filter (empirically unnecessary).
- Do not switch to a 4-TR backward difference (no justification for TR=1.8s).
- Do not include excluded subjects (06/08/12/26/36) in calculation or analysis.
- Do not change the FD threshold (0.3mm) without consulting the supervisor.
- Do not modify fMRIPrep outputs. Do not use the AROMA BOLD (`desc-smoothAROMAnonaggr`).
- Do not install heavy pipeline libraries (XCP-D, fMRIPost-AROMA) or use GUI/interactive plots.

## Conventions
- Figures 300 dpi PNG (`plt.savefig`, never `plt.show`). Numerical outputs `.tsv`/`.csv`.
- Subject-level filenames include `sub-XX_ses-YY`. Print summary stats per script.
