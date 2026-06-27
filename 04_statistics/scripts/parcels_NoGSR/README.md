# parcels_NoGSR statistics — what's where

Two metrics are computed on the **same** denoised Glasser-parcel timeseries and carried
through the **same** self-vs-nonself / pre-vs-post / drug-group design:

| Metric | What it measures | Scripts |
|--------|------------------|---------|
| **INT** (ACW-AUC) | Intrinsic neural timescale (area under the autocorrelation function) | `intrinsic_timescale/` |
| **SampEn** | Sample entropy (temporal irregularity) | `sample_entropy/` |

Each metric runs **two LMMs** (category-averaged + per-parcel) plus a figures script.
Outputs land in metric-tagged folders under `04_statistics/results/parcels_NoGSR/`:
`int_tables/` + `int_figures/` and `sampen_tables/` + `sampen_figures/`.

---

## Scripts & run order

### INT — `intrinsic_timescale/`
```
01_build_dataframe.jl     ACW-AUC JLD2s  -> int_tables/glasser_full_dataframe.csv
02_lmm_category_avg.R     LMM #1: auc ~ session*group*self_layer + (1|subject)   (category means)
03_lmm_per_parcel.R       LMM #2: auc ~ group*session + (1|subject), one fit per parcel
04_figures.R              12 figures + descriptive_pvalues.csv  (run 02 & 03 first)
```
Upstream: ACW itself is computed in `03_intrinsic_neural_metrics/scripts/01_compute_acw.jl`.

### SampEn — `sample_entropy/`
```
(build step is FROZEN — see "SampEn dataframe provenance" below; no build script)
01_lmm_category_avg.R     LMM #1: sampen ~ session*group*self_layer + (1|subject) (category means)
02_lmm_per_parcel.R       LMM #2: sampen ~ group*session + (1|subject), one fit per parcel
03_figures.R              12 figures + descriptive_pvalues.csv  (run 01 & 02 first)
```
Upstream: SampEn is computed in `03_intrinsic_neural_metrics/scripts/compute_sampen.py`.

Run R with `C:\Program Files\R\R-4.6.0\bin\Rscript.exe` (not on PATH); Julia 1.12 local.
The dataframes already exist, so re-running only the LMM + figures scripts regenerates
every table and figure. Example:
```
Rscript 04_statistics/scripts/parcels_NoGSR/intrinsic_timescale/02_lmm_category_avg.R
Rscript 04_statistics/scripts/parcels_NoGSR/intrinsic_timescale/03_lmm_per_parcel.R
Rscript 04_statistics/scripts/parcels_NoGSR/intrinsic_timescale/04_figures.R
```

---

## Tables (identical names in `int_tables/` and `sampen_tables/`)

| File | From | Content |
|------|------|---------|
| `<metric>_full_dataframe.csv` | build | one row per subject×session×parcel (the data) |
| `catavg_fixed_effects.csv` | LMM #1 | fixed-effect coefficients of the category-averaged model |
| `catavg_prepost_by_arm.csv` | LMM #1 | pre→post change within each group×layer |
| `catavg_drug_effect.csv` | LMM #1 | **drug effect** = (verum pre→post) − (placebo pre→post), per layer; raw `p` + `p_holm` + `p_fdr` (corrected across the 4 layers) |
| `catavg_3way_interaction_Ftest.csv` | LMM #1 | Type-III F-tests incl. session:group:self_layer (single tests, not corrected) |
| `perparcel_drug_effect.csv` | LMM #2 | group:session estimate + raw `p_value` + `p_fdr` (Benjamini-Hochberg across all parcels) |
| `descriptive_pvalues.csv` | figures | base-R paired-t (pre→post) + baseline-balance p-values |

(INT dataframe is `glasser_full_dataframe.csv`; SampEn is `sampen_full_dataframe.csv`.)

---

## Figures — naming = `<METRIC>_<analysis>_<what>.png`

Each filename is self-identifying, so a figure shown on its own is unambiguous.
The **three INT analyses** and the **two SampEn LMMs** map to the figures as follows
(same 12 figures exist for both metrics, INT_* / SampEn_*):

**Analysis 1 — Self vs Non-self** (`selfVSnonself`)
- `<M>_selfVSnonself_density.png` — self vs nonself distribution
- `<M>_selfVSnonself_pooled_density.png` — all parcels pooled
- `<M>_selfVSnonself_per_subject.png` — per-subject boxplots, atlas × session

**Analysis 2 — Retreat effect, pre→post** (`retreat`)
- `<M>_retreat_per_category.png` — placebo group, ses-01 vs ses-02 (paired-t bracket)
- `<M>_retreat_paired_prepost.png` — per-subject pre→post lines, both arms

**Analysis 3 — Drug effect, DMT vs placebo** (`drug`) — this is where the two LMMs land
- `<M>_drug_delta_per_category.png` — Δ(post−pre) per arm, descriptive
- `<M>_drug_catavg_significance.png` — **LMM #1** drug effect; bracket shows raw `p` and FDR `q` (corrected across the 4 layers), star reflects `q`
- `<M>_drug_perparcel_forest.png` — **LMM #2** per-parcel forest; orange = raw p<0.05, red = BH-FDR q<0.05

> **Multiple-comparison correction.** Both LMMs are FDR-corrected: the 4 per-layer drug
> contrasts (LMM #1) carry Holm + BH-FDR; the per-parcel tests (LMM #2) carry BH-FDR across
> all parcels. The omnibus `session:group` F-test is a single test and is **not** in either
> family. As of the latest run nothing survives FDR in either metric at the per-layer or
> per-parcel level; the only significant drug signal is the SampEn omnibus (p=0.009).

**Overview / QC** (not a primary analysis)
- `<M>_overview_raincloud_4conditions.png` — all 4 conditions × category
- `<M>_QC_baseline_balance_pooled.png`, `<M>_QC_baseline_balance_per_category.png`
- `<M>_QC_lmm_diagnostics.png` — residual QQ / resid-vs-fitted / histogram (from LMM #1)

---

## `_old/` — archived, not used by the pipeline
- `detrend_bandpass_filter_UNUSED.py` — a SampEn preprocessing experiment; the SampEn
  compute detrends internally and reads the *unfiltered* timeseries, so this was never wired in.
- `compute_sampen_rewrite_UNUSED.py` — a cleaner per-atlas rewrite of the SampEn compute
  that was never run (writes a different output layout than the data on disk).

Superseded INT tables from the old config-driven pipeline are archived in
`results/parcels_NoGSR/_old_tables/`.

---

## ⚠ Provenance & caveats (read before presenting)

**SampEn dataframe is frozen.** `sampen_full_dataframe.csv` was produced by an earlier
build step whose script is lost. Its parcel counts (nonself = 100 parcels / 6998 rows,
Interoception = 11, Cognition = 10) **do not match** the current per-run SampEn CSVs in
`03_intrinsic_neural_metrics/results/sampen/` (which give nonself = 320, Interoception = 14,
Cognition = 12 — the same structure as INT). It is preserved verbatim because it is what
the existing SampEn results come from; it is **not** regenerated here. The nonself layer is
labelled `somatomotor` in this file, but that 100-parcel set is **not** the CAB-NP
Somatomotor network (67 parcels) — the label is a misnomer of unknown origin.

**INT nonself = all ~304 non-self parcels.** Figures label it "Sensory-Motor", but the INT
`glasser_full_dataframe.csv` nonself layer contains every non-self Glasser parcel, not a
somatomotor subset. So INT and SampEn use *different* nonself references and are not
directly comparable on the nonself layer.

If consistency between the two metrics matters for the thesis, regenerate the SampEn
dataframe from the current per-run CSVs with the same logic as the INT builder.
