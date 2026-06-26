# Implementation Prompt — Three Parallel Denoising Pipelines

> **Scope guard:** This task covers the **denoising stage only**
> (`01_denoising/`). Do **NOT** touch timeseries extraction,
> ACW computation, or the statistics pipeline. Those come in a later, separate step.

## Goal

Produce three independent denoised BOLD versions of the same data, so that the
final self-vs-nonself / pre-vs-post / drug results can be shown to be robust to
denoising choice (remove denoising as a source of bias). All three run on the
**same subject sample** so the only thing that varies is the denoising.

## Common sample (all three pipelines)

- **n = 39 — exclude `sub-12` only.** Keep the 4 high-motion subjects
  (`sub-06`, `sub-08`, `sub-26`, `sub-36`) that the legacy n=35 rule had dropped.
- 39 subjects × 2 sessions = **78 runs** per pipeline.
- Do **not** edit `utils/subject_filter.py:get_included_subjects()` (n=35) — the
  legacy analysis still depends on it. **Add a new function** instead, e.g.
  `get_pipeline_subjects()` returning all subjects minus `sub-12`.
- Motion caveat to record now (handled later in stats, not here): pipelines 1 and
  2 do no censoring, so the 4 high-motion subjects keep all contaminated frames.
  Plan to carry **mean FD as a subject-level covariate** and/or a sensitivity
  analysis dropping those 4 — but that is a STATS-stage concern, not this task.

## The three pipelines (step matrix)

| Step | 1. `detrend` (Keskin) | 2. `glm` | 3. `maximal` |
|---|---|---|---|
| Polynomial detrend | **yes** (order = CONFIRM, default polort 1) | within GLM | within GLM |
| WM + CSF regressors | no | **yes** | yes |
| Motion regressors (6 + 6 deriv) | no | **yes** | yes |
| FD>0.3 frame censoring | no | **no** | **yes** |
| Lomb-Scargle interpolation | no | no | **yes** |
| Bandpass 0.01–0.1 Hz | no | **no** | **yes** |
| FPC + FPCsq regressors | no | no | **yes (NEW — see open items)** |
| GSR | no | no (default) | no (NoGSR) |

- **Pipeline 1 (`detrend`)** = Keskin et al. 2025 style. Detrend only, nothing else.
- **Pipeline 2 (`glm`)** = full nuisance GLM **minus** censoring, interpolation,
  and bandpass. Same regressor list as pipeline 3 except FPC/FPCsq.
- **Pipeline 3 (`maximal`)** = current pipeline **plus** the new FPC + FPCsq
  regressors. Note: because FPC/FPCsq are new, pipeline 3 must be **re-run from
  scratch** (existing `desc-denoisedNoGSR` outputs are superseded).

## Distinct folder + naming convention (must be unambiguous)

Outputs (NIfTIs + per-pipeline log) go to **separate sibling folders**, one per
pipeline — no shared directory, no overlap:

```
01_denoising/results/
├── detrend/
│   ├── sub-XX_ses-YY_task-rest_desc-detrend_bold.nii.gz
│   └── _batch_log.tsv
├── glm/
│   ├── sub-XX_ses-YY_task-rest_desc-glm_bold.nii.gz
│   └── _batch_log.tsv
└── maximal/
    ├── sub-XX_ses-YY_task-rest_desc-maximal_bold.nii.gz
    └── _batch_log.tsv
```

- The pipeline name appears **both** in the folder and in the `desc-` filename tag,
  so a file is self-identifying even if moved.
- Legacy `desc-denoisedGSR` / `desc-denoisedNoGSR` files in the flat `results/`
  root: leave in place but treat as obsolete (do not delete in this task; flag for
  archival to `_old/` later).

## Code changes

Prefer **one parameterized core** over three copy-pasted scripts (avoids drift):

- `denoise_pipelines.py` — add a `pipeline` argument (`"detrend" | "glm" | "maximal"`)
  that toggles each step as a boolean config: `do_nuisance`, `do_motion`,
  `do_censor`, `do_interp`, `do_bandpass`, `do_gsr`, `do_fpc`, `detrend_order`.
  Each pipeline is just one preset of these flags (the step matrix above).
- `01_denoise_all.py` — accept `--pipeline {detrend,glm,maximal}`; iterate the
  n=39 sample from `get_pipeline_subjects()`; write to the matching folder; skip
  existing outputs; append to that pipeline's `_batch_log.tsv`.
- `00_test_one_subject.py` — same `--pipeline` flag for the sub-01 ses-01 test.
- Keep the existing Lomb-Scargle / bandpass implementation for the `maximal` path
  unchanged except for adding FPC/FPCsq to the regressor matrix.

## Environment / where it runs

- Runs **on the server** (raw fMRIPrep NIfTIs are server-only): `conda activate fmri`.
- fMRIPrep outputs are READ-ONLY. Print summary stats to console per run.

## Open items — CONFIRM before running

1. **FPC / FPCsq definition (blocking for pipeline 3).** These are not defined
   anywhere in CLAUDE.md. State exactly what they are (regressor source + how
   computed) and "sq" = element-wise square. Pipeline 3 cannot be written correctly
   without this.
2. **Detrend order for pipeline 1** — linear (polort 1) or linear+quadratic
   (polort 2)? Default assumed: polort 1.
3. **Pipeline 2 motion model** — confirm 6 + 6-derivative (12 params), matching
   pipeline 3 minus FPC/FPCsq. (Not the full Friston-24.)
4. **GSR** — assumed OFF for all three (NoGSR). Confirm you don't also want GSR
   variants of each at this stage.

## Definition of done

- `get_pipeline_subjects()` added (n=39), legacy filter untouched.
- `denoise_pipelines.py` parameterized; batch + single-subject scripts take `--pipeline`.
- Three populated folders (`detrend/`, `glm/`, `maximal/`), 78 runs each, each with
  its own `_batch_log.tsv`.
- CLAUDE.md "Decisions Made So Far" updated to document the three-pipeline design,
  the n=39 common sample, and the FPC/FPCsq addition.
- No changes to timeseries / ACW / statistics code.
