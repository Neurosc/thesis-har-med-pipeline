# qc_residualize_auc.R — Group-blind OLS residualization of AUC_diff
#
# For each of 6 qinspheres layers (visual, auditory, motor, extero, intero,
# mental) fit OLS across subjects:
#
#   AUC_diff ~ 1 + pcf_diff + pcf_sq_diff + mean_fd_retained_diff
#
# AUC_diff = mean AUC (ses-02) − mean AUC (ses-01), averaged across ROIs
# within the layer.  Residuals are the drug-group-blind, quality-corrected
# per-subject deltas passed to downstream Welch / Mann-Whitney tests.
#
# Inputs
#   99_QC/01_motion_qc/results/fd_covariates_wide_thresh03.csv
#   04_statistics/results/qinspheres/tables/qinspheres_auc.csv
#
# Outputs
#   04_statistics/results/qinspheres/tables/auc_diff_quality_residuals.csv
#   04_statistics/results/qinspheres/tables/auc_diff_quality_model_summary.txt
#
# Run from repo root:
#   Rscript 04_statistics/scripts/qinspheres/qc_residualize_auc.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
REPO_ROOT <- if (length(file_arg) > 0) {
  normalizePath(file.path(dirname(normalizePath(sub("--file=", "", file_arg))),
                          "..", "..", ".."))
} else {
  normalizePath(".")
}

COV_CSV <- file.path(REPO_ROOT, "99_QC", "01_motion_qc", "results",
                     "fd_covariates_wide_thresh03.csv")
AUC_CSV <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres",
                     "tables", "qinspheres_auc.csv")
OUT_DIR  <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres", "tables")
OUT_CSV  <- file.path(OUT_DIR, "auc_diff_quality_residuals.csv")
OUT_LOG  <- file.path(OUT_DIR, "auc_diff_quality_model_summary.txt")

LAYER_ORDER    <- c("visual", "auditory", "motor", "extero", "intero", "mental")
COVARIATE_COLS <- c("pcf_diff", "pcf_sq_diff", "mean_fd_retained_diff")
MIN_N_MODEL    <- 5   # refuse to fit with fewer subjects than this

# ── Logging helper (mirrors output to console + log buffer) ──────────────────
log_lines <- character(0)
lg <- function(...) {
  msg <- paste0(...)
  cat(msg, "\n", sep = "")
  log_lines <<- c(log_lines, msg)
}

lg("═══════════════════════════════════════════════════════════════════════")
lg("QC residualization: AUC_diff ~ 1 + pcf_diff + pcf_sq_diff + mean_fd_retained_diff")
lg("Run: ", format(Sys.time(), "%Y-%m-%d %H:%M:%S"))
lg("═══════════════════════════════════════════════════════════════════════")

# ── Load inputs ───────────────────────────────────────────────────────────────
if (!file.exists(COV_CSV)) stop("Covariates file not found:\n  ", COV_CSV)
if (!file.exists(AUC_CSV)) stop("AUC file not found:\n  ", AUC_CSV)

cov_raw <- read.csv(COV_CSV, stringsAsFactors = FALSE)
auc_raw <- read.csv(AUC_CSV, stringsAsFactors = FALSE)

lg("\n── Input files ──────────────────────────────────────────────────────────")
lg("  Covariates:  ", COV_CSV)
lg(sprintf("    %d rows × %d columns; %d unique subjects",
           nrow(cov_raw), ncol(cov_raw), n_distinct(cov_raw$subject)))
lg("  AUC data:    ", AUC_CSV)
lg(sprintf("    %d rows × %d columns; %d unique subjects",
           nrow(auc_raw), ncol(auc_raw), n_distinct(auc_raw$subject)))

# ── Validate required columns ─────────────────────────────────────────────────
required_cov  <- c("subject", COVARIATE_COLS)
missing_cov   <- setdiff(required_cov, names(cov_raw))
if (length(missing_cov) > 0)
  stop("ERROR: missing columns in covariates CSV: ", paste(missing_cov, collapse = ", "))

required_auc  <- c("subject", "session", "category", "auc")
missing_auc   <- setdiff(required_auc, names(auc_raw))
if (length(missing_auc) > 0)
  stop("ERROR: missing columns in AUC CSV: ", paste(missing_auc, collapse = ", "))

missing_cats  <- setdiff(LAYER_ORDER, unique(auc_raw$category))
if (length(missing_cats) > 0)
  stop("ERROR: missing categories in AUC data: ", paste(missing_cats, collapse = ", "))

missing_sess <- setdiff(c("ses-01", "ses-02"), unique(auc_raw$session))
if (length(missing_sess) > 0)
  stop("ERROR: missing sessions in AUC data: ", paste(missing_sess, collapse = ", "))

# ── Subject reconciliation ────────────────────────────────────────────────────
lg("\n── Subject inventory ────────────────────────────────────────────────────")
cov_subjects <- sort(unique(cov_raw$subject))
auc_subjects <- sort(unique(auc_raw$subject))

lg(sprintf("  Covariates file:  %d subjects", length(cov_subjects)))
lg(sprintf("  AUC file:         %d subjects", length(auc_subjects)))

only_in_cov <- setdiff(cov_subjects, auc_subjects)
only_in_auc <- setdiff(auc_subjects, cov_subjects)

if (length(only_in_cov) > 0) {
  lg(sprintf("  WARNING — in covariates but not AUC (%d): %s",
             length(only_in_cov), paste(only_in_cov, collapse = ", ")))
}
if (length(only_in_auc) > 0) {
  lg(sprintf("  WARNING — in AUC but not covariates (%d): %s",
             length(only_in_auc), paste(only_in_auc, collapse = ", ")))
}

shared_subjects <- intersect(cov_subjects, auc_subjects)
lg(sprintf("  Intersection (will enter analysis): %d subjects", length(shared_subjects)))

# ── Covariate table: restrict to shared subjects, check for NAs ──────────────
cov <- cov_raw %>%
  filter(subject %in% shared_subjects) %>%
  select(all_of(c("subject", COVARIATE_COLS)))

na_per_col <- sapply(COVARIATE_COLS, function(col) sum(is.na(cov[[col]])))
if (any(na_per_col > 0)) {
  bad_cols <- names(na_per_col[na_per_col > 0])
  lg("\n  NA covariate values:")
  for (col in bad_cols)
    lg(sprintf("    %s: %d NAs", col, na_per_col[col]))
}

cov_na_subs <- cov %>%
  filter(if_any(all_of(COVARIATE_COLS), is.na)) %>%
  pull(subject)

if (length(cov_na_subs) > 0) {
  lg(sprintf("  DROPPED (missing covariate values, all layers): %d subjects — %s",
             length(cov_na_subs), paste(sort(cov_na_subs), collapse = ", ")))
  cov             <- cov %>% filter(!subject %in% cov_na_subs)
  shared_subjects <- setdiff(shared_subjects, cov_na_subs)
}

lg(sprintf("  Subjects available for modelling: %d", length(shared_subjects)))

# ── Per-subject × layer AUC_diff ─────────────────────────────────────────────
# Mean AUC across ROIs per subject × session × layer (drop non-finite first)
n_nonfinite <- sum(!is.finite(auc_raw$auc))
if (n_nonfinite > 0)
  lg(sprintf("\n  Note: dropped %d non-finite AUC values before aggregation", n_nonfinite))

subj_ses_layer <- auc_raw %>%
  filter(subject %in% shared_subjects, is.finite(auc)) %>%
  group_by(subject, session, category) %>%
  summarise(mean_auc = mean(auc), .groups = "drop")

# Pivot to pre / post
auc_wide <- subj_ses_layer %>%
  pivot_wider(names_from = session, values_from = mean_auc,
              names_prefix = "ses_")

# Rename ses_ prefix to valid names (ses-01 → pre, ses-02 → post)
names(auc_wide) <- gsub("^ses_ses-01$", "pre",  names(auc_wide))
names(auc_wide) <- gsub("^ses_ses-02$", "post", names(auc_wide))

if (!all(c("pre", "post") %in% names(auc_wide)))
  stop("ERROR: failed to pivot AUC to pre/post columns — check session labels.")

auc_wide <- auc_wide %>%
  mutate(auc_diff = post - pre)

# ── Per-layer OLS ─────────────────────────────────────────────────────────────
lg("\n── Per-layer OLS model fits ─────────────────────────────────────────────")

resid_dfs <- lapply(LAYER_ORDER, function(layer) {
  lg(sprintf("\n  ── %s ──", toupper(layer)))

  layer_data <- auc_wide %>%
    filter(category == layer)

  n_all <- nrow(layer_data)

  # Subjects with complete pre AND post
  n_complete_auc <- sum(!is.na(layer_data$pre) & !is.na(layer_data$post))
  n_missing_pre  <- sum( is.na(layer_data$pre))
  n_missing_post <- sum( is.na(layer_data$post))

  if (n_missing_pre > 0 || n_missing_post > 0) {
    missing_subs <- layer_data %>%
      filter(is.na(pre) | is.na(post)) %>%
      pull(subject)
    lg(sprintf("    Subjects missing ses-01 or ses-02 AUC (%d): %s",
               length(missing_subs), paste(sort(missing_subs), collapse = ", ")))
  }

  layer_data <- layer_data %>%
    filter(!is.na(pre), !is.na(post), !is.na(auc_diff))

  # Inner join with covariates (drops subjects missing from cov)
  model_data <- inner_join(layer_data, cov, by = "subject")
  n_dropped  <- nrow(layer_data) - nrow(model_data)

  if (n_dropped > 0) {
    dropped <- setdiff(layer_data$subject, model_data$subject)
    lg(sprintf("    Dropped (no covariate match, %d): %s",
               n_dropped, paste(sort(dropped), collapse = ", ")))
  }

  n_model <- nrow(model_data)
  lg(sprintf("    N entering model: %d", n_model))

  if (n_model < MIN_N_MODEL) {
    lg(sprintf("    ERROR: only %d subjects — cannot fit OLS (minimum %d). Layer skipped.",
               n_model, MIN_N_MODEL))
    warning(sprintf("Layer '%s': %d subjects after filtering — skipped.", layer, n_model),
            call. = FALSE)
    return(NULL)
  }

  # Fit OLS
  fit <- lm(auc_diff ~ pcf_diff + pcf_sq_diff + mean_fd_retained_diff,
            data = model_data)
  sm  <- summary(fit)

  r2     <- sm$r.squared
  adj_r2 <- sm$adj.r.squared
  ct     <- sm$coefficients  # Estimate, Std.Error, t value, Pr(>|t|)

  lg(sprintf("    R²          = %.4f  (%.1f%% variance explained)", r2, 100 * r2))
  lg(sprintf("    Adjusted R² = %.4f", adj_r2))
  lg("")
  lg(sprintf("    %-28s  %9s  %9s  %8s  %9s",
             "Coefficient", "Estimate", "Std.Err", "t", "p"))
  lg(sprintf("    %s", strrep("─", 68)))
  for (i in seq_len(nrow(ct))) {
    p_str <- formatC(ct[i, "Pr(>|t|)"], format = "f", digits = 4)
    lg(sprintf("    %-28s  %9.5f  %9.5f  %8.3f  %9s",
               rownames(ct)[i],
               ct[i, "Estimate"],
               ct[i, "Std. Error"],
               ct[i, "t value"],
               p_str))
  }

  data.frame(subject  = model_data$subject,
             residual = unname(residuals(fit)),
             stringsAsFactors = FALSE)
})
names(resid_dfs) <- LAYER_ORDER

# ── Assemble wide residuals table ─────────────────────────────────────────────
lg("\n── Output summary ───────────────────────────────────────────────────────")

all_subs <- sort(unique(unlist(lapply(resid_dfs, function(d) {
  if (!is.null(d)) d$subject else character(0)
}))))

resid_wide <- data.frame(subject = all_subs, stringsAsFactors = FALSE)

for (layer in LAYER_ORDER) {
  col <- paste0(layer, "_resid")
  d   <- resid_dfs[[layer]]
  if (is.null(d)) {
    resid_wide[[col]] <- NA_real_
  } else {
    resid_wide[[col]] <- d$residual[match(all_subs, d$subject)]
  }
}

n_full <- sum(complete.cases(resid_wide[, LAYER_ORDER %>% paste0("_resid"), drop = FALSE]))
lg(sprintf("  Output rows:                   %d subjects", nrow(resid_wide)))
lg(sprintf("  Rows with all 6 residuals:     %d", n_full))

skipped_layers <- names(which(sapply(resid_dfs, is.null)))
if (length(skipped_layers) > 0)
  lg(sprintf("  Layers skipped (too few obs):  %s", paste(skipped_layers, collapse = ", ")))

# ── Join drug group ───────────────────────────────────────────────────────────
drug_map <- auc_raw %>% distinct(subject, drug_group)
resid_wide <- resid_wide %>%
  left_join(drug_map, by = "subject") %>%
  select(subject, drug_group, everything())

n_missing_group <- sum(is.na(resid_wide$drug_group))
if (n_missing_group > 0)
  lg(sprintf("  WARNING: %d subjects without drug_group label", n_missing_group))

# ── Write outputs ─────────────────────────────────────────────────────────────
dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)
write.csv(resid_wide, OUT_CSV, row.names = FALSE)
writeLines(log_lines, OUT_LOG)

lg("")
lg(sprintf("Residuals CSV → %s", OUT_CSV))
lg(sprintf("Model log     → %s", OUT_LOG))
