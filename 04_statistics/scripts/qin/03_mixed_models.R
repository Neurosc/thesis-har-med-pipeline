# 03_mixed_models.R — per-region mixed models for the DRUG effect and the TIME effect.
#
# We fit ONE linear mixed model PER REGION, on that region's 70 rows (35 subjects x 2 sessions):
#
#       AUC ~ session * arm + pcf + pcf_sq + mean_fd + (1 | subject)
#
#   - session (pre/post) and arm (placebo/verum) are the effects of interest
#   - pcf, pcf_sq, mean_fd are motion/quality covariates (adjust for head motion)
#   - (1 | subject) is a random intercept: each subject is their own baseline
#
# From each model we read three things:
#   1. DiD  (drug effect) : the session:arm interaction — does verum change post-pre differently than placebo?
#   2. Overall (time)     : the session and arm main effects, averaged over the other factor
#   3. Global             : the same effects from ONE pooled model across all 5 regions
# p-values use Kenward-Roger degrees of freedom; we BH-FDR correct across the 5 regions.
# We also produce two fit checks: R-squared per region, and residual diagnostic plots.
#
# Usage : Rscript 04_statistics/scripts/qin/03_mixed_models.R [pipeline] [atlas] [metric]
#         defaults: maximal qinspheres auc
# Output: results/mixed_models/per_config/{atlas}_{pipeline}{_metric}_{longtable,did,simple,overall,overall_global,r2}.csv
#         + _model_summaries.txt ; residual plots ->
#         results/{acw|sampen}/{spheres|parcels/self_regions}/{pipeline}/figures/qin_diag_*.png


# ===========================================================================
# 0. Packages
# ===========================================================================
needed <- c("lme4", "lmerTest", "pbkrtest", "dplyr", "tidyr", "MuMIn")
for (pkg in needed)
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
suppressPackageStartupMessages({
  library(lmerTest)   # lmer() plus Kenward-Roger / Satterthwaite p-values
  library(pbkrtest)
  library(dplyr)
  library(tidyr)
})


# ===========================================================================
# 1. Settings: which pipeline / atlas / metric to run
#    (from command-line args, or defaults; can also be set before source())
# ===========================================================================
args <- commandArgs(trailingOnly = TRUE)
PIPELINE <- if (exists(".pipeline")) .pipeline else if (length(args) >= 1) args[1] else "maximal"
ATLAS    <- if (exists(".atlas"))    .atlas    else if (length(args) >= 2) args[2] else "qinspheres"
METRIC   <- if (exists(".metric"))   .metric   else if (length(args) >= 3) args[3] else "auc"
MSUF     <- if (METRIC == "auc") "" else paste0("_", METRIC)   # filename infix; auc keeps the short names


# ===========================================================================
# 2. File paths
# ===========================================================================
# Find the repository root so paths work no matter where the script is launched from.
find_repo_root <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)     # set when run with Rscript
  script_path <- if (length(file_arg)) sub("^--file=", "", file_arg[1]) else {
    path <- NULL                                                     # set when the file is source()d
    for (i in rev(seq_len(sys.nframe())))
      if (!is.null(sys.frame(i)$ofile)) { path <- sys.frame(i)$ofile; break }
    path
  }
  if (is.null(script_path) || is.na(script_path)) normalizePath(getwd())
  else normalizePath(file.path(dirname(normalizePath(script_path)), "..", "..", ".."))
}
REPO <- find_repo_root()

# Results tree mirrors 03_intrinsic_neural_metrics/results — one shape for every metric:
#   results/{acw|sampen|fc}/{spheres|parcels/self_regions}/{pipeline}/{tables,figures}
METRIC_DIR <- if (METRIC == "auc") "acw" else METRIC
ATLAS_DIR  <- if (ATLAS == "qinspheres") "spheres" else file.path("parcels", "self_regions")
CONFIG_DIR <- file.path(REPO, "04_statistics", "results", METRIC_DIR, ATLAS_DIR, PIPELINE)
# Flat filename token for the per_config outputs, so they carry the same vocabulary as the
# results tree: "spheres" | "parcels_self_regions" (the CLI arg stays qinspheres/qinparcels).
ATLAS_TAG  <- gsub("/", "_", ATLAS_DIR)

# Input: the per-ROI metric table (built by 01_build_auc.jl / 02_build_sampen.py).
INPUT_TABLE <- file.path(CONFIG_DIR, "tables", sprintf("%s.csv", METRIC))

# Per-session motion covariates.
COVARIATES_TABLE <- file.path(REPO, "99_QC", "01_motion_qc", "results", "fd_covariates_wide_thresh03.csv")

# Where results and figures go.
OUT_DIR <- file.path(REPO, "04_statistics", "results", "mixed_models", "per_config")
FIG_DIR <- file.path(CONFIG_DIR, "figures")
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# Helper for output file names, e.g. out_path("did") -> ".../spheres_maximal_did.csv"
out_path <- function(name, ext = "csv")
  file.path(OUT_DIR, sprintf("%s_%s%s_%s.%s", ATLAS_TAG, PIPELINE, MSUF, name, ext))

# The 6 category labels map to region names; we analyse 5 (motor is excluded — its
# broadband AUC is degraded — and FDR correction is over the remaining 5).
CATEGORY_TO_REGION <- c(intero = "intero", extero = "extero", mental = "cognition",
                        visual = "visual", motor = "motor", auditory = "auditory")
REGIONS <- setdiff(unname(CATEGORY_TO_REGION), "motor")


# ===========================================================================
# 3. Build the long table: one row per subject x session x region
# ===========================================================================
# 3a. read per-ROI values and average them within each region
raw <- read.csv(INPUT_TABLE)
raw <- raw[is.finite(raw$auc), ]
if (METRIC == "auc") raw <- raw[raw$auc > 0, ]   # AUC <= 0 = collapsed autocorrelation, not a valid timescale

region_means <- raw %>%
  group_by(subject, session, drug_group, category) %>%
  summarise(AUC = mean(auc), .groups = "drop")

# 3b. reshape motion covariates from wide (pre/post columns) to one row per subject x session
covariates <- read.csv(COVARIATES_TABLE) %>%
  transmute(subject,
            pre.pcf     = pcf_pre,               post.pcf     = pcf_post,
            pre.mean_fd = mean_fd_retained_pre,  post.mean_fd = mean_fd_retained_post) %>%
  pivot_longer(-subject, names_to = c("session", "var"), names_sep = "\\.") %>%
  pivot_wider(names_from = var, values_from = value)

# 3c. join everything and set up the factors the model needs
dat <- region_means %>%
  mutate(session = ifelse(session == "ses-01", "pre", "post"),
         region  = CATEGORY_TO_REGION[category]) %>%
  left_join(covariates, by = c("subject", "session")) %>%
  mutate(pcf_sq  = pcf^2,
         subject = factor(subject),
         session = factor(session, levels = c("pre", "post")),      # reference = pre
         arm     = factor(drug_group, levels = c("placebo", "verum")),  # reference = placebo
         region  = factor(region, levels = REGIONS)) %>%
  filter(!is.na(region)) %>%                                        # drops the motor rows
  select(AUC, subject, session, arm, region, pcf, pcf_sq, mean_fd)

write.csv(dat, out_path("longtable"), row.names = FALSE)
cat(sprintf("Long table: %d rows (%d subjects x 2 sessions x %d regions)\n",
            nrow(dat), nlevels(dat$subject), length(REGIONS)))


# ===========================================================================
# 4. The model, and a few small helper functions
# ===========================================================================
# The one model we fit per region:
REGION_MODEL <- AUC ~ session * arm + pcf + pcf_sq + mean_fd + (1 | subject)
# The pooled model across all regions (adds region as a fixed effect):
GLOBAL_MODEL <- AUC ~ session * arm + region + pcf + pcf_sq + mean_fd + (1 | subject)

# Fit an LMM quietly (warnings are handled separately in section 8).
fit_lmm <- function(formula, data)
  suppressWarnings(suppressMessages(lmer(formula, data = data, REML = TRUE)))

# Get the fixed-effect coefficient table, using Kenward-Roger df when available.
coef_table <- function(model)
  tryCatch(summary(model, ddf = "Kenward-Roger")$coefficients,
           error = function(e) summary(model)$coefficients)

# Pull one term's estimate / SE / df / p-value out of a coefficient table.
term_stats <- function(coefs, term) list(
  est = coefs[term, 1],
  se  = coefs[term, 2],
  df  = if (ncol(coefs) >= 5) coefs[term, 3] else NA,   # KR adds a df column; plain summary does not
  p   = coefs[term, ncol(coefs)])                        # p-value is always the last column

# Permutation p-value for the drug effect, used ONLY when an LMM fit is singular
# (unreliable). Shuffle the verum/placebo labels 10,000 times and see how often the
# between-arm difference in change is as large as the one we observed.
permutation_pvalue <- function(change, arm) {
  set.seed(42)
  observed <- mean(change[arm == "verum"]) - mean(change[arm == "placebo"])
  n_verum  <- sum(arm == "verum")
  null_diffs <- replicate(10000, {
    picked <- sample(length(change), n_verum)
    mean(change[picked]) - mean(change[-picked])
  })
  (sum(abs(null_diffs) >= abs(observed)) + 1) / 10001
}


# ===========================================================================
# 5. Analysis 1 — DRUG effect (DiD): the session:arm interaction, per region
# ===========================================================================
did_one_region <- function(region_name) {
  d <- dat %>% filter(region == region_name)
  model <- fit_lmm(REGION_MODEL, d)                         # treatment coding -> term "sessionpost:armverum"
  stats <- term_stats(coef_table(model), "sessionpost:armverum")

  # 95% confidence interval (t-based if we have df, else normal approximation)
  crit    <- if (!is.na(stats$df)) qt(0.975, stats$df) else 1.96
  ci_low  <- stats$est - crit * stats$se
  ci_high <- stats$est + crit * stats$se

  # standardized effect size (Cohen's dz): each subject's post-pre change, compared between arms
  changes <- d %>%
    select(subject, arm, session, AUC) %>%
    pivot_wider(names_from = session, values_from = AUC) %>%
    filter(!is.na(pre), !is.na(post)) %>%
    mutate(change = post - pre)
  verum_change   <- changes$change[changes$arm == "verum"]
  placebo_change <- changes$change[changes$arm == "placebo"]
  pooled_sd <- sqrt(((length(verum_change)   - 1) * var(verum_change) +
                     (length(placebo_change) - 1) * var(placebo_change)) /
                    (length(verum_change) + length(placebo_change) - 2))
  dz <- (mean(verum_change) - mean(placebo_change)) / pooled_sd

  # if the model was singular, trust the permutation test instead of the LMM p-value
  p_value <- if (isSingular(model)) permutation_pvalue(changes$change, changes$arm) else stats$p

  data.frame(region = region_name,
             effect   = round(stats$est, 4),
             ci_low   = round(ci_low, 4), ci_high = round(ci_high, 4),
             dz       = round(dz, 3),
             KR_df    = round(stats$df, 1),
             p        = round(p_value, 4),
             n        = nrow(changes),
             singular = isSingular(model))
}

run_did <- function() {
  results <- do.call(rbind, lapply(REGIONS, did_one_region))
  results$q        <- round(p.adjust(results$p, "BH"), 4)      # FDR across the 5 regions
  results$survives <- ifelse(results$q < 0.05, "yes", "no")
  results <- results[, c("region", "effect", "dz", "ci_low", "ci_high",
                         "p", "q", "survives", "n", "singular", "KR_df")]
  write.csv(results, out_path("did"), row.names = FALSE)
  cat(sprintf("\n==== DRUG EFFECT (DiD) — %s/%s/%s (n=%d) ====\n",
              ATLAS, PIPELINE, METRIC, results$n[1]))
  print(results, row.names = FALSE)
  results
}


# ===========================================================================
# 5b. Analysis 1b — WITHIN-ARM simple effects: post-pre inside each arm, per region
# ===========================================================================
# With treatment coding (session ref = pre, arm ref = placebo), the "sessionpost"
# coefficient IS the placebo post-pre change. Releveling arm so verum is the reference
# makes "sessionpost" the verum post-pre change. Both use Kenward-Roger df. These are the
# LMM equivalents of the per-group paired-t brackets on qin_retreat_paired_prepost.png.
simple_one_arm <- function(region_name, ref_arm) {
  d <- dat %>% filter(region == region_name)
  d$arm <- relevel(d$arm, ref = ref_arm)              # sessionpost = post-pre inside ref_arm
  model <- fit_lmm(REGION_MODEL, d)
  s     <- term_stats(coef_table(model), "sessionpost")
  crit  <- if (!is.na(s$df)) qt(0.975, s$df) else 1.96
  # standardized within-arm effect: each subject's post-pre change inside this arm
  changes <- d %>% filter(arm == ref_arm) %>% select(subject, session, AUC) %>%
    pivot_wider(names_from = session, values_from = AUC) %>%
    mutate(change = post - pre)
  data.frame(region = region_name, arm = ref_arm,
             est     = round(s$est, 4),
             dz      = round(paired_dz(changes$change), 3),
             ci_low  = round(s$est - crit * s$se, 4),
             ci_high = round(s$est + crit * s$se, 4),
             KR_df   = round(s$df, 1),
             p       = round(s$p, 4),
             singular = isSingular(model))
}

run_simple <- function() {
  results <- do.call(rbind, unlist(
    lapply(REGIONS, function(r) lapply(c("placebo", "verum"),
                                       function(a) simple_one_arm(r, a))), recursive = FALSE))
  results$q <- NA_real_                                # BH-FDR across the 5 regions, within each arm
  for (a in c("placebo", "verum")) {
    idx <- results$arm == a
    results$q[idx] <- round(p.adjust(results$p[idx], "BH"), 4)
  }
  results$survives <- ifelse(results$q < 0.05, "yes", "no")
  results <- results[, c("region", "arm", "est", "dz", "ci_low", "ci_high",
                         "p", "q", "survives", "KR_df", "singular")]
  write.csv(results, out_path("simple"), row.names = FALSE)
  cat(sprintf("\n==== WITHIN-ARM SIMPLE EFFECTS (post-pre) — %s/%s/%s ====\n", ATLAS, PIPELINE, METRIC))
  cat("est = post-pre inside each arm (LMM, KR df); q = BH-FDR across 5 regions within arm\n")
  print(results, row.names = FALSE)
  results
}


# ===========================================================================
# 6. Analysis 2 — OVERALL effects (time & baseline), per region
# ===========================================================================
# With SUM coding, each main effect is averaged over the other factor. For a 2-level
# factor the "post - pre" (and "verum - placebo") effect equals -2 * the coefficient.
scaled_effect_ci <- function(stats) {
  effect <- -2 * stats$est
  crit   <- if (!is.na(stats$df)) qt(0.975, stats$df) else 1.96
  half   <- crit * 2 * stats$se
  c(effect - half, effect + half)
}

# Cohen's dz for a within-subject post-pre change vector (mean change / SD of change).
paired_dz <- function(change) {
  change <- change[is.finite(change)]
  if (length(change) < 2 || sd(change) == 0) return(NA_real_)
  mean(change) / sd(change)
}

overall_one_region <- function(region_name) {
  d <- dat %>% filter(region == region_name)
  d$session <- C(d$session, "contr.sum")   # switch to sum coding
  d$arm     <- C(d$arm, "contr.sum")
  coefs <- coef_table(model <- fit_lmm(REGION_MODEL, d))
  session <- term_stats(coefs, "session1")   # -> post - pre  (time effect)
  arm     <- term_stats(coefs, "arm1")       # -> verum - placebo  (baseline difference)
  session_ci <- scaled_effect_ci(session)
  arm_ci     <- scaled_effect_ci(arm)
  # standardized time effect: each subject's post-pre change, pooled over both arms
  changes <- d %>% select(subject, session, AUC) %>%
    pivot_wider(names_from = session, values_from = AUC) %>%
    mutate(change = post - pre)
  data.frame(region = region_name,
             session_est     = round(-2 * session$est, 4),
             session_dz      = round(paired_dz(changes$change), 3),
             session_ci_low  = round(session_ci[1], 4),
             session_ci_high = round(session_ci[2], 4),
             session_p       = round(session$p, 4),
             arm_est     = round(-2 * arm$est, 4),
             arm_ci_low  = round(arm_ci[1], 4),
             arm_ci_high = round(arm_ci[2], 4),
             arm_p       = round(arm$p, 4),
             singular    = isSingular(model))
}

run_overall <- function() {
  results <- do.call(rbind, lapply(REGIONS, overall_one_region))
  results$session_q <- round(p.adjust(results$session_p, "BH"), 4)
  results$arm_q     <- round(p.adjust(results$arm_p, "BH"), 4)
  results <- results[, c("region",
                         "session_est", "session_dz", "session_ci_low", "session_ci_high", "session_p", "session_q",
                         "arm_est", "arm_ci_low", "arm_ci_high", "arm_p", "arm_q", "singular")]
  write.csv(results, out_path("overall"), row.names = FALSE)
  cat(sprintf("\n==== OVERALL EFFECTS — %s/%s/%s ====\n", ATLAS, PIPELINE, METRIC))
  cat("session_est = post-pre (time, averaged over arm); arm_est = verum-placebo (baseline)\n")
  print(results, row.names = FALSE)
  results
}


# ===========================================================================
# 7. Analysis 3 — one POOLED model across all 5 regions
# ===========================================================================
run_global <- function() {
  # sum-coded fit gives the averaged session / arm main effects
  d_sum <- dat
  d_sum$session <- C(d_sum$session, "contr.sum")
  d_sum$arm     <- C(d_sum$arm, "contr.sum")
  model_sum <- fit_lmm(GLOBAL_MODEL, d_sum)
  session <- term_stats(coef_table(model_sum), "session1")
  arm     <- term_stats(coef_table(model_sum), "arm1")

  # treatment-coded fit gives an unambiguous DiD interaction
  model_trt <- fit_lmm(GLOBAL_MODEL, dat)
  did <- term_stats(coef_table(model_trt), "sessionpost:armverum")

  results <- data.frame(
    term = c("session_post_minus_pre", "arm_verum_minus_placebo", "DiD_session:arm"),
    est  = round(c(-2 * session$est, -2 * arm$est, did$est), 4),
    p    = round(c(session$p, arm$p, did$p), 4),
    singular = c(isSingular(model_sum), isSingular(model_sum), isSingular(model_trt)))
  write.csv(results, out_path("overall_global"), row.names = FALSE)
  cat(sprintf("\n==== POOLED GLOBAL (5 regions) — %s/%s/%s ====\n", ATLAS, PIPELINE, METRIC))
  print(results, row.names = FALSE)
  results
}


# ===========================================================================
# 8. Fit the 5 per-region models ONCE (treatment coding), keeping any warnings,
#    so the fit checks below can reuse the same fitted objects.
# ===========================================================================
fit_all_regions <- function() {
  models <- lapply(REGIONS, function(region_name) {
    warnings_seen <- character(0)
    model <- withCallingHandlers(
      suppressMessages(lmer(REGION_MODEL, data = dat %>% filter(region == region_name), REML = TRUE)),
      warning = function(w) {                                  # catch (and silence) warnings so we can report them
        warnings_seen <<- c(warnings_seen, conditionMessage(w))
        invokeRestart("muffleWarning")
      })
    convergence_msgs <- unlist(model@optinfo$conv$lme4$messages)
    list(region = region_name, model = model,
         warnings = unique(c(warnings_seen, convergence_msgs)))
  })
  setNames(models, REGIONS)
}


# ===========================================================================
# 9. Fit report: short lines to the console, full summaries to a text file
# ===========================================================================
report_fits <- function(models) {
  # (a) full summary() of every model -> text file (durable record; not printed)
  summary_file <- out_path("model_summaries", "txt")
  con <- file(summary_file, "w"); on.exit(close(con))
  write_block <- function(title, model)
    cat(strrep("=", 70), title, strrep("=", 70),
        paste(capture.output(print(summary(model))), collapse = "\n"), "",
        sep = "\n", file = con)
  for (region_name in REGIONS)
    write_block(sprintf("REGION: %s | %s", region_name,
                        "AUC ~ session*arm + pcf + pcf_sq + mean_fd + (1|subject)"),
                models[[region_name]]$model)
  write_block("POOLED GLOBAL (5 regions) | AUC ~ session*arm + region + covars + (1|subject)",
              fit_lmm(GLOBAL_MODEL, dat))

  # (b) compact per-region report -> console
  cat(sprintf("\n######## PER-REGION FIT REPORT — %s/%s/%s ########\n", ATLAS, PIPELINE, METRIC))
  for (region_name in REGIONS) {
    model <- models[[region_name]]$model
    coefs <- coef_table(model)
    p_col <- ncol(coefs)
    session_post <- coefs["sessionpost", ]
    did          <- coefs["sessionpost:armverum", ]
    cat(sprintf("- %-10s sessionpost: est=%+.4f p=%.4f | DiD: est=%+.4f p=%.4f | singular=%s\n",
                region_name, session_post[1], session_post[p_col],
                did[1], did[p_col], isSingular(model)))
    warns <- models[[region_name]]$warnings
    if (length(warns))
      cat(sprintf("    warning: %s\n", paste(gsub("\\s+", " ", warns), collapse = " ; ")))
  }
  cat(sprintf("(full summaries saved to %s)\n", basename(summary_file)))
}


# ===========================================================================
# 10. Fit check 1 — R-squared per region
#     marginal    = variance explained by the fixed effects
#     conditional = fixed effects + subject random intercept
# ===========================================================================
report_r2 <- function(models) {
  rows <- lapply(models, function(item) {
    r2 <- suppressWarnings(MuMIn::r.squaredGLMM(item$model))
    data.frame(region = item$region,
               R2_marginal    = round(as.numeric(r2[1, "R2m"]), 3),
               R2_conditional = round(as.numeric(r2[1, "R2c"]), 3))
  })
  table <- do.call(rbind, rows); rownames(table) <- NULL
  write.csv(table, out_path("r2"), row.names = FALSE)
  cat(sprintf("\n==== R-SQUARED per region — %s/%s/%s ====\n", ATLAS, PIPELINE, METRIC))
  cat("marginal = fixed effects only; conditional = fixed effects + subject random intercept\n")
  print(table, row.names = FALSE)
  table
}


# ===========================================================================
# 11. Fit check 2 — residuals-vs-fitted and Q-Q plots (one panel per region)
# ===========================================================================
report_diagnostics <- function(models) {
  save_grid <- function(file_name, title, draw_one) {
    png(file.path(FIG_DIR, file_name), width = 1600, height = 1050, res = 140)
    old <- par(mfrow = c(2, 3), mar = c(4, 4, 2.6, 1), oma = c(0, 0, 2, 0))
    for (region_name in REGIONS) draw_one(models[[region_name]]$model, region_name)
    mtext(sprintf("%s — %s/%s/%s", title, ATLAS, PIPELINE, METRIC), outer = TRUE, cex = 1)
    par(old); dev.off()
  }
  save_grid("qin_diag_resid_vs_fitted.png", "Residuals vs fitted", function(model, region_name) {
    plot(fitted(model), resid(model), main = region_name,
         xlab = "Fitted", ylab = "Residuals", pch = 16, col = rgb(0, 0, 0, 0.4))
    abline(h = 0, col = "red", lty = 2)
  })
  save_grid("qin_diag_qq.png", "Normal Q-Q of residuals", function(model, region_name) {
    r <- resid(model)
    qqnorm(r, main = region_name, pch = 16, col = rgb(0, 0, 0, 0.4))
    qqline(r, col = "red", lwd = 2)
  })
  cat("Saved diagnostic plots: qin_diag_resid_vs_fitted.png, qin_diag_qq.png\n")
}


# ===========================================================================
# 12. Run everything
# ===========================================================================
cat(sprintf("\n######## %s / %s / %s ########\n", ATLAS, PIPELINE, METRIC))
models <- fit_all_regions()
invisible(run_did())
invisible(run_simple())
invisible(run_overall())
invisible(run_global())
invisible(report_fits(models))
invisible(report_r2(models))
invisible(report_diagnostics(models))
