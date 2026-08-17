# 03_global_mixed_model.R — whole-cortex mixed model, same LMM as the qin analysis.
#
# ONE REML LMM on the whole cortex, 70 rows (35 subjects x 2 sessions):
#
#       AUC ~ session * arm + pcf + pcf_sq + mean_fd + (1 | subject)
#
# AUC = mean over ALL 360 Glasser parcels, per subject x session — i.e. one value per run,
# so the model has exactly one row per observation. Kenward-Roger df (lmerTest + pbkrtest).
#
# NOTE — this replaces the former per-network script and its "pooled global" row. That row
# fitted 12 networks x 70 rows (840) under a single subject intercept, treating correlated
# rows as independent, and its p-values were anticonservative. Averaging to one value per
# run first removes the pseudo-replication; no FDR is needed either, since there is now a
# single test per effect rather than 12.
#
# Three effects, exactly as in 04_statistics/scripts/qin/03_mixed_models.R:
#   1. session main effect (sum coding)       -> overall post-pre time effect
#   2. arm main effect (sum coding)           -> verum-placebo baseline
#   3. session:arm (treatment coding)         -> DiD / drug effect
#   + sessionpost within one arm (treatment)  -> within-arm post-pre (placebo by default)
#
# In : 04_statistics/results/{acw|sampen}/parcels/all_parcels/tables/glasser360_{metric}_tidy.csv
#      99_QC/01_motion_qc/results/fd_covariates_wide_thresh03.csv
# Out: .../tables/glasser360_lmm_{pipeline}{_metric}_{overall,simple_{arm},did}.csv
#      + glasser360_lmm_{pipeline}{_metric}_model_summaries.txt
# Run: Rscript 04_statistics/scripts/all_parcels/03_global_mixed_model.R [pipeline] [arms] [metric]
#      defaults: maximal placebo auc   (arms is comma-separated, e.g. "placebo,verum")

suppressPackageStartupMessages({
  library(lmerTest); library(pbkrtest); library(dplyr); library(tidyr)
})

args     <- commandArgs(trailingOnly = TRUE)
PIPELINE <- if (length(args) >= 1) args[1] else "maximal"
ARMS     <- if (length(args) >= 2) strsplit(args[2], ",")[[1]] else "placebo"
METRIC   <- if (length(args) >= 3) args[3] else "auc"          # auc | sampen
MSUF     <- if (METRIC == "auc") "" else paste0("_", METRIC)   # auc keeps the short filenames
MLAB     <- if (METRIC == "sampen") "SampEn" else "AUC"

REPO <- normalizePath(file.path(dirname(sub("^--file=", "",
        grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "..", "..", ".."))
FD   <- file.path(REPO, "99_QC", "01_motion_qc", "results", "fd_covariates_wide_thresh03.csv")
METRIC_DIR <- if (METRIC == "auc") "acw" else METRIC
TAB  <- file.path(REPO, "04_statistics", "results", METRIC_DIR, "parcels", "all_parcels", "tables")
dir.create(TAB, recursive = TRUE, showWarnings = FALSE)
out_path <- function(name, ext = "csv")
  file.path(TAB, sprintf("glasser360_lmm_%s%s_%s.%s", PIPELINE, MSUF, name, ext))
TIDY <- file.path(TAB, sprintf("glasser360_%s_tidy.csv", METRIC))
if (!file.exists(TIDY))
  stop("Missing ", TIDY, " — run 01_build_tidy.py (or 01b for sampen) first.")


# ---------------------------------------------------------------------------
# 1. Long table: one whole-cortex value per subject x session
# ---------------------------------------------------------------------------
# The metric column is named after the metric (auc / sampen); rename to `value` so the model
# formula is metric-agnostic. AUC <= 0 (collapsed autocorrelation, not a valid timescale) is
# dropped as in 01_build_auc.jl; SampEn has no such filter.
raw <- read.csv(TIDY, stringsAsFactors = FALSE) %>%
  rename(value = !!METRIC) %>%
  filter(pipeline == !!PIPELINE, is.finite(value))
if (METRIC == "auc") raw <- raw %>% filter(value > 0)

covariates <- read.csv(FD, stringsAsFactors = FALSE) %>%
  transmute(subject,
            pre.pcf     = pcf_pre,               post.pcf     = pcf_post,
            pre.mean_fd = mean_fd_retained_pre,  post.mean_fd = mean_fd_retained_post) %>%
  pivot_longer(-subject, names_to = c("session", "var"), names_sep = "\\.") %>%
  pivot_wider(names_from = var, values_from = value)

dat <- raw %>%
  group_by(subject, session, arm) %>%
  summarise(AUC = mean(value), .groups = "drop") %>%   # mean over all 360 parcels
  mutate(session = tolower(session)) %>%               # "Pre"/"Post" -> "pre"/"post"
  left_join(covariates, by = c("subject", "session")) %>%
  mutate(pcf_sq  = pcf^2,
         subject = factor(subject),
         session = factor(session, levels = c("pre", "post")),
         arm     = factor(arm, levels = c("placebo", "verum"))) %>%
  select(AUC, subject, session, arm, pcf, pcf_sq, mean_fd)

cat(sprintf("\n######## whole-cortex Glasser 360 / %s / %s ########\n", PIPELINE, METRIC))
cat(sprintf("Long table: %d rows (%d subjects x 2 sessions)\n",
            nrow(dat), nlevels(dat$subject)))


# ---------------------------------------------------------------------------
# 2. Model + helpers (identical to qin/03_mixed_models.R)
# ---------------------------------------------------------------------------
MODEL <- AUC ~ session * arm + pcf + pcf_sq + mean_fd + (1 | subject)

fit_lmm <- function(formula, data)
  suppressWarnings(suppressMessages(lmer(formula, data = data, REML = TRUE)))

coef_table <- function(model)
  tryCatch(summary(model, ddf = "Kenward-Roger")$coefficients,
           error = function(e) summary(model)$coefficients)

term_stats <- function(coefs, term) list(
  est = coefs[term, 1], se = coefs[term, 2],
  df  = if (ncol(coefs) >= 5) coefs[term, 3] else NA,
  p   = coefs[term, ncol(coefs)])

# sum coding: for a 2-level factor the post-pre effect equals -2 * the coefficient
scaled_effect_ci <- function(s) {
  effect <- -2 * s$est
  crit   <- if (!is.na(s$df)) qt(0.975, s$df) else 1.96
  half   <- crit * 2 * s$se
  c(effect - half, effect + half)
}

paired_dz <- function(change) {
  change <- change[is.finite(change)]
  if (length(change) < 2 || sd(change) == 0) return(NA_real_)
  mean(change) / sd(change)
}


# ---------------------------------------------------------------------------
# 3. OVERALL: post-pre time effect + verum-placebo baseline (sum coding)
# ---------------------------------------------------------------------------
d_sum <- dat
d_sum$session <- C(d_sum$session, "contr.sum")
d_sum$arm     <- C(d_sum$arm, "contr.sum")
m_sum   <- fit_lmm(MODEL, d_sum)
coefs   <- coef_table(m_sum)
session <- term_stats(coefs, "session1")   # -> post - pre
arm     <- term_stats(coefs, "arm1")       # -> verum - placebo
s_ci    <- scaled_effect_ci(session); a_ci <- scaled_effect_ci(arm)

changes <- dat %>% select(subject, session, AUC) %>%
  pivot_wider(names_from = session, values_from = AUC) %>%
  mutate(change = post - pre)

overall <- data.frame(
  session_est     = round(-2 * session$est, 4),
  session_dz      = round(paired_dz(changes$change), 3),
  session_ci_low  = round(s_ci[1], 4),
  session_ci_high = round(s_ci[2], 4),
  session_KR_df   = round(session$df, 1),
  session_p       = round(session$p, 4),
  arm_est     = round(-2 * arm$est, 4),
  arm_ci_low  = round(a_ci[1], 4), arm_ci_high = round(a_ci[2], 4),
  arm_p       = round(arm$p, 4),
  singular    = isSingular(m_sum))
write.csv(overall, out_path("overall"), row.names = FALSE)
cat(sprintf("\n==== OVERALL (post-pre time effect, %s) ====\n", MLAB))
print(overall, row.names = FALSE)


# ---------------------------------------------------------------------------
# 3b. WITHIN-ARM post-pre (treatment coding, arm releveled)
# ---------------------------------------------------------------------------
for (a in ARMS) {
  d_arm <- dat; d_arm$arm <- relevel(d_arm$arm, ref = a)
  m     <- fit_lmm(MODEL, d_arm)
  s     <- term_stats(coef_table(m), "sessionpost")
  crit  <- if (!is.na(s$df)) qt(0.975, s$df) else 1.96
  ch    <- dat %>% filter(arm == a) %>% select(subject, session, AUC) %>%
    pivot_wider(names_from = session, values_from = AUC) %>% mutate(change = post - pre)
  simple <- data.frame(
    arm      = a,
    est      = round(s$est, 4),
    dz       = round(paired_dz(ch$change), 3),
    ci_low   = round(s$est - crit * s$se, 4),
    ci_high  = round(s$est + crit * s$se, 4),
    KR_df    = round(s$df, 1),
    p        = round(s$p, 4),
    n        = nrow(ch),
    singular = isSingular(m))
  write.csv(simple, out_path(sprintf("simple_%s", a)), row.names = FALSE)
  cat(sprintf("\n==== WITHIN-ARM post-pre — %s (n=%d) ====\n", a, nrow(ch)))
  print(simple, row.names = FALSE)
}


# ---------------------------------------------------------------------------
# 4. DiD (drug effect) — session:arm, treatment coding
# ---------------------------------------------------------------------------
m_trt <- fit_lmm(MODEL, dat)
d     <- term_stats(coef_table(m_trt), "sessionpost:armverum")
crit  <- if (!is.na(d$df)) qt(0.975, d$df) else 1.96
did <- data.frame(
  est      = round(d$est, 4),
  ci_low   = round(d$est - crit * d$se, 4),
  ci_high  = round(d$est + crit * d$se, 4),
  KR_df    = round(d$df, 1),
  p        = round(d$p, 4),
  singular = isSingular(m_trt))
write.csv(did, out_path("did"), row.names = FALSE)
cat("\n==== DiD (drug effect: session:arm) ====\n")
print(did, row.names = FALSE)


# ---------------------------------------------------------------------------
# 5. Full model summaries -> text file
# ---------------------------------------------------------------------------
con <- file(out_path("model_summaries", "txt"), "w")
sink(con)
cat("Whole-cortex Glasser 360 —", PIPELINE, "/", METRIC, "\n")
cat("Model:", deparse(MODEL), "\n")
cat("AUC = mean over all 360 parcels, one value per subject x session.\n\n")
cat(strrep("=", 70), "\n--- sum coding (overall session / arm) ---\n")
print(summary(m_sum, ddf = "Kenward-Roger"))
cat("\n", strrep("=", 70), "\n--- treatment coding (DiD) ---\n")
print(summary(m_trt, ddf = "Kenward-Roger"))
sink(); close(con)
cat(sprintf("\n(full summaries saved to %s)\n", basename(out_path("model_summaries", "txt"))))
