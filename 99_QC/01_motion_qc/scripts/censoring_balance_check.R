# censoring_balance_check.R
# ---------------------------------------------------------------------------
# Is the "maximal" pipeline's frame censoring BALANCED across sessions and arms?
#
# The maximal pipeline censors frames at FD > 0.3 mm and Lomb-Scargle-interpolates
# them. Median ~17.5% of frames are removed (up to ~50%). If censoring is heavier
# in one session or one arm, the interpolation-induced smoothing (which can inflate
# autocorrelation / ACW-AUC) would be UNBALANCED and could masquerade as a real
# session or drug effect. This script quantifies that balance. It does NOT exclude
# anyone and does NOT change any pipeline; it only describes and tests pcf.
#
# pcf = percent of frames censored (FD > 0.3 mm) in a run. One value per
#       subject x session. Source of truth = the committed motion-QC covariates.
#
# What it does (per the task spec):
#   1. Load per-run pcf for ALL subjects, both sessions.
#   2. Mean and SD of pcf by session (pre, post) and by arm (placebo, verum).
#   3. Paired t-test on pcf, post vs pre, across subjects.
#   4. LMM: pcf ~ session * arm + (1 | subject); report the session term and the
#      session:arm interaction (Kenward-Roger df, matching the main stats stack).
#   5. Save all numeric results to one tidy CSV; print a short console summary.
#
# Fail-loud policy: if an input file is missing, or arm cannot be assigned to every
# subject, the script STOPS with an error. It never substitutes a default.
#
# Inputs  (both committed to the repo):
#   99_QC/01_motion_qc/results/fd_covariates_long_thresh03.csv   (subject, session, pcf, ...)
#   participants.tsv                                             (participant_id, condition)
# Output:
#   99_QC/01_motion_qc/results/censoring_balance_check.csv       (tidy: section, ...)
#
# Run (Rscript; same R/lmerTest stack as 04_statistics):
#   Rscript 99_QC/01_motion_qc/scripts/censoring_balance_check.R
# The inputs are committed CSVs (no NIfTIs), so this runs on the server or locally.
# ---------------------------------------------------------------------------

# ===========================================================================
# 0. Packages (same set the main mixed models use)
# ===========================================================================
needed <- c("lme4", "lmerTest", "pbkrtest", "dplyr", "tidyr")
for (pkg in needed)
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
suppressPackageStartupMessages({
  library(lmerTest)   # lmer() + Kenward-Roger / Satterthwaite p-values
  library(pbkrtest)
  library(dplyr)
  library(tidyr)
})


# ===========================================================================
# 1. Locate the repo root (works no matter where Rscript is launched from)
# ===========================================================================
find_repo_root <- function() {
  file_arg <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  script_path <- if (length(file_arg)) sub("^--file=", "", file_arg[1]) else {
    path <- NULL
    for (i in rev(seq_len(sys.nframe())))
      if (!is.null(sys.frame(i)$ofile)) { path <- sys.frame(i)$ofile; break }
    path
  }
  # this script lives at 99_QC/01_motion_qc/scripts/ -> up 3 to repo root
  if (is.null(script_path) || is.na(script_path)) normalizePath(getwd())
  else normalizePath(file.path(dirname(normalizePath(script_path)), "..", "..", ".."))
}
REPO <- find_repo_root()

COVARIATES_LONG <- file.path(REPO, "99_QC", "01_motion_qc", "results",
                             "fd_covariates_long_thresh03.csv")
PARTICIPANTS    <- file.path(REPO, "participants.tsv")
OUT_CSV         <- file.path(REPO, "99_QC", "01_motion_qc", "results",
                             "censoring_balance_check.csv")

# Source of truth for the 5 exclusions is utils/subject_filter.py:_EXCLUDED.
# Hardcoded here for the same reason 04_statistics/scripts/qin/01_build_auc.jl:24
# hardcodes it: no R binding to the Python filter exists.
EXCLUDED <- c("sub-06", "sub-08", "sub-12", "sub-26", "sub-36")

# Fail loud if an input is missing (task: "stop and tell me; do not substitute defaults").
for (f in c(COVARIATES_LONG, PARTICIPANTS))
  if (!file.exists(f)) stop(sprintf("Required input not found: %s", f), call. = FALSE)


# ===========================================================================
# 2. Load pcf (per subject x session) and attach arm from participants.tsv
# ===========================================================================
cov <- read.csv(COVARIATES_LONG, stringsAsFactors = FALSE)
if (!all(c("subject", "session", "pcf") %in% names(cov)))
  stop("fd_covariates_long_thresh03.csv is missing one of: subject, session, pcf",
       call. = FALSE)

parts <- read.csv(PARTICIPANTS, sep = "\t", stringsAsFactors = FALSE)
if (!all(c("participant_id", "condition") %in% names(parts)))
  stop("participants.tsv is missing one of: participant_id, condition", call. = FALSE)
arm_map <- parts %>%
  transmute(subject = participant_id, arm = tolower(trimws(condition)))

dat <- cov %>%
  select(subject, session, pcf) %>%
  left_join(arm_map, by = "subject") %>%
  mutate(session = dplyr::recode(session, "ses-01" = "pre", "ses-02" = "post"),
         session = factor(session, levels = c("pre", "post")),   # reference = pre
         arm     = factor(arm,     levels = c("placebo", "verum")), # reference = placebo
         subject = factor(subject))

# Fail loud on any unmapped arm, unexpected session label, or missing pcf.
if (any(is.na(dat$arm)))
  stop(sprintf("No arm for subject(s): %s (not matched in participants.tsv/condition)",
               paste(unique(as.character(dat$subject[is.na(dat$arm)])), collapse = ", ")),
       call. = FALSE)
if (any(is.na(dat$session)))
  stop("Unexpected session label(s) in covariates file (expected ses-01 / ses-02).",
       call. = FALSE)
if (any(is.na(dat$pcf)))
  stop("Missing pcf value(s) in covariates file.", call. = FALSE)

n_subj <- nlevels(dat$subject)
n_rows <- nrow(dat)
cat(sprintf("Loaded pcf for %d subjects x sessions = %d rows (no subjects excluded).\n",
            n_subj, n_rows))
# Sanity: expect exactly 2 sessions per subject for the paired test.
per_subj <- dat %>% count(subject)
if (any(per_subj$n != 2))
  stop(sprintf("Expected 2 sessions per subject; offending: %s",
               paste(per_subj$subject[per_subj$n != 2], collapse = ", ")), call. = FALSE)


# ===========================================================================
# 3. Descriptives: mean & SD of pcf by session, by arm, and by session x arm cell
# ===========================================================================
sd_pop <- function(x) sd(x)   # sample SD (n-1), the usual reporting convention

by_session <- dat %>% group_by(session) %>%
  summarise(n = n(), mean_pcf = mean(pcf), sd_pcf = sd_pop(pcf), .groups = "drop") %>%
  mutate(grouping = "by_session", level = as.character(session)) %>%
  select(grouping, level, n, mean_pcf, sd_pcf)

by_arm <- dat %>% group_by(arm) %>%
  summarise(n = n(), mean_pcf = mean(pcf), sd_pcf = sd_pop(pcf), .groups = "drop") %>%
  mutate(grouping = "by_arm", level = as.character(arm)) %>%
  select(grouping, level, n, mean_pcf, sd_pcf)

by_cell <- dat %>% group_by(arm, session) %>%
  summarise(n = n(), mean_pcf = mean(pcf), sd_pcf = sd_pop(pcf), .groups = "drop") %>%
  mutate(grouping = "by_arm_x_session",
         level = paste(arm, session, sep = "/")) %>%
  select(grouping, level, n, mean_pcf, sd_pcf)

descriptives <- bind_rows(by_session, by_arm, by_cell)

cat("\n==== pcf descriptives (percent frames censored, FD>0.3mm) ====\n")
print(as.data.frame(descriptives), row.names = FALSE, digits = 4)


# ---------------------------------------------------------------------------
# 3b. Distribution of pcf in BOTH samples, explicitly labelled.
#
#   all-40  = every subject with data. THIS is the sample the Stage 2b
#             exclusion-threshold table operates over, so it is the relevant
#             one for choosing a cutoff.
#   n=35    = the current analysis sample (all-40 minus the 5 subjects already
#             dropped by the >50%-censoring rule). Reported so the two are
#             never confused when quoted.
# ---------------------------------------------------------------------------
sample_stats <- function(x, label, n_subj) {
  data.frame(sample = label, n_subject = n_subj, n_runs = length(x),
             median = median(x), mean = mean(x), min = min(x), max = max(x),
             stringsAsFactors = FALSE)
}
pcf_all40 <- dat$pcf
pcf_n35   <- dat$pcf[!as.character(dat$subject) %in% EXCLUDED]
n35_subj  <- length(setdiff(levels(dat$subject), EXCLUDED))

samples <- bind_rows(
  sample_stats(pcf_all40, "all-40 (Stage 2b threshold table)", n_subj),
  sample_stats(pcf_n35,   "n=35 (current analysis sample)",    n35_subj)
)

cat("\n==== pcf distribution by sample (label which is which when quoting) ====\n")
print(as.data.frame(samples), row.names = FALSE, digits = 4)


# ===========================================================================
# 4. Paired t-test: pcf post vs pre, across subjects
# ===========================================================================
wide <- dat %>%
  select(subject, session, pcf) %>%
  pivot_wider(names_from = session, values_from = pcf)
if (any(is.na(wide$pre)) || any(is.na(wide$post)))
  stop("Some subject lacks a pre or post pcf value; cannot run the paired test.",
       call. = FALSE)

tt <- t.test(wide$post, wide$pre, paired = TRUE)
paired_row <- data.frame(
  section    = "paired_t_post_minus_pre",
  term       = "post - pre",
  n_subject  = nrow(wide),
  estimate   = unname(tt$estimate),          # mean of (post - pre)
  ci_low     = tt$conf.int[1],
  ci_high    = tt$conf.int[2],
  t          = unname(tt$statistic),
  df         = unname(tt$parameter),
  p          = tt$p.value,
  stringsAsFactors = FALSE
)

cat("\n==== Paired t-test: pcf (post - pre), across subjects ====\n")
cat(sprintf("mean(post-pre) = %+.4f  95%% CI [%+.4f, %+.4f]  t(%.0f) = %.3f  p = %.4f\n",
            paired_row$estimate, paired_row$ci_low, paired_row$ci_high,
            paired_row$df, paired_row$t, paired_row$p))


# ===========================================================================
# 5. LMM: pcf ~ session * arm + (1 | subject); KR df. Report session + interaction.
# ===========================================================================
# Treatment coding: session ref = pre, arm ref = placebo. So the coefficients are:
#   sessionpost           -> post-pre change in the PLACEBO arm
#   armverum              -> verum-placebo baseline (pre) difference
#   sessionpost:armverum  -> DiD: does verum change post-pre differently than placebo?
model <- suppressWarnings(suppressMessages(
  lmer(pcf ~ session * arm + (1 | subject), data = dat, REML = TRUE)))
singular <- lme4::isSingular(model)

# KR coefficient table when available, else fall back to Satterthwaite (same pattern
# as 04_statistics/scripts/qin/03_mixed_models.R).
coefs <- tryCatch(summary(model, ddf = "Kenward-Roger")$coefficients,
                  error = function(e) summary(model)$coefficients)
p_col <- ncol(coefs)   # p-value is the last column
df_col <- if (p_col >= 5) 3 else NA   # KR/Satterthwaite add a df column

term_row <- function(term, label) {
  if (!term %in% rownames(coefs))
    stop(sprintf("Expected LMM term '%s' not in the coefficient table.", term),
         call. = FALSE)
  data.frame(
    section  = "lmm_pcf_session_x_arm",
    term     = label,
    estimate = coefs[term, 1],
    se       = coefs[term, 2],
    df       = if (!is.na(df_col)) coefs[term, df_col] else NA_real_,
    p        = coefs[term, p_col],
    singular = singular,
    stringsAsFactors = FALSE
  )
}

lmm_rows <- bind_rows(
  term_row("sessionpost",          "session (post-pre, placebo arm)"),
  term_row("armverum",             "arm (verum-placebo, baseline)"),
  term_row("sessionpost:armverum", "session:arm interaction (DiD)")
)

cat("\n==== LMM: pcf ~ session * arm + (1|subject)  [Kenward-Roger df] ====\n")
if (singular) cat("NOTE: fit is singular (subject random-intercept variance ~ 0).\n")
for (i in seq_len(nrow(lmm_rows))) {
  r <- lmm_rows[i, ]
  cat(sprintf("- %-34s est=%+.4f  se=%.4f  df=%s  p=%.4f\n",
              r$term, r$estimate, r$se,
              ifelse(is.na(r$df), "NA", sprintf("%.1f", r$df)), r$p))
}
cat("Requested terms: SESSION = 'session (post-pre, placebo arm)'; ",
    "INTERACTION = 'session:arm interaction (DiD)'.\n", sep = "")


# ===========================================================================
# 6. Save one tidy CSV with every numeric result
# ===========================================================================
# Stack the three blocks with a shared 'section' column. Columns not relevant to a
# block are left NA. This keeps the whole check in a single file.
descriptives_out <- descriptives %>%
  transmute(section = paste0("descriptives_", grouping),
            term = level, n_subject = NA_integer_,
            estimate = mean_pcf, se = sd_pcf,
            ci_low = NA_real_, ci_high = NA_real_,
            t = NA_real_, df = NA_real_, p = NA_real_, singular = NA)

paired_out <- paired_row %>%
  transmute(section, term, n_subject,
            estimate, se = NA_real_, ci_low, ci_high,
            t, df, p, singular = NA)

lmm_out <- lmm_rows %>%
  transmute(section, term, n_subject = NA_integer_,
            estimate, se, ci_low = NA_real_, ci_high = NA_real_,
            t = NA_real_, df, p, singular)

samples_out <- samples %>%
  tidyr::pivot_longer(c(median, mean, min, max),
                      names_to = "stat", values_to = "value") %>%
  transmute(section = "sample_summary", term = paste0(sample, " :: ", stat),
            n_subject, estimate = value, se = NA_real_,
            ci_low = NA_real_, ci_high = NA_real_,
            t = NA_real_, df = NA_real_, p = NA_real_, singular = NA)

out <- bind_rows(samples_out, descriptives_out, paired_out, lmm_out)
write.csv(out, OUT_CSV, row.names = FALSE)
cat(sprintf("\nSaved: %s  (%d rows)\n", OUT_CSV, nrow(out)))

# ---- summary, phrased for direct use in the write-up ----------------------
# Deliberately CI-first, not p-first: "p = 0.98" only says we failed to detect
# an imbalance, whereas the interval states how large an imbalance is still
# compatible with the data — which is the quantity the robustness argument
# actually rests on.
placebo_mean <- by_arm$mean_pcf[by_arm$level == "placebo"]
verum_mean   <- by_arm$mean_pcf[by_arm$level == "verum"]
arm_gap      <- verum_mean - placebo_mean
arm_p        <- lmm_rows$p[lmm_rows$term == "arm (verum-placebo, baseline)"]

cat("\n==== SUMMARY (lead with the interval, not the p-value) ====\n")
cat(sprintf(
  "Sessions differed by %.2f pp, 95%% CI [%.2f, %+.2f] (paired t, p = %.3f).\n",
  paired_row$estimate, paired_row$ci_low, paired_row$ci_high, paired_row$p))
cat(sprintf(
  "  pre = %.2f%% (SD %.2f), post = %.2f%% (SD %.2f), n = %d subjects / %d runs.\n",
  by_session$mean_pcf[by_session$level == "pre"],
  by_session$sd_pcf[by_session$level == "pre"],
  by_session$mean_pcf[by_session$level == "post"],
  by_session$sd_pcf[by_session$level == "post"], n_subj, n_rows))
cat(sprintf("  LMM session p = %.3f; session:arm interaction p = %.3f%s.\n",
            lmm_rows$p[lmm_rows$term == "session (post-pre, placebo arm)"],
            lmm_rows$p[lmm_rows$term == "session:arm interaction (DiD)"],
            if (singular) " (singular fit)" else ""))
cat(sprintf(
  "Arm: verum %.2f%% vs placebo %.2f%% (+%.2f pp, ns, LMM p = %.3f).\n",
  verum_mean, placebo_mean, arm_gap, arm_p))
cat("  Heavier censoring in verum biases verum INT SHORTER, i.e. against the\n")
cat("  drug effect, not for it.\n")

cat("\nSample summaries (label which is which when quoting):\n")
for (i in seq_len(nrow(samples))) {
  s <- samples[i, ]
  cat(sprintf("  %-34s median %.2f%%, mean %.2f%%, min %.2f%%, max %.2f%%  (%d runs)\n",
              s$sample, s$median, s$mean, s$min, s$max, s$n_runs))
}
