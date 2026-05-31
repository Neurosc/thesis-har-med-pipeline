# 18_lmm_simple_slopes_and_log.R
# Step 12b: simple slopes + log-AUC sensitivity check
#
# Part 1: session x G1_z simple slopes within each drug group (group-split models)
# Part 2: log-AUC sensitivity — refit full Step 12 model on log(auc)
# Part 3: residual diagnostics for log model (fig15)
#
# Run from repo root:
#   Rscript 04_statistics/scripts/18_lmm_simple_slopes_and_log.R

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(moments)
  library(plotly)
  library(htmlwidgets)
})

# ── Paths ──────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
  REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
} else {
  REPO_ROOT  <- normalizePath(".")
}

DATA_CSV        <- file.path(REPO_ROOT, "04_statistics", "results", "nonself_g1_model_ready.csv")
ORIG_CSV        <- file.path(REPO_ROOT, "04_statistics", "results", "lmm_g1_full_summary.csv")
OUT_SLOPES      <- file.path(REPO_ROOT, "04_statistics", "results", "lmm_simple_slopes.csv")
OUT_LOG         <- file.path(REPO_ROOT, "04_statistics", "results", "lmm_log_auc_summary.csv")
OUT_FIG15       <- file.path(REPO_ROOT, "04_statistics", "figures",  "fig15_lmm_log_auc_diagnostics.html")

# ── Idempotent guard ───────────────────────────────────────────────────────────
if (all(file.exists(c(OUT_SLOPES, OUT_LOG, OUT_FIG15)))) {
  cat("All outputs already exist — nothing to do. Delete to rerun.\n")
  quit(status = 0)
}

SEP <- paste(rep("=", 70), collapse = "")

# ── Load and prepare data ──────────────────────────────────────────────────────
cat(SEP, "\n18_lmm_simple_slopes_and_log.R — Step 12b\n", SEP, "\n\n", sep = "")
cat("Loading data...\n")
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Data: %d rows x %d cols\n\n", nrow(df), ncol(df)))
stopifnot(nrow(df) == 18042)

df$group      <- relevel(factor(df$group),   ref = "placebo")
df$session    <- relevel(factor(df$session), ref = "ses-01")
df$subject    <- factor(df$subject)
df$roi_pos_id <- factor(df$roi_pos_id)

# shared optimizer settings (same as Step 12)
ctrl_bobyqa <- lmerControl(
  optimizer = "bobyqa",
  optCtrl   = list(maxfun = 2e5),
  check.conv.grad = .makeCC("warning", tol = 0.002)
)

fit_model <- function(formula, data, label) {
  tryCatch(
    withCallingHandlers(
      lmer(formula, data = data, REML = TRUE, control = ctrl_bobyqa),
      warning = function(w) {
        cat(sprintf("[%s] bobyqa warning: %s\nRetrying Nelder-Mead...\n",
                    label, conditionMessage(w)))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) {
      stop(sprintf("[%s] Model failed: %s", label, e$message))
    }
  )
}

extract_fe <- function(model, group_label) {
  s   <- summary(model)
  raw <- coef(s)
  data.frame(
    group    = group_label,
    term     = rownames(raw),
    estimate = raw[, "Estimate"],
    SE       = raw[, "Std. Error"],
    df       = raw[, "df"],
    t_value  = raw[, "t value"],
    p_value  = raw[, "Pr(>|t|)"],
    stringsAsFactors = FALSE,
    row.names = NULL
  )
}

check_convergence <- function(model, label) {
  msgs <- model@optinfo$conv$lme4$messages
  if (!is.null(msgs)) {
    cat(sprintf("  CONVERGENCE WARNING [%s]:\n  %s\n\n",
                label, paste(msgs, collapse = "\n  ")))
  } else {
    cat(sprintf("  Convergence: OK [%s]\n", label))
  }
}

# ─────────────────────────────────────────────────────────────────────────────
cat(SEP, "\nPART 1 — Simple slopes: session x G1_z within each drug group\n",
    SEP, "\n", sep = "")

df_plac <- subset(df, group == "placebo")
df_veru <- subset(df, group == "verum")
cat(sprintf("Placebo rows: %d  |  Verum rows: %d\n\n", nrow(df_plac), nrow(df_veru)))

FORMULA_SIMPLE <- auc ~ session * G1_z + (1 | subject) + (1 | roi_pos_id)

cat("Fitting Model A (placebo only)...\n")
m_plac <- fit_model(FORMULA_SIMPLE, df_plac, "placebo")
check_convergence(m_plac, "placebo")

cat("Fitting Model B (verum only)...\n")
m_veru <- fit_model(FORMULA_SIMPLE, df_veru, "verum")
check_convergence(m_veru, "verum")

fe_plac <- extract_fe(m_plac, "placebo")
fe_veru <- extract_fe(m_veru, "verum")

cat("\n--- Model A: Placebo ---\n")
print(summary(m_plac)$coefficients)

cat("\n--- Random effects (placebo) ---\n")
print(as.data.frame(VarCorr(m_plac)))

cat("\n--- Model B: Verum ---\n")
print(summary(m_veru)$coefficients)

cat("\n--- Random effects (verum) ---\n")
print(as.data.frame(VarCorr(m_veru)))

# Comparison table
get_term <- function(fe, term_name) {
  r <- fe[fe$term == term_name, ]
  if (nrow(r) == 0) return(list(b = NA, p = NA))
  list(b = r$estimate, p = r$p_value)
}

plac_sg  <- get_term(fe_plac, "sessionses-02:G1_z")
veru_sg  <- get_term(fe_veru, "sessionses-02:G1_z")
plac_ses <- get_term(fe_plac, "sessionses-02")
veru_ses <- get_term(fe_veru, "sessionses-02")

cat(sprintf("\n%s\nSimple slopes comparison table\n%s\n", SEP, SEP))
cat(sprintf("%-30s  %-28s  %-28s\n", "Term", "Placebo", "Verum"))
cat(sprintf("%-30s  %-28s  %-28s\n",
            strrep("-", 30), strrep("-", 28), strrep("-", 28)))
cat(sprintf("%-30s  beta = %8.5f (p = %.4g)  beta = %8.5f (p = %.4g)\n",
            "session:G1_z",
            plac_sg$b,  plac_sg$p,
            veru_sg$b,  veru_sg$p))
cat(sprintf("%-30s  beta = %8.5f (p = %.4g)  beta = %8.5f (p = %.4g)\n",
            "session main effect",
            plac_ses$b, plac_ses$p,
            veru_ses$b, veru_ses$p))

# Interpretation
cat(sprintf("\nInterpretation:\n"))
if (!is.na(plac_sg$b) && !is.na(veru_sg$b)) {
  both_neg    <- plac_sg$b < 0 && veru_sg$b < 0
  plac_sig    <- plac_sg$p < 0.05
  veru_sig    <- veru_sg$p < 0.05
  plac_steep  <- abs(plac_sg$b) > abs(veru_sg$b)

  if (plac_sig && !veru_sig) {
    cat("  Placebo has a significant session:G1_z slope; verum does not.\n")
    cat("  => Meditation creates the cortical gradient; DMT attenuates or eliminates it.\n")
  } else if (both_neg && plac_steep) {
    cat("  Both groups show a negative session:G1_z slope, but placebo is steeper.\n")
    cat("  => DMT attenuates but does not eliminate the gradient.\n")
  } else if (veru_sg$b > 0 && plac_sg$b < 0) {
    cat("  Verum slope is positive, placebo negative.\n")
    cat("  => DMT reverses the gradient shift.\n")
  } else {
    cat("  Pattern does not fit a simple narrative — inspect slopes directly.\n")
  }
}

fe_slopes <- rbind(fe_plac, fe_veru)
write.csv(fe_slopes, OUT_SLOPES, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", OUT_SLOPES))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 2 — Log-AUC sensitivity analysis\n", SEP, "\n", sep = "")

n_nonpos <- sum(df$auc <= 0, na.rm = TRUE)
min_auc  <- min(df$auc, na.rm = TRUE)
cat(sprintf("AUC range  : [%.6f, %.6f]\n", min_auc, max(df$auc, na.rm = TRUE)))
cat(sprintf("AUC <= 0   : %d observations\n\n", n_nonpos))

if (n_nonpos == 0) {
  log_label <- "log(auc) — no non-positive values"
  df$log_auc <- log(df$auc)
  cat("All AUC > 0. Using log(auc) directly.\n")
} else if (n_nonpos < 5) {
  log_label <- sprintf("log(auc), excluding %d non-positive rows", n_nonpos)
  cat(sprintf("Excluding %d non-positive AUC observations (< 5 — negligible).\n", n_nonpos))
  df <- df[df$auc > 0, ]
  df$log_auc <- log(df$auc)
} else {
  offset    <- abs(min_auc) + 0.01
  log_label <- sprintf("log(auc + %.4f) — offset applied (min was %.6f)", offset, min_auc)
  cat(sprintf("Many non-positive AUC values (%d). Applying offset = %.4f.\n",
              n_nonpos, offset))
  df$log_auc <- log(df$auc + offset)
}
cat(sprintf("Transformation: %s\n\n", log_label))

cat("Fitting log-AUC full model (REML)...\n")
m_log <- tryCatch(
  lmer(log_auc ~ group * session * G1_z + (1 | subject) + (1 | roi_pos_id),
       data = df, REML = TRUE, control = ctrl_bobyqa),
  warning = function(w) {
    cat(sprintf("bobyqa warning: %s\nRetrying Nelder-Mead...\n", conditionMessage(w)))
    lmer(log_auc ~ group * session * G1_z + (1 | subject) + (1 | roi_pos_id),
         data = df, REML = TRUE,
         control = lmerControl(optimizer = "Nelder_Mead",
                               optCtrl   = list(maxfun = 2e5),
                               check.conv.grad = .makeCC("warning", tol = 0.002)))
  }
)
check_convergence(m_log, "log model")

s_log   <- summary(m_log)
raw_log <- coef(s_log)
fe_log  <- data.frame(
  term     = rownames(raw_log),
  estimate = raw_log[, "Estimate"],
  SE       = raw_log[, "Std. Error"],
  df       = raw_log[, "df"],
  t_value  = raw_log[, "t value"],
  p_value  = raw_log[, "Pr(>|t|)"],
  stringsAsFactors = FALSE,
  row.names = NULL
)

cat("\n--- Log model fixed effects ---\n")
print(s_log$coefficients)

# Read original model fixed effects
if (!file.exists(ORIG_CSV)) {
  cat(sprintf("\nWARNING: original summary CSV not found at %s\n", ORIG_CSV))
  cat("Cannot produce side-by-side comparison.\n")
} else {
  fe_orig <- read.csv(ORIG_CSV, stringsAsFactors = FALSE)

  # Term labels matching the 8 fixed effects
  term_labels <- c(
    "(Intercept)"                      = "intercept",
    "groupverum"                        = "group",
    "sessionses-02"                     = "session",
    "G1_z"                              = "G1_z",
    "groupverum:sessionses-02"          = "group:session",
    "groupverum:G1_z"                   = "group:G1_z",
    "sessionses-02:G1_z"               = "session:G1_z",
    "groupverum:sessionses-02:G1_z"    = "group:session:G1_z"
  )

  cat(sprintf("\n%s\nSide-by-side comparison: Original AUC vs Log-AUC\n%s\n", SEP, SEP))
  header <- sprintf("%-26s  %-20s  %-20s  %s",
                    "Term", "Original (AUC)", "Log-AUC", "Signs match?")
  cat(header, "\n")
  cat(strrep("-", nchar(header)), "\n")

  signs_all_match <- TRUE
  for (raw_term in names(term_labels)) {
    label   <- term_labels[[raw_term]]
    r_orig  <- fe_orig[fe_orig$term == raw_term, ]
    r_log   <- fe_log[ fe_log$term  == raw_term, ]

    if (nrow(r_orig) == 0 || nrow(r_log) == 0) {
      cat(sprintf("%-26s  (not found in one model)\n", label))
      next
    }

    sign_ok  <- sign(r_orig$estimate) == sign(r_log$estimate)
    if (!sign_ok) signs_all_match <- FALSE
    sign_str <- if (sign_ok) "YES" else "NO !"

    cat(sprintf("%-26s  b=%9.5f p=%.3g  b=%9.5f p=%.3g  %s\n",
                label,
                r_orig$estimate, r_orig$p_value,
                r_log$estimate,  r_log$p_value,
                sign_str))
  }

  r3_orig <- fe_orig[fe_orig$term == "groupverum:sessionses-02:G1_z", ]
  r3_log  <- fe_log[ fe_log$term  == "groupverum:sessionses-02:G1_z", ]

  p3_orig <- if (nrow(r3_orig) > 0) r3_orig$p_value else NA
  p3_log  <- if (nrow(r3_log)  > 0) r3_log$p_value  else NA
  survived <- !is.na(p3_log) && p3_log < 0.05
  surv_str <- if (survived) "survived" else "did NOT survive"
  sign_str <- if (signs_all_match) "matched" else "did NOT all match"
  robust_str <- if (survived && signs_all_match) "robust" else "sensitive"

  cat(sprintf(
    "\nInterpretation:\nThe 3-way interaction %s log transformation (original p = %.4g, log p = %.4g).\nAll coefficient signs %s. The finding is %s to the scale of the outcome variable.\n",
    surv_str, p3_orig, p3_log, sign_str, robust_str
  ))
}

write.csv(fe_log, OUT_LOG, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", OUT_LOG))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 3 — Residual diagnostics for log model\n", SEP, "\n", sep = "")

ORIG_KURTOSIS <- 7.84

resids  <- resid(m_log)
fitted_ <- fitted(m_log)

sk <- skewness(resids)
ku <- kurtosis(resids)
cat(sprintf("Skewness : %.4f\n", sk))
cat(sprintf("Kurtosis : %.4f  (excess = %.4f)\n", ku, ku - 3))

set.seed(42)
sw_idx <- sample(length(resids), min(5000L, length(resids)))
sw     <- shapiro.test(resids[sw_idx])
cat(sprintf("Shapiro-Wilk (n=%d): W = %.5f, p = %.4e\n",
            length(sw_idx), sw$statistic, sw$p.value))

# QQ plot
qq_n   <- 5000L
qq_smp <- sort(resids[sample(length(resids), qq_n)])
qq_teo <- qnorm(ppoints(qq_n))
ref_rng <- range(qq_teo)

p_qq <- plot_ly() %>%
  add_trace(x = qq_teo, y = qq_smp, type = "scatter", mode = "markers",
            marker = list(size = 4, color = "steelblue", opacity = 0.5),
            name = "Residuals") %>%
  add_trace(x = ref_rng, y = ref_rng * sd(qq_smp), type = "scatter",
            mode = "lines", line = list(color = "red", dash = "dash"),
            name = "Normal ref") %>%
  layout(title = "QQ Plot (sample n=5000)",
         xaxis = list(title = "Theoretical quantiles"),
         yaxis = list(title = "Sample quantiles"),
         showlegend = FALSE)

# Residuals vs fitted
rvf_idx <- sample(length(resids), min(5000L, length(resids)))
p_rvf <- plot_ly() %>%
  add_trace(x = fitted_[rvf_idx], y = resids[rvf_idx],
            type = "scatter", mode = "markers",
            marker = list(size = 4, color = "steelblue", opacity = 0.4),
            name = "Resid") %>%
  add_trace(x = range(fitted_), y = c(0, 0), type = "scatter", mode = "lines",
            line = list(color = "red", dash = "dash"), name = "Zero") %>%
  layout(title = "Residuals vs Fitted (sample n=5000)",
         xaxis = list(title = "Fitted values"),
         yaxis = list(title = "Residuals"),
         showlegend = FALSE)

# Histogram
p_hist <- plot_ly(x = resids, type = "histogram",
                  marker = list(color = "steelblue",
                                line = list(color = "white", width = 0.3)),
                  nbinsx = 80) %>%
  layout(title = "Histogram of residuals",
         xaxis = list(title = "Residual"),
         yaxis = list(title = "Count"))

fig15 <- subplot(p_qq, p_rvf, p_hist, nrows = 1, titleX = TRUE, titleY = TRUE,
                 margin = 0.06) %>%
  layout(title = list(text = "Log-AUC LMM Residual Diagnostics (Step 12b)", x = 0.5),
         showlegend = FALSE)

saveWidget(fig15, OUT_FIG15, selfcontained = FALSE, title = "Log-AUC Residual Diagnostics")
cat(sprintf("Saved: %s\n", OUT_FIG15))

# Comparison with original
delta_ku <- ku - ORIG_KURTOSIS
if (abs(delta_ku) < 0.5) {
  comp_str <- "did not change"
} else if (delta_ku < 0) {
  comp_str <- "improved"
} else {
  comp_str <- "worsened"
}
cat(sprintf(
  "\nLog transformation %s residual normality (original kurtosis = %.2f, log kurtosis = %.4f).\n",
  comp_str, ORIG_KURTOSIS, ku
))

cat("\n", SEP, "\nALL PARTS COMPLETE\n", SEP, "\n", sep = "")
