# 16_lmm_g1_drug_session.R
# Step 12: LMM drug x session x G1 gradient interaction
#
# Model: auc ~ group * session * G1_z + (1|subject) + (1|roi_pos_id)
# P-values via Satterthwaite df approximation (lmerTest).
#
# Run from repo root:
#   Rscript 04_statistics/scripts/16_lmm_g1_drug_session.R

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(MuMIn)
  library(plotly)
  library(dplyr)
  library(tidyr)
  library(moments)
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

DATA_CSV   <- file.path(REPO_ROOT, "04_statistics", "results", "nonself_g1_model_ready.csv")
OUT_FIXED  <- file.path(REPO_ROOT, "04_statistics", "results", "lmm_g1_full_summary.csv")
OUT_RANDOM <- file.path(REPO_ROOT, "04_statistics", "results", "lmm_g1_random_effects.csv")
OUT_FIG13  <- file.path(REPO_ROOT, "04_statistics", "figures", "fig13_lmm_residual_diagnostics.html")
OUT_FIG14  <- file.path(REPO_ROOT, "04_statistics", "figures", "fig14_g1_drug_session_scatter.html")

# ── Idempotent guard ───────────────────────────────────────────────────────────
if (all(file.exists(c(OUT_FIXED, OUT_RANDOM, OUT_FIG13, OUT_FIG14)))) {
  cat("All outputs already exist — nothing to do. Delete to rerun.\n")
  quit(status = 0)
}

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n16_lmm_g1_drug_session.R\n", SEP, "\n\n", sep = "")

# ── Load and prepare data ──────────────────────────────────────────────────────
cat("Loading data...\n")
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Data: %d rows x %d cols\n", nrow(df), ncol(df)))
stopifnot(nrow(df) == 18042)

df$group      <- relevel(factor(df$group),      ref = "placebo")
df$session    <- relevel(factor(df$session),    ref = "ses-01")
df$subject    <- factor(df$subject)
df$roi_pos_id <- factor(df$roi_pos_id)

cat(sprintf("group levels   : %s\n", paste(levels(df$group),   collapse = ", ")))
cat(sprintf("session levels : %s\n", paste(levels(df$session), collapse = ", ")))
cat(sprintf("Subjects       : %d\n", nlevels(df$subject)))
cat(sprintf("ROIs           : %d\n", nlevels(df$roi_pos_id)))

# ── Fit full model ─────────────────────────────────────────────────────────────
cat("\nFitting full LMM (REML, bobyqa, maxfun=200000)...\n")
cat("Formula: auc ~ group * session * G1_z + (1|subject) + (1|roi_pos_id)\n\n")

ctrl_bobyqa <- lmerControl(
  optimizer = "bobyqa",
  optCtrl   = list(maxfun = 2e5),
  check.conv.grad = .makeCC("warning", tol = 0.002)
)

m_full <- tryCatch(
  lmer(auc ~ group * session * G1_z + (1 | subject) + (1 | roi_pos_id),
       data = df, REML = TRUE, control = ctrl_bobyqa),
  warning = function(w) {
    cat(sprintf("bobyqa warning: %s\n", conditionMessage(w)))
    cat("Retrying with Nelder-Mead...\n")
    lmer(auc ~ group * session * G1_z + (1 | subject) + (1 | roi_pos_id),
         data = df, REML = TRUE,
         control = lmerControl(optimizer  = "Nelder_Mead",
                               optCtrl    = list(maxfun = 2e5),
                               check.conv.grad = .makeCC("warning", tol = 0.002)))
  }
)

conv_msgs <- m_full@optinfo$conv$lme4$messages
if (!is.null(conv_msgs)) {
  cat("CONVERGENCE WARNINGS:\n", paste(conv_msgs, collapse = "\n"), "\n\n")
} else {
  cat("Convergence: OK\n\n")
}

# ─────────────────────────────────────────────────────────────────────────────
cat(SEP, "\nPART 1 — Model fit summary\n", SEP, "\n", sep = "")
cat("P-values via Satterthwaite df approximation (lmerTest)\n\n")

sum_full <- summary(m_full)
print(sum_full)

cat(sprintf("\nAIC    : %.4f\n", AIC(m_full)))
cat(sprintf("BIC    : %.4f\n", BIC(m_full)))
cat(sprintf("logLik : %.4f\n", as.numeric(logLik(m_full))))

# Fixed effects table (lmerTest gives df and p-value columns)
fe_raw <- coef(sum_full)
fe_tbl <- data.frame(
  term    = rownames(fe_raw),
  estimate = fe_raw[, "Estimate"],
  SE       = fe_raw[, "Std. Error"],
  df       = fe_raw[, "df"],
  t_value  = fe_raw[, "t value"],
  p_value  = fe_raw[, "Pr(>|t|)"],
  stringsAsFactors = FALSE,
  row.names = NULL
)

cat("\nRandom effects (variance components):\n")
vc <- as.data.frame(VarCorr(m_full))
print(vc)

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 2 — Key contrasts\n", SEP, "\n", sep = "")

key_terms <- c(
  "groupverum:sessionses-02",
  "sessionses-02:G1_z",
  "groupverum:sessionses-02:G1_z"
)
key_labels <- c(
  "group x session                   (global drug x session)",
  "session x G1_z                    (does pre->post change vary by hierarchy?)",
  "group x session x G1_z  [PRIMARY] (does drug effect vary along cortex?)"
)

for (i in seq_along(key_terms)) {
  row <- fe_tbl[fe_tbl$term == key_terms[i], ]
  if (nrow(row) == 0) {
    cat(sprintf("\nWARNING: term '%s' not found in model\n", key_terms[i]))
    next
  }
  cat(sprintf("\n%s\n", key_labels[i]))
  cat(sprintf("  beta = %9.6f  SE = %9.6f  t(%7.1f) = %8.4f  p = %.4g\n",
              row$estimate, row$SE, row$df, row$t_value, row$p_value))
}

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 3 — Effect size and interpretation\n", SEP, "\n", sep = "")

r2 <- tryCatch(r.squaredGLMM(m_full), error = function(e) { cat("MuMIn r2 failed:", e$message, "\n"); NULL })
if (!is.null(r2)) {
  cat(sprintf("Marginal  R2 (fixed effects only) : %.4f\n", r2[1, "R2m"]))
  cat(sprintf("Conditional R2 (fixed + random)   : %.4f\n", r2[1, "R2c"]))
}

b3way <- fe_tbl$estimate[fe_tbl$term == "groupverum:sessionses-02:G1_z"]
cat(sprintf(
  "\nFor each 1-SD increase in G1 (toward transmodal pole),\nthe drug x session effect on AUC changes by %.6f seconds.\n",
  b3way
))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 4 — Residual diagnostics\n", SEP, "\n", sep = "")

resids  <- resid(m_full)
fitted_ <- fitted(m_full)

sk <- skewness(resids)
ku <- kurtosis(resids)
cat(sprintf("Skewness : %.4f\n", sk))
cat(sprintf("Kurtosis : %.4f  (excess = %.4f)\n", ku, ku - 3))

set.seed(42)
sw_idx <- sample(length(resids), min(5000L, length(resids)))
sw     <- shapiro.test(resids[sw_idx])
cat(sprintf("Shapiro-Wilk (n=%d): W = %.5f, p = %.4e\n",
            length(sw_idx), sw$statistic, sw$p.value))

# QQ plot — sample 5000 to keep HTML responsive
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

# Residuals vs fitted — sample 5000
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

# Histogram (all residuals)
p_hist <- plot_ly(x = resids, type = "histogram",
                  marker = list(color = "steelblue", line = list(color = "white", width = 0.3)),
                  nbinsx = 80) %>%
  layout(title = "Histogram of residuals",
         xaxis = list(title = "Residual"),
         yaxis = list(title = "Count"))

fig13 <- subplot(p_qq, p_rvf, p_hist, nrows = 1, titleX = TRUE, titleY = TRUE,
                 margin = 0.06) %>%
  layout(title = list(text = "LMM Residual Diagnostics (Step 12)", x = 0.5),
         showlegend = FALSE)

saveWidget(fig13, OUT_FIG13, selfcontained = TRUE, title = "LMM Residual Diagnostics")
cat(sprintf("Saved: %s\n", OUT_FIG13))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 5 — 3-way interaction visualization\n", SEP, "\n", sep = "")

# Per-ROI drug x session double-difference
roi_means <- df %>%
  mutate(ses_lbl = ifelse(session == "ses-01", "pre", "post")) %>%
  group_by(roi_pos_id, ses_lbl, group) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE),
            G1_z = first(G1_z), G1 = first(G1),
            roi_name = first(roi_name),
            .groups = "drop")

roi_wide <- roi_means %>%
  pivot_wider(id_cols     = c(roi_pos_id, G1_z, G1, roi_name),
              names_from  = c(group, ses_lbl),
              values_from = mean_auc)

# Verify expected columns exist
expected <- c("placebo_pre", "placebo_post", "verum_pre", "verum_post")
missing  <- setdiff(expected, names(roi_wide))
if (length(missing) > 0) {
  cat(sprintf("WARNING: pivot columns missing: %s\n  Got: %s\n",
              paste(missing, collapse = ", "),
              paste(names(roi_wide), collapse = ", ")))
} else {
  roi_wide <- roi_wide %>%
    mutate(drug_session_delta = (verum_post - verum_pre) - (placebo_post - placebo_pre))
  cat(sprintf("Per-ROI drug x session effects computed for %d ROIs\n", nrow(roi_wide)))

  # Model regression line + pointwise 95% CI
  beta_gs  <- fe_tbl$estimate[fe_tbl$term == "groupverum:sessionses-02"]
  beta_gsg <- b3way

  vcov_m   <- as.matrix(vcov(m_full))
  coef_nm  <- rownames(vcov_m)
  gs_idx   <- which(coef_nm == "groupverum:sessionses-02")
  gsg_idx  <- which(coef_nm == "groupverum:sessionses-02:G1_z")

  g1_grid <- seq(min(df$G1_z), max(df$G1_z), length.out = 200)
  pred_y   <- beta_gs + beta_gsg * g1_grid
  pred_se  <- vapply(g1_grid, function(g) {
    cv         <- numeric(length(coef_nm))
    cv[gs_idx]  <- 1
    cv[gsg_idx] <- g
    sqrt(as.numeric(t(cv) %*% vcov_m %*% cv))
  }, numeric(1))
  pred_lo <- pred_y - 1.96 * pred_se
  pred_hi <- pred_y + 1.96 * pred_se

  fig14 <- plot_ly() %>%
    # CI ribbon (filled polygon)
    add_trace(
      x         = c(g1_grid, rev(g1_grid)),
      y         = c(pred_hi, rev(pred_lo)),
      type      = "scatter", mode = "lines",
      fill      = "toself",
      fillcolor = "rgba(70,130,180,0.15)",
      line      = list(color = "transparent"),
      name      = "95% CI",
      showlegend = TRUE
    ) %>%
    # Regression line
    add_trace(
      x    = g1_grid, y = pred_y,
      type = "scatter", mode = "lines",
      line = list(color = "steelblue", width = 2),
      name = sprintf("Model (beta=%.4f)", beta_gsg)
    ) %>%
    # Per-ROI scatter
    add_trace(
      x            = roi_wide$G1_z,
      y            = roi_wide$drug_session_delta,
      type         = "scatter", mode = "markers",
      marker       = list(size = 8, color = "navy", opacity = 0.65),
      text         = roi_wide$roi_name,
      hovertemplate = "ROI: %{text}<br>G1_z: %{x:.3f}<br>delta AUC: %{y:.4f}<extra></extra>",
      name         = "Per-ROI delta AUC"
    ) %>%
    layout(
      title  = list(text = "Drug x Session Effect Along G1 Cortical Gradient (Step 12)", x = 0.5),
      xaxis  = list(title = "G1_z  (sensory  <--  -->  transmodal)"),
      yaxis  = list(title = "Drug x Session delta AUC (s): (verum post-pre) - (placebo post-pre)"),
      legend = list(x = 0.02, y = 0.98)
    )

  saveWidget(fig14, OUT_FIG14, selfcontained = TRUE, title = "G1 Drug x Session Scatter")
  cat(sprintf("Saved: %s\n", OUT_FIG14))
}

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 6 — Reduced model + likelihood ratio test\n", SEP, "\n", sep = "")
cat("Reduced: auc ~ group*session + G1_z + session:G1_z + (1|subject) + (1|roi_pos_id)\n")
cat("Drops group:G1_z and group:session:G1_z (2 df) from the full model\n\n")

ctrl_ml <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

m_full_ml <- update(m_full, REML = FALSE, control = ctrl_ml)
m_red_ml  <- lmer(
  auc ~ group * session + G1_z + session:G1_z + (1 | subject) + (1 | roi_pos_id),
  data = df, REML = FALSE, control = ctrl_ml
)

lrt     <- anova(m_red_ml, m_full_ml)
cat("LRT output:\n")
print(lrt)

chi2  <- lrt$Chisq[2]
df_lr <- lrt$Df[2]
pval  <- lrt$`Pr(>Chisq)`[2]
cat(sprintf("\nLRT: chi2(%d) = %.4f, p = %.4g\n", df_lr, chi2, pval))
if (pval < 0.05) {
  cat("=> SIGNIFICANT: the 3-way interaction improves model fit.\n")
} else {
  cat("=> Non-significant: 3-way interaction does not significantly improve fit.\n")
}

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSaving CSV outputs\n", SEP, "\n", sep = "")

write.csv(fe_tbl, OUT_FIXED,  row.names = FALSE)
cat(sprintf("Saved: %s\n", OUT_FIXED))

write.csv(vc, OUT_RANDOM, row.names = FALSE)
cat(sprintf("Saved: %s\n", OUT_RANDOM))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nINTERPRETATION\n", SEP, "\n", sep = "")

b3  <- fe_tbl$estimate[fe_tbl$term == "groupverum:sessionses-02:G1_z"]
se3 <- fe_tbl$SE[      fe_tbl$term == "groupverum:sessionses-02:G1_z"]
t3  <- fe_tbl$t_value[ fe_tbl$term == "groupverum:sessionses-02:G1_z"]
p3  <- fe_tbl$p_value[ fe_tbl$term == "groupverum:sessionses-02:G1_z"]

sig_str  <- if (p3 < 0.05) "significant" else "not significant"
does_str <- if (p3 < 0.05) "does" else "does not"
tail_str <- if (p3 < 0.05) {
  sprintf("The effect indicates that for each 1-SD increase in G1 toward the transmodal pole, the drug x session effect on AUC increased by %.6f seconds.", b3)
} else {
  "The exploratory ROI-level pattern observed in Step 11 was not confirmed by the mixed-effects model."
}

cat(sprintf(
  "\nThe group x session x G1_z interaction was %s (beta = %.6f, SE = %.6f, t = %.4f, p = %.4g), suggesting that the drug effect %s vary along the sensory-to-transmodal cortical hierarchy. %s\n",
  sig_str, b3, se3, t3, p3, does_str, tail_str
))

cat("\n", SEP, "\nALL PARTS COMPLETE\n", SEP, "\n", sep = "")
