# 20_self_lmm_layer.R
# Step 13: self atlas LMM — drug × session × layer (ordered) interaction
#
# NOTE: task requested filename 19_self_lmm_layer.R but
#   19_lmm_g1_influence_check.R already exists. Using 20_.
#
# NOTE: self_coordinates.txt not found at the specified canonical path
#   (02_timeseries_extraction/results/atlases/self_coordinates.txt).
#   Layer mapping is hardcoded below from _old/Thesis/01_atlases/self_coordinates.txt
#   (verified: 37 ROIs, Qin et al. 2020).
#
# Model: auc ~ group * session * layer + random
# layer is an ordered factor: Interoception < Exteroception < Cognition
# Primary: group:session:layer.L (monotonic 3-way interaction)
#
# Run from repo root:
#   Rscript 04_statistics/scripts/20_self_lmm_layer.R

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(MuMIn)
  library(dplyr)
  library(tidyr)
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

DATA_CSV    <- file.path(REPO_ROOT, "04_statistics", "results", "analysis_long_format_auc.csv")
OUT_FIXED   <- file.path(REPO_ROOT, "04_statistics", "results", "self_lmm_layer_summary.csv")
OUT_RANDOM  <- file.path(REPO_ROOT, "04_statistics", "results", "self_lmm_layer_random_effects.csv")
OUT_LOO_ROI <- file.path(REPO_ROOT, "04_statistics", "results", "self_lmm_loo_roi_robustness.csv")
OUT_LOO_SUB <- file.path(REPO_ROOT, "04_statistics", "results", "self_lmm_loo_subject_robustness.csv")
OUT_FIG16   <- file.path(REPO_ROOT, "04_statistics", "figures",  "fig16_self_lmm_residual_diagnostics.html")
OUT_FIG17   <- file.path(REPO_ROOT, "04_statistics", "figures",  "fig17_self_layer_drug_session.html")

# ── Idempotent guard ───────────────────────────────────────────────────────────
all_out <- c(OUT_FIXED, OUT_RANDOM, OUT_LOO_ROI, OUT_LOO_SUB, OUT_FIG16, OUT_FIG17)
if (all(file.exists(all_out))) {
  cat("All outputs already exist — nothing to do. Delete to rerun.\n")
  quit(status = 0)
}

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n20_self_lmm_layer.R — Step 13: self atlas LMM\n", SEP, "\n\n", sep = "")

# ── Hardcoded layer mapping (Qin et al. 2020, 37 ROIs) ────────────────────────
LAYER_MAP <- data.frame(
  roi_pos_id = 1:37,
  roi_name = c(
    "Interoception_R_Insula","Interoception_L_DACC","Interoception_R_Thalamus",
    "Interoception_R_Parahippocampal","Interoception_L_Parahippocampal",
    "Interoception_L_Insula_1","Interoception_L_Insula_2","Interoception_R_IPL",
    "Interoception_R_SFG","Interoception_L_STG","Interoception_L_Postcentral",
    "Exteroception_R_Fusiform","Exteroception_R_IFG","Exteroception_R_Premotor",
    "Exteroception_R_Insula","Exteroception_L_Fusiform","Exteroception_R_SPL",
    "Exteroception_R_Postcentral","Exteroception_R_IPL","Exteroception_L_Insula",
    "Exteroception_R_IOG","Exteroception_L_IPL","Exteroception_L_SPL",
    "Exteroception_R_Cingulate","Exteroception_L_mPFC",
    "Cognition_L_ACC","Cognition_L_PCC","Cognition_L_Insula",
    "Cognition_L_MTG","Cognition_L_Thalamus","Cognition_L_SFG_1",
    "Cognition_Cingulate","Cognition_L_ITG","Cognition_R_MTG",
    "Cognition_R_Insula","Cognition_L_SFG_2","Cognition_R_Premotor"
  ),
  layer = c(rep("Interoception", 11), rep("Exteroception", 14), rep("Cognition", 12)),
  stringsAsFactors = FALSE
)

# ─────────────────────────────────────────────────────────────────────────────
cat(SEP, "\nSTEP 1 — Load and prepare data\n", SEP, "\n", sep = "")

df_all <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Full AUC data: %d rows\n", nrow(df_all)))

df <- df_all[df_all$atlas == "self", ]
cat(sprintf("Self atlas rows: %d\n", nrow(df)))
stopifnot(nrow(df) == 2590)

# Rename drug_group → group (consistent with Step 12)
names(df)[names(df) == "drug_group"] <- "group"

# Merge layer info
df <- merge(df, LAYER_MAP, by = "roi_pos_id", all.x = TRUE)
stopifnot(!any(is.na(df$layer)))

# Ordered factor and reference levels
df$layer   <- ordered(df$layer, levels = c("Interoception", "Exteroception", "Cognition"))
df$group   <- relevel(factor(df$group),   ref = "placebo")
df$session <- relevel(factor(df$session), ref = "ses-01")
df$subject <- factor(df$subject)
df$roi_pos_id <- factor(df$roi_pos_id)

cat(sprintf("group    levels: %s\n", paste(levels(df$group),   collapse = ", ")))
cat(sprintf("session  levels: %s\n", paste(levels(df$session), collapse = ", ")))
cat(sprintf("layer    levels: %s\n", paste(levels(df$layer),   collapse = " < ")))
cat(sprintf("Subjects: %d  |  ROIs: %d  |  Rows: %d\n",
            nlevels(df$subject), nlevels(df$roi_pos_id), nrow(df)))
cat(sprintf("ROIs per layer: %s\n",
            paste(tapply(LAYER_MAP$roi_pos_id, LAYER_MAP$layer, length), collapse = " / ")))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSTEP 2 — Fit primary model\n", SEP, "\n", sep = "")

FORMULA_FULL  <- auc ~ group * session * layer + (1 + session | subject) + (1 | roi_pos_id)
FORMULA_SIMP  <- auc ~ group * session * layer + (1 | subject) + (1 | roi_pos_id)

ctrl_main <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))
ctrl_nm   <- lmerControl(optimizer = "Nelder_Mead", optCtrl = list(maxfun = 1e5),
                         check.conv.grad = .makeCC("warning", tol = 0.002))

try_fit <- function(formula, ctrl, label) {
  tryCatch(
    withCallingHandlers(
      lmer(formula, data = df, REML = TRUE, control = ctrl),
      warning = function(w) {
        cat(sprintf("  [%s] warning: %s\n", label, conditionMessage(w)))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) { cat(sprintf("  [%s] ERROR: %s\n", label, e$message)); NULL }
  )
}

# Attempt 1: random slope
cat("Attempting: (1 + session | subject) + (1 | roi_pos_id) ...\n")
m_self <- try_fit(FORMULA_FULL, ctrl_main, "random-slope/bobyqa")

random_slope_used <- TRUE
if (is.null(m_self)) {
  cat("Failed. Trying Nelder-Mead...\n")
  m_self <- try_fit(FORMULA_FULL, ctrl_nm, "random-slope/NM")
}

# Check for singular fit (common with random slopes on small data)
if (!is.null(m_self)) {
  sing <- isSingular(m_self)
  if (sing) {
    cat("  Singular fit detected with random slope. Falling back to (1|subject).\n")
    m_self <- NULL
  }
}

if (is.null(m_self)) {
  cat("Fallback: (1 | subject) + (1 | roi_pos_id) ...\n")
  m_self <- try_fit(FORMULA_SIMP, ctrl_main, "intercept-only/bobyqa")
  random_slope_used <- FALSE
  if (is.null(m_self)) {
    m_self <- try_fit(FORMULA_SIMP, ctrl_nm, "intercept-only/NM")
  }
}

if (is.null(m_self)) stop("All model-fitting attempts failed. Investigate data.")

FORMULA_USED <- if (random_slope_used) FORMULA_FULL else FORMULA_SIMP
re_str       <- ifelse(random_slope_used,
                       "(1 + session | subject) + (1 | roi_pos_id)",
                       "(1 | subject) + (1 | roi_pos_id)")
cat(sprintf("\nRandom effects used: %s\n", re_str))

conv_msgs <- m_self@optinfo$conv$lme4$messages
if (!is.null(conv_msgs)) {
  cat(sprintf("CONVERGENCE WARNINGS:\n%s\n", paste(conv_msgs, collapse = "\n")))
} else {
  cat("Convergence: OK\n")
}

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSTEP 3 — Fixed effects (Satterthwaite p-values)\n", SEP, "\n", sep = "")

s_main  <- summary(m_self)
print(s_main)

raw_fe  <- coef(s_main)
fe_tbl  <- data.frame(
  term     = rownames(raw_fe),
  estimate = raw_fe[, "Estimate"],
  SE       = raw_fe[, "Std. Error"],
  df       = raw_fe[, "df"],
  t_value  = raw_fe[, "t value"],
  p_value  = raw_fe[, "Pr(>|t|)"],
  stringsAsFactors = FALSE, row.names = NULL
)

# Key terms
key_terms <- c(
  "groupverum:sessionses-02:layer.L",
  "groupverum:sessionses-02:layer.Q",
  "groupverum:sessionses-02",
  "sessionses-02:layer.L"
)
key_labels <- c(
  "group:session:layer.L  [PRIMARY — monotonic 3-way]",
  "group:session:layer.Q  [SECONDARY — quadratic 3-way]",
  "group:session           [global drug x session]",
  "session:layer.L         [session gradient]"
)

cat("\n--- Key terms ---\n")
for (i in seq_along(key_terms)) {
  r <- fe_tbl[fe_tbl$term == key_terms[i], ]
  if (nrow(r) == 0) { cat(sprintf("  %s: NOT FOUND\n", key_labels[i])); next }
  cat(sprintf("  %s\n  beta = %.6f  SE = %.6f  t(%6.1f) = %7.4f  p = %.4g\n\n",
              key_labels[i], r$estimate, r$SE, r$df, r$t_value, r$p_value))
}

# ─────────────────────────────────────────────────────────────────────────────
cat(SEP, "\nSTEP 4 — Effect size and interpretation\n", SEP, "\n", sep = "")

r2 <- tryCatch(r.squaredGLMM(m_self),
               error = function(e) { cat("MuMIn R² failed:", e$message, "\n"); NULL })
if (!is.null(r2)) {
  cat(sprintf("Marginal  R² (fixed only)   : %.4f\n", r2[1, "R2m"]))
  cat(sprintf("Conditional R² (fixed+random): %.4f\n", r2[1, "R2c"]))
}

b_3L <- fe_tbl$estimate[fe_tbl$term == "groupverum:sessionses-02:layer.L"]
b_3Q <- fe_tbl$estimate[fe_tbl$term == "groupverum:sessionses-02:layer.Q"]
p_3L <- fe_tbl$p_value[ fe_tbl$term == "groupverum:sessionses-02:layer.L"]
p_3Q <- fe_tbl$p_value[ fe_tbl$term == "groupverum:sessionses-02:layer.Q"]

if (length(b_3L) > 0) {
  cat(sprintf(
    "\nFor each step up the self-processing hierarchy\n(Interoception → Exteroception → Cognition),\nthe drug × session effect on AUC changes by %.6f seconds (linear contrast).\n",
    b_3L))
}

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSTEP 5 — Residual diagnostics (fig16)\n", SEP, "\n", sep = "")

resids  <- resid(m_self)
fitted_ <- fitted(m_self)

sk <- skewness(resids);  ku <- kurtosis(resids)
cat(sprintf("Skewness : %.4f\n", sk))
cat(sprintf("Kurtosis : %.4f  (excess = %.4f)\n", ku, ku - 3))

set.seed(42)
sw_idx <- sample(length(resids), min(5000L, length(resids)))
sw     <- shapiro.test(resids[sw_idx])
cat(sprintf("Shapiro-Wilk (n=%d): W = %.5f, p = %.4e\n",
            length(sw_idx), sw$statistic, sw$p.value))

qq_n   <- min(5000L, length(resids))
qq_smp <- sort(resids[sample(length(resids), qq_n)])
qq_teo <- qnorm(ppoints(qq_n))

p_qq <- plot_ly() %>%
  add_trace(x = qq_teo, y = qq_smp, type = "scatter", mode = "markers",
            marker = list(size = 4, color = "steelblue", opacity = 0.5)) %>%
  add_trace(x = range(qq_teo), y = range(qq_teo) * sd(qq_smp),
            type = "scatter", mode = "lines",
            line = list(color = "red", dash = "dash")) %>%
  layout(title = "QQ Plot", xaxis = list(title = "Theoretical quantiles"),
         yaxis = list(title = "Sample quantiles"), showlegend = FALSE)

rvf_idx <- sample(length(resids), min(5000L, length(resids)))
p_rvf <- plot_ly() %>%
  add_trace(x = fitted_[rvf_idx], y = resids[rvf_idx],
            type = "scatter", mode = "markers",
            marker = list(size = 4, color = "steelblue", opacity = 0.4)) %>%
  add_trace(x = range(fitted_), y = c(0, 0), type = "scatter", mode = "lines",
            line = list(color = "red", dash = "dash")) %>%
  layout(title = "Residuals vs Fitted",
         xaxis = list(title = "Fitted"), yaxis = list(title = "Residuals"),
         showlegend = FALSE)

p_hist <- plot_ly(x = resids, type = "histogram",
                  marker = list(color = "steelblue",
                                line = list(color = "white", width = 0.3)),
                  nbinsx = 60) %>%
  layout(title = "Histogram", xaxis = list(title = "Residual"),
         yaxis = list(title = "Count"))

fig16 <- subplot(p_qq, p_rvf, p_hist, nrows = 1, titleX = TRUE, titleY = TRUE,
                 margin = 0.06) %>%
  layout(title = list(text = "Self LMM Residual Diagnostics (Step 13)", x = 0.5),
         showlegend = FALSE)

saveWidget(fig16, OUT_FIG16, selfcontained = FALSE,
           title = "Self LMM Residual Diagnostics")
cat(sprintf("Saved: %s\n", OUT_FIG16))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSTEP 6 — Visualization (fig17)\n", SEP, "\n", sep = "")

LAYER_COLORS <- c(
  Interoception = "#1f77b4",
  Exteroception = "#ff7f0e",
  Cognition     = "#2ca02c"
)

# Per-ROI means by (roi_pos_id, group, session) — averaged over subjects
roi_means <- df %>%
  group_by(roi_pos_id, group, session) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop") %>%
  left_join(LAYER_MAP[, c("roi_pos_id", "layer", "roi_name")],
            by = "roi_pos_id") %>%
  mutate(roi_pos_id = as.integer(as.character(roi_pos_id)),
         ses_lbl = ifelse(session == "ses-01", "pre", "post"))

roi_wide <- roi_means %>%
  pivot_wider(id_cols = c(roi_pos_id, layer, roi_name),
              names_from = c(group, ses_lbl), values_from = mean_auc)

expected_cols <- c("placebo_pre", "placebo_post", "verum_pre", "verum_post")
missing_cols  <- setdiff(expected_cols, names(roi_wide))

if (length(missing_cols) > 0) {
  cat(sprintf("WARNING: pivot columns missing: %s\n", paste(missing_cols, collapse = ", ")))
} else {
  roi_wide <- roi_wide %>%
    mutate(drug_ses_delta = (verum_post - verum_pre) - (placebo_post - placebo_pre))

  # Layer-level summaries with 95% CI (t-distribution)
  ci95 <- function(x) qt(0.975, df = length(x) - 1) * sd(x, na.rm = TRUE) / sqrt(sum(!is.na(x)))

  layer_delta <- roi_wide %>%
    group_by(layer) %>%
    summarise(mean_d = mean(drug_ses_delta, na.rm = TRUE),
              ci     = ci95(drug_ses_delta),
              n      = sum(!is.na(drug_ses_delta)),
              .groups = "drop") %>%
    mutate(layer = factor(layer, levels = c("Interoception", "Exteroception", "Cognition")))

  # ── Left panel: drug×session delta per layer ───────────────────────────────
  p_left <- plot_ly()

  # individual ROI dots (jittered)
  set.seed(1)
  for (lyr in c("Interoception", "Exteroception", "Cognition")) {
    roi_sub <- roi_wide[roi_wide$layer == lyr, ]
    jitter  <- runif(nrow(roi_sub), -0.15, 0.15)
    p_left  <- p_left %>%
      add_trace(x = rep(lyr, nrow(roi_sub)) ,
                y = roi_sub$drug_ses_delta,
                type = "scatter", mode = "markers",
                marker = list(size = 7,
                              color = LAYER_COLORS[[lyr]],
                              opacity = 0.5,
                              symbol = "circle"),
                text  = roi_sub$roi_name,
                hovertemplate = "%{text}<br>delta=%{y:.4f}<extra></extra>",
                name  = lyr, legendgroup = lyr, showlegend = FALSE)
  }

  # mean ± CI bars
  for (i in seq_len(nrow(layer_delta))) {
    r   <- layer_delta[i, ]
    lyr <- as.character(r$layer)
    p_left <- p_left %>%
      add_trace(x = lyr, y = r$mean_d,
                type = "scatter", mode = "markers",
                marker = list(size = 14, color = LAYER_COLORS[[lyr]],
                              symbol = "diamond"),
                error_y = list(type = "data", array = r$ci,
                               color = LAYER_COLORS[[lyr]], thickness = 2),
                name = lyr, legendgroup = lyr, showlegend = TRUE)
  }

  # trend line connecting layer means
  layer_order <- c("Interoception", "Exteroception", "Cognition")
  trend_y <- layer_delta$mean_d[match(layer_order, as.character(layer_delta$layer))]
  p_left <- p_left %>%
    add_trace(x = layer_order, y = trend_y,
              type = "scatter", mode = "lines",
              line = list(color = "grey40", dash = "dash", width = 1.5),
              name = "trend", showlegend = FALSE)

  # annotation
  annot_text <- if (length(b_3L) > 0 && length(p_3L) > 0) {
    sprintf("Linear: β = %.4f, p = %.3g", b_3L, p_3L)
  } else { "" }

  p_left <- p_left %>%
    layout(title = "Drug × Session Effect per Layer",
           xaxis = list(title = "Self-processing layer",
                        categoryorder = "array",
                        categoryarray = layer_order),
           yaxis = list(title = "Drug × Session delta AUC (s)"),
           annotations = list(list(
             x = 0.5, y = 1.08, xref = "paper", yref = "paper",
             text = annot_text, showarrow = FALSE,
             font = list(size = 11)
           )))

  # ── Right panel: pre→post change per group per layer ──────────────────────
  # Per-subject means by (subject, layer, group, session)
  subj_layer <- df %>%
    group_by(subject, layer, group, session) %>%
    summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop")

  subj_wide <- subj_layer %>%
    pivot_wider(id_cols = c(subject, layer, group),
                names_from = session, values_from = mean_auc,
                names_prefix = "ses_") %>%
    rename(pre = `ses_ses-01`, post = `ses_ses-02`) %>%
    mutate(change = post - pre)

  layer_grp_change <- subj_wide %>%
    group_by(layer, group) %>%
    summarise(mean_change = mean(change, na.rm = TRUE),
              ci = ci95(change),
              .groups = "drop") %>%
    mutate(layer = factor(layer, levels = c("Interoception", "Exteroception", "Cognition")))

  p_right <- plot_ly()
  for (grp in c("placebo", "verum")) {
    grp_col <- if (grp == "placebo") "#636EFA" else "#EF553B"
    sub_grp <- layer_grp_change[layer_grp_change$group == grp, ]
    sub_grp <- sub_grp[order(match(as.character(sub_grp$layer), layer_order)), ]

    p_right <- p_right %>%
      add_trace(x = as.character(sub_grp$layer),
                y = sub_grp$mean_change,
                type = "scatter", mode = "markers+lines",
                marker = list(size = 10, color = grp_col),
                line   = list(color = grp_col, width = 2),
                error_y = list(type = "data", array = sub_grp$ci,
                               color = grp_col, thickness = 2),
                name = grp)
  }
  p_right <- p_right %>%
    layout(title = "Pre→Post Change per Group per Layer",
           xaxis = list(title = "Self-processing layer",
                        categoryorder = "array",
                        categoryarray = layer_order),
           yaxis = list(title = "Mean session change in AUC (s)"))

  fig17 <- subplot(p_left, p_right, nrows = 1, titleX = TRUE, titleY = TRUE,
                   margin = 0.07, shareY = FALSE) %>%
    layout(title = list(text = "Self Atlas: Drug × Session Effect by Layer (Step 13)",
                        x = 0.5),
           legend = list(x = 0.02, y = 0.98))

  saveWidget(fig17, OUT_FIG17, selfcontained = FALSE,
             title = "Self Layer Drug x Session")
  cat(sprintf("Saved: %s\n", OUT_FIG17))
}

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSTEP 7 — Leave-one-ROI-out robustness\n", SEP, "\n", sep = "")

roi_ids     <- levels(df$roi_pos_id)
n_roi       <- length(roi_ids)
ctrl_loo    <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5),
                           check.conv.singular = .makeCC("ignore", tol = 1e-4))
TARGET_TERM <- "groupverum:sessionses-02:layer.L"

loo_roi_results <- data.frame(
  dropped_roi   = character(n_roi),
  roi_name      = character(n_roi),
  layer         = character(n_roi),
  b_3way_L      = numeric(n_roi),
  p_3way_L      = numeric(n_roi),
  converged     = logical(n_roi),
  stringsAsFactors = FALSE
)

cat(sprintf("Refitting model %d times (one per ROI)...\n", n_roi))

for (i in seq_along(roi_ids)) {
  roi_i  <- roi_ids[i]
  info_i <- LAYER_MAP[LAYER_MAP$roi_pos_id == as.integer(roi_i), ]
  dsub   <- df[df$roi_pos_id != roi_i, ]
  dsub$roi_pos_id <- droplevels(dsub$roi_pos_id)
  dsub$subject    <- droplevels(dsub$subject)

  m_i <- tryCatch(
    suppressWarnings(lme4::lmer(FORMULA_USED, data = dsub,
                                REML = TRUE, control = ctrl_loo)),
    error = function(e) NULL
  )

  if (is.null(m_i)) {
    b_i <- NA_real_;  p_i <- NA_real_;  conv_i <- FALSE
  } else {
    raw_i  <- coef(summary(m_i))
    r_i    <- raw_i[rownames(raw_i) == TARGET_TERM, , drop = FALSE]
    b_i    <- if (nrow(r_i) > 0) r_i[1, "Estimate"]     else NA_real_
    p_i    <- if (nrow(r_i) > 0) r_i[1, "Pr(>|t|)"]     else NA_real_
    conv_i <- is.null(m_i@optinfo$conv$lme4$messages)
  }

  loo_roi_results[i, ] <- list(
    roi_i, info_i$roi_name, info_i$layer, b_i, p_i, conv_i
  )

  if (i %% 10 == 0 || i == n_roi)
    cat(sprintf("  ... %d / %d done\n", i, n_roi))
}

b_loo_roi    <- loo_roi_results$b_3way_L[!is.na(loo_roi_results$b_3way_L)]
n_sig_roi    <- sum(loo_roi_results$p_3way_L < 0.05, na.rm = TRUE)
b_base       <- fe_tbl$estimate[fe_tbl$term == TARGET_TERM]
most_influen <- loo_roi_results$dropped_roi[which.min(abs(loo_roi_results$b_3way_L - b_base))]

cat(sprintf("\nLOO-ROI summary:\n"))
cat(sprintf("  β range across 37 refits: [%.5f, %.5f]\n",
            min(b_loo_roi), max(b_loo_roi)))
cat(sprintf("  Refits with p < 0.05: %d / %d\n", n_sig_roi, n_roi))
cat(sprintf("  Most influential ROI (largest β change): %s\n",
            loo_roi_results$roi_name[loo_roi_results$dropped_roi == most_influen]))

loo_stable_roi <- (n_sig_roi == n_roi)
cat(sprintf("  Linear contrast %s across LOO-ROI refits.\n",
            if (loo_stable_roi) "IS stable" else "is NOT stable"))

write.csv(loo_roi_results, OUT_LOO_ROI, row.names = FALSE)
cat(sprintf("Saved: %s\n", OUT_LOO_ROI))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSTEP 8 — Leave-one-subject-out robustness\n", SEP, "\n", sep = "")

subj_ids <- levels(df$subject)
n_subj   <- length(subj_ids)

loo_sub_results <- data.frame(
  dropped_subject = character(n_subj),
  b_3way_L        = numeric(n_subj),
  p_3way_L        = numeric(n_subj),
  converged       = logical(n_subj),
  stringsAsFactors = FALSE
)

cat(sprintf("Refitting model %d times (one per subject)...\n", n_subj))

for (i in seq_along(subj_ids)) {
  subj_i <- subj_ids[i]
  dsub   <- df[df$subject != subj_i, ]
  dsub$subject    <- droplevels(dsub$subject)
  dsub$roi_pos_id <- droplevels(dsub$roi_pos_id)

  m_i <- tryCatch(
    suppressWarnings(lme4::lmer(FORMULA_USED, data = dsub,
                                REML = TRUE, control = ctrl_loo)),
    error = function(e) NULL
  )

  if (is.null(m_i)) {
    b_i <- NA_real_;  p_i <- NA_real_;  conv_i <- FALSE
  } else {
    raw_i  <- coef(summary(m_i))
    r_i    <- raw_i[rownames(raw_i) == TARGET_TERM, , drop = FALSE]
    b_i    <- if (nrow(r_i) > 0) r_i[1, "Estimate"]     else NA_real_
    p_i    <- if (nrow(r_i) > 0) r_i[1, "Pr(>|t|)"]     else NA_real_
    conv_i <- is.null(m_i@optinfo$conv$lme4$messages)
  }

  loo_sub_results[i, ] <- list(subj_i, b_i, p_i, conv_i)

  if (i %% 10 == 0 || i == n_subj)
    cat(sprintf("  ... %d / %d done\n", i, n_subj))
}

b_loo_sub  <- loo_sub_results$b_3way_L[!is.na(loo_sub_results$b_3way_L)]
n_sig_sub  <- sum(loo_sub_results$p_3way_L < 0.05, na.rm = TRUE)

cat(sprintf("\nLOO-subject summary:\n"))
cat(sprintf("  β range across 35 refits: [%.5f, %.5f]\n",
            min(b_loo_sub), max(b_loo_sub)))
cat(sprintf("  Refits with p < 0.05: %d / %d\n", n_sig_sub, n_subj))

loo_stable_sub <- (n_sig_sub == n_subj)
cat(sprintf("  Linear contrast %s across LOO-subject refits.\n",
            if (loo_stable_sub) "IS stable" else "is NOT stable"))

write.csv(loo_sub_results, OUT_LOO_SUB, row.names = FALSE)
cat(sprintf("Saved: %s\n", OUT_LOO_SUB))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSTEP 9 — LRT: full vs reduced (drops group:layer + group:session:layer)\n",
    SEP, "\n", sep = "")

# Reduced: drops group:layer.L, group:layer.Q, group:session:layer.L, group:session:layer.Q (4 df)
FORMULA_RED_LRT <- if (random_slope_used) {
  auc ~ group * session + layer + session:layer + (1 + session | subject) + (1 | roi_pos_id)
} else {
  auc ~ group * session + layer + session:layer + (1 | subject) + (1 | roi_pos_id)
}

ctrl_ml <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 1e5))

cat("Fitting full model (ML)...\n")
m_full_ml <- suppressWarnings(
  lme4::lmer(FORMULA_USED, data = df, REML = FALSE, control = ctrl_ml)
)

cat("Fitting reduced model (ML)...\n")
m_red_ml <- suppressWarnings(
  lme4::lmer(FORMULA_RED_LRT, data = df, REML = FALSE, control = ctrl_ml)
)

lrt     <- anova(m_red_ml, m_full_ml)
cat("\nLRT output:\n")
print(lrt)

chi2_lrt  <- lrt$Chisq[2]
df_lrt    <- lrt$Df[2]
pval_lrt  <- lrt$`Pr(>Chisq)`[2]
cat(sprintf("\nLRT: chi2(%d) = %.4f, p = %.4g\n", df_lrt, chi2_lrt, pval_lrt))
lrt_sig <- pval_lrt < 0.05
cat(if (lrt_sig) "=> SIGNIFICANT: group:layer terms improve model fit.\n"
    else          "=> Non-significant.\n")

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSaving CSV outputs\n", SEP, "\n", sep = "")

write.csv(fe_tbl, OUT_FIXED,  row.names = FALSE);  cat(sprintf("Saved: %s\n", OUT_FIXED))
vc <- as.data.frame(VarCorr(m_self))
write.csv(vc,     OUT_RANDOM, row.names = FALSE);  cat(sprintf("Saved: %s\n", OUT_RANDOM))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSUMMARY\n", SEP, "\n", sep = "")

b_3L_v  <- if (length(b_3L) > 0) b_3L else NA
b_3Q_v  <- if (length(b_3Q) > 0) b_3Q else NA
p_3L_v  <- if (length(p_3L) > 0) p_3L else NA
p_3Q_v  <- if (length(p_3Q) > 0) p_3Q else NA
se_3L   <- fe_tbl$SE[    fe_tbl$term == "groupverum:sessionses-02:layer.L"]
t_3L    <- fe_tbl$t_value[fe_tbl$term == "groupverum:sessionses-02:layer.L"]

sig_3L_str <- if (!is.na(p_3L_v) && p_3L_v < 0.05) "significant" else "not significant"
sig_3Q_str <- if (!is.na(p_3Q_v) && p_3Q_v < 0.05) "significant" else "not significant"
sug_str    <- if (!is.na(p_3L_v) && p_3L_v < 0.05) "suggesting"  else "not supporting"
loo_r_str  <- if (loo_stable_roi) "stable" else "unstable"
loo_s_str  <- if (loo_stable_sub) "stable" else "unstable"
lrt_str    <- if (lrt_sig)        "significant" else "not significant"

cat(sprintf(
  "Self atlas primary finding: the linear contrast on group × session × layer was %s\n(β = %.6f, SE = %.6f, t = %.4f, p = %.4g), %s a monotonic change\nin drug × session effect across the self-processing hierarchy.\nThe quadratic contrast was %s (β = %.6f, p = %.4g).\nRobustness: leave-one-ROI-out %s (%d / %d p < 0.05),\nleave-one-subject-out %s (%d / %d p < 0.05).\nLRT %s (chi2(%d) = %.4f, p = %.4g).\n",
  sig_3L_str, b_3L_v, if (length(se_3L) > 0) se_3L else NA,
  if (length(t_3L) > 0) t_3L else NA, p_3L_v,
  sug_str, sig_3Q_str, b_3Q_v, p_3Q_v,
  loo_r_str, n_sig_roi, n_roi,
  loo_s_str, n_sig_sub, n_subj,
  lrt_str, df_lrt, chi2_lrt, pval_lrt
))

cat("\n", SEP, "\nALL STEPS COMPLETE\n", SEP, "\n", sep = "")
