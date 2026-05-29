# 22_cat_flattening_analysis.R
# Step 15: categorical self-vs-nonself flattening analysis with EMMs
#
# NOTE: task requested filename 21_cat_flattening_analysis.R but
#   21_self_vs_sensorymotor_visual.R already exists. Using 22_.
#
# Tests whether the drug × session interaction differs across 4 region
# categories (nonself sensory-motor, Interoception, Exteroception, Cognition)
# and whether DMT "flattens" the self–nonself AUC gap.
#
# Run from repo root:
#   Rscript 04_statistics/scripts/22_cat_flattening_analysis.R

# ── Auto-install emmeans if needed ────────────────────────────────────────────
if (!requireNamespace("emmeans", quietly = TRUE)) {
  message("Installing emmeans...")
  install.packages("emmeans", repos = "https://cloud.r-project.org", quiet = TRUE)
}

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
  library(emmeans)
  library(moments)
  library(dplyr)
  library(tidyr)
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

COMBINED_CSV <- file.path(REPO_ROOT, "04_statistics", "results",
                           "sensory_motor_self_combined.csv")
OUT_DATA     <- file.path(REPO_ROOT, "04_statistics", "results", "cat_model_data.csv")
OUT_FE       <- file.path(REPO_ROOT, "04_statistics", "results", "cat_lmm_fixed_effects.csv")
OUT_EMM      <- file.path(REPO_ROOT, "04_statistics", "results", "cat_lmm_emm_contrasts.csv")
OUT_FLAT     <- file.path(REPO_ROOT, "04_statistics", "results", "cat_lmm_flattening.csv")
OUT_FIG19    <- file.path(REPO_ROOT, "04_statistics", "figures",
                           "fig19_cat_lmm_residual_diagnostics.html")
OUT_FIG20    <- file.path(REPO_ROOT, "04_statistics", "figures",
                           "fig20_cat_emm_profile.html")
OUT_FIG21    <- file.path(REPO_ROOT, "04_statistics", "figures",
                           "fig21_cat_flattening_dotplot.html")

# ── Idempotent guard ───────────────────────────────────────────────────────────
all_out <- c(OUT_DATA, OUT_FE, OUT_EMM, OUT_FLAT, OUT_FIG19, OUT_FIG20, OUT_FIG21)
if (all(file.exists(all_out))) {
  cat("All outputs already exist — nothing to do. Delete to rerun.\n")
  quit(status = 0)
}

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n22_cat_flattening_analysis.R — Step 15\n", SEP, "\n\n", sep = "")

# ─────────────────────────────────────────────────────────────────────────────
cat(SEP, "\nPART 1 — Assemble categorical table\n", SEP, "\n", sep = "")

raw <- read.csv(COMBINED_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows from %s\n", nrow(raw), basename(COMBINED_CSV)))

# Recode to task-specified column names
df <- raw %>%
  mutate(
    layer        = ifelse(region_category == "Sensory-Motor", "nonself", region_category),
    atlas_source = ifelse(atlas_source == "self", "sphere", "parcel"),
    group        = drug_group
  ) %>%
  select(subject, session, roi_pos_id, roi_uid, roi_name,
         auc, group, layer, atlas_source)

cat("\nRows per layer:\n")
for (lyr in c("nonself", "Interoception", "Exteroception", "Cognition")) {
  cat(sprintf("  %-15s : %d\n", lyr, sum(df$layer == lyr)))
}
cat(sprintf("  %-15s : %d\n", "TOTAL", nrow(df)))
cat("\nRows per atlas_source:\n")
cat(sprintf("  sphere  : %d\n", sum(df$atlas_source == "sphere")))
cat(sprintf("  parcel  : %d\n", sum(df$atlas_source == "parcel")))

expected <- 35 * 2 * (37 + 99)
if (abs(nrow(df) - expected) > 5) {
  cat(sprintf("WARNING: expected ~%d rows, got %d\n", expected, nrow(df)))
} else {
  cat(sprintf("\nRow count OK (%d vs expected ~%d)\n", nrow(df), expected))
}

write.csv(df, OUT_DATA, row.names = FALSE)
cat(sprintf("Saved: %s\n", OUT_DATA))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 2 — Factor variables and reference levels\n", SEP, "\n", sep = "")

LAYER_LEVELS <- c("nonself", "Interoception", "Exteroception", "Cognition")
df$layer      <- factor(df$layer,   levels = LAYER_LEVELS)
df$group      <- factor(df$group,   levels = c("placebo", "verum"))
df$session    <- factor(df$session, levels = c("ses-01", "ses-02"))
df$subject    <- factor(df$subject)
df$roi_uid    <- factor(df$roi_uid)

cat(sprintf("layer    levels (ref = '%s'): %s\n",
            levels(df$layer)[1], paste(levels(df$layer), collapse = ", ")))
cat(sprintf("group    levels (ref = '%s'): %s\n",
            levels(df$group)[1], paste(levels(df$group), collapse = ", ")))
cat(sprintf("session  levels (ref = '%s'): %s\n",
            levels(df$session)[1], paste(levels(df$session), collapse = ", ")))
cat(sprintf("Subjects : %d  |  ROI UIDs : %d\n",
            nlevels(df$subject), nlevels(df$roi_uid)))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 3 — Baseline gradient check (placebo, ses-01)\n", SEP, "\n", sep = "")

df_base <- df[df$group == "placebo" & df$session == "ses-01", ]
cat(sprintf("Baseline subset: %d rows\n\n", nrow(df_base)))

layer_means <- tapply(df_base$auc, df_base$layer, mean, na.rm = TRUE)
cat("Mean AUC per layer at baseline (placebo, ses-01):\n")
for (lyr in LAYER_LEVELS)
  cat(sprintf("  %-15s : %.4f\n", lyr, layer_means[lyr]))

aov_base <- aov(auc ~ layer, data = df_base)
aov_sum  <- summary(aov_base)[[1]]
F_base   <- aov_sum["layer", "F value"]
p_base   <- aov_sum["layer", "Pr(>F)"]
df1_base <- aov_sum["layer", "Df"]
df2_base <- aov_sum["Residuals", "Df"]

cat(sprintf("\nOne-way ANOVA: F(%d,%d) = %.4f, p = %.4g\n",
            df1_base, df2_base, F_base, p_base))

tukey <- TukeyHSD(aov_base, "layer")$layer
cat("\nTukey HSD pairwise comparisons:\n")
print(round(tukey, 4))

if (p_base < 0.05) {
  cat(sprintf(
    "\nBaseline AUC DOES differ across layers (F = %.4f, p = %.4g). The gradient exists at baseline.\n",
    F_base, p_base))
} else {
  cat(sprintf(
    "\nWARNING: Baseline AUC does NOT significantly differ across layers (F = %.4f, p = %.4g).\nLayers do not separate at baseline — proceed with caution.\n",
    F_base, p_base))
}

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 4 — Fit primary model\n", SEP, "\n", sep = "")

FORMULA_FULL <- auc ~ group * session * layer +
                      (1 + session | subject) + (1 | roi_uid)
FORMULA_SIMP <- auc ~ group * session * layer +
                      (1 | subject) + (1 | roi_uid)

ctrl_bobyqa  <- lmerControl(optimizer = "bobyqa",
                             optCtrl   = list(maxfun = 2e5),
                             check.conv.grad = .makeCC("warning", tol = 0.002))
ctrl_nlopt   <- lmerControl(optimizer = "nloptwrap",
                             optCtrl   = list(maxfun = 2e5),
                             check.conv.grad = .makeCC("warning", tol = 0.002))

try_fit <- function(formula, ctrl, label, df) {
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

cat("Attempt 1: (1+session|subject) + (1|roi_uid), bobyqa...\n")
m_cat <- try_fit(FORMULA_FULL, ctrl_bobyqa, "full/bobyqa", df)

if (!is.null(m_cat) && isSingular(m_cat)) {
  cat("  Singular fit. Trying nloptwrap...\n")
  m_cat <- try_fit(FORMULA_FULL, ctrl_nlopt, "full/nloptwrap", df)
}
if (!is.null(m_cat) && isSingular(m_cat)) {
  cat("  Still singular. Falling back to (1|subject) + (1|roi_uid).\n")
  m_cat <- NULL
}

re_label <- "(1 + session | subject) + (1 | roi_uid)"
if (is.null(m_cat)) {
  cat("Attempt 2 (fallback): (1|subject) + (1|roi_uid), bobyqa...\n")
  m_cat <- try_fit(FORMULA_SIMP, ctrl_bobyqa, "simple/bobyqa", df)
  re_label <- "(1 | subject) + (1 | roi_uid)"
}
if (is.null(m_cat)) stop("All model-fitting attempts failed.")

FORMULA_USED <- if (re_label == "(1 + session | subject) + (1 | roi_uid)") FORMULA_FULL else FORMULA_SIMP
cat(sprintf("\nRandom effects used: %s\n", re_label))

conv_msgs <- m_cat@optinfo$conv$lme4$messages
if (!is.null(conv_msgs)) {
  cat(sprintf("CONVERGENCE WARNINGS:\n%s\n\n", paste(conv_msgs, collapse = "\n")))
} else {
  cat("Convergence: OK\n\n")
}

s_main <- summary(m_cat)
print(s_main)

raw_fe <- coef(s_main)
fe_tbl <- data.frame(
  term     = rownames(raw_fe),
  estimate = raw_fe[, "Estimate"],
  SE       = raw_fe[, "Std. Error"],
  df       = raw_fe[, "df"],
  t_value  = raw_fe[, "t value"],
  p_value  = raw_fe[, "Pr(>|t|)"],
  stringsAsFactors = FALSE, row.names = NULL
)

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 5 — Residual diagnostics\n", SEP, "\n", sep = "")

resids  <- resid(m_cat)
fitted_ <- fitted(m_cat)

sk <- skewness(resids);  ku <- kurtosis(resids)
cat(sprintf("Skewness : %.4f\n", sk))
cat(sprintf("Kurtosis : %.4f  (excess = %.4f)\n", ku, ku - 3))
set.seed(42)
sw_idx <- sample(length(resids), min(5000L, length(resids)))
sw     <- shapiro.test(resids[sw_idx])
cat(sprintf("Shapiro-Wilk (n=%d): W = %.5f, p = %.4e\n",
            length(sw_idx), sw$statistic, sw$p.value))
if (ku > 10) cat("NOTE: excess kurtosis > 7 — heavy tails, normality violated.\n")

qq_n   <- min(5000L, length(resids))
qq_smp <- sort(resids[sample(length(resids), qq_n)])
qq_teo <- qnorm(ppoints(qq_n))

p_qq <- plot_ly() %>%
  add_trace(x = qq_teo, y = qq_smp, type = "scatter", mode = "markers",
            marker = list(size = 4, color = "steelblue", opacity = 0.5)) %>%
  add_trace(x = range(qq_teo), y = range(qq_teo) * sd(qq_smp),
            type = "scatter", mode = "lines",
            line = list(color = "red", dash = "dash")) %>%
  layout(title = "QQ Plot",
         xaxis = list(title = "Theoretical quantiles"),
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
                  nbinsx = 80) %>%
  layout(title = "Histogram", xaxis = list(title = "Residual"),
         yaxis = list(title = "Count"))

fig19 <- subplot(p_qq, p_rvf, p_hist, nrows = 1, titleX = TRUE, titleY = TRUE,
                 margin = 0.06) %>%
  layout(title = list(text = "Categorical LMM Residual Diagnostics (Step 15)", x = 0.5),
         showlegend = FALSE)
saveWidget(fig19, OUT_FIG19, selfcontained = FALSE,
           title = "Categorical LMM Diagnostics")
cat(sprintf("Saved: %s\n", OUT_FIG19))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 6 — Omnibus flattening test (LRT)\n", SEP, "\n", sep = "")

# Reduced: drops 3-way group:session:layer (keeps all 2-way interactions)
FORMULA_RED_LRT <- if (re_label == "(1 + session | subject) + (1 | roi_uid)") {
  auc ~ group * session + layer + session:layer + group:layer +
        (1 + session | subject) + (1 | roi_uid)
} else {
  auc ~ group * session + layer + session:layer + group:layer +
        (1 | subject) + (1 | roi_uid)
}

ctrl_ml <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
cat("Fitting full model (ML)...\n")
m_full_ml <- suppressWarnings(
  lme4::lmer(FORMULA_USED, data = df, REML = FALSE, control = ctrl_ml))
cat("Fitting reduced model (ML)...\n")
m_red_ml  <- suppressWarnings(
  lme4::lmer(FORMULA_RED_LRT, data = df, REML = FALSE, control = ctrl_ml))

lrt    <- anova(m_red_ml, m_full_ml)
print(lrt)
chi2_lrt <- lrt$Chisq[2]
df_lrt   <- lrt$Df[2]
p_lrt    <- lrt$`Pr(>Chisq)`[2]
lrt_sig  <- p_lrt < 0.05

cat(sprintf(
  "\nThe group × session × layer three-way interaction %s significant\n(χ²(%d) = %.4f, p = %.4g), indicating that the across-layer pattern\n%s change differently under DMT.\n",
  if (lrt_sig) "IS" else "is NOT", df_lrt, chi2_lrt, p_lrt,
  if (lrt_sig) "DOES" else "does NOT"
))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 7 — Estimated marginal means and contrasts\n", SEP, "\n", sep = "")

emm_main <- emmeans(m_cat, ~ group * session * layer)

cat("\n--- 7a: Cell means (16 cells: 2 groups × 2 sessions × 4 layers) ---\n")
emm_df <- as.data.frame(emm_main)
print(emm_df[, c("group", "session", "layer", "emmean", "SE", "df",
                  "lower.CL", "upper.CL")], row.names = FALSE)

# 7b: Self–nonself gap per group × session
cat("\n--- 7b: Self–nonself gap per group × session ---\n")
emm_by <- emmeans(m_cat, ~ layer | group + session)
gaps <- contrast(emm_by,
  method = list(
    "Interoception - nonself" = c(-1, 1, 0, 0),
    "Exteroception - nonself" = c(-1, 0, 1, 0),
    "Cognition - nonself"     = c(-1, 0, 0, 1)
  ),
  adjust = "none"
)
gaps_df <- as.data.frame(summary(gaps, infer = TRUE))
cat("(gap = self_layer EMM − nonself EMM)\n\n")
print(gaps_df[, c("contrast", "group", "session",
                   "estimate", "SE", "df", "lower.CL", "upper.CL",
                   "t.ratio", "p.value")], row.names = FALSE)

# 7c: Flattening = Δgap_verum − Δgap_placebo
cat("\n--- 7c: Flattening contrasts (Δgap_verum − Δgap_placebo) ---\n")
flat <- contrast(gaps,
  interaction = list(group = "consec", session = "consec"),
  adjust = "none"
)
flat_df <- as.data.frame(summary(flat, infer = TRUE))
# Rename interaction columns for clarity
names(flat_df)[names(flat_df) == "group_consec"]   <- "group_contrast"
names(flat_df)[names(flat_df) == "session_consec"] <- "session_contrast"

cat("(positive = gap widens under DMT; negative = gap narrows = flattening)\n\n")
print(flat_df[, c("contrast", "group_contrast", "session_contrast",
                   "estimate", "SE", "df", "lower.CL", "upper.CL",
                   "t.ratio", "p.value")], row.names = FALSE)

# Average flattening across all 3 self layers
flat_avg <- contrast(flat, method = list("avg_3layers" = c(1/3, 1/3, 1/3)))
flat_avg_df <- as.data.frame(summary(flat_avg, infer = TRUE))
cat("\nAverage flattening across 3 self layers:\n")
print(flat_avg_df[, c("estimate", "SE", "df", "lower.CL", "upper.CL",
                       "t.ratio", "p.value")], row.names = FALSE)
avg_flat_p <- flat_avg_df$p.value[1]
avg_flat_b <- flat_avg_df$estimate[1]

# 7d: "Meet at a point" — gap_verum_post = 0?
cat("\n--- 7d: Does the self–nonself gap disappear after DMT? ---\n")
cat("Test: gap_verum_post = 0 for each self layer\n\n")
meet_df <- gaps_df[gaps_df$group == "verum" & gaps_df$session == "ses-02", ]
print(meet_df[, c("contrast", "estimate", "SE", "df",
                   "lower.CL", "upper.CL", "t.ratio", "p.value")], row.names = FALSE)

# Save all EMM contrasts
emm_contrasts_all <- rbind(
  cbind(test = "gap", gaps_df[, c("contrast", "group", "session",
                                   "estimate", "SE", "df",
                                   "lower.CL", "upper.CL", "t.ratio", "p.value")]),
  cbind(test = "meet_at_point",
        group = "verum", session = "ses-02",
        meet_df[, c("contrast", "estimate", "SE", "df",
                    "lower.CL", "upper.CL", "t.ratio", "p.value")])
)
write.csv(emm_contrasts_all, OUT_EMM, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", OUT_EMM))

flat_save <- flat_df
flat_save$avg_flattening_p <- NA
flat_save$avg_flattening_b <- NA
flat_save$avg_flattening_b[1] <- avg_flat_b
flat_save$avg_flattening_p[1] <- avg_flat_p
write.csv(flat_save, OUT_FLAT, row.names = FALSE)
cat(sprintf("Saved: %s\n", OUT_FLAT))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 8 — Visualization\n", SEP, "\n", sep = "")

LAYER_ORDER <- c("nonself", "Interoception", "Exteroception", "Cognition")
LAYER_X     <- setNames(1:4, LAYER_ORDER)
SES_COLORS  <- c("ses-01" = "#3D5A6C", "ses-02" = "#8B7355")

# ── fig20: Profile line plot ───────────────────────────────────────────────────
make_profile_panel <- function(grp_label, show_legend) {
  sub <- emm_df[emm_df$group == grp_label, ]

  p <- plot_ly()
  for (ses in c("ses-01", "ses-02")) {
    d   <- sub[sub$session == ses, ]
    d   <- d[match(LAYER_ORDER, as.character(d$layer)), ]
    xv  <- seq_along(LAYER_ORDER)
    col <- SES_COLORS[[ses]]
    lty <- if (ses == "ses-01") "solid" else "dash"
    ses_lbl <- if (ses == "ses-01") "Pre (ses-01)" else "Post (ses-02)"

    # CI ribbon (polygon)
    p <- p %>%
      add_trace(
        x     = c(xv, rev(xv)),
        y     = c(d$upper.CL, rev(d$lower.CL)),
        type  = "scatter", mode = "lines",
        fill  = "toself",
        fillcolor = paste0(substr(col, 1, 7), "22"),
        line  = list(color = "transparent"),
        showlegend = FALSE, hoverinfo = "skip"
      )
    # Line + markers
    p <- p %>%
      add_trace(
        x    = xv, y = d$emmean,
        type = "scatter", mode = "lines+markers",
        line   = list(color = col, width = 2, dash = lty),
        marker = list(size = 8, color = col),
        error_y = list(type = "data",
                       array    = d$upper.CL - d$emmean,
                       arrayminus = d$emmean - d$lower.CL,
                       color = col, thickness = 1.5, width = 4),
        name = ses_lbl, legendgroup = ses_lbl,
        showlegend = show_legend
      )
  }
  p <- p %>%
    layout(
      title = list(text = grp_label, font = list(size = 14)),
      xaxis = list(
        tickvals = 1:4,
        ticktext = LAYER_ORDER,
        title    = "Region category",
        showgrid = TRUE, gridcolor = "#eeeeee"
      ),
      yaxis = list(
        title    = "Est. marginal mean AUC (s)",
        showgrid = TRUE, gridcolor = "#eeeeee"
      ),
      plot_bgcolor  = "white",
      paper_bgcolor = "white"
    )
  p
}

p_plac <- make_profile_panel("placebo", show_legend = TRUE)
p_veru <- make_profile_panel("verum",   show_legend = FALSE)

fig20 <- subplot(p_plac, p_veru, nrows = 1, titleX = TRUE, titleY = TRUE,
                 shareY = TRUE, margin = 0.07) %>%
  layout(
    title = list(
      text = "Estimated marginal means by region category: Placebo vs Verum",
      x = 0.5
    ),
    legend = list(x = 0.01, y = 0.99)
  )
saveWidget(fig20, OUT_FIG20, selfcontained = FALSE, title = "EMM Profile Plot")
cat(sprintf("Saved: %s\n", OUT_FIG20))

# ── fig21: Flattening dotplot ──────────────────────────────────────────────────
SELF_LAYERS   <- c("Interoception", "Exteroception", "Cognition")
LAYER_COLORS  <- c(Interoception = "#1f77b4",
                   Exteroception = "#ff7f0e",
                   Cognition     = "#2ca02c")

# Extract per-layer flattening from flat_df
# The 'contrast' column in flat_df contains the layer contrast label
flat_plot <- flat_df
# Clean contrast column name (may contain layer name)
flat_plot$self_layer <- sub(" - nonself.*", "", flat_plot$contrast)

fig21 <- plot_ly() %>%
  add_trace(x = c(0.5, 3.5), y = c(0, 0),
            type = "scatter", mode = "lines",
            line = list(color = "black", dash = "dash", width = 1),
            showlegend = FALSE, hoverinfo = "skip")

for (i in seq_along(SELF_LAYERS)) {
  lyr <- SELF_LAYERS[i]
  r   <- flat_plot[flat_plot$self_layer == lyr, ]
  if (nrow(r) == 0) next
  col <- LAYER_COLORS[[lyr]]
  sig_marker <- if (!is.na(r$p.value) && r$p.value < 0.05) "asterisk" else "circle"

  fig21 <- fig21 %>%
    add_trace(x = i, y = r$estimate,
              type = "scatter", mode = "markers",
              marker = list(size = 12, color = col, symbol = sig_marker),
              error_y = list(type = "data",
                             array     = r$upper.CL - r$estimate,
                             arrayminus = r$estimate - r$lower.CL,
                             color = col, thickness = 2, width = 8),
              name = lyr,
              hovertemplate = sprintf(
                "%s<br>flattening = %%{y:.4f}<br>p = %.4g<extra></extra>",
                lyr, r$p.value))
}

fig21 <- fig21 %>%
  layout(
    title = list(
      text = "DMT flattening of self–nonself gap per self layer",
      x = 0.5
    ),
    xaxis = list(
      tickvals  = 1:3,
      ticktext  = SELF_LAYERS,
      title     = "Self-processing layer",
      showgrid  = FALSE,
      range     = c(0.4, 3.6)
    ),
    yaxis = list(
      title    = "Flattening: Δgap_verum − Δgap_placebo (AUC, s)",
      zeroline = FALSE,
      showgrid = TRUE, gridcolor = "#eeeeee"
    ),
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    annotations = list(list(
      x = 0.5, y = 1.06, xref = "paper", yref = "paper",
      text = "Negative = gap narrows (flattening); * = p < 0.05",
      showarrow = FALSE, font = list(size = 11, color = "grey40")
    ))
  )
saveWidget(fig21, OUT_FIG21, selfcontained = FALSE, title = "Flattening Dotplot")
cat(sprintf("Saved: %s\n", OUT_FIG21))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 9 — Robustness\n", SEP, "\n", sep = "")

# 9a: atlas_source as fixed effect
cat("--- 9a: Does atlas_source (sphere vs parcel) confound the 3-way interaction? ---\n")

FORMULA_SRC <- if (re_label == "(1 + session | subject) + (1 | roi_uid)") {
  auc ~ group * session * layer + atlas_source +
        (1 + session | subject) + (1 | roi_uid)
} else {
  auc ~ group * session * layer + atlas_source +
        (1 | subject) + (1 | roi_uid)
}

m_src <- tryCatch(
  suppressWarnings(lmer(FORMULA_SRC, data = df, REML = TRUE,
                        control = ctrl_bobyqa)),
  error = function(e) { cat(sprintf("atlas_source model failed: %s\n", e$message)); NULL }
)

if (!is.null(m_src)) {
  s_src   <- summary(m_src)
  fe_src  <- coef(s_src)
  src_b   <- fe_src["atlas_sourcesphere", "Estimate"]
  src_p   <- fe_src["atlas_sourcesphere", "Pr(>|t|)"]
  orig_3b <- fe_tbl$estimate[grepl("group.*session.*layer", fe_tbl$term) |
                              grepl("layer.*session.*group", fe_tbl$term)]

  cat(sprintf("atlas_source (sphere): beta = %.5f, p = %.4g\n", src_b, src_p))
  cat(sprintf("atlas_source %s significant.\n",
              if (src_p < 0.05) "IS" else "is NOT"))

  fe_src_df <- as.data.frame(fe_src)
  twi_orig <- fe_tbl[fe_tbl$term == "groupverum:sessionses-02:layerInteroception" |
                     grepl("groupverum:sessionses-02:layer", fe_tbl$term), ]
  twi_src  <- fe_src_df[grepl("groupverum:sessionses-02:layer", rownames(fe_src_df)), ]

  cat("\n3-way interaction terms before/after adding atlas_source:\n")
  for (tm in rownames(twi_src)) {
    b_orig <- fe_tbl$estimate[fe_tbl$term == tm]
    b_new  <- twi_src[tm, "Estimate"]
    if (length(b_orig) == 0) next
    cat(sprintf("  %s: original β=%.5f → with atlas_source β=%.5f (Δ=%.5f)\n",
                gsub("groupverum:sessionses-02:", "", tm),
                b_orig, b_new, b_new - b_orig))
  }
} else {
  cat("Could not fit atlas_source model — skipping 9a.\n")
}

# 9b: Agreement with G1 slope finding
cat("\n--- 9b: Qualitative comparison with G1 slope (Step 12) ---\n")
g1_b    <-  0.031928  # from Step 12: groupverum:sessionses-02:G1_z
g1_p    <-  0.002030
lrt_p_display <- p_lrt

cat(sprintf("Step 12 (G1 continuous): group:session:G1_z β = %.5f, p = %.4g\n",
            g1_b, g1_p))
cat(sprintf("Step 15 (categorical LRT): group:session:layer p = %.4g\n", lrt_p_display))

if ((g1_p < 0.05) == (lrt_p_display < 0.05)) {
  cat("AGREEMENT: both analyses %s a significant group:session interaction across cortical space.\n",
      if (g1_p < 0.05) "SHOW" else "do NOT show")
} else {
  cat("DISCREPANCY: G1 analysis and categorical analysis give different significance conclusions.\n")
}
cat(sprintf(
  "The G1 result (continuous sensory→transmodal gradient) and the categorical\n result (nonself sensory-motor vs self layers) test related but distinct questions.\n G1 measures a linear modulation along the cortical hierarchy; the categorical\nanalysis asks whether the gap between nonself and self contracts under DMT.\n"
))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSaving fixed effects CSV\n", SEP, "\n", sep = "")
write.csv(fe_tbl, OUT_FE, row.names = FALSE)
cat(sprintf("Saved: %s\n", OUT_FE))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSUMMARY\n", SEP, "\n", sep = "")

flat_any_sig <- any(flat_df$p.value < 0.05, na.rm = TRUE)
meet_any_sig <- any(meet_df$p.value < 0.05, na.rm = TRUE)

cat(sprintf(
  "The categorical flattening analysis %s a significant group × session × layer\ninteraction (LRT p = %.4g). Estimated marginal means show that [see fig20 for\nthe pattern across 4 layers for each group and session].\nThe self–nonself gap %s contract under DMT (average flattening = %.5f, p = %.4g).\n%sThis %s consistent with the G1 gradient finding (Step 12 p = %.4g).\n",
  if (lrt_sig) "FOUND" else "did NOT find",
  p_lrt,
  if (avg_flat_b < 0) "DOES" else "does NOT",
  avg_flat_b, avg_flat_p,
  if (meet_any_sig)
    sprintf("The self–nonself gap in the verum post-session condition IS significantly different from zero for at least one layer (see 7d). ") else "",
  if ((g1_p < 0.05) == (lrt_p_display < 0.05)) "IS" else "is NOT",
  g1_p
))

cat("\n", SEP, "\nALL PARTS COMPLETE\n", SEP, "\n", sep = "")
