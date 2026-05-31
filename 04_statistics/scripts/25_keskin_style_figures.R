# 25_keskin_style_figures.R
# Step 17 (redesigned): Keskin-style raincloud + fixed interaction dot plot
#
# Figure 1 (fig25): 2×2 raincloud panels per category (ggdist)
# Figure 2 (fig26): drug × session interaction dot + CI (fixed from bar chart)
#
# Run from repo root:
#   Rscript 04_statistics/scripts/25_keskin_style_figures.R

# ── Packages ───────────────────────────────────────────────────────────────────
required_pkgs <- c("ggplot2", "ggdist", "patchwork", "dplyr",
                   "lme4", "lmerTest", "emmeans", "plotly", "htmlwidgets")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
}

suppressPackageStartupMessages({
  library(ggplot2); library(ggdist); library(patchwork)
  library(dplyr)
  library(lme4); library(lmerTest); library(emmeans)
  library(plotly); library(htmlwidgets)
})

# ── Paths ──────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
  REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
} else {
  REPO_ROOT <- normalizePath(".")
}

DATA_CSV      <- file.path(REPO_ROOT, "04_statistics", "results",
                            "glasser_self_nonself_model_ready.csv")
OUT_FIG25_PNG  <- file.path(REPO_ROOT, "04_statistics", "figures",
                             "fig25_raincloud_per_category.png")
OUT_FIG25_HTML <- file.path(REPO_ROOT, "04_statistics", "figures",
                             "fig25_raincloud_per_category.html")
OUT_FIG26_PNG  <- file.path(REPO_ROOT, "04_statistics", "figures",
                             "fig26_drug_session_interaction_fixed.png")
OUT_FIG26_HTML <- file.path(REPO_ROOT, "04_statistics", "figures",
                             "fig26_drug_session_interaction_fixed.html")

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n25_keskin_style_figures.R — Step 17 (Redesigned)\n", SEP, "\n\n", sep = "")

# ── Load & prepare data ────────────────────────────────────────────────────────
df_raw <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n\n", nrow(df_raw)))

# Display labels for each self_layer value
CAT_MAP <- c(
  Interoception = "Interoceptive Self",
  Exteroception = "Exteroceptive Self",
  Cognition     = "Mental Self",
  nonself       = "Sensory-Motor"
)

# Panel order for figures
CAT_ORDER <- c("Interoceptive Self", "Exteroceptive Self",
               "Mental Self", "Sensory-Motor")

# Condition display labels and order (pre grouped, post grouped)
COND_LEVELS <- c("Placebo Pre", "Verum Pre", "Placebo Post", "Verum Post")
COND_LABELS <- c("Placebo\nPre", "Verum\nPre", "Placebo\nPost", "Verum\nPost")

# Colors for four conditions (Keskin style: red, blue, orange, teal)
COND_COLORS <- c(
  "Placebo Pre"  = "#CD5C5C",
  "Verum Pre"    = "#4682B4",
  "Placebo Post" = "#E8963E",
  "Verum Post"   = "#2E8B8B"
)

# Category colors for Figure 2
CAT_COLORS <- c(
  "Interoceptive Self" = "#1f77b4",
  "Exteroceptive Self" = "#ff7f0e",
  "Mental Self"        = "#2ca02c",
  "Sensory-Motor"      = "#888780"
)

# Factor relevels for LMMs (must match reference coding in script 26)
df_raw$group      <- relevel(factor(df_raw$group),      ref = "placebo")
df_raw$session    <- relevel(factor(df_raw$session),    ref = "ses-01")
df_raw$subject    <- factor(df_raw$subject)
df_raw$roi_uid    <- factor(df_raw$roi_uid)
df_raw$self_layer <- relevel(factor(df_raw$self_layer), ref = "nonself")

# Add display category and condition columns
df_raw$category  <- CAT_MAP[as.character(df_raw$self_layer)]
df_raw$condition <- paste0(
  ifelse(df_raw$group == "placebo", "Placebo", "Verum"), " ",
  ifelse(df_raw$session == "ses-01", "Pre", "Post")
)
df_raw$condition <- factor(df_raw$condition, levels = COND_LEVELS)

# Subject-level means per (category × condition)
subj_means <- df_raw %>%
  group_by(category, subject, group, session, condition) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop")
subj_means$condition <- factor(subj_means$condition, levels = COND_LEVELS)

cat("=== Figure 1 source: subject-level means per category × condition ===\n")
summary_tbl <- subj_means %>%
  group_by(category, condition) %>%
  summarise(n = n(), mean = mean(mean_auc), sd = sd(mean_auc), .groups = "drop")
cat(sprintf("  %-22s %-14s %4s %8s %8s\n", "Category", "Condition", "n", "mean", "sd"))
cat(strrep("-", 62), "\n")
for (i in seq_len(nrow(summary_tbl))) {
  r <- summary_tbl[i, ]
  cat(sprintf("  %-22s %-14s %4d %8.4f %8.4f\n",
              r$category, as.character(r$condition), r$n, r$mean, r$sd))
}
cat("\n")

# ── Per-category LMMs ──────────────────────────────────────────────────────────
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5),
                    check.conv.grad = .makeCC("warning", tol = 0.002))

int_rows <- list()
for (cat in CAT_ORDER) {
  sub <- df_raw[df_raw$category == cat, ]
  sub$roi_uid <- droplevels(sub$roi_uid)

  m <- tryCatch(
    suppressWarnings(
      lmer(auc ~ group * session + (1 | subject) + (1 | roi_uid),
           data = sub, REML = TRUE, control = ctrl)
    ),
    error = function(e) {
      cat(sprintf("  [%s] LMM error: %s\n", cat, e$message)); NULL
    }
  )
  if (is.null(m)) next

  fe    <- coef(summary(m))
  b     <- fe["groupverum:sessionses-02", "Estimate"]
  se    <- fe["groupverum:sessionses-02", "Std. Error"]
  t_val <- fe["groupverum:sessionses-02", "t value"]
  p_val <- fe["groupverum:sessionses-02", "Pr(>|t|)"]

  int_rows[[cat]] <- data.frame(
    category = cat,
    beta     = b,
    SE       = se,
    t_val    = t_val,
    p_val    = p_val,
    ci_lo    = b - 1.96 * se,
    ci_hi    = b + 1.96 * se,
    stringsAsFactors = FALSE
  )
}
int_tbl <- do.call(rbind, int_rows)
rownames(int_tbl) <- NULL

cat("=== Figure 2 source: Drug × Session interaction β per category ===\n")
cat(sprintf("  %-22s %10s %8s %7s %10s %10s %8s\n",
            "Category", "β", "SE", "t", "CI_lo", "CI_hi", "p"))
cat(strrep("-", 78), "\n")
for (i in seq_len(nrow(int_tbl))) {
  r <- int_tbl[i, ]
  cat(sprintf("  %-22s %10.6f %8.6f %7.3f %10.6f %10.6f %8.4g%s\n",
              r$category, r$beta, r$SE, r$t_val, r$ci_lo, r$ci_hi, r$p_val,
              if (!is.na(r$p_val) && r$p_val < 0.05) " *" else ""))
}
cat("\n")

# ── Exteroception vs Sensory-Motor formal contrast (full model) ────────────────
cat(SEP, "\nExteroceptive Self vs Sensory-Motor formal contrast\n", SEP, "\n", sep = "")

emm_options(lmerTest.limit = nrow(df_raw), pbkrtest.limit = nrow(df_raw))

m_full <- tryCatch(
  suppressWarnings(
    lmer(auc ~ group * session * self_layer +
           (1 + session | subject) + (1 | roi_uid),
         data = df_raw, REML = TRUE, control = ctrl)
  ),
  error = function(e) {
    cat(sprintf("  Full RE failed: %s\n  Falling back to (1|subject)...\n", e$message))
    suppressWarnings(
      lmer(auc ~ group * session * self_layer + (1 | subject) + (1 | roi_uid),
           data = df_raw, REML = TRUE, control = ctrl)
    )
  }
)
if (!is.null(m_full) && isSingular(m_full)) {
  cat("  Singular — refitting with (1|subject)+(1|roi_uid)...\n")
  m_full <- suppressWarnings(
    lmer(auc ~ group * session * self_layer + (1 | subject) + (1 | roi_uid),
         data = df_raw, REML = TRUE, control = ctrl))
}
conv <- m_full@optinfo$conv$lme4$messages
cat(if (!is.null(conv)) sprintf("  Convergence: %s\n", paste(conv, collapse = "; "))
    else "  Convergence: OK\n")

# self_layer factor levels after relevel(ref="nonself"):
#   1=nonself, 2=Cognition, 3=Exteroception, 4=Interoception
emm_by   <- emmeans(m_full, ~ self_layer | group + session)
gaps_ext <- contrast(emm_by,
  list("Exteroception - nonself" = c(-1, 0, 1, 0)), adjust = "none")
gap_ses  <- contrast(gaps_ext, "consec",
  by = c("contrast", "group"), adjust = "none")
flat_ext <- contrast(gap_ses, "consec", by = "contrast", adjust = "none")
flat_df  <- as.data.frame(summary(flat_ext, infer = TRUE))
if (!"lower.CL" %in% names(flat_df)) {
  flat_df$lower.CL <- flat_df$estimate - 1.96 * flat_df$SE
  flat_df$upper.CL <- flat_df$estimate + 1.96 * flat_df$SE
}
ext_p <- flat_df$p.value[1]
ext_b <- flat_df$estimate[1]
cat(sprintf("  Exteroception vs Sensory-Motor: β=%.6f, p=%.4g%s\n\n",
            ext_b, ext_p,
            if (!is.na(ext_p) && ext_p < 0.05) " *" else ""))

# ── Figure 1: Keskin-style raincloud ──────────────────────────────────────────
cat(SEP, "\nFIGURE 1 — Keskin-style raincloud (fig25)\n", SEP, "\n", sep = "")

make_panel <- function(cat_label) {
  d <- subj_means[subj_means$category == cat_label, ]
  irow <- int_tbl[int_tbl$category == cat_label, ]
  p_val <- if (nrow(irow) > 0) irow$p_val[1] else NA

  # x spacing: multiply by 2 so conditions are 2 units apart
  set.seed(42)
  d <- d %>%
    group_by(condition) %>%
    mutate(
      x_cond  = as.numeric(condition) * 2,
      x_jitter = x_cond - 0.40 + runif(n(), -0.2, 0.2)
    ) %>%
    ungroup()

  p <- ggplot(d, aes(x = x_cond, y = mean_auc,
                     fill = condition, color = condition,
                     group = condition)) +
    # 1. Subject dots INSIDE the box, jittered horizontally
    geom_point(
      aes(x = x_jitter, color = condition),
      size = 1.5, alpha = 0.60, shape = 16
    ) +
    # 2. Box plot: transparent fill, condition-colored outline (on top of dots)
    geom_boxplot(
      fill = NA,
      width = 0.9, # size of the boxplot
      outlier.shape = NA,
      linewidth = 0.70,
      position = position_nudge(x = -0.4) # boxplotla violin arasindaki bosluk
    ) +
    # 3. Mean diamond (black) at the mean of each condition
    stat_summary(
      fun = mean, geom = "point",
      shape = 18, size = 2, color = "black",
      position = position_nudge(x = -0.4) # boxplotla violin arasindaki bosluk - diamond
    ) +
    # 4. Half-violin to the RIGHT, with clear gap from box
    stat_halfeye(
      adjust = 0.7, width = 0.55,
      justification = -0.38,
      .width = 0, point_colour = NA,
      slab_alpha = 0.65
    ) +
    scale_fill_manual(values = COND_COLORS, guide = "none") +
    scale_color_manual(values = COND_COLORS, guide = "none") +
    scale_x_continuous(
      breaks = (1:4) * 2,
      labels = COND_LABELS,
      expand = expansion(add = c(0.65, 1.00))
    ) +
    labs(title = cat_label, y = "Mean AUC (s)", x = NULL) +
    theme_minimal(base_size = 11, base_family = "Times New Roman") +
    theme(
      panel.background   = element_rect(fill = "white", color = NA),
      plot.background    = element_rect(fill = "white", color = NA),
      panel.grid.major.y = element_line(color = "#e8e8e8", linewidth = 0.4),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      plot.title   = element_text(face = "plain", hjust = 0.5, size = 12),
      axis.text.x  = element_text(size = 9, lineheight = 0.85),
      
      axis.title.y = element_text(size = 10),
      legend.position = "none"
    )

  p
}

panels <- lapply(CAT_ORDER, make_panel)

fig25_gg <- (panels[[1]] | panels[[2]]) / (panels[[3]] | panels[[4]]) +
  plot_annotation(
    title = "Drug × Session effect per region category (Glasser self ROIs)",
    theme = theme(
      plot.title = element_text(face = "plain", hjust = 0.5, size = 14)
    )
  )

ggsave(OUT_FIG25_PNG, fig25_gg, width = 10, height = 8, dpi = 300, bg = "white")
cat(sprintf("Saved PNG: %s\n", OUT_FIG25_PNG))

# HTML: attempt ggplotly (ggdist slabs may render partially)
tryCatch({
  fig25_pl <- ggplotly(fig25_gg) %>%
    layout(legend = list(orientation = "h", y = -0.08, x = 0.3))
  saveWidget(fig25_pl, OUT_FIG25_HTML, selfcontained = FALSE,
             title = "Raincloud per Category")
  cat(sprintf("Saved HTML: %s\n", OUT_FIG25_HTML))
}, error = function(e) {
  cat(sprintf("  HTML fig25 skipped (ggplotly conversion error): %s\n", e$message))
})


cat("\n", SEP, "\nFIGURE 3 — Delta plot (fig27)\n", SEP, "\n", sep = "")

OUT_FIG27_PNG <- file.path(REPO_ROOT, "04_statistics", "figures",
                            "fig27_delta_plot_per_category.png")

# Load drug × session p-values per category
pval_csv <- file.path(REPO_ROOT, "04_statistics", "results",
                       "per_category_drug_session.csv")
pval_tbl <- read.csv(pval_csv, stringsAsFactors = FALSE)
PVAL_CAT_MAP <- c(
  "Sensory-Motor" = "Sensory-Motor",
  "Interoception" = "Interoceptive Self",
  "Exteroception" = "Exteroceptive Self",
  "Cognition"     = "Mental Self"
)
pval_tbl$cat_display <- PVAL_CAT_MAP[pval_tbl$category]

# Compute delta = post − pre per subject × category
pre_df  <- subj_means[as.character(subj_means$session) == "ses-01",
                       c("category", "subject", "group", "mean_auc")]
post_df <- subj_means[as.character(subj_means$session) == "ses-02",
                       c("category", "subject", "group", "mean_auc")]
names(pre_df)[4]  <- "pre_auc"
names(post_df)[4] <- "post_auc"
delta_df <- merge(pre_df, post_df, by = c("category", "subject", "group"))
delta_df$delta <- delta_df$post_auc - delta_df$pre_auc
delta_df$group <- factor(delta_df$group, levels = c("placebo", "verum"))

cat("=== Figure 3 source: delta (post − pre) per category × group ===\n")
delta_summ <- delta_df %>%
  group_by(category, group) %>%
  summarise(n = n(), mean_d = mean(delta), sd_d = sd(delta), .groups = "drop")
cat(sprintf("  %-22s %-8s %4s %9s %9s\n", "Category", "Group", "n", "mean_Δ", "sd_Δ"))
cat(strrep("-", 58), "\n")
for (i in seq_len(nrow(delta_summ))) {
  r <- delta_summ[i, ]
  cat(sprintf("  %-22s %-8s %4d %9.4f %9.4f\n",
              r$category, as.character(r$group), r$n, r$mean_d, r$sd_d))
}
cat("\n")

DELTA_COLORS <- c(placebo = "#CD5C5C", verum = "#4682B4")
GROUP_LABELS <- c("Placebo", "Verum")

make_delta_panel <- function(cat_label) {
  d    <- delta_df[delta_df$category == cat_label, ]
  prow <- pval_tbl[!is.na(pval_tbl$cat_display) &
                     pval_tbl$cat_display == cat_label, ]
  p_val <- if (nrow(prow) > 0) prow$drug_session_p[1] else NA

  set.seed(42)
  d <- d %>%
    mutate(
      x_cond   = as.numeric(group) * 2,
      x_jitter = x_cond - 0.40 + runif(n(), -0.10, 0.10)
    )

  p <- ggplot(d, aes(x = x_cond, y = delta,
                     fill = group, color = group,
                     group = group)) +
    geom_hline(yintercept = 0, linetype = "dashed",
               color = "gray50", linewidth = 0.5) +
    geom_point(
      aes(x = x_jitter, color = group),
      size = 1.5, alpha = 0.60, shape = 16
    ) +
    geom_boxplot(
      fill = NA, width = 0.90,
      outlier.shape = NA, linewidth = 0.70,
      position = position_nudge(x = -0.4)
    ) +
    stat_halfeye(
      adjust = 0.7, width = 0.55,
      justification = -0.38,
      .width = 0, point_colour = NA,
      slab_alpha = 0.65
    ) +
    scale_fill_manual(values = DELTA_COLORS, guide = "none") +
    scale_color_manual(values = DELTA_COLORS, guide = "none") +
    scale_x_continuous(
      breaks = (1:2) * 2,
      labels = GROUP_LABELS,
      expand = expansion(add = c(0.65, 1.00))
    ) +
    labs(title = cat_label, y = "ΔAUC (post − pre, s)", x = NULL) +
    theme_minimal(base_size = 11, base_family = "Times New Roman") +
    theme(
      panel.background   = element_rect(fill = "white", color = NA),
      plot.background    = element_rect(fill = "white", color = NA),
      panel.grid.major.y = element_line(color = "#e8e8e8", linewidth = 0.4),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      plot.title   = element_text(face = "plain", hjust = 0.5, size = 12),
      axis.text.x  = element_text(size = 9, lineheight = 0.85),
      axis.title.y = element_text(size = 10),
      legend.position = "none"
    )

  if (!is.na(p_val) && p_val < 0.05) {
    y_max  <- max(d$delta, na.rm = TRUE)
    y_span <- diff(range(d$delta, na.rm = TRUE))
    y_br   <- y_max + y_span * 0.06
    y_tick <- y_br  - y_span * 0.025
    ast    <- if (p_val < 0.01) "**" else "*"
    p_str  <- sprintf("%s p=%.3f", ast, p_val)

    p <- p +
      annotate("segment", x = 1.6, xend = 3.6, y = y_br, yend = y_br,
               color = "black", linewidth = 0.5) +
      annotate("segment", x = 1.6, xend = 1.6, y = y_br, yend = y_tick,
               color = "black", linewidth = 0.5) +
      annotate("segment", x = 3.6, xend = 3.6, y = y_br, yend = y_tick,
               color = "black", linewidth = 0.5) +
      annotate("text", x = 2.6, y = y_br + y_span * 0.04,
               label = p_str, size = 3.5, hjust = 0.5, color = "black") +
      coord_cartesian(ylim = c(NA, y_br + y_span * 0.10))
  }
  p
}

delta_panels <- lapply(CAT_ORDER, make_delta_panel)

fig27_gg <- (delta_panels[[1]] | delta_panels[[2]]) /
            (delta_panels[[3]] | delta_panels[[4]]) +
  plot_annotation(
    title = "Post − Pre AUC change per region category (Glasser self ROIs)",
    theme = theme(
      plot.title = element_text(face = "plain", hjust = 0.5, size = 14)
    )
  )

ggsave(OUT_FIG27_PNG, fig27_gg, width = 10, height = 8, dpi = 300, bg = "white")
cat(sprintf("Saved PNG: %s\n", OUT_FIG27_PNG))

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
