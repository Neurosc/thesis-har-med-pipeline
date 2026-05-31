# 06b_figures_sphere.R
# Keskin-style figures for sphere-based analysis
#
# Figure 1: 2×2 raincloud panels per category (ggdist)
# Figure 2: delta plot (post−pre per drug group)
# Figure 3: retreat effect (placebo pre vs post)
# Figure 4: baseline balance
#
# Run from repo root:
#   Rscript 04_statistics/scripts/06b_figures_sphere.R

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
                            "sphere_lmm_model_ready.csv")
OUT_FIG25_PNG  <- file.path(REPO_ROOT, "04_statistics", "figures",
                             "fig_sphere_raincloud.png")

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n06b_figures_sphere.R\n", SEP, "\n\n", sep = "")

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

# Factor relevels for LMMs (must match reference coding in 05b_per_category_sphere.R)
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
    theme_minimal(base_size = 11, base_family = "serif") +
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
    title = "Drug × Session effect per region category (sphere self ROIs)",
    theme = theme(
      plot.title = element_text(face = "plain", hjust = 0.5, size = 12,
                                family = "serif")
    )
  )

ggsave(OUT_FIG25_PNG, fig25_gg, width = 10, height = 8, dpi = 300, bg = "white")
cat(sprintf("Saved PNG: %s\n", OUT_FIG25_PNG))


cat("\n", SEP, "\nFIGURE 3 — Delta plot (fig27)\n", SEP, "\n", sep = "")

OUT_FIG27_PNG <- file.path(REPO_ROOT, "04_statistics", "figures",
                            "fig_sphere_delta_plot.png")

# Load drug × session p-values per category
pval_csv <- file.path(REPO_ROOT, "04_statistics", "results",
                       "sphere_per_category_drug_session.csv")
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
    theme_minimal(base_size = 11, base_family = "serif") +
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
      plot.title = element_text(face = "plain", hjust = 0.5, size = 12,
                                family = "serif")
    )
  )

ggsave(OUT_FIG27_PNG, fig27_gg, width = 10, height = 8, dpi = 300, bg = "white")
cat(sprintf("Saved PNG: %s\n", OUT_FIG27_PNG))

cat("\n", SEP, "\nFIGURE 4 — Retreat effect plot (fig28)\n", SEP, "\n", sep = "")

OUT_FIG28_PNG <- file.path(REPO_ROOT, "04_statistics", "figures",
                            "fig_sphere_retreat_effect.png")

# Subset to placebo group
df_plac <- df_raw[df_raw$group == "placebo", ]

# Subject-level means for placebo only (reuse session column from subj_means)
plac_means <- subj_means[subj_means$group == "placebo", ]

# ── Per-category LMMs (placebo only): auc ~ session + RE ──────────────────────
retreat_rows <- list()
for (cat in CAT_ORDER) {
  sub <- df_plac[df_plac$category == cat, ]
  sub$roi_uid <- droplevels(sub$roi_uid)

  m <- tryCatch(
    suppressWarnings(
      lmer(auc ~ session + (1 | subject) + (1 | roi_uid),
           data = sub, REML = TRUE, control = ctrl)
    ),
    error = function(e) {
      cat(sprintf("  [%s] LMM error: %s\n", cat, e$message)); NULL
    }
  )
  if (is.null(m)) next

  fe    <- coef(summary(m))
  b     <- fe["sessionses-02", "Estimate"]
  se    <- fe["sessionses-02", "Std. Error"]
  t_val <- fe["sessionses-02", "t value"]
  p_val <- fe["sessionses-02", "Pr(>|t|)"]

  retreat_rows[[cat]] <- data.frame(
    category = cat, beta = b, SE = se, t_val = t_val, p_val = p_val,
    stringsAsFactors = FALSE
  )
}
retreat_tbl <- do.call(rbind, retreat_rows)
rownames(retreat_tbl) <- NULL

cat("=== Figure 4 source: session β (placebo only) per category ===\n")
cat(sprintf("  %-22s %10s %8s %7s %8s\n", "Category", "β", "SE", "t", "p"))
cat(strrep("-", 60), "\n")
for (i in seq_len(nrow(retreat_tbl))) {
  r <- retreat_tbl[i, ]
  ast <- if (!is.na(r$p_val) && r$p_val < 0.001) " ***"  else
         if (!is.na(r$p_val) && r$p_val < 0.01)  " **"   else
         if (!is.na(r$p_val) && r$p_val < 0.05)  " *"    else ""
  cat(sprintf("  %-22s %10.6f %8.6f %7.3f %8.4g%s\n",
              r$category, r$beta, r$SE, r$t_val, r$p_val, ast))
}
cat("\n")

# ── Panel function ─────────────────────────────────────────────────────────────
SESSION_COLORS <- c("ses-01" = "#CD5C5C", "ses-02" = "#E8963E")
SESSION_LABELS <- c("Pre", "Post")

make_retreat_panel <- function(cat_label) {
  d    <- plac_means[plac_means$category == cat_label, ]
  rrow <- retreat_tbl[retreat_tbl$category == cat_label, ]
  p_val <- if (nrow(rrow) > 0) rrow$p_val[1] else NA

  d$session_f <- factor(as.character(d$session), levels = c("ses-01", "ses-02"))

  set.seed(42)
  d <- d %>%
    mutate(
      x_cond   = as.numeric(session_f) * 2,
      x_jitter = x_cond - 0.40 + runif(n(), -0.10, 0.10)
    )

  p <- ggplot(d, aes(x = x_cond, y = mean_auc,
                     fill = session_f, color = session_f,
                     group = session_f)) +
    geom_point(
      aes(x = x_jitter, color = session_f),
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
    scale_fill_manual(values = SESSION_COLORS, guide = "none") +
    scale_color_manual(values = SESSION_COLORS, guide = "none") +
    scale_x_continuous(
      breaks = (1:2) * 2,
      labels = SESSION_LABELS,
      expand = expansion(add = c(0.65, 1.00))
    ) +
    labs(title = cat_label, y = "Mean AUC (s)", x = NULL) +
    theme_minimal(base_size = 11, base_family = "serif") +
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

  # Bracket always shown; asterisk label depends on p-value
  if (!is.na(p_val)) {
    y_max  <- max(d$mean_auc, na.rm = TRUE)
    y_span <- diff(range(d$mean_auc, na.rm = TRUE))
    y_br   <- y_max + y_span * 0.06
    y_tick <- y_br  - y_span * 0.025
    ast    <- if (p_val < 0.001) "***" else
              if (p_val < 0.01)  "**"  else
              if (p_val < 0.05)  "*"   else "n.s."
    p_str  <- if (p_val < 0.05) sprintf("%s p=%.3f", ast, p_val) else
              sprintf("n.s. p=%.3f", p_val)

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

retreat_panels <- lapply(CAT_ORDER, make_retreat_panel)

fig28_gg <- (retreat_panels[[1]] | retreat_panels[[2]]) /
            (retreat_panels[[3]] | retreat_panels[[4]]) +
  plot_annotation(
    title = "Meditation retreat effect — sphere (placebo group, n=18)",
    theme = theme(
      plot.title = element_text(face = "plain", hjust = 0.5, size = 12,
                                family = "serif")
    )
  )

ggsave(OUT_FIG28_PNG, fig28_gg, width = 10, height = 8, dpi = 300, bg = "white")
cat(sprintf("Saved PNG: %s\n", OUT_FIG28_PNG))

# ── Shared helpers for baseline figures ───────────────────────────────────────
cohens_d <- function(x, y) {
  sp <- sqrt(((length(x) - 1) * var(x) + (length(y) - 1) * var(y)) /
               (length(x) + length(y) - 2))
  if (sp == 0) return(0)
  (mean(x) - mean(y)) / sp
}
d_label <- function(d) {
  a <- abs(d)
  if (a < 0.2) "negligible" else if (a < 0.5) "small" else
  if (a < 0.8) "medium"     else "large"
}

BASE_COLORS <- c(placebo = "#CD5C5C", verum = "#4682B4")
BASE_LABELS <- c("Placebo", "Verum")

# Derive ses-01 subject-level means from df_raw (already loaded)
ses01_df <- df_raw[as.character(df_raw$session) == "ses-01", ]

make_base_panel <- function(d_in, panel_title) {
  d <- data.frame(group = factor(as.character(d_in$group),
                                  levels = c("placebo", "verum")),
                  value = d_in$value,
                  stringsAsFactors = FALSE)

  plac <- d$value[d$group == "placebo"]
  verm <- d$value[d$group == "verum"]
  t_r  <- t.test(plac, verm, var.equal = FALSE)
  mw_r <- wilcox.test(plac, verm, exact = FALSE)
  dv   <- cohens_d(plac, verm)
  stats_txt <- sprintf(
    "t=%.3f (df=%.1f), p=%.3f\nU=%.0f, p=%.3f\nd=%+.3f (%s)",
    unname(t_r$statistic), unname(t_r$parameter), t_r$p.value,
    unname(mw_r$statistic), mw_r$p.value, dv, d_label(dv)
  )

  set.seed(42)
  d <- d %>%
    mutate(x_cond   = as.numeric(group) * 2,
           x_jitter = x_cond - 0.40 + runif(n(), -0.10, 0.10))

  ggplot(d, aes(x = x_cond, y = value,
                fill = group, color = group, group = group)) +
    geom_point(aes(x = x_jitter, color = group),
               size = 1.5, alpha = 0.60, shape = 16) +
    geom_boxplot(fill = NA, width = 0.90, outlier.shape = NA,
                 linewidth = 0.70, position = position_nudge(x = -0.4)) +
    stat_halfeye(adjust = 0.7, width = 0.55, justification = -0.38,
                 .width = 0, point_colour = NA, slab_alpha = 0.65) +
    annotate("label", x = Inf, y = -Inf, label = stats_txt,
             hjust = 1.05, vjust = -0.08, size = 2.8, family = "serif",
             fill = "white", color = "gray30",
             label.padding = unit(0.15, "cm")) +
    scale_fill_manual(values = BASE_COLORS, guide = "none") +
    scale_color_manual(values = BASE_COLORS, guide = "none") +
    scale_x_continuous(breaks = (1:2) * 2, labels = BASE_LABELS,
                       expand = expansion(add = c(0.65, 1.00))) +
    labs(title = panel_title, y = "Mean AUC (s)", x = NULL) +
    theme_minimal(base_size = 11, base_family = "serif") +
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
}

# ── Figure 5A — Pooled baseline (fig29a) ──────────────────────────────────────
cat("\n", SEP, "\nFIGURE 5A — Pooled baseline balance (fig29a)\n",
    SEP, "\n", sep = "")

OUT_FIG29A_PNG <- file.path(REPO_ROOT, "04_statistics", "figures",
                             "fig_sphere_baseline_balance_pooled.png")

pooled_means <- ses01_df %>%
  group_by(subject, group) %>%
  summarise(value = mean(auc, na.rm = TRUE), .groups = "drop")

cat("=== Pooled baseline: group means ===\n")
cat(sprintf("  n_placebo=%d  n_verum=%d\n",
            sum(pooled_means$group == "placebo"),
            sum(pooled_means$group == "verum")))

fig29a_gg <- make_base_panel(pooled_means,
                              "Baseline balance — all regions pooled (ses-01)") +
  theme(plot.title = element_text(face = "plain", hjust = 0.5, size = 12,
                                   family = "serif"))

ggsave(OUT_FIG29A_PNG, fig29a_gg, width = 5, height = 6,
       dpi = 300, bg = "white")
cat(sprintf("Saved PNG: %s\n", OUT_FIG29A_PNG))

# ── Figure 5B — Per-category baseline (fig29b) ────────────────────────────────
cat("\n", SEP, "\nFIGURE 5B — Per-category baseline balance (fig29b)\n",
    SEP, "\n", sep = "")

OUT_FIG29B_PNG <- file.path(REPO_ROOT, "04_statistics", "figures",
                             "fig_sphere_baseline_balance_per_category.png")

cat("=== Per-category baseline ===\n")
base_panels <- lapply(CAT_ORDER, function(cat) {
  d_cat <- ses01_df[ses01_df$category == cat, ] %>%
    group_by(subject, group) %>%
    summarise(value = mean(auc, na.rm = TRUE), .groups = "drop")
  cat(sprintf("  [%s] n_placebo=%d  n_verum=%d\n", cat,
              sum(d_cat$group == "placebo"), sum(d_cat$group == "verum")))
  make_base_panel(d_cat, cat)
})

fig29b_gg <- (base_panels[[1]] | base_panels[[2]]) /
             (base_panels[[3]] | base_panels[[4]]) +
  plot_annotation(
    title = "Baseline balance by region category (ses-01)",
    theme = theme(plot.title = element_text(face = "plain", hjust = 0.5,
                                            size = 12, family = "serif"))
  )

ggsave(OUT_FIG29B_PNG, fig29b_gg, width = 10, height = 8,
       dpi = 300, bg = "white")
cat(sprintf("Saved PNG: %s\n", OUT_FIG29B_PNG))

cat("\n", SEP, "\nFIGURE 6 — Paired pre→post plot (significant categories)\n",
    SEP, "\n", sep = "")

OUT_FIG_PAIRED <- file.path(REPO_ROOT, "04_statistics", "figures",
                             "fig_glasser_paired_prepost.png")

# x positions: Placebo pair on left, Verum pair on right, gap between
XPOS_PAIRED <- c("Placebo Pre"  = 1.0, "Placebo Post" = 2.0,
                 "Verum Pre"    = 3.2, "Verum Post"   = 4.2)

fmt_p_paired <- function(p) {
  if (is.na(p))     return("n.s.")
  if (p < 0.001)    return("p<0.001")
  sprintf("p=%.3f", p)
}

make_paired_panel <- function(cat_label) {
  d <- subj_means[subj_means$category == cat_label, ]
  d$x_pos <- XPOS_PAIRED[as.character(d$condition)]

  # Paired line segments: one per subject × group
  make_lines <- function(grp) {
    pre  <- d[d$group == grp & d$session == "ses-01",
              c("subject", "mean_auc", "x_pos")]
    post <- d[d$group == grp & d$session == "ses-02",
              c("subject", "mean_auc", "x_pos")]
    merge(pre, post, by = "subject", suffixes = c("_pre", "_post"))
  }
  plac_segs <- make_lines("placebo")
  verm_segs <- make_lines("verum")

  # Paired t-tests (Post − Pre, matched by subject)
  paired_t <- function(grp) {
    pre_d  <- d[d$group == grp & d$session == "ses-01", c("subject","mean_auc")]
    post_d <- d[d$group == grp & d$session == "ses-02", c("subject","mean_auc")]
    both   <- merge(pre_d, post_d, by = "subject", suffixes = c("_pre","_post"))
    if (nrow(both) < 3) return(list(p.value = NA))
    t.test(both$mean_auc_post, both$mean_auc_pre, paired = TRUE)
  }
  t_plac <- paired_t("placebo")
  t_verm <- paired_t("verum")

  # Drug × session p-value for this category
  ds_p <- {
    r <- pval_tbl[!is.na(pval_tbl$cat_display) & pval_tbl$cat_display == cat_label, ]
    if (nrow(r) > 0) r$drug_session_p[1] else NA
  }

  y_max  <- max(d$x_pos, d$mean_auc, na.rm = TRUE)   # just to trigger; recalc below
  y_max  <- max(d$mean_auc, na.rm = TRUE)
  y_span <- diff(range(d$mean_auc, na.rm = TRUE))

  p <- ggplot() +
    # 1. Paired lines
    geom_segment(data = plac_segs,
                 aes(x = x_pos_pre, xend = x_pos_post,
                     y = mean_auc_pre, yend = mean_auc_post),
                 color = "#CD5C5C", alpha = 0.30, linewidth = 0.5) +
    geom_segment(data = verm_segs,
                 aes(x = x_pos_pre, xend = x_pos_post,
                     y = mean_auc_pre, yend = mean_auc_post),
                 color = "#4682B4", alpha = 0.30, linewidth = 0.5) +
    # 2. Individual dots
    geom_point(data = d[d$group == "placebo", ],
               aes(x = x_pos, y = mean_auc),
               color = "#CD5C5C", alpha = 0.60, size = 2, shape = 16) +
    geom_point(data = d[d$group == "verum", ],
               aes(x = x_pos, y = mean_auc),
               color = "#4682B4", alpha = 0.60, size = 2, shape = 16) +
    # 3. Boxplots (transparent fill, colored outline)
    geom_boxplot(data = d[d$group == "placebo", ],
                 aes(x = x_pos, y = mean_auc, group = condition),
                 fill = NA, color = "#CD5C5C",
                 width = 0.22, outlier.shape = NA, linewidth = 0.60) +
    geom_boxplot(data = d[d$group == "verum", ],
                 aes(x = x_pos, y = mean_auc, group = condition),
                 fill = NA, color = "#4682B4",
                 width = 0.22, outlier.shape = NA, linewidth = 0.60) +
    scale_x_continuous(
      breaks = unname(XPOS_PAIRED),
      labels = c("Placebo\nPre", "Placebo\nPost", "Verum\nPre", "Verum\nPost"),
      expand = expansion(add = c(0.5, 0.5))
    ) +
    labs(title = cat_label, y = "Mean AUC (s)", x = NULL) +
    theme_minimal(base_size = 11, base_family = "serif") +
    theme(
      panel.background   = element_rect(fill = "white", color = NA),
      plot.background    = element_rect(fill = "white", color = NA),
      panel.grid.major.x = element_blank(),
      panel.grid.minor   = element_blank(),
      panel.grid.major.y = element_line(color = "#e8e8e8", linewidth = 0.4),
      plot.title   = element_text(face = "bold", hjust = 0.5, size = 12,
                                  family = "serif"),
      axis.text.x  = element_text(size = 9, lineheight = 0.85),
      axis.title.y = element_text(size = 10),
      legend.position = "none"
    )

  # 4. Significance brackets
  y_br1 <- y_max + y_span * 0.05
  y_t1  <- y_br1 - y_span * 0.02
  y_br2 <- y_br1 + y_span * 0.05    # verum bracket slightly higher
  y_t2  <- y_br2 - y_span * 0.02

  # Placebo Pre → Post
  p <- p +
    annotate("segment", x = 1.0, xend = 2.0, y = y_br1, yend = y_br1,
             color = "#CD5C5C", linewidth = 0.5) +
    annotate("segment", x = 1.0, xend = 1.0, y = y_br1, yend = y_t1,
             color = "#CD5C5C", linewidth = 0.5) +
    annotate("segment", x = 2.0, xend = 2.0, y = y_br1, yend = y_t1,
             color = "#CD5C5C", linewidth = 0.5) +
    annotate("text", x = 1.5, y = y_br1 + y_span * 0.02,
             label = fmt_p_paired(t_plac$p.value),
             size = 3, hjust = 0.5, color = "#CD5C5C")

  # Verum Pre → Post
  p <- p +
    annotate("segment", x = 3.2, xend = 4.2, y = y_br2, yend = y_br2,
             color = "#4682B4", linewidth = 0.5) +
    annotate("segment", x = 3.2, xend = 3.2, y = y_br2, yend = y_t2,
             color = "#4682B4", linewidth = 0.5) +
    annotate("segment", x = 4.2, xend = 4.2, y = y_br2, yend = y_t2,
             color = "#4682B4", linewidth = 0.5) +
    annotate("text", x = 3.7, y = y_br2 + y_span * 0.02,
             label = fmt_p_paired(t_verm$p.value),
             size = 3, hjust = 0.5, color = "#4682B4")

  # Drug × session bracket: Placebo Post (x=2) → Verum Post (x=4.2)
  y_br3 <- max(y_br1, y_br2) + y_span * 0.08
  y_t3  <- y_br3 - y_span * 0.02
  p <- p +
    annotate("segment", x = 2.0, xend = 4.2, y = y_br3, yend = y_br3,
             color = "black", linewidth = 0.5) +
    annotate("segment", x = 2.0, xend = 2.0, y = y_br3, yend = y_t3,
             color = "black", linewidth = 0.5) +
    annotate("segment", x = 4.2, xend = 4.2, y = y_br3, yend = y_t3,
             color = "black", linewidth = 0.5) +
    annotate("text", x = 3.1, y = y_br3 + y_span * 0.02,
             label = fmt_p_paired(ds_p),
             size = 3, hjust = 0.5, color = "black") +
    coord_cartesian(ylim = c(NA, y_br3 + y_span * 0.08))

  p
}

PAIRED_CATS <- c("Exteroceptive Self", "Sensory-Motor")
paired_panels <- lapply(PAIRED_CATS, make_paired_panel)

fig_paired <- (paired_panels[[1]] | paired_panels[[2]]) +
  plot_annotation(
    title = "Pre → Post AUC change by drug group (significant categories)",
    theme = theme(plot.title = element_text(face = "plain", hjust = 0.5,
                                            size = 12, family = "serif"))
  )

ggsave(OUT_FIG_PAIRED, fig_paired, width = 10, height = 5,
       dpi = 300, bg = "white")
cat(sprintf("Saved PNG: %s\n", OUT_FIG_PAIRED))

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
