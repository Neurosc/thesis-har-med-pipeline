# 06_figures_glasser.R — Internal–external ACW difference figure
#
# Produces fig_internal_external_difference.png: two Keskin-style raincloud
# panels (IE_diff_1 = Interoception−Exteroception, IE_diff_2 = Interoception−
# Sensory-Motor) across 4 conditions (Placebo Pre/Post, Verum Pre/Post).
#
# Run from repo root:
#   Rscript 04_statistics/scripts/06_figures_glasser.R

required_pkgs <- c("ggplot2", "ggdist", "patchwork", "dplyr", "tidyr")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
}
suppressPackageStartupMessages({
  library(ggplot2); library(ggdist); library(patchwork)
  library(dplyr);   library(tidyr)
})

args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
  REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
} else {
  REPO_ROOT <- normalizePath(".")
}

TAG       <- "parcels_NoGSR"
MODEL_CSV <- file.path(REPO_ROOT, "04_statistics", "results", TAG, "tables", "model_ready.csv")
FIGS_DIR  <- file.path(REPO_ROOT, "04_statistics", "figures")
OUT_PNG   <- file.path(FIGS_DIR, "fig_internal_external_difference.png")

dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)
if (!file.exists(MODEL_CSV)) stop("Missing input: ", MODEL_CSV)

# ── Load and prepare difference scores ───────────────────────────────────────
df <- read.csv(MODEL_CSV, stringsAsFactors = FALSE)
df$group   <- factor(df$group,   levels = c("placebo", "verum"))
df$session <- factor(df$session, levels = c("ses-01", "ses-02"))

COND_LEVELS <- c("Placebo Pre", "Verum Pre", "Placebo Post", "Verum Post")
COND_COLORS <- c("Placebo Pre"  = "#CD5C5C", "Verum Pre"   = "#4682B4",
                 "Placebo Post" = "#E8963E", "Verum Post"  = "#2E8B8B")

external_means <- df %>%
  filter(self_layer %in% c("Exteroception", "nonself")) %>%
  group_by(subject, session, group) %>%
  summarise(External = mean(auc, na.rm = TRUE), .groups = "drop")

diff_df <- df %>%
  filter(self_layer %in% c("Interoception", "Exteroception", "nonself")) %>%
  group_by(subject, session, group, self_layer) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = self_layer, values_from = mean_auc) %>%
  left_join(external_means, by = c("subject", "session", "group")) %>%
  mutate(
    IE_diff_1 = Interoception - Exteroception,
    IE_diff_2 = Interoception - nonself,
    IE_diff_3 = Interoception - External,
    condition = factor(
      paste0(ifelse(group == "placebo", "Placebo", "Verum"), " ",
             ifelse(session == "ses-01", "Pre", "Post")),
      levels = COND_LEVELS)
  )

# ── Drug × session interaction p-values (for bracket annotations) ─────────────
inter_p <- function(diff_col) {
  pre  <- diff_df[diff_df$session == "ses-01", c("subject", "group", diff_col)]
  post <- diff_df[diff_df$session == "ses-02", c("subject", diff_col)]
  names(pre)[3] <- "val_pre"; names(post)[2] <- "val_post"
  both       <- merge(pre, post, by = "subject")
  both$delta <- both$val_post - both$val_pre
  plac <- both$delta[both$group == "placebo"]
  verm <- both$delta[both$group == "verum"]
  tryCatch(t.test(verm, plac, var.equal = FALSE)$p.value, error = function(e) NA_real_)
}

p1 <- inter_p("IE_diff_1")
p2 <- inter_p("IE_diff_2")
p3 <- inter_p("IE_diff_3")

cat(sprintf("Drug × session p  —  IE_diff_1: %.4f  |  IE_diff_2: %.4f  |  IE_diff_3: %.4f\n",
            p1, p2, p3))

# ── Raincloud panel builder ───────────────────────────────────────────────────
make_panel <- function(diff_col, panel_title, y_lab, inter_pval) {
  d <- diff_df[, c("subject", "condition", diff_col)]
  names(d)[3] <- "y"

  set.seed(42)
  d <- d %>%
    group_by(condition) %>%
    mutate(
      x_cond   = as.numeric(condition) * 2,
      x_jitter = x_cond - 0.40 + runif(n(), -0.20, 0.20)
    ) %>%
    ungroup()

  p <- ggplot(d, aes(x = x_cond, y = y,
                     fill = condition, color = condition, group = condition)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50",
               linewidth = 0.5) +
    geom_point(aes(x = x_jitter, color = condition),
               size = 1.5, alpha = 0.60, shape = 16) +
    geom_boxplot(fill = NA, width = 0.9, outlier.shape = NA, linewidth = 0.70,
                 position = position_nudge(x = -0.4)) +
    stat_halfeye(adjust = 0.7, width = 0.55, justification = -0.38,
                 .width = 0, point_colour = NA, slab_alpha = 0.65) +
    scale_fill_manual(values  = COND_COLORS, guide = "none") +
    scale_color_manual(values = COND_COLORS, guide = "none") +
    scale_x_continuous(breaks = (1:4) * 2,
                       labels = c("Placebo\nPre", "Verum\nPre",
                                  "Placebo\nPost", "Verum\nPost"),
                       expand = expansion(add = c(0.65, 1.00))) +
    labs(title = panel_title, y = y_lab, x = NULL) +
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

  # Drug × session interaction bracket (only when p < 0.05)
  if (!is.na(inter_pval) && inter_pval < 0.05) {
    y_vals <- d$y[is.finite(d$y)]
    y_max  <- max(y_vals);  y_span <- diff(range(y_vals))
    y_br   <- y_max + y_span * 0.10
    y_tick <- y_br  - y_span * 0.025
    ast    <- if (inter_pval < 0.01) "**" else "*"
    p_str  <- sprintf("%s p=%.3f (group×session)", ast, inter_pval)
    p <- p +
      annotate("segment", x = 1.6, xend = 7.6,
               y = y_br, yend = y_br, color = "black", linewidth = 0.5) +
      annotate("segment", x = 1.6, xend = 1.6,
               y = y_br, yend = y_tick, color = "black", linewidth = 0.5) +
      annotate("segment", x = 7.6, xend = 7.6,
               y = y_br, yend = y_tick, color = "black", linewidth = 0.5) +
      annotate("text", x = 4.6, y = y_br + y_span * 0.04,
               label = p_str, size = 3.2, hjust = 0.5, color = "black",
               family = "serif") +
      coord_cartesian(ylim = c(NA, y_br + y_span * 0.12))
  }
  p
}

# ── Build and save figure ─────────────────────────────────────────────────────
panel1 <- make_panel("IE_diff_1",
                     "Interoception − Exteroception",
                     "ACW difference (s)", p1)
panel2 <- make_panel("IE_diff_2",
                     "Interoception − Sensory-Motor",
                     "ACW difference (s)", p2)
panel3 <- make_panel("IE_diff_3",
                     "Interoception − (Extero + Sensory-Motor)",
                     "ACW difference (s)", p3)

fig <- (panel1 | panel2 | panel3) +
  plot_annotation(
    title = "Internal–external ACW differences by drug group and session",
    theme = theme(
      plot.title = element_text(face = "plain", hjust = 0.5,
                                size = 13, family = "serif")
    )
  )

ggsave(OUT_PNG, fig, width = 12, height = 5, dpi = 300, bg = "white")
cat(sprintf("Saved PNG: %s\n", OUT_PNG))
cat("Done.\n")
