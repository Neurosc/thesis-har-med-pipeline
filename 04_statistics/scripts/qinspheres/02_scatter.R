# 02_scatter.R — AUC distribution across 6 Qin-sphere layers, pre and post.
#
# One point per ROI = mean AUC across all 35 subjects.
# Both sessions share the same y-axis for direct comparison.
#
# Run from repo root:
#   Rscript 04_statistics/scripts/qinspheres/02_scatter.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
REPO_ROOT <- if (length(file_arg) > 0) {
  normalizePath(file.path(dirname(normalizePath(sub("--file=", "", file_arg))),
                          "..", "..", ".."))
} else {
  normalizePath(".")
}

DATA_CSV <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres",
                      "tables", "qinspheres_auc.csv")
FIGS_DIR <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres",
                      "figures")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

if (!file.exists(DATA_CSV))
  stop("Input not found — run 01_build_df.jl first:\n  ", DATA_CSV)

# ── Load ──────────────────────────────────────────────────────────────────────
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n", nrow(df)))

# ── Constants ─────────────────────────────────────────────────────────────────
LAYER_ORDER  <- c("visual", "auditory", "motor", "extero", "intero", "mental")
LAYER_LABELS <- c(
  intero   = "Interoception",
  extero   = "Exteroception",
  mental   = "Cognition",
  visual   = "Visual",
  auditory = "Auditory",
  motor    = "Motor"
)
LAYER_COLORS <- c(
  intero   = "#4682B4",
  extero   = "#E8963E",
  mental   = "#8B4682",
  visual   = "#2E8B57",
  auditory = "#B8860B",
  motor    = "#CD5C5C"
)
EXPECTED_ROIS <- c(intero = 11, extero = 14, mental = 12,
                   visual = 55, auditory = 14, motor = 36)

base_theme <- theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#e8e8e8", linewidth = 0.35),
    axis.title       = element_text(size = 10),
    axis.text.x      = element_text(size = 10.5),
    plot.title       = element_text(size = 12, face = "bold", hjust = 0.5),
    plot.subtitle    = element_text(size = 9, hjust = 0.5, color = "grey40")
  )

# ── Per-ROI means (both sessions) ────────────────────────────────────────────
roi_means_all <- df %>%
  filter(is.finite(auc)) %>%
  group_by(session, category, roi_id) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop")

shared_ylim <- range(roi_means_all$mean_auc, na.rm = TRUE)
cat(sprintf("Shared y-axis range: [%.3f, %.3f]\n\n", shared_ylim[1], shared_ylim[2]))

# ── Plot function ─────────────────────────────────────────────────────────────
make_layer_scatter <- function(ses, ses_label) {
  roi_means <- roi_means_all %>%
    filter(session == ses) %>%
    mutate(category = factor(category, levels = LAYER_ORDER))

  roi_counts <- table(roi_means$category)
  for (cat in names(EXPECTED_ROIS)) {
    n <- if (cat %in% names(roi_counts)) roi_counts[[cat]] else 0
    if (n != EXPECTED_ROIS[[cat]])
      stop(sprintf("ROI count mismatch for '%s': got %d, expected %d",
                   cat, n, EXPECTED_ROIS[[cat]]))
  }
  cat(sprintf("%s: ROI counts OK (%s)\n", ses,
              paste(names(roi_counts), roi_counts, sep = "=", collapse = ", ")))

  layer_stats <- roi_means %>%
    group_by(category) %>%
    summarise(med = median(mean_auc), .groups = "drop")

  ggplot(roi_means, aes(x = category, y = mean_auc, color = category)) +
    geom_jitter(size = 2.8, alpha = 0.8, width = 0.18, height = 0) +
    geom_crossbar(data = layer_stats,
                  aes(x = category, y = med, ymin = med, ymax = med),
                  width = 0.45, linewidth = 0.8, color = "black",
                  inherit.aes = FALSE) +
    scale_x_discrete(labels = LAYER_LABELS) +
    scale_color_manual(values = LAYER_COLORS, guide = "none") +
    scale_y_continuous(limits = shared_ylim) +
    labs(
      x        = NULL,
      y        = "Mean ACW-AUC (across subjects)",
      title    = sprintf("AUC across Qin-sphere layers — %s", ses_label),
      subtitle = "Both groups pooled (n = 35). One point = one ROI. Black bar = median."
    ) +
    base_theme
}

# ── Save both session figures ─────────────────────────────────────────────────
p_ses1 <- make_layer_scatter("ses-01", "Pre-retreat (ses-01)")
ggsave(file.path(FIGS_DIR, "qinspheres_layer_scatter_ses01.png"),
       p_ses1, width = 8, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_layer_scatter_ses01.png")))

p_ses2 <- make_layer_scatter("ses-02", "Post-retreat (ses-02)")
ggsave(file.path(FIGS_DIR, "qinspheres_layer_scatter_ses02.png"),
       p_ses2, width = 8, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_layer_scatter_ses02.png")))

# ── Pre minus Post per ROI, by group ─────────────────────────────────────────
# Per-ROI mean per (drug_group, session), then delta = ses-01 − ses-02.
roi_means_grp <- df %>%
  filter(is.finite(auc)) %>%
  group_by(drug_group, session, category, roi_id) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop")

roi_delta <- roi_means_grp %>%
  tidyr::pivot_wider(names_from = session, values_from = mean_auc) %>%
  rename(pre = `ses-01`, post = `ses-02`) %>%
  filter(!is.na(pre), !is.na(post)) %>%
  mutate(
    delta    = post - pre,
    category = factor(category, levels = LAYER_ORDER)
  )

# Shared y-axis across both groups
delta_ylim <- range(roi_delta$delta, na.rm = TRUE)
cat(sprintf("\nPre-minus-post shared y range: [%.3f, %.3f]\n",
            delta_ylim[1], delta_ylim[2]))

make_delta_scatter <- function(grp, grp_label, pt_color) {
  d <- roi_delta %>%
    filter(drug_group == grp) %>%
    mutate(category = factor(category, levels = LAYER_ORDER))
  med_df <- d %>%
    group_by(category) %>%
    summarise(med = median(delta), .groups = "drop")

  ggplot(d, aes(x = category, y = delta, color = category)) +
    geom_hline(yintercept = 0, linetype = "dashed",
               color = "grey50", linewidth = 0.5) +
    geom_jitter(size = 2.8, alpha = 0.8, width = 0.18, height = 0) +
    geom_crossbar(data = med_df,
                  aes(x = category, y = med, ymin = med, ymax = med),
                  width = 0.45, linewidth = 0.8, color = "black",
                  inherit.aes = FALSE) +
    scale_x_discrete(labels = LAYER_LABELS) +
    scale_color_manual(values = LAYER_COLORS, guide = "none") +
    scale_y_continuous(limits = delta_ylim) +
    labs(
      x        = NULL,
      y        = "Post − Pre ACW-AUC",
      title    = sprintf("Post − Pre AUC per layer — %s", grp_label),
      subtitle = sprintf("%s only. One point = one ROI. Black bar = median.",
                         grp_label)
    ) +
    base_theme
}

p_plac <- make_delta_scatter("placebo", "Placebo",  "#CD5C5C")
ggsave(file.path(FIGS_DIR, "qinspheres_prepost_delta_placebo.png"),
       p_plac, width = 8, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_prepost_delta_placebo.png")))

p_verm <- make_delta_scatter("verum", "Verum (DMT)", "#4682B4")
ggsave(file.path(FIGS_DIR, "qinspheres_prepost_delta_verum.png"),
       p_verm, width = 8, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_prepost_delta_verum.png")))

# ── Placebo minus Verum (drug effect on pre-post delta) ──────────────────────
# Join placebo and verum deltas on (category, roi_id); subtract verum from placebo.
drug_diff <- roi_delta %>%
  select(drug_group, category, roi_id, delta) %>%
  tidyr::pivot_wider(names_from = drug_group, values_from = delta) %>%
  filter(!is.na(placebo), !is.na(verum)) %>%
  mutate(
    diff     = placebo - verum,
    category = factor(category, levels = LAYER_ORDER)
  )

med_diff <- drug_diff %>%
  group_by(category) %>%
  summarise(med = median(diff), .groups = "drop")

cat(sprintf("\nPlacebo − Verum medians per layer:\n"))
print(as.data.frame(med_diff), row.names = FALSE, digits = 3)

p_diff <- ggplot(drug_diff, aes(x = category, y = diff, color = category)) +
  geom_hline(yintercept = 0, linetype = "dashed",
             color = "grey50", linewidth = 0.5) +
  geom_jitter(size = 2.8, alpha = 0.8, width = 0.18, height = 0) +
  geom_crossbar(data = med_diff,
                aes(x = category, y = med, ymin = med, ymax = med),
                width = 0.45, linewidth = 0.8, color = "black",
                inherit.aes = FALSE) +
  scale_x_discrete(labels = LAYER_LABELS) +
  scale_color_manual(values = LAYER_COLORS, guide = "none") +
  labs(
    x        = NULL,
    y        = "Placebo − Verum (Pre−Post ACW-AUC)",
    title    = "Drug effect: Placebo − Verum post-pre delta per layer",
    subtitle = "One point = one ROI. Black bar = median. Positive = placebo increased more."
  ) +
  base_theme

ggsave(file.path(FIGS_DIR, "qinspheres_drug_diff.png"),
       p_diff, width = 8, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_drug_diff.png")))

# ── Baseline balance: Placebo vs Verum at ses-01 ─────────────────────────────
# Aggregate AUC across ALL layers and ROIs per subject, baseline session only.
baseline <- df %>%
  filter(session == "ses-01", is.finite(auc)) %>%
  group_by(subject, drug_group) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop")

cat(sprintf("\nBaseline (ses-01) group means:\n"))
print(baseline %>% group_by(drug_group) %>%
        summarise(n = n(), mean = mean(mean_auc), sd = sd(mean_auc), .groups = "drop"),
      digits = 3)

tt <- t.test(mean_auc ~ drug_group, data = baseline)
cat(sprintf("Welch t-test: t(%.1f) = %.3f, p = %.4f\n",
            tt$parameter, tt$statistic, tt$p.value))

p_label <- ifelse(tt$p.value < 0.001, "p < 0.001",
           ifelse(tt$p.value < 0.01,  sprintf("p = %.3f", tt$p.value),
                                       sprintf("p = %.2f",  tt$p.value)))

GROUP_COLORS <- c(placebo = "#CD5C5C", verum = "#4682B4")

p_base <- ggplot(baseline, aes(x = drug_group, y = mean_auc, color = drug_group)) +
  geom_jitter(size = 3, alpha = 0.8, width = 0.12, height = 0) +
  stat_summary(fun = mean, geom = "crossbar",
               aes(ymin = after_stat(y), ymax = after_stat(y)),
               width = 0.4, linewidth = 0.9, color = "black") +
  annotate("text", x = 1.5,
           y = max(baseline$mean_auc) + 0.01 * diff(range(baseline$mean_auc)),
           label = p_label, size = 4, hjust = 0.5) +
  scale_color_manual(values = GROUP_COLORS, guide = "none") +
  scale_x_discrete(labels = c(placebo = "Placebo", verum = "Verum (DMT)")) +
  labs(
    x        = NULL,
    y        = "Mean ACW-AUC (all layers, all ROIs)",
    title    = "Baseline balance: Placebo vs Verum at ses-01",
    subtitle = "One point = one subject. Black bar = group mean."
  ) +
  base_theme

ggsave(file.path(FIGS_DIR, "qinspheres_baseline_balance.png"),
       p_base, width = 5, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_baseline_balance.png")))
