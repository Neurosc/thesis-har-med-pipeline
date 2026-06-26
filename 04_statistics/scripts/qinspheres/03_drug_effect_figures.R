# 03_drug_effect_figures.R — Supervisor-facing drug-effect figures (qinspheres)
#
# Three figures answering "is the verum vs placebo effect real or a motion artifact?"
#
#   Fig 1  qinspheres_drug_effect_raw_vs_resid.png
#     Per-layer drug effect (mean verum delta − mean placebo delta) shown twice:
#     raw AUC_diff vs the quality-residualized AUC_diff. A dumbbell per layer makes
#     the shift under motion correction a one-glance read. Permutation p annotated;
#     black ring = uncorrected p < .05.
#
#   Fig 2  qinspheres_extero_significance.png
#     The exteroception result up close: per-subject verum vs placebo boxplots, raw
#     delta (left) and residual (right). Each panel is annotated with the permutation
#     p under all three outlier schemes (Full / -2.5 SD / Tukey-IQR); open rings mark
#     the IQR outliers that the IQR row drops.
#
#   Fig 3  qinspheres_drug_effect_boxplots_alllayers.png
#     Drug-effect boxplots across ALL 6 layers (placebo vs verum), raw (top row) and
#     residual (bottom row), with per-panel permutation p (red * = uncorrected p<.05).
#
# Inputs (run statistics.R, statistics_resid.R, qc_residualize_auc.R first)
#   04_statistics/results/qinspheres/tables/layer_drug_effect.csv            (+ _sd25, _iqr)
#   04_statistics/results/qinspheres/tables/layer_drug_effect_resid.csv      (+ _sd25, _iqr)
#   04_statistics/results/qinspheres/tables/auc_diff_quality_residuals.csv
#   04_statistics/results/qinspheres/tables/qinspheres_auc.csv
#
# Outputs
#   04_statistics/results/qinspheres/figures/qinspheres_drug_effect_raw_vs_resid.png
#   04_statistics/results/qinspheres/figures/qinspheres_extero_significance.png
#   04_statistics/results/qinspheres/figures/qinspheres_drug_effect_boxplots_alllayers.png
#
# Run from repo root:
#   Rscript 04_statistics/scripts/qinspheres/03_drug_effect_figures.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(ggplot2)
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

TBL_DIR  <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres", "tables")
FIGS_DIR <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres", "figures")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

RAW_TBL   <- file.path(TBL_DIR, "layer_drug_effect.csv")
RESID_TBL <- file.path(TBL_DIR, "layer_drug_effect_resid.csv")
RESID_CSV <- file.path(TBL_DIR, "auc_diff_quality_residuals.csv")
AUC_CSV   <- file.path(TBL_DIR, "qinspheres_auc.csv")

for (f in c(RAW_TBL, RESID_TBL, RESID_CSV, AUC_CSV))
  if (!file.exists(f)) stop("Input not found — run the upstream scripts first:\n  ", f)

# ── Constants / styling (shared with 02_scatter.R) ────────────────────────────
LAYER_ORDER  <- c("visual", "auditory", "motor", "extero", "intero", "mental")
LAYER_LABELS <- c(intero = "Interoception", extero = "Exteroception",
                  mental = "Cognition",     visual = "Visual",
                  auditory = "Auditory",    motor = "Motor")
GROUP_COLORS <- c(placebo = "#CD5C5C", verum = "#4682B4")
TYPE_COLORS  <- c(Raw = "#9E9E9E", Residual = "#1A5276")
ALPHA        <- 0.05

base_theme <- theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#e8e8e8", linewidth = 0.35),
    axis.title       = element_text(size = 10),
    plot.title       = element_text(size = 12, face = "bold", hjust = 0.5),
    plot.subtitle    = element_text(size = 9, hjust = 0.5, color = "grey40"),
    strip.text       = element_text(size = 10.5, face = "bold"),
    strip.background = element_rect(fill = "#f0f0f0", color = NA)
  )

fmt_p <- function(p) ifelse(p < 0.001, "p < .001", sprintf("p = %.3f", p))

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 — raw vs residual drug effect per layer (dumbbell)
# ══════════════════════════════════════════════════════════════════════════════
raw_tbl   <- read.csv(RAW_TBL,   stringsAsFactors = FALSE)
resid_tbl <- read.csv(RESID_TBL, stringsAsFactors = FALSE)

eff <- bind_rows(
  raw_tbl   %>% mutate(type = "Raw"),
  resid_tbl %>% mutate(type = "Residual")
) %>%
  select(layer, type, observed_diff, perm_p_raw, perm_p_fdr) %>%
  mutate(
    layer = factor(layer, levels = rev(LAYER_ORDER)),   # rev → first layer on top
    type  = factor(type, levels = c("Raw", "Residual")),
    sig   = perm_p_raw < ALPHA
  )

# Connect the two types per layer
seg <- eff %>%
  select(layer, type, observed_diff) %>%
  pivot_wider(names_from = type, values_from = observed_diff)

# p-value labels: centered above each dumbbell (residual permutation p)
lab <- eff %>% filter(type == "Residual") %>%
  select(layer, perm_p_raw, sig) %>%
  left_join(seg, by = "layer") %>%
  mutate(
    mid   = (Raw + Residual) / 2,
    yn    = as.numeric(layer) + 0.30,        # nudge text above the markers
    label = sprintf("%s%s", fmt_p(perm_p_raw), ifelse(sig, " *", ""))
  )

cat("── Figure 1: drug effect raw vs residual ──\n")
cat(sprintf("  %-10s  %10s  %10s  %12s  %12s\n",
            "layer", "raw_diff", "res_diff", "raw_perm_p", "res_perm_p"))
for (l in LAYER_ORDER) {
  r <- raw_tbl[raw_tbl$layer == l, ]; s <- resid_tbl[resid_tbl$layer == l, ]
  cat(sprintf("  %-10s  %+10.4f  %+10.4f  %12.4f  %12.4f\n",
              l, r$observed_diff, s$observed_diff, r$perm_p_raw, s$perm_p_raw))
}

p1 <- ggplot(eff, aes(x = observed_diff, y = layer)) +
  geom_vline(xintercept = 0, linetype = "dashed", color = "grey55", linewidth = 0.5) +
  geom_segment(data = seg,
               aes(x = Raw, xend = Residual, y = layer, yend = layer),
               inherit.aes = FALSE, color = "grey75", linewidth = 1) +
  # black ring marking the layer that is significant after motion correction
  geom_point(data = eff %>% filter(sig & type == "Residual"),
             aes(x = observed_diff, y = layer),
             inherit.aes = FALSE, shape = 21, fill = NA,
             color = "black", size = 6.4, stroke = 1) +
  geom_point(aes(fill = type), shape = 21, color = "white", size = 4.2, stroke = 0.7) +
  geom_text(data = lab, aes(x = mid, y = yn, label = label, color = sig),
            inherit.aes = FALSE, size = 3.2, fontface = "plain") +
  scale_fill_manual(values = TYPE_COLORS, name = NULL) +
  scale_color_manual(values = c(`TRUE` = "#1A5276", `FALSE` = "grey40"),
                     guide = "none") +
  scale_y_discrete(labels = LAYER_LABELS) +
  scale_x_continuous(expand = expansion(mult = 0.12)) +
  labs(
    x = "Drug effect: mean(verum Δ) − mean(placebo Δ)",
    y = NULL,
    title = "Drug effect per layer: raw vs motion-residualized",
    subtitle = paste0("Grey = raw AUC change, blue = quality-residualized. ",
                      "Left of 0 = verum decreased more. p = permutation (uncorrected); * p < .05.")
  ) +
  base_theme +
  theme(legend.position = "top")

ggsave(file.path(FIGS_DIR, "qinspheres_drug_effect_raw_vs_resid.png"),
       p1, width = 8.5, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_drug_effect_raw_vs_resid.png")))

# ── Per-subject long data, all layers, raw + residual ─────────────────────────
M_RAW <- "Raw ΔAUC"
M_RES <- "Residual (motion-corrected)"

resid_raw <- read.csv(RESID_CSV, stringsAsFactors = FALSE)
auc_raw   <- read.csv(AUC_CSV,   stringsAsFactors = FALSE)
drug_map  <- auc_raw %>% distinct(subject, drug_group)

# Raw per-subject pre→post delta per layer (aggregated as in statistics.R)
raw_long <- auc_raw %>%
  filter(is.finite(auc), category %in% LAYER_ORDER) %>%
  group_by(subject, drug_group, session, category) %>%
  summarise(mean_auc = mean(auc), .groups = "drop") %>%
  pivot_wider(names_from = session, values_from = mean_auc) %>%
  rename(pre = `ses-01`, post = `ses-02`) %>%
  filter(!is.na(pre), !is.na(post)) %>%
  transmute(subject, drug_group, layer = category,
            value = post - pre, metric = M_RAW)

# Residual per-subject value per layer (re-join drug group from AUC, like statistics_resid.R)
resid_long <- resid_raw %>%
  select(subject, all_of(paste0(LAYER_ORDER, "_resid"))) %>%
  pivot_longer(ends_with("_resid"), names_to = "layer", values_to = "value") %>%
  mutate(layer = sub("_resid$", "", layer), metric = M_RES) %>%
  filter(!is.na(value)) %>%
  inner_join(drug_map, by = "subject")

long_all <- bind_rows(raw_long, resid_long) %>%
  mutate(layer      = factor(layer, levels = LAYER_ORDER),
         metric     = factor(metric, levels = c(M_RAW, M_RES)),
         drug_group = factor(drug_group, levels = c("placebo", "verum")))

# ── Permutation p for every (metric × scheme × layer) from the saved tables ────
load_summary <- function(fname, metric, scheme) {
  read.csv(file.path(TBL_DIR, fname), stringsAsFactors = FALSE) %>%
    transmute(layer, perm_p = perm_p_raw, perm_p_fdr,
              n_verum, n_placebo, metric = metric, scheme = scheme)
}
psum <- bind_rows(
  load_summary("layer_drug_effect.csv",            M_RAW, "Full"),
  load_summary("layer_drug_effect_sd25.csv",       M_RAW, "-2.5 SD"),
  load_summary("layer_drug_effect_iqr.csv",        M_RAW, "IQR"),
  load_summary("layer_drug_effect_resid.csv",      M_RES, "Full"),
  load_summary("layer_drug_effect_resid_sd25.csv", M_RES, "-2.5 SD"),
  load_summary("layer_drug_effect_resid_iqr.csv",  M_RES, "IQR")
) %>%
  mutate(metric = factor(metric, levels = c(M_RAW, M_RES)),
         scheme = factor(scheme, levels = c("Full", "-2.5 SD", "IQR")))

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — exteroception, with outlier-removed statistics
# ══════════════════════════════════════════════════════════════════════════════
extero <- long_all %>% filter(layer == "extero")

# Flag Tukey-IQR outliers per metric (the points the IQR scheme drops) and fix a
# deterministic jitter so the rings land exactly on their points.
extero <- extero %>%
  group_by(metric) %>%
  mutate(q1 = quantile(value, 0.25), q3 = quantile(value, 0.75), iqr = q3 - q1,
         iqr_outlier = value < (q1 - 1.5 * iqr) | value > (q3 + 1.5 * iqr)) %>%
  ungroup()
set.seed(42)
extero$xpos <- as.integer(extero$drug_group) + runif(nrow(extero), -0.13, 0.13)

# Annotation: the three outlier schemes' permutation p per panel
ext_p <- psum %>% filter(layer == "extero")
mk_label <- function(m) {
  d <- ext_p %>% filter(metric == m)
  g <- function(sc) d[d$scheme == sc, ]
  f <- g("Full"); s <- g("-2.5 SD"); q <- g("IQR")
  paste(
    sprintf("Full:     perm p = %.3f  (n %d/%d)", f$perm_p, f$n_verum, f$n_placebo),
    sprintf("-2.5 SD:  perm p = %.3f  (n %d/%d)", s$perm_p, s$n_verum, s$n_placebo),
    sprintf("IQR:      perm p = %.3f  (n %d/%d)", q$perm_p, q$n_verum, q$n_placebo),
    sep = "\n")
}
annot2 <- data.frame(
  metric = factor(c(M_RAW, M_RES), levels = c(M_RAW, M_RES)),
  label  = c(mk_label(M_RAW), mk_label(M_RES)),
  stringsAsFactors = FALSE
) %>%
  left_join(extero %>% group_by(metric) %>%
              summarise(y = max(value) + 0.05 * diff(range(value)), .groups = "drop"),
            by = "metric")

cat("\n── Figure 2: exteroception (outlier-sensitivity) ──\n")
print(as.data.frame(ext_p[, c("metric", "scheme", "perm_p", "n_verum", "n_placebo")]),
      digits = 3, row.names = FALSE)

p2 <- ggplot(extero, aes(x = xpos, y = value)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey55", linewidth = 0.5) +
  geom_boxplot(aes(x = as.integer(drug_group), group = drug_group, color = drug_group),
               width = 0.5, outlier.shape = NA, fill = NA, linewidth = 0.6,
               inherit.aes = FALSE) +
  geom_point(aes(color = drug_group), size = 2.6, alpha = 0.8) +
  geom_point(data = extero %>% filter(iqr_outlier),
             shape = 21, fill = NA, color = "black", size = 4.6, stroke = 1) +
  geom_text(data = annot2, aes(x = 1.5, y = y, label = label),
            inherit.aes = FALSE, size = 2.9, lineheight = 0.95, vjust = 0) +
  facet_wrap(~ metric, scales = "free_y") +
  scale_color_manual(values = GROUP_COLORS, guide = "none") +
  scale_x_continuous(breaks = c(1, 2), labels = c("Placebo", "Verum (DMT)"),
                     limits = c(0.5, 2.5)) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.30))) +
  labs(
    x = NULL, y = "Exteroception ACW-AUC change",
    title = "Exteroception: drug effect with outlier sensitivity",
    subtitle = paste0("One point = one subject. Box = median / IQR. ",
                      "Open rings = Tukey-IQR outliers (dropped in the IQR row of stats).")
  ) +
  base_theme

ggsave(file.path(FIGS_DIR, "qinspheres_extero_significance.png"),
       p2, width = 9, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_extero_significance.png")))

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — drug-effect boxplots across ALL layers (statistics.R style)
# ══════════════════════════════════════════════════════════════════════════════
# Per-panel permutation p (no-exclusion / Full scheme); red label = p < .05.
annot3 <- psum %>%
  filter(scheme == "Full") %>%
  transmute(metric, layer = factor(layer, levels = LAYER_ORDER),
            label = sprintf("p=%.3f%s", perm_p, ifelse(perm_p < ALPHA, " *", "")),
            sig   = perm_p < ALPHA) %>%
  left_join(long_all %>% group_by(metric) %>%
              summarise(y = max(value) + 0.06 * diff(range(value)), .groups = "drop"),
            by = "metric")
annot3$col <- ifelse(annot3$sig, "#B22222", "grey25")

cat("\n── Figure 3: all-layer drug-effect boxplots ──\n")
psum %>% filter(scheme == "Full") %>%
  select(metric, layer, perm_p) %>%
  pivot_wider(names_from = metric, values_from = perm_p) %>%
  as.data.frame() %>% print(digits = 3, row.names = FALSE)

p3 <- ggplot(long_all, aes(x = drug_group, y = value, color = drug_group)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "grey55", linewidth = 0.45) +
  geom_boxplot(width = 0.6, outlier.shape = NA, fill = NA, linewidth = 0.55) +
  geom_jitter(width = 0.16, height = 0, size = 1.5, alpha = 0.6) +
  geom_text(data = annot3, aes(x = 1.5, y = y, label = label),
            inherit.aes = FALSE, color = annot3$col, size = 2.9) +
  facet_grid(metric ~ layer, scales = "free_y",
             labeller = labeller(layer = LAYER_LABELS)) +
  scale_color_manual(values = GROUP_COLORS, guide = "none") +
  scale_x_discrete(labels = c(placebo = "Plac", verum = "Verum")) +
  scale_y_continuous(expand = expansion(mult = c(0.05, 0.16))) +
  labs(
    x = NULL, y = "ACW-AUC change (post − pre)",
    title = "Drug effect across all layers: Placebo vs Verum (DMT)",
    subtitle = paste0("Per subject. Box = median / IQR. Top row = raw ΔAUC, ",
                      "bottom row = motion-residualized. p = permutation (uncorrected); * p < .05.")
  ) +
  base_theme +
  theme(axis.text.x = element_text(size = 8.5))

ggsave(file.path(FIGS_DIR, "qinspheres_drug_effect_boxplots_alllayers.png"),
       p3, width = 13.5, height = 6.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_drug_effect_boxplots_alllayers.png")))

cat(sprintf("\nFigures saved to %s:\n", FIGS_DIR))
cat("  qinspheres_drug_effect_raw_vs_resid.png\n")
cat("  qinspheres_extero_significance.png\n")
cat("  qinspheres_drug_effect_boxplots_alllayers.png\n")
