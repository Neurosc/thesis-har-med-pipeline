# 04_figures.R — qinspheres ACW-AUC thesis figures (6 layers, maximal pipeline).
#
# Ports the parcels_NoGSR/intrinsic_timescale/04_figures.R aesthetic (ggdist
# raincloud + serif theme_minimal + significance brackets + patchwork) to the
# 6-layer qinspheres design. Self-contained: reads qinspheres_auc.csv, computes
# its own paired-t (retreat) and baseline p-values, and reads the per-layer
# drug-effect permutation p from layer_drug_effect.csv (run statistics.R first).
#
# Input : 04_statistics/results/qinspheres/tables/qinspheres_auc.csv
#         04_statistics/results/qinspheres/tables/layer_drug_effect.csv  (drug bracket)
# Output: 04_statistics/results/qinspheres/figures/qin_*.png  (300 dpi)
#         04_statistics/results/qinspheres/tables/descriptive_pvalues.csv
#
# Run from repo root:
#   Rscript 04_statistics/scripts/qinspheres/04_figures.R

for (pkg in c("ggplot2", "ggdist", "patchwork", "dplyr", "tidyr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
}
suppressPackageStartupMessages({
  library(ggplot2); library(ggdist); library(patchwork); library(dplyr); library(tidyr)
})

# ── Paths ──────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
REPO_ROOT <- if (length(file_arg))
  normalizePath(file.path(dirname(normalizePath(sub("^--file=", "", file_arg[1]))),
                          "..", "..", "..")) else normalizePath(".")

PIPELINE   <- { a <- commandArgs(trailingOnly = TRUE); if (length(a) >= 1) a[1] else "maximal" }
cat(sprintf("Pipeline: %s\n", PIPELINE))
QIN_DIR    <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres", PIPELINE)
TABLES_DIR <- file.path(QIN_DIR, "tables")
FIGS_DIR   <- file.path(QIN_DIR, "figures")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

DATA_CSV <- file.path(TABLES_DIR, "qinspheres_auc.csv")
DRUG_CSV <- file.path(TABLES_DIR, "layer_drug_effect.csv")
if (!file.exists(DATA_CSV)) stop("Input not found — run 01_build_df.jl first: ", DATA_CSV)

SEP <- strrep("=", 70)
out_png <- function(name, gg, w, h) {
  path <- file.path(FIGS_DIR, name)
  ggsave(path, gg, width = w, height = h, dpi = 300, bg = "white")
  cat(sprintf("Saved: %s\n", path))
}
cat(SEP, "\n04_figures.R — qinspheres (6 layers, maximal)\n", SEP, "\n\n", sep = "")

# ── Constants ──────────────────────────────────────────────────────────────────
# Display order: self layers (row 1) then nonself (row 2)
LAYER_ORDER  <- c("intero", "extero", "mental", "visual", "motor", "auditory")
LAYER_LABELS <- c(intero = "Interoception", extero = "Exteroception",
                  mental = "Cognition",     visual = "Visual",
                  motor  = "Motor",         auditory = "Auditory")

COND_LEVELS <- c("Placebo Pre", "Verum Pre", "Placebo Post", "Verum Post")
COND_LABELS <- c("Placebo\nPre", "Verum\nPre", "Placebo\nPost", "Verum\nPost")
COND_COLORS <- c("Placebo Pre" = "#CD5C5C", "Verum Pre" = "#4682B4",
                 "Placebo Post" = "#E8963E", "Verum Post" = "#2E8B8B")
DELTA_COLORS   <- c(placebo = "#CD5C5C", verum = "#4682B4")
SESSION_COLORS <- c("ses-01" = "#CD5C5C", "ses-02" = "#E8963E")
BASE_COLORS    <- c(placebo = "#CD5C5C", verum = "#4682B4")

# ── Shared helpers (parcels aesthetic) ─────────────────────────────────────────
panel_theme <- function(title_face = "plain")
  theme_minimal(base_size = 11, base_family = "serif") +
  theme(panel.background   = element_rect(fill = "white", color = NA),
        plot.background    = element_rect(fill = "white", color = NA),
        panel.grid.major.y = element_line(color = "#e8e8e8", linewidth = 0.4),
        panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
        plot.title   = element_text(face = title_face, hjust = 0.5, size = 12, family = "serif"),
        axis.text.x  = element_text(size = 9, lineheight = 0.85),
        axis.title.y = element_text(size = 10), legend.position = "none")

rain_layers <- function() list(
  geom_point(aes(x = x_jitter), size = 1.3, alpha = 0.55, shape = 16),
  geom_boxplot(fill = NA, width = 0.90, outlier.shape = NA, linewidth = 0.65,
               position = position_nudge(x = -0.4)),
  stat_halfeye(adjust = 0.7, width = 0.55, justification = -0.38, .width = 0,
               point_colour = NA, slab_alpha = 0.65))

add_x <- function(d, col, jit = 0.10) {
  set.seed(42)
  d$x_cond   <- as.numeric(d[[col]]) * 2
  d$x_jitter <- d$x_cond - 0.40 + runif(nrow(d), -jit, jit)
  d
}

# 6 panels → 2 rows × 3 cols (self on top, nonself below)
assemble6 <- function(panels, title)
  ((panels[[1]] | panels[[2]] | panels[[3]]) /
   (panels[[4]] | panels[[5]] | panels[[6]])) +
  plot_annotation(title = title,
    theme = theme(plot.title = element_text(face = "plain", hjust = 0.5,
                                            size = 13, family = "serif")))

sig_bracket <- function(p, x1, x2, yvals, label, color = "black", lab_size = 3) {
  y_max <- max(yvals, na.rm = TRUE); y_span <- diff(range(yvals, na.rm = TRUE))
  if (!is.finite(y_span) || y_span == 0) y_span <- abs(y_max) + 1
  y_br  <- y_max + y_span * 0.06; y_tick <- y_br - y_span * 0.025
  p +
    annotate("segment", x = x1, xend = x2, y = y_br, yend = y_br,  color = color, linewidth = 0.5) +
    annotate("segment", x = x1, xend = x1, y = y_br, yend = y_tick, color = color, linewidth = 0.5) +
    annotate("segment", x = x2, xend = x2, y = y_br, yend = y_tick, color = color, linewidth = 0.5) +
    annotate("text", x = (x1 + x2) / 2, y = y_br + y_span * 0.035, label = label,
             size = lab_size, hjust = 0.5, color = color) +
    coord_cartesian(ylim = c(NA, y_br + y_span * 0.12))
}

fmt_p    <- function(p) if (is.na(p)) "n.s." else if (p < 0.001) "p<0.001" else sprintf("p=%.3f", p)
sig_star <- function(p) if (is.na(p)) "" else if (p < 0.001) "***" else
                        if (p < 0.01) "**" else if (p < 0.05) "*" else ""
cohens_d <- function(x, y) {
  sp <- sqrt(((length(x)-1)*var(x) + (length(y)-1)*var(y)) / (length(x)+length(y)-2))
  if (sp == 0) 0 else (mean(x) - mean(y)) / sp
}
paired_test <- function(d, grp) {  # paired post vs pre on subject means
  both <- merge(d[d$group == grp & d$session == "ses-01", c("subject", "mean_auc")],
                d[d$group == grp & d$session == "ses-02", c("subject", "mean_auc")],
                by = "subject", suffixes = c("_pre", "_post"))
  if (nrow(both) < 3) return(c(n = nrow(both), t = NA, p = NA))
  tt <- t.test(both$mean_auc_post, both$mean_auc_pre, paired = TRUE)
  c(n = nrow(both), t = unname(tt$statistic), p = tt$p.value)
}
base_test <- function(plac, verm) {  # placebo vs verum at baseline
  t_r <- t.test(plac, verm, var.equal = FALSE); mw <- wilcox.test(plac, verm, exact = FALSE)
  list(n = length(plac) + length(verm), t = unname(t_r$statistic), df = unname(t_r$parameter),
       p_t = t_r$p.value, U = unname(mw$statistic), p_mw = mw$p.value, d = cohens_d(plac, verm))
}

# ── Load, prepare ──────────────────────────────────────────────────────────────
df_raw <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded qinspheres_auc: %d rows, %d subjects\n",
            nrow(df_raw), dplyr::n_distinct(df_raw$subject)))

df_raw <- df_raw %>%
  filter(category %in% LAYER_ORDER, is.finite(auc)) %>%
  mutate(group   = factor(drug_group, levels = c("placebo", "verum")),
         session = factor(session, levels = c("ses-01", "ses-02")),
         category = factor(category, levels = LAYER_ORDER),
         condition = factor(paste0(ifelse(group == "placebo", "Placebo", "Verum"), " ",
                                   ifelse(session == "ses-01", "Pre", "Post")),
                            levels = COND_LEVELS))

subj_means <- df_raw %>%
  group_by(category, subject, group, session, condition) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop") %>%
  mutate(condition = factor(condition, levels = COND_LEVELS),
         category  = factor(category, levels = LAYER_ORDER))

ses01_df <- df_raw[as.character(df_raw$session) == "ses-01", ]
ses01_subj <- function(cat = NULL) {
  d <- if (is.null(cat)) ses01_df else ses01_df[as.character(ses01_df$category) == cat, ]
  d %>% group_by(subject, group) %>% summarise(value = mean(auc, na.rm = TRUE), .groups = "drop")
}

# ── Descriptive p-values → descriptive_pvalues.csv ─────────────────────────────
cat("\n", SEP, "\nComputing descriptive p-values (base-R)\n", SEP, "\n", sep = "")
pair_stats <- do.call(rbind, lapply(LAYER_ORDER, function(cat) {
  d <- as.data.frame(subj_means[as.character(subj_means$category) == cat, ])
  do.call(rbind, lapply(c("placebo", "verum"), function(g) {
    s <- paired_test(d, g)
    data.frame(layer = cat, test = paste0(g, "_prepost_pairedt"),
               n = s["n"], statistic = s["t"], p = s["p"], row.names = NULL)
  }))
}))
pp <- function(cat, grp) {
  r <- pair_stats[pair_stats$layer == cat & pair_stats$test == paste0(grp, "_prepost_pairedt"), ]
  if (nrow(r)) r$p[1] else NA
}
base_stats <- do.call(rbind, lapply(c("Pooled", LAYER_ORDER), function(lab) {
  d <- if (lab == "Pooled") ses01_subj() else ses01_subj(lab)
  s <- base_test(d$value[d$group == "placebo"], d$value[d$group == "verum"])
  data.frame(layer = lab,
             test  = c("baseline_welch_t", "baseline_mannwhitney", "baseline_cohens_d"),
             n     = s$n,
             statistic = c(s$t, s$U, s$d),
             p     = c(s$p_t, s$p_mw, NA), row.names = NULL)
}))
pvalues <- rbind(pair_stats, base_stats)
write.csv(pvalues, file.path(TABLES_DIR, "descriptive_pvalues.csv"), row.names = FALSE)
cat(sprintf("Saved: descriptive_pvalues.csv (%d rows)\n", nrow(pvalues)))

# Drug-effect permutation p (from statistics.R)
drug_p <- function(layer) c(p = NA_real_, q = NA_real_)
if (file.exists(DRUG_CSV)) {
  drug_tbl <- read.csv(DRUG_CSV, stringsAsFactors = FALSE)
  drug_p <- function(layer) {
    r <- drug_tbl[drug_tbl$layer == layer, ]
    if (!nrow(r)) return(c(p = NA_real_, q = NA_real_))
    c(p = r$perm_p_raw[1], q = if ("perm_p_fdr" %in% names(r)) r$perm_p_fdr[1] else NA_real_)
  }
} else {
  cat(sprintf("Note: %s not found — drug-effect brackets show n/a (run statistics.R)\n", DRUG_CSV))
}

# ── Figure 1: overview raincloud (4 conditions) per layer ──────────────────────
make_overview <- function(cat) {
  d <- add_x(as.data.frame(subj_means[as.character(subj_means$category) == cat, ]),
             "condition", jit = 0.2)
  ggplot(d, aes(x = x_cond, y = mean_auc, fill = condition, color = condition, group = condition)) +
    rain_layers() +
    stat_summary(fun = mean, geom = "point", shape = 18, size = 2, color = "black",
                 position = position_nudge(x = -0.4)) +
    scale_fill_manual(values = COND_COLORS, guide = "none") +
    scale_color_manual(values = COND_COLORS, guide = "none") +
    scale_x_continuous(breaks = (1:4) * 2, labels = COND_LABELS,
                       expand = expansion(add = c(0.65, 1.00))) +
    labs(title = LAYER_LABELS[[cat]], y = "Mean AUC (s)", x = NULL) + panel_theme()
}
out_png("qin_overview_raincloud_4conditions.png",
        assemble6(lapply(LAYER_ORDER, make_overview),
                  "AUC per condition and layer (qinspheres, maximal)"), 14, 8)

# ── Delta (post − pre) per group ───────────────────────────────────────────────
pre_df  <- subj_means[subj_means$session == "ses-01", c("category", "subject", "group", "mean_auc")]
post_df <- subj_means[subj_means$session == "ses-02", c("category", "subject", "group", "mean_auc")]
names(pre_df)[4] <- "pre_auc"; names(post_df)[4] <- "post_auc"
delta_df <- merge(pre_df, post_df, by = c("category", "subject", "group"))
delta_df$delta <- delta_df$post_auc - delta_df$pre_auc
delta_df$group <- factor(delta_df$group, levels = c("placebo", "verum"))

make_delta <- function(cat, with_sig = FALSE) {
  d <- add_x(delta_df[as.character(delta_df$category) == cat, ], "group")
  p <- ggplot(d, aes(x = x_cond, y = delta, fill = group, color = group, group = group)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
    rain_layers() +
    scale_fill_manual(values = DELTA_COLORS, guide = "none") +
    scale_color_manual(values = DELTA_COLORS, guide = "none") +
    scale_x_continuous(breaks = (1:2) * 2, labels = c("Placebo", "Verum"),
                       expand = expansion(add = c(0.65, 1.00))) +
    labs(title = LAYER_LABELS[[cat]], y = "ΔAUC (post − pre, s)", x = NULL) + panel_theme()
  if (!with_sig) return(p)
  pq  <- drug_p(cat); star <- sig_star(pq[["q"]])
  lab <- if (is.na(pq[["p"]])) "n/a"
         else sprintf("%sp=%.3f (q=%.3f)", if (star != "") paste0(star, " ") else "",
                      pq[["p"]], pq[["q"]])
  sig_bracket(p, 1.6, 3.6, d$delta, lab)
}
out_png("qin_drug_delta_per_layer.png",
        assemble6(lapply(LAYER_ORDER, make_delta),
                  "Post − Pre AUC change per layer (qinspheres, maximal)"), 14, 8)
out_png("qin_drug_effect_significance.png",
        assemble6(lapply(LAYER_ORDER, function(c) make_delta(c, with_sig = TRUE)),
                  "Drug effect per layer — permutation p (q = FDR across 6 layers)"), 14, 8)

# ── Retreat effect (placebo, pre vs post; paired-t bracket) ────────────────────
plac_means <- subj_means[subj_means$group == "placebo", ]
make_retreat <- function(cat) {
  d <- plac_means[as.character(plac_means$category) == cat, ]
  d$session_f <- factor(as.character(d$session), levels = c("ses-01", "ses-02"))
  d <- add_x(as.data.frame(d), "session_f")
  p <- ggplot(d, aes(x = x_cond, y = mean_auc, fill = session_f, color = session_f, group = session_f)) +
    rain_layers() +
    scale_fill_manual(values = SESSION_COLORS, guide = "none") +
    scale_color_manual(values = SESSION_COLORS, guide = "none") +
    scale_x_continuous(breaks = (1:2) * 2, labels = c("Pre", "Post"),
                       expand = expansion(add = c(0.65, 1.00))) +
    labs(title = LAYER_LABELS[[cat]], y = "Mean AUC (s)", x = NULL) + panel_theme()
  sig_bracket(p, 1.6, 3.6, d$mean_auc, fmt_p(pp(cat, "placebo")))
}
out_png("qin_retreat_per_layer.png",
        assemble6(lapply(LAYER_ORDER, make_retreat),
                  "Meditation retreat effect (placebo group, paired t)"), 14, 8)

# ── Baseline balance (placebo vs verum at ses-01) ──────────────────────────────
make_base <- function(d_in, panel_title) {
  d <- data.frame(group = factor(as.character(d_in$group), levels = c("placebo", "verum")),
                  value = d_in$value)
  s <- base_test(d$value[d$group == "placebo"], d$value[d$group == "verum"])
  stats_txt <- sprintf("t=%.2f (df=%.1f), p=%.3f\nU=%.0f, p=%.3f\nd=%+.2f",
                       s$t, s$df, s$p_t, s$U, s$p_mw, s$d)
  d <- add_x(d, "group")
  ggplot(d, aes(x = x_cond, y = value, fill = group, color = group, group = group)) +
    rain_layers() +
    annotate("label", x = Inf, y = -Inf, label = stats_txt, hjust = 1.05, vjust = -0.08,
             size = 2.6, family = "serif", fill = "white", color = "gray30",
             label.padding = unit(0.15, "cm")) +
    scale_fill_manual(values = BASE_COLORS, guide = "none") +
    scale_color_manual(values = BASE_COLORS, guide = "none") +
    scale_x_continuous(breaks = (1:2) * 2, labels = c("Placebo", "Verum"),
                       expand = expansion(add = c(0.65, 1.00))) +
    labs(title = panel_title, y = "Mean AUC (s)", x = NULL) + panel_theme()
}
out_png("qin_QC_baseline_balance_pooled.png",
        make_base(ses01_subj(), "Baseline balance — all layers pooled (ses-01)"), 5, 6)
out_png("qin_QC_baseline_balance_per_layer.png",
        assemble6(lapply(LAYER_ORDER, function(cat) make_base(ses01_subj(cat), LAYER_LABELS[[cat]])),
                  "Baseline balance by layer (ses-01)"), 14, 8)

# ── Paired pre→post lines per layer (within-group paired-t brackets) ───────────
XPOS_PAIRED <- c("Placebo Pre" = 1.0, "Placebo Post" = 2.0, "Verum Pre" = 3.2, "Verum Post" = 4.2)
make_paired <- function(cat) {
  d <- as.data.frame(subj_means[as.character(subj_means$category) == cat, ])
  d$x_pos <- XPOS_PAIRED[as.character(d$condition)]
  seg <- function(grp) merge(
    d[d$group == grp & d$session == "ses-01", c("subject", "mean_auc", "x_pos")],
    d[d$group == grp & d$session == "ses-02", c("subject", "mean_auc", "x_pos")],
    by = "subject", suffixes = c("_pre", "_post"))
  plac_segs <- seg("placebo"); verm_segs <- seg("verum")
  y_max <- max(d$mean_auc, na.rm = TRUE); y_span <- diff(range(d$mean_auc, na.rm = TRUE))
  p <- ggplot() +
    geom_segment(data = plac_segs, aes(x = x_pos_pre, xend = x_pos_post, y = mean_auc_pre, yend = mean_auc_post),
                 color = "#CD5C5C", alpha = 0.30, linewidth = 0.5) +
    geom_segment(data = verm_segs, aes(x = x_pos_pre, xend = x_pos_post, y = mean_auc_pre, yend = mean_auc_post),
                 color = "#4682B4", alpha = 0.30, linewidth = 0.5) +
    geom_point(data = d[d$group == "placebo", ], aes(x = x_pos, y = mean_auc),
               color = "#CD5C5C", alpha = 0.60, size = 1.8, shape = 16) +
    geom_point(data = d[d$group == "verum", ], aes(x = x_pos, y = mean_auc),
               color = "#4682B4", alpha = 0.60, size = 1.8, shape = 16) +
    geom_boxplot(data = d[d$group == "placebo", ], aes(x = x_pos, y = mean_auc, group = condition),
                 fill = NA, color = "#CD5C5C", width = 0.22, outlier.shape = NA, linewidth = 0.55) +
    geom_boxplot(data = d[d$group == "verum", ], aes(x = x_pos, y = mean_auc, group = condition),
                 fill = NA, color = "#4682B4", width = 0.22, outlier.shape = NA, linewidth = 0.55) +
    scale_x_continuous(breaks = unname(XPOS_PAIRED),
                       labels = c("Placebo\nPre", "Placebo\nPost", "Verum\nPre", "Verum\nPost"),
                       expand = expansion(add = c(0.5, 0.5))) +
    labs(title = LAYER_LABELS[[cat]], y = "Mean AUC (s)", x = NULL) + panel_theme(title_face = "bold")
  y_br1 <- y_max + y_span * 0.05; y_t1 <- y_br1 - y_span * 0.02
  y_br2 <- y_br1 + y_span * 0.05; y_t2 <- y_br2 - y_span * 0.02
  p +
    annotate("segment", x = 1.0, xend = 2.0, y = y_br1, yend = y_br1, color = "#CD5C5C", linewidth = 0.5) +
    annotate("text", x = 1.5, y = y_br1 + y_span * 0.02, label = fmt_p(pp(cat, "placebo")),
             size = 2.8, hjust = 0.5, color = "#CD5C5C") +
    annotate("segment", x = 3.2, xend = 4.2, y = y_br2, yend = y_br2, color = "#4682B4", linewidth = 0.5) +
    annotate("text", x = 3.7, y = y_br2 + y_span * 0.02, label = fmt_p(pp(cat, "verum")),
             size = 2.8, hjust = 0.5, color = "#4682B4") +
    coord_cartesian(ylim = c(NA, y_br2 + y_span * 0.10))
}
out_png("qin_retreat_paired_prepost.png",
        assemble6(lapply(LAYER_ORDER, make_paired), "Pre → Post AUC change by drug group"), 14, 9)

# ── AUC distribution (overall) ─────────────────────────────────────────────────
dist_theme <- function() theme_minimal(base_size = 12, base_family = "serif") +
  theme(panel.background = element_rect(fill = "white", color = NA),
        plot.background  = element_rect(fill = "white", color = NA),
        panel.grid.minor = element_blank(),
        plot.title = element_text(face = "plain", size = 13, hjust = 0.5))
all_auc <- df_raw$auc[is.finite(df_raw$auc)]
out_png("qin_overall_auc_density.png",
  ggplot(data.frame(auc = all_auc), aes(x = auc)) +
    geom_density(alpha = 0.20, adjust = 1.8, linewidth = 1.0, fill = "#2E8B8B", color = "#2E8B8B") +
    labs(title = "Overall AUC distribution (qinspheres, maximal)",
         x = "AUC (seconds)", y = "Density") + dist_theme(), 8, 6)

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
