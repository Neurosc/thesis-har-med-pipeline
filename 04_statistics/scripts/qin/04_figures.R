# 04_figures.R — qinspheres ACW-AUC thesis figures (6 layers, maximal pipeline).
#
# Ports the parcels_NoGSR/intrinsic_timescale/04_figures.R aesthetic (ggdist
# raincloud + serif theme_minimal + significance brackets + patchwork) to the
# 6-layer qinspheres design. Reads the metric table ({metric}.csv), computes its own within-group
# paired (placebo/verum) and baseline p-values, and reads the per-region **LMM DiD**
# (session:arm) p/q from results/mixed_models/per_config/{spheres|parcels_self_regions}_{pipeline}_did.csv
# (run 03_mixed_models.R first).
#
# Input : 04_statistics/results/acw/{spheres|parcels/self_regions}/{pipeline}/tables/auc.csv
#         04_statistics/results/mixed_models/per_config/{atlas_tag}_{pipeline}_did.csv  (drug-effect bracket = LMM DiD)
# Output: 04_statistics/results/{acw|sampen}/{spheres|parcels/self_regions}/{pipeline}/figures/qin_*.png  (300 dpi)
#         …/{pipeline}/tables/descriptive_pvalues.csv
#
# Run from repo root:
#   Rscript 04_statistics/scripts/qin/04_figures.R

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
REPO_ROOT <- local({                                  # robust to Rscript (--file=) AND source()
  f <- if (length(file_arg)) sub("^--file=", "", file_arg[1]) else {
    p <- NULL; for (i in rev(seq_len(sys.nframe()))) if (!is.null(sys.frame(i)$ofile)) { p <- sys.frame(i)$ofile; break }; p }
  if (is.null(f) || is.na(f)) normalizePath(".")
  else normalizePath(file.path(dirname(normalizePath(f)), "..", "..", "..")) })

.args      <- commandArgs(trailingOnly = TRUE)   # Rscript positional: [pipeline] [metric] [atlas]
# Interactive source(): set .pipeline / .metric / .atlas in the console BEFORE source() to override.
PIPELINE   <- if (exists(".pipeline")) .pipeline else if (length(.args) >= 1) .args[1] else "maximal"
METRIC     <- if (exists(".metric"))   .metric   else if (length(.args) >= 2) .args[2] else "auc"
MLAB       <- if (METRIC == "acw50") "ACW-50" else if (METRIC == "sampen") "SampEn" else if (METRIC == "tau") "τ" else "AUC"
MUNIT      <- if (METRIC == "sampen") MLAB else sprintf("%s (s)", MLAB)   # SampEn is unitless
DUNIT      <- if (METRIC == "sampen") "" else ", s"                       # delta-axis unit suffix
XUNIT      <- if (METRIC == "sampen") "" else " (seconds)"               # density x-axis unit
cat(sprintf("Pipeline: %s  Metric: %s\n", PIPELINE, METRIC))
ATLAS      <- if (exists(".atlas")) .atlas else if (length(.args) >= 3) .args[3] else "qinspheres"
METRIC_DIR <- if (METRIC == "auc") "acw" else METRIC
ATLAS_DIR  <- if (ATLAS == "qinspheres") "spheres" else file.path("parcels", "self_regions")
ATLAS_TAG  <- gsub("/", "_", ATLAS_DIR)   # per_config filename token: spheres | parcels_self_regions
QIN_DIR    <- file.path(REPO_ROOT, "04_statistics", "results", METRIC_DIR, ATLAS_DIR, PIPELINE)
TABLES_DIR <- file.path(QIN_DIR, "tables")
FIGS_DIR   <- file.path(QIN_DIR, "figures")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

MSUF     <- if (METRIC == "auc") "" else paste0("_", METRIC)   # AUC keeps legacy mixed-model filenames
DATA_CSV <- file.path(TABLES_DIR, paste0(METRIC, ".csv"))
if (!file.exists(DATA_CSV)) stop(
  "Input not found — run ",
  if (METRIC == "auc") "01_build_auc.jl" else "02_build_sampen.py",
  " first: ", DATA_CSV)

SEP <- strrep("=", 70)
out_png <- function(name, gg, w, h) {
  path <- file.path(FIGS_DIR, name)
  ggsave(path, gg, width = w, height = h, dpi = 300, bg = "white")
  cat(sprintf("Saved: %s\n", path))
}
cat(SEP, sprintf("\n04_figures.R — %s / %s / %s (6 layers → 5 analysed)\n", ATLAS, PIPELINE, METRIC), SEP, "\n\n", sep = "")

# ── Constants ──────────────────────────────────────────────────────────────────
# Display order: self layers (row 1) then nonself (row 2)
LAYER_ORDER  <- c("intero", "extero", "mental", "visual", "auditory")   # motor excluded from analysis
LAYER_LABELS <- c(intero = "Interoception", extero = "Exteroception",
                  mental = "Cognition",     visual = "Visual",
                  motor  = "Motor",         auditory = "Auditory")

COND_LEVELS <- c("Placebo Pre", "Verum Pre", "Placebo Post", "Verum Post")
COND_LABELS <- c("Placebo\nPre", "Verum\nPre", "Placebo\nPost", "Verum\nPost")

# ── Region × session palette (the single source of colour) ─────────────────────
# Warm = self (intero/extero/cognition), cool = nonself (visual/auditory).
# Every figure encodes SESSION with these two-tone-per-region colours; where a panel
# also distinguishes arm, arm is shown by x-position (not colour).
REGION_SESSION <- list(
  intero   = c(Pre = "#DDB3DE", Post = "#9B4DA1"),   # purple
  extero   = c(Pre = "#F6BBA4", Post = "#EB6834"),
  mental   = c(Pre = "#F7D58C", Post = "#EDA100"),
  visual   = c(Pre = "#9FC2ED", Post = "#2A78D6"),
  auditory = c(Pre = "#98DBC3", Post = "#1BAF7A"))
rs_col       <- function(cat, sess) unname(REGION_SESSION[[cat]][[sess]])   # region+session -> hex
BASE_NEUTRAL <- c(placebo = "#8A8A8A", verum = "#3A3A3A")   # pooled baseline only (no single region)

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

# 6 panels → 2 rows × 3 cols (self on top, nonself below); ncol flexible for subsets
assemble6 <- function(panels, title, ncol = 3)   # flexible layout (5 regions, motor excluded)
  patchwork::wrap_plots(panels, ncol = ncol) +
  plot_annotation(title = title,
    theme = theme(plot.title = element_text(face = "plain", hjust = 0.5,
                                            size = 13, family = "serif")))


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
cat(sprintf("Loaded %s: %d rows, %d subjects\n", basename(DATA_CSV),
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

# Overall time-effect table (for the forest figure below), from 03_mixed_models.R.
OVERALL_CSV <- file.path(REPO_ROOT, "04_statistics", "results", "mixed_models", "per_config",
                         sprintf("%s_%s%s_overall.csv", ATLAS_TAG, PIPELINE, MSUF))
# Per-region LMM DiD (session:arm interaction = drug effect), from 03_mixed_models.R.
DID_CSV <- file.path(REPO_ROOT, "04_statistics", "results", "mixed_models", "per_config",
                     sprintf("%s_%s%s_did.csv", ATLAS_TAG, PIPELINE, MSUF))

# ── Figure 1: overview raincloud (4 conditions) per layer ──────────────────────
make_overview <- function(cat) {
  d <- add_x(as.data.frame(subj_means[as.character(subj_means$category) == cat, ]),
             "condition", jit = 0.2)
  # colour by session in this region's two-tone palette (both Pre conds = pre hue, both Post = post hue)
  cond_cols <- c("Placebo Pre" = rs_col(cat, "Pre"),  "Verum Pre"  = rs_col(cat, "Pre"),
                 "Placebo Post" = rs_col(cat, "Post"), "Verum Post" = rs_col(cat, "Post"))
  ggplot(d, aes(x = x_cond, y = mean_auc, fill = condition, color = condition, group = condition)) +
    rain_layers() +
    stat_summary(fun = mean, geom = "point", shape = 18, size = 2, color = "black",
                 position = position_nudge(x = -0.4)) +
    scale_fill_manual(values = cond_cols, guide = "none") +
    scale_color_manual(values = cond_cols, guide = "none") +
    scale_x_continuous(breaks = (1:4) * 2, labels = COND_LABELS,
                       expand = expansion(add = c(0.65, 1.00))) +
    labs(title = LAYER_LABELS[[cat]], y = paste("Mean", MUNIT), x = NULL) + panel_theme()
}
out_png("qin_overview_raincloud_4conditions.png",
        assemble6(lapply(LAYER_ORDER, make_overview),
                  sprintf("%s per condition and layer (%s %s)", MLAB, ATLAS, PIPELINE)), 14, 8)

# ── Time-effect boxplots: Pre vs Post per region (both arms), LMM session significance ──
ov_tbl  <- if (file.exists(OVERALL_CSV)) read.csv(OVERALL_CSV, stringsAsFactors = FALSE) else NULL
CAT2REG <- c(intero="intero", extero="extero", mental="cognition", visual="visual", auditory="auditory")
time_pq <- function(cat) {                              # session (time) p/q from _overall.csv
  if (is.null(ov_tbl)) return(c(p = NA_real_, q = NA_real_))
  r <- ov_tbl[ov_tbl$region == CAT2REG[[cat]], ]
  if (!nrow(r)) return(c(p = NA_real_, q = NA_real_))
  c(p = r$session_p[1], q = r$session_q[1])
}
did_tbl <- if (file.exists(DID_CSV)) read.csv(DID_CSV, stringsAsFactors = FALSE) else NULL
did_pq  <- function(cat) {                              # DiD (session:arm) est/p/q from _did.csv
  if (is.null(did_tbl)) return(c(est = NA_real_, p = NA_real_, q = NA_real_))
  r <- did_tbl[did_tbl$region == CAT2REG[[cat]], ]
  if (!nrow(r)) return(c(est = NA_real_, p = NA_real_, q = NA_real_))
  c(est = r$effect[1], p = r$p[1], q = r$q[1])
}
SIMPLE_CSV <- file.path(REPO_ROOT, "04_statistics", "results", "mixed_models", "per_config",
                        sprintf("%s_%s%s_simple.csv", ATLAS_TAG, PIPELINE, MSUF))
simple_tbl <- if (file.exists(SIMPLE_CSV)) read.csv(SIMPLE_CSV, stringsAsFactors = FALSE) else NULL
simple_pq  <- function(cat, arm) {                      # within-arm post-pre est/p/q from _simple.csv (LMM)
  if (is.null(simple_tbl)) return(c(est = NA_real_, p = NA_real_, q = NA_real_))
  r <- simple_tbl[simple_tbl$region == CAT2REG[[cat]] & simple_tbl$arm == arm, ]
  if (!nrow(r)) return(c(est = NA_real_, p = NA_real_, q = NA_real_))
  c(est = r$est[1], p = r$p[1], q = r$q[1])
}
# Pre/Post boxplots + per-subject connecting lines. arm = NULL (both arms) or "placebo"/"verum".
box_base <- function(cat, arm = NULL) {
  box_cols <- c(Pre = rs_col(cat, "Pre"), Post = rs_col(cat, "Post"))   # this region's two tones
  d <- as.data.frame(subj_means[as.character(subj_means$category) == cat, ])
  if (!is.null(arm)) d <- d[as.character(d$group) == arm, ]
  d$sess <- factor(ifelse(d$session == "ses-01", "Pre", "Post"), levels = c("Pre", "Post"))
  d$x <- as.numeric(d$sess) * 2
  subs <- unique(as.character(d$subject))               # one x-offset per subject (same for pre & post)
  set.seed(42); off <- setNames(runif(length(subs), -0.12, 0.12), subs)
  d$xj <- d$x + off[as.character(d$subject)]
  w <- merge(d[d$sess == "Pre",  c("subject", "xj", "mean_auc")],   # connecting-line endpoints
             d[d$sess == "Post", c("subject", "xj", "mean_auc")], by = "subject", suffixes = c("_pre", "_post"))
  p <- ggplot(d, aes(x = x, y = mean_auc)) +
    geom_segment(data = w, aes(x = xj_pre, xend = xj_post, y = mean_auc_pre, yend = mean_auc_post),
                 color = "gray45", alpha = 0.6, linewidth = 0.45, inherit.aes = FALSE) +   # each person's line
    geom_point(aes(x = xj, color = sess), size = 1.2, alpha = 0.65, shape = 16) +
    geom_boxplot(aes(color = sess, fill = sess, group = sess), alpha = 0.15, width = 1.0,
                 outlier.shape = NA, linewidth = 0.65) +
    scale_color_manual(values = box_cols, guide = "none") +
    scale_fill_manual(values = box_cols, guide = "none") +
    scale_x_continuous(breaks = c(2, 4), labels = c("Pre", "Post"), expand = expansion(add = c(1.2, 1.2))) +
    labs(title = LAYER_LABELS[[cat]], y = paste("Mean", MUNIT), x = NULL) + panel_theme()
  list(cat = cat, p = p, ext = range(d$mean_auc, na.rm = TRUE))
}
# Assemble a Pre/Post boxplot figure with a shared y-axis and per-region p/q bracket.
#   pqfun(cat) -> c(p=, q=) supplies the LMM significance for each panel's bracket.
#   regions: which layers to show (default all 5). w/h: figure size (auto-scaled if NULL).
make_box_figure <- function(fname, title, arm, pqfun, n_lab, regions = LAYER_ORDER,
                            w = NULL, h = NULL) {
  bases <- lapply(regions, function(cat) box_base(cat, arm))
  lo <- min(vapply(bases, function(b) b$ext[1], numeric(1)))
  hi <- max(vapply(bases, function(b) b$ext[2], numeric(1)))
  span <- hi - lo; ybr <- hi + span * 0.06
  tick <- ybr - span * 0.02; top <- ybr + span * 0.13
  panels <- lapply(bases, function(b) {
    pq <- pqfun(b$cat); star <- sig_star(pq[["q"]])
    lab <- if (is.na(pq[["p"]])) "n.s."
           else sprintf("%sp=%.3f (q=%.3f)", if (nzchar(star)) paste0(star, " ") else "", pq[["p"]], pq[["q"]])
    b$p +
      annotate("segment", x = 2, xend = 4, y = ybr, yend = ybr,  color = "black", linewidth = 0.5) +
      annotate("segment", x = 2, xend = 2, y = ybr, yend = tick, color = "black", linewidth = 0.5) +
      annotate("segment", x = 4, xend = 4, y = ybr, yend = tick, color = "black", linewidth = 0.5) +
      annotate("text", x = 3, y = ybr + span * 0.045, label = lab, size = 3, family = "serif") +
      coord_cartesian(ylim = c(lo - span * 0.05, top))
  })
  ncol <- min(3, length(panels))
  if (is.null(w)) w <- 4.7 * ncol
  if (is.null(h)) h <- 4 * ceiling(length(panels) / ncol)
  out_png(fname, assemble6(panels, sprintf(title, MLAB, n_lab), ncol = ncol), w, h)
}
# Single-panel version: all 5 regions on one x-axis, Pre/Post boxplots side-by-side per region.
# Colour = the region two-tone palette (Pre = light, Post = saturated). A separate colour-key
# panel names which category is which colour. pqfun(cat) -> c(p=, q=) labels each region.
make_box_single <- function(fname, title, arm, pqfun) {
  d <- as.data.frame(subj_means[as.character(subj_means$category) %in% LAYER_ORDER, ])
  if (!is.null(arm)) d <- d[as.character(d$group) == arm, ]
  d$cat  <- factor(as.character(d$category), levels = LAYER_ORDER)
  d$sess <- factor(ifelse(d$session == "ses-01", "Pre", "Post"), levels = c("Pre", "Post"))
  d$xpos <- as.integer(d$cat) + ifelse(d$sess == "Pre", -0.19, 0.19)   # two boxes per region
  d$key  <- paste(as.character(d$cat), d$sess, sep = ".")
  set.seed(42); d$xj <- d$xpos + runif(nrow(d), -0.07, 0.07)
  keycols <- unlist(lapply(LAYER_ORDER, function(cat)                 # region+session -> hex
    setNames(c(rs_col(cat, "Pre"), rs_col(cat, "Post")), paste(cat, c("Pre", "Post"), sep = "."))))
  ymax <- max(d$mean_auc, na.rm = TRUE); yspan <- diff(range(d$mean_auc, na.rm = TRUE))
  labs_df <- data.frame(x = seq_along(LAYER_ORDER), y = ymax + yspan * 0.07,
    lab = vapply(LAYER_ORDER, function(cat) {
      pq <- pqfun(cat); star <- sig_star(pq[["q"]])
      if (is.na(pq[["p"]])) "n.s."
      else sprintf("%sp=%.3f\n(q=%.3f)", if (nzchar(star)) paste0(star, " ") else "", pq[["p"]], pq[["q"]])
    }, character(1)))
  boxx <- sort(unique(d$xpos))                                       # 10 box centres (Pre,Post per region)
  main <- ggplot(d, aes(x = xpos, y = mean_auc, group = key, fill = key, color = key)) +
    geom_boxplot(width = 0.34, alpha = 0.25, outlier.shape = NA, linewidth = 0.6) +
    geom_point(aes(x = xj), size = 1.1, alpha = 0.5, shape = 16) +
    geom_text(data = labs_df, aes(x = x, y = y, label = lab), inherit.aes = FALSE,
              size = 3, family = "serif", lineheight = 0.85) +
    scale_fill_manual(values = keycols, guide = "none") +
    scale_color_manual(values = keycols, guide = "none") +
    scale_x_continuous(breaks = boxx, labels = rep(c("Pre", "Post"), length(LAYER_ORDER)),
                       expand = expansion(add = 0.4)) +
    labs(title = sprintf(title, MLAB), x = NULL, y = paste("Mean", MUNIT)) +
    panel_theme() +
    theme(axis.text.x = element_text(size = 8)) +
    coord_cartesian(ylim = c(NA, ymax + yspan * 0.19))
  # compact colour key (scientific-legend style): one swatch per region + name to the right
  ny <- length(LAYER_ORDER)
  key_df <- data.frame(ry = ny:1, region = unname(LAYER_LABELS[LAYER_ORDER]),
                       col = vapply(LAYER_ORDER, function(cat) rs_col(cat, "Post"), character(1)),
                       stringsAsFactors = FALSE)
  key <- ggplot(key_df) +
    geom_tile(aes(x = 0, y = ry, fill = col), width = 0.62, height = 0.62, color = NA) +
    scale_fill_identity() +
    geom_text(aes(x = 0.55, y = ry, label = region), hjust = 0, size = 2.9, family = "serif") +
    coord_fixed(ratio = 1, xlim = c(-0.4, 4.0), ylim = c(0.4, ny + 0.6), clip = "off") +
    theme_void()
  out_png(fname, main + key + patchwork::plot_layout(widths = c(6.5, 1)), 12, 6)
}

# Both arms: bracket = LMM session main effect (averaged over arm).
make_box_figure("qin_time_effect_boxplots.png",
                "Time effect — %s Pre vs Post per region (both arms, n=%s; LMM session p, q = BH-FDR / 5 regions)",
                NULL, time_pq, "35")
# Both arms, FDR survivors only (Interoception + Visual).
make_box_figure("qin_time_effect_boxplots_intero_visual.png",
                "Time effect — %s Pre vs Post (both arms, n=%s; LMM session p, q = BH-FDR / 5 regions)",
                NULL, time_pq, "35", regions = c("intero", "visual"))
# Placebo only (one panel per region): the per-region companion to the single-panel version
# below (the 12-network whole-cortex counterpart was dropped with the network analysis).
make_box_figure("qin_prepost_boxplots_placebo.png",
                "Retreat (placebo) time effect — %s Pre vs Post per region (placebo only, n=%s; LMM within-arm p, q = BH-FDR / 5 regions)",
                "placebo", function(cat) simple_pq(cat, "placebo"), "18")
# Placebo only (single panel): label = LMM placebo within-arm post-pre simple effect (the retreat effect).
make_box_single("qin_time_effect_boxplots_placebo.png",
                "Retreat (placebo) time effect — %s Pre vs Post, 5 regions (placebo only, n=18; LMM within-arm p, q = BH-FDR / 5 regions)",
                "placebo", function(cat) simple_pq(cat, "placebo"))
# Placebo, FDR survivors only (Exteroception + Visual).
make_box_figure("qin_time_effect_boxplots_placebo_extero_visual.png",
                "Retreat (placebo) time effect — %s Pre vs Post (placebo only, n=%s; LMM within-arm p, q = BH-FDR / 5 regions)",
                "placebo", function(cat) simple_pq(cat, "placebo"), "18", regions = c("extero", "visual"))
# Both arms (single panel): label = LMM session main effect (averaged over arm) — same as the multi-panel version.
make_box_single("qin_time_effect_boxplots_single.png",
                "Time effect — %s Pre vs Post, 5 regions (both arms, n=35; LMM session p, q = BH-FDR / 5 regions)",
                NULL, time_pq)

# ── ΔAUC by region: one panel, 5 categories, post−pre only (both arms) ──────────
dpre  <- as.data.frame(subj_means[subj_means$session == "ses-01", c("category", "subject", "mean_auc")])
dpost <- as.data.frame(subj_means[subj_means$session == "ses-02", c("category", "subject", "mean_auc")])
names(dpre)[3] <- "pre"; names(dpost)[3] <- "post"
dlt <- merge(dpre, dpost, by = c("category", "subject"))
dlt$delta <- dlt$post - dlt$pre
ord <- names(sort(tapply(dlt$delta, as.character(dlt$category), median), decreasing = TRUE))  # regions by median Δ
dlt$region_f <- factor(LAYER_LABELS[as.character(dlt$category)], levels = LAYER_LABELS[ord])
dlt$xpos <- as.numeric(dlt$region_f)
set.seed(42); dlt$xj <- dlt$xpos + runif(nrow(dlt), -0.14, 0.14)
REGION_COLORS <- c(Interoception = rs_col("intero", "Post"), Exteroception = rs_col("extero", "Post"),
                   Cognition = rs_col("mental", "Post"), Visual = rs_col("visual", "Post"),
                   Auditory = rs_col("auditory", "Post"))   # each region's saturated tone
star_df <- data.frame(xpos = seq_along(ord),
                      star = vapply(ord, function(cc) sig_star(time_pq(cc)[["q"]]), character(1)),
                      y = max(dlt$delta, na.rm = TRUE) + diff(range(dlt$delta, na.rm = TRUE)) * 0.05)
delta_region <- ggplot(dlt, aes(x = xpos, y = delta)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray55", linewidth = 0.5) +
  geom_point(aes(x = xj, color = region_f), size = 1.2, alpha = 0.55, shape = 16) +
  geom_boxplot(aes(color = region_f, fill = region_f, group = region_f), alpha = 0.15,
               width = 0.6, outlier.shape = NA, linewidth = 0.65) +
  geom_text(data = star_df, aes(x = xpos, y = y, label = star), inherit.aes = FALSE,
            size = 6, color = "gray20") +
  scale_color_manual(values = REGION_COLORS, guide = "none") +
  scale_fill_manual(values = REGION_COLORS, guide = "none") +
  scale_x_continuous(breaks = seq_along(ord), labels = levels(dlt$region_f), expand = expansion(add = 0.6)) +
  labs(title = sprintf("Δ%s (post − pre) by region — both arms, n=35", MLAB),
       subtitle = "ordered by median Δ · star = time effect survives BH-FDR (q<0.05, LMM session / 5 regions)",
       x = NULL, y = paste0("Δ", MLAB, " (post − pre", DUNIT, ")")) +
  panel_theme() +
  theme(plot.subtitle = element_text(hjust = 0.5, size = 8, family = "serif", color = "gray35"))
out_png("qin_delta_by_region.png", delta_region, 8, 5)

# ── MAIN FIGURE: per-region overall time effect (post−pre) — forest / coefficient plot ──
# Reads the LMM session main effect (averaged over arm) from _overall.csv (OVERALL_CSV set above).
REGION_LABELS <- c(intero = "Interoception", extero = "Exteroception", cognition = "Cognition",
                   visual = "Visual", auditory = "Auditory")
REG2CAT <- setNames(names(CAT2REG), CAT2REG)                  # region name -> layer key (cognition->mental)
# Generic per-region forest. fd needs columns: region, est, ci_low, ci_high, q.
# FDR survivors get their region colour + diamond + star; everything else is grey.
make_forest <- function(fd, fname, title, subtitle) {
  fd$label    <- REGION_LABELS[fd$region]
  fd$survives <- !is.na(fd$q) & fd$q < 0.05
  fd$star     <- vapply(fd$q, sig_star, character(1))
  fd$col      <- mapply(function(rg, sv) if (sv) rs_col(REG2CAT[[rg]], "Post") else "gray70",
                        fd$region, fd$survives)
  fd$region_f <- reorder(factor(fd$label), fd$est)           # ascending est → largest region at top
  xr  <- range(c(fd$ci_low, fd$ci_high, 0), na.rm = TRUE); pad <- diff(xr) * 0.14
  forest <- ggplot(fd, aes(x = est, y = region_f)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray55", linewidth = 0.5) +
    geom_errorbar(aes(xmin = ci_low, xmax = ci_high, color = col),
                  orientation = "y", width = 0.20, linewidth = 0.75) +
    geom_point(aes(color = col, shape = survives, size = survives)) +
    geom_text(aes(x = ci_high, label = star), hjust = -0.35, vjust = 0.78,
              size = 5, color = "gray20", family = "serif") +
    scale_color_identity(guide = "none") +
    scale_shape_manual(values = c(`FALSE` = 16, `TRUE` = 18), guide = "none") +
    scale_size_manual(values  = c(`FALSE` = 2.6, `TRUE` = 4.0), guide = "none") +
    scale_x_continuous(limits = c(xr[1] - pad * 0.3, xr[2] + pad),
                       expand = expansion(mult = c(0.02, 0.02))) +
    labs(title = title, subtitle = subtitle,
         x = sprintf("Δ%s (post − pre%s)", MLAB, DUNIT), y = NULL) +
    panel_theme() +
    theme(panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(color = "#e8e8e8", linewidth = 0.4),
          axis.text.y   = element_text(size = 11),
          plot.subtitle = element_text(hjust = 0.5, size = 8, family = "serif", color = "gray35"))
  out_png(fname, forest, 7, 4)
}

# Both arms: LMM session main effect (averaged over arm), from _overall.csv.
if (file.exists(OVERALL_CSV)) {
  ov <- read.csv(OVERALL_CSV, stringsAsFactors = FALSE)
  make_forest(data.frame(region = ov$region, est = ov$session_est,
                         ci_low = ov$session_ci_low, ci_high = ov$session_ci_high, q = ov$session_q),
              "qin_time_effect_forest.png",
              sprintf("Per-region time effect (post − pre) — %s, %s pipeline", MLAB, PIPELINE),
              "LMM session main effect (averaged over arm) · 95% CI · coloured ◆ + star = survives BH-FDR (q<0.05); grey = n.s.")
} else {
  cat(sprintf("Note: %s not found — time-effect forest skipped (run 03_mixed_models.R %s %s)\n",
              OVERALL_CSV, PIPELINE, ATLAS))
}

# Placebo only: LMM within-placebo post−pre simple effect (the retreat effect), from _simple.csv.
if (file.exists(SIMPLE_CSV)) {
  sm <- read.csv(SIMPLE_CSV, stringsAsFactors = FALSE); sm <- sm[sm$arm == "placebo", ]
  make_forest(data.frame(region = sm$region, est = sm$est,
                         ci_low = sm$ci_low, ci_high = sm$ci_high, q = sm$q),
              "qin_time_effect_forest_placebo.png",
              sprintf("Per-region retreat (placebo) time effect (post − pre) — %s, %s pipeline", MLAB, PIPELINE),
              "LMM within-placebo post−pre · 95% CI · coloured ◆ + star = survives BH-FDR (q<0.05); grey = n.s.")
} else {
  cat(sprintf("Note: %s not found — placebo forest skipped (run 03_mixed_models.R %s %s)\n",
              SIMPLE_CSV, PIPELINE, ATLAS))
}

# ── Baseline balance (placebo vs verum at ses-01) ──────────────────────────────
# ses-01 is all "pre", so per-layer panels use the region's PRE hue; arm is shown by
# fill (placebo = solid) vs outline (verum = hollow). The pooled panel has no single
# region, so it keeps a neutral placebo/verum two-tone.
make_base <- function(d_in, panel_title, cat = NULL) {
  d <- data.frame(group = factor(as.character(d_in$group), levels = c("placebo", "verum")),
                  value = d_in$value)
  s <- base_test(d$value[d$group == "placebo"], d$value[d$group == "verum"])
  stats_txt <- sprintf("t=%.2f (df=%.1f), p=%.3f\nU=%.0f, p=%.3f\nd=%+.2f",
                       s$t, s$df, s$p_t, s$U, s$p_mw, s$d)
  d <- add_x(d, "group")
  if (is.null(cat)) {                            # pooled — neutral two-tone, arm by colour
    colorv <- BASE_NEUTRAL; fillv <- BASE_NEUTRAL
    slab <- stat_halfeye(aes(fill = group), adjust = 0.7, width = 0.55, justification = -0.38,
                         .width = 0, point_colour = NA, slab_alpha = 0.55)
  } else {                                       # per-layer — region pre hue, arm by fill vs outline
    hue <- rs_col(cat, "Pre")
    colorv <- c(placebo = hue, verum = hue)
    fillv  <- c(placebo = hue, verum = "white")
    slab <- stat_halfeye(fill = hue, adjust = 0.7, width = 0.55, justification = -0.38,
                         .width = 0, point_colour = NA, slab_alpha = 0.45)
  }
  ggplot(d, aes(x = x_cond, y = value, color = group, group = group)) +
    geom_point(aes(x = x_jitter, fill = group), shape = 21, size = 1.5, alpha = 0.65, stroke = 0.4) +
    geom_boxplot(aes(fill = group), width = 0.90, outlier.shape = NA, linewidth = 0.65,
                 position = position_nudge(x = -0.4), alpha = 0.5) +
    slab +
    annotate("label", x = Inf, y = -Inf, label = stats_txt, hjust = 1.05, vjust = -0.08,
             size = 2.6, family = "serif", fill = "white", color = "gray30",
             label.padding = unit(0.15, "cm")) +
    scale_fill_manual(values = fillv, guide = "none") +
    scale_color_manual(values = colorv, guide = "none") +
    scale_x_continuous(breaks = (1:2) * 2, labels = c("Placebo", "Verum"),
                       expand = expansion(add = c(0.65, 1.00))) +
    labs(title = panel_title, y = paste("Mean", MUNIT), x = NULL) + panel_theme()
}
out_png("qin_QC_baseline_balance_pooled.png",
        make_base(ses01_subj(), "Baseline balance — all layers pooled (ses-01)"), 5, 6)
out_png("qin_QC_baseline_balance_per_layer.png",
        assemble6(lapply(LAYER_ORDER, function(cat) make_base(ses01_subj(cat), LAYER_LABELS[[cat]], cat)),
                  "Baseline balance by layer (ses-01)"), 14, 8)

# ── Paired pre→post lines per layer (within-group paired-t brackets) ───────────
XPOS_PAIRED <- c("Placebo Pre" = 1.0, "Placebo Post" = 2.0, "Verum Pre" = 3.2, "Verum Post" = 4.2)
make_paired <- function(cat) {
  cpre <- rs_col(cat, "Pre"); cpost <- rs_col(cat, "Post")   # colour encodes SESSION; arm is by x-position
  sess_cols <- c(Pre = cpre, Post = cpost)
  d <- as.data.frame(subj_means[as.character(subj_means$category) == cat, ])
  d$x_pos <- XPOS_PAIRED[as.character(d$condition)]
  d$sess  <- factor(ifelse(d$session == "ses-01", "Pre", "Post"), levels = c("Pre", "Post"))
  seg <- function(grp) merge(
    d[d$group == grp & d$session == "ses-01", c("subject", "mean_auc", "x_pos")],
    d[d$group == grp & d$session == "ses-02", c("subject", "mean_auc", "x_pos")],
    by = "subject", suffixes = c("_pre", "_post"))
  plac_segs <- seg("placebo"); verm_segs <- seg("verum")
  y_max <- max(d$mean_auc, na.rm = TRUE); y_span <- diff(range(d$mean_auc, na.rm = TRUE))
  p <- ggplot() +
    geom_segment(data = plac_segs, aes(x = x_pos_pre, xend = x_pos_post, y = mean_auc_pre, yend = mean_auc_post),
                 color = "gray55", alpha = 0.35, linewidth = 0.5) +
    geom_segment(data = verm_segs, aes(x = x_pos_pre, xend = x_pos_post, y = mean_auc_pre, yend = mean_auc_post),
                 color = "gray55", alpha = 0.35, linewidth = 0.5) +
    geom_point(data = d, aes(x = x_pos, y = mean_auc, color = sess),
               alpha = 0.60, size = 1.8, shape = 16) +
    geom_boxplot(data = d, aes(x = x_pos, y = mean_auc, group = condition, color = sess),
                 fill = NA, width = 0.22, outlier.shape = NA, linewidth = 0.55) +
    scale_color_manual(values = sess_cols, guide = "none") +
    scale_x_continuous(breaks = unname(XPOS_PAIRED),
                       labels = c("Placebo\nPre", "Placebo\nPost", "Verum\nPre", "Verum\nPost"),
                       expand = expansion(add = c(0.5, 0.5))) +
    labs(title = LAYER_LABELS[[cat]], y = paste("Mean", MUNIT), x = NULL) + panel_theme(title_face = "bold")
  y_br1 <- y_max + y_span * 0.05; y_t1 <- y_br1 - y_span * 0.02
  y_br2 <- y_br1 + y_span * 0.05; y_t2 <- y_br2 - y_span * 0.02
  # DiD bracket (LMM session:arm) spans placebo → verum, above the two within-group brackets
  y_brD <- y_br2 + y_span * 0.16; y_tD <- y_brD - y_span * 0.02
  did <- did_pq(cat); did_star <- sig_star(did[["q"]])
  did_lab <- if (is.na(did[["p"]])) "n.s."
             else sprintf("%sp=%.3f (q=%.3f)",
                          if (nzchar(did_star)) paste0(did_star, " ") else "", did[["p"]], did[["q"]])
  # within-arm post-pre brackets: LMM simple effects (KR df), q = BH-FDR across 5 regions within arm
  simple_lab <- function(arm) {
    s <- simple_pq(cat, arm); st <- sig_star(s[["q"]])
    if (is.na(s[["p"]])) "n.s."
    else sprintf("%sp=%.3f (q=%.3f)", if (nzchar(st)) paste0(st, " ") else "", s[["p"]], s[["q"]])
  }
  p +
    annotate("segment", x = 1.0, xend = 2.0, y = y_br1, yend = y_br1, color = "gray35", linewidth = 0.5) +
    annotate("text", x = 1.5, y = y_br1 + y_span * 0.02, label = simple_lab("placebo"),
             size = 3.7, hjust = 0.5, color = "gray35") +
    annotate("segment", x = 3.2, xend = 4.2, y = y_br2, yend = y_br2, color = "gray35", linewidth = 0.5) +
    annotate("text", x = 3.7, y = y_br2 + y_span * 0.02, label = simple_lab("verum"),
             size = 3.7, hjust = 0.5, color = "gray35") +
    annotate("segment", x = 1.5, xend = 3.7, y = y_brD, yend = y_brD, color = "gray20", linewidth = 0.5) +
    annotate("segment", x = 1.5, xend = 1.5, y = y_brD, yend = y_tD, color = "gray20", linewidth = 0.5) +
    annotate("segment", x = 3.7, xend = 3.7, y = y_brD, yend = y_tD, color = "gray20", linewidth = 0.5) +
    annotate("text", x = 2.6, y = y_brD + y_span * 0.03, label = did_lab,
             size = 3.7, hjust = 0.5, color = "gray20") +
    coord_cartesian(ylim = c(NA, y_brD + y_span * 0.11))
}
out_png("qin_retreat_paired_prepost.png",
        assemble6(lapply(LAYER_ORDER, make_paired),
                  sprintf("Pre → Post %s change by drug group (within-arm & DiD from LMM; q = BH-FDR / 5 regions)", MLAB)), 14, 9)

# ── AUC distribution (overall) ─────────────────────────────────────────────────
dist_theme <- function() theme_minimal(base_size = 12, base_family = "serif") +
  theme(panel.background = element_rect(fill = "white", color = NA),
        plot.background  = element_rect(fill = "white", color = NA),
        panel.grid.minor = element_blank(),
        plot.title = element_text(face = "plain", size = 13, hjust = 0.5))
all_auc <- df_raw$auc[is.finite(df_raw$auc)]
out_png("qin_overall_auc_histogram.png",
  ggplot(data.frame(auc = all_auc), aes(x = auc)) +
    geom_histogram(bins = 50, fill = "#8E5B9B", color = "white", linewidth = 0.2) +
    labs(title = sprintf("Overall %s distribution (%s %s)", MLAB, ATLAS, PIPELINE),
         x = paste0(MLAB, XUNIT), y = "Count") + dist_theme(), 8, 6)

# ── BOLD autocorrelation per parcel (single subject) ───────────────────────────
# One ACF curve per parcel for one subject/session, all on a single panel (muted rainbow).
# Reads the raw denoised timeseries (independent of METRIC) — only built on the auc pass.
if (METRIC == "auc") local({
  TR <- 1.8; N_DUMMY <- 6; MAX_LAG <- 15; XMAX <- 25   # 15 lags ≈ 27 s (axis clipped to 25)
  SUBJECT <- "01"; SESSION <- "ses-01"
  ts_dir <- file.path(REPO_ROOT, "02_timeseries_extraction", "results", ATLAS_DIR, PIPELINE)
  series <- list()
  for (layer in names(LAYER_LABELS)) {                  # all 6 layers (motor included here)
    f <- file.path(ts_dir, layer, sprintf("sub-%s_%s_%s_timeseries.csv", SUBJECT, SESSION, layer))
    if (!file.exists(f)) next
    m <- as.matrix(read.csv(f)[, -1, drop = FALSE])[-(1:N_DUMMY), , drop = FALSE]
    for (j in seq_len(ncol(m))) series[[sprintf("%s_%02d", layer, j)]] <- m[, j]
  }
  if (!length(series)) { cat("ACF: no timeseries under", ts_dir, "— skipping\n"); return(invisible()) }
  np <- length(series)
  acf_df <- do.call(rbind, lapply(names(series), function(nm)
    data.frame(parcel = nm, sec = (0:MAX_LAG) * TR,
               acf = acf(series[[nm]], lag.max = MAX_LAG, plot = FALSE)$acf[, 1, 1])))
  acf_df$parcel <- factor(acf_df$parcel, levels = names(series))
  out_png("qin_acf_single_subject.png",
    ggplot(acf_df, aes(sec, acf, group = parcel, color = parcel)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray60", linewidth = 0.4) +
      geom_line(alpha = 0.75, linewidth = 0.18) +
      scale_color_manual(values = rainbow(np, s = 0.50, v = 0.95), guide = "none") +   # muted rainbow
      coord_cartesian(xlim = c(0, XMAX)) +
      scale_x_continuous(expand = expansion(mult = c(0, 0.02))) +
      labs(title = sprintf("BOLD autocorrelation per parcel — sub-%s %s (%s %s, %d parcels)",
                           SUBJECT, SESSION, ATLAS, PIPELINE, np),
           x = "Lag (s)", y = "Autocorrelation") + dist_theme(), 8, 6)
})

# (The old motion-residualized drug-effect figure was removed — the LMM DiD in
# `03_mixed_models.R` already partials out pcf/pcf_sq/mean_fd, so the drug-effect figure
# above IS the covariate-adjusted effect.)

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
