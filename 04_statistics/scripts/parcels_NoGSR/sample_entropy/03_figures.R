# 03_figures.R — SampEn thesis figures, parcels / NoGSR (self-contained).
# Mirror of intrinsic_timescale/04_figures.R with AUC -> SampEn. Reads the frozen
# sampen_full_dataframe.csv (see README for provenance), drops the 58 low-tSNR parcels,
# computes its own base-R paired-t / baseline p-values (-> descriptive_pvalues.csv), and
# saves 12 PNGs (300 dpi) to sampen_figures/, named SampEn_<analysis>_<what>.png. See the
# folder README for the figure -> analysis map (selfVSnonself / retreat / drug / QC).
#   SampEn_drug_catavg_significance.png  reads catavg_drug_effect.csv     (run 01 first)
#     catavg_drug_effect.csv comes from the parcel-level LMM in 01_lmm_category_avg.R
#     (sampen ~ session * group * self_layer + (1|subject) + (1|parcel)); emmeans
#     marginalises over parcels within each category to give the per-category drug effect.
#   SampEn_drug_perparcel_forest.png     reads perparcel_drug_effect.csv  (run 02 first)
# Run from repo root:  Rscript 04_statistics/scripts/parcels_NoGSR/sample_entropy/03_figures.R

for (pkg in c("ggplot2", "ggdist", "patchwork", "dplyr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
}
suppressPackageStartupMessages({
  library(ggplot2); library(ggdist); library(patchwork); library(dplyr)
})

# ── Paths (repo root from this script's location: works for Rscript and source()) ──
script_dir <- function() {
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa)) return(dirname(normalizePath(sub("^--file=", "", fa[1]), winslash = "/")))
  for (i in rev(seq_len(sys.nframe()))) {
    of <- get0("ofile", envir = sys.frame(i), inherits = FALSE)
    if (!is.null(of)) return(dirname(normalizePath(of, winslash = "/")))
  }
  normalizePath(getwd(), winslash = "/")
}
REPO_ROOT   <- normalizePath(file.path(script_dir(), "..", "..", "..", ".."), winslash = "/", mustWork = FALSE)
ATLAS_LABEL <- "Glasser self ROIs"
TABLES_DIR  <- file.path(REPO_ROOT, "04_statistics", "results", "parcels_NoGSR", "sampen_tables")
FIGS_DIR    <- file.path(REPO_ROOT, "04_statistics", "results", "parcels_NoGSR", "sampen_figures")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

DATA_CSV <- file.path(TABLES_DIR, "sampen_full_dataframe.csv")
TSNR_TSV <- file.path(REPO_ROOT, "excluded_rois_low_tsnr.tsv")
if (!file.exists(DATA_CSV)) stop("Input not found — provide sampen_full_dataframe.csv (frozen artifact; see README): ", DATA_CSV)

SEP <- strrep("=", 70)
out_png <- function(name, gg, w, h) {
  path <- file.path(FIGS_DIR, name)
  ggsave(path, gg, width = w, height = h, dpi = 300, bg = "white")
  cat(sprintf("Saved: %s\n", path))
}
cat(SEP, "\n03_figures.R — parcels_NoGSR\n", SEP, "\n\n", sep = "")

# ── Constants ────────────────────────────────────────────────────────────────────
COND_LEVELS  <- c("Placebo Pre", "Verum Pre", "Placebo Post", "Verum Post")
COND_LABELS  <- c("Placebo\nPre", "Verum\nPre", "Placebo\nPost", "Verum\nPost")
COND_COLORS  <- c("Placebo Pre" = "#CD5C5C", "Verum Pre" = "#4682B4",
                  "Placebo Post" = "#E8963E", "Verum Post" = "#2E8B8B")
CAT_MAP      <- c(Interoception = "Interoceptive Self", Exteroception = "Exteroceptive Self",
                  Cognition = "Mental Self", somatomotor = "Sensory-Motor")
CAT_ORDER    <- c("Interoceptive Self", "Exteroceptive Self", "Mental Self", "Sensory-Motor")
DELTA_COLORS   <- c(placebo = "#CD5C5C", verum = "#4682B4")
SESSION_COLORS <- c("ses-01" = "#CD5C5C", "ses-02" = "#E8963E")
BASE_COLORS    <- c(placebo = "#CD5C5C", verum = "#4682B4")

# ── Shared helpers ───────────────────────────────────────────────────────────────
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
  geom_point(aes(x = x_jitter), size = 1.5, alpha = 0.60, shape = 16),
  geom_boxplot(fill = NA, width = 0.90, outlier.shape = NA, linewidth = 0.70,
               position = position_nudge(x = -0.4)),
  stat_halfeye(adjust = 0.7, width = 0.55, justification = -0.38, .width = 0,
               point_colour = NA, slab_alpha = 0.65))

add_x <- function(d, col, jit = 0.10) {
  set.seed(42)
  d$x_cond   <- as.numeric(d[[col]]) * 2
  d$x_jitter <- d$x_cond - 0.40 + runif(nrow(d), -jit, jit)
  d
}

assemble <- function(panels, title)
  ((panels[[1]] | panels[[2]]) / (panels[[3]] | panels[[4]])) +
  plot_annotation(title = title,
    theme = theme(plot.title = element_text(face = "plain", hjust = 0.5,
                                            size = 12, family = "serif")))

sig_bracket <- function(p, x1, x2, yvals, label, color = "black", lab_size = 3) {
  y_max <- max(yvals, na.rm = TRUE); y_span <- diff(range(yvals, na.rm = TRUE))
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
  both <- merge(d[d$group == grp & d$session == "ses-01", c("subject", "mean_sampen")],
                d[d$group == grp & d$session == "ses-02", c("subject", "mean_sampen")],
                by = "subject", suffixes = c("_pre", "_post"))
  if (nrow(both) < 3) return(c(n = nrow(both), t = NA, p = NA))
  tt <- t.test(both$mean_sampen_post, both$mean_sampen_pre, paired = TRUE)
  c(n = nrow(both), t = unname(tt$statistic), p = tt$p.value)
}
base_test <- function(plac, verm) {  # placebo vs verum at baseline
  t_r <- t.test(plac, verm, var.equal = FALSE); mw <- wilcox.test(plac, verm, exact = FALSE)
  list(n = length(plac) + length(verm), t = unname(t_r$statistic), df = unname(t_r$parameter),
       p_t = t_r$p.value, U = unname(mw$statistic), p_mw = mw$p.value, d = cohens_d(plac, verm))
}

# ── Load, drop low-tSNR parcels, prepare ─────────────────────────────────────────
df_raw <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded sampen_full_dataframe: %d rows\n", nrow(df_raw)))

excl <- read.delim(TSNR_TSV, stringsAsFactors = FALSE)$roi_id
n0 <- nrow(df_raw); df_raw <- df_raw[!(df_raw$roi_pos_id %in% excl), ]
cat(sprintf("Dropped %d rows from %d low-tSNR parcels (%d -> %d)\n",
            n0 - nrow(df_raw), length(excl), n0, nrow(df_raw)))

df_raw$self_layer <- relevel(factor(df_raw$self_layer), ref = "somatomotor")
df_raw$category   <- CAT_MAP[as.character(df_raw$self_layer)]
df_raw$atlas      <- ifelse(df_raw$self_layer == "somatomotor", "nonself", "self")
df_raw$group      <- relevel(factor(df_raw$drug_group), ref = "placebo")
df_raw$session    <- relevel(factor(df_raw$session), ref = "ses-01")
df_raw$condition  <- factor(
  paste0(ifelse(df_raw$group == "placebo", "Placebo", "Verum"), " ",
         ifelse(df_raw$session == "ses-01", "Pre", "Post")), levels = COND_LEVELS)

subj_means <- df_raw %>%
  group_by(category, subject, group, session, condition) %>%
  summarise(mean_sampen = mean(sampen, na.rm = TRUE), .groups = "drop")
subj_means$condition <- factor(subj_means$condition, levels = COND_LEVELS)

ses01_df <- df_raw[as.character(df_raw$session) == "ses-01", ]
ses01_subj <- function(cat = NULL) {
  d <- if (is.null(cat)) ses01_df else ses01_df[ses01_df$category == cat, ]
  d %>% group_by(subject, group) %>% summarise(value = mean(sampen, na.rm = TRUE), .groups = "drop")
}

# ── Compute p-values and write figure_pvalues.csv ────────────────────────────────
cat("\n", SEP, "\nComputing p-values (base-R)\n", SEP, "\n", sep = "")
pair_stats <- do.call(rbind, lapply(CAT_ORDER, function(cat) {
  d <- subj_means[subj_means$category == cat, ]
  do.call(rbind, lapply(c("placebo", "verum"), function(g) {
    s <- paired_test(d, g)
    data.frame(category = cat, test = paste0(g, "_prepost_pairedt"),
               n = s["n"], statistic = s["t"], p = s["p"], row.names = NULL)
  }))
}))
pp <- function(cat, grp) {  # paired p lookup for the bracket annotations
  r <- pair_stats[pair_stats$category == cat & pair_stats$test == paste0(grp, "_prepost_pairedt"), ]
  if (nrow(r)) r$p[1] else NA
}
base_stats <- do.call(rbind, lapply(c("Pooled", CAT_ORDER), function(lab) {
  d <- if (lab == "Pooled") ses01_subj() else ses01_subj(lab)
  s <- base_test(d$value[d$group == "placebo"], d$value[d$group == "verum"])
  data.frame(category = lab,
             test      = c("baseline_welch_t", "baseline_mannwhitney", "baseline_cohens_d"),
             n         = s$n,
             statistic = c(s$t, s$U, s$d),
             p         = c(s$p_t, s$p_mw, NA), row.names = NULL)
}))
pvalues <- rbind(pair_stats, base_stats)
write.csv(pvalues, file.path(TABLES_DIR, "descriptive_pvalues.csv"), row.names = FALSE)
cat(sprintf("Saved: %s (%d rows)\n", file.path(TABLES_DIR, "descriptive_pvalues.csv"), nrow(pvalues)))
print(pvalues, digits = 4)

# ── Figure: raincloud per category (4 conditions) ────────────────────────────────
make_panel <- function(cat_label) {
  d <- add_x(subj_means[subj_means$category == cat_label, ], "condition", jit = 0.2)
  ggplot(d, aes(x = x_cond, y = mean_sampen, fill = condition, color = condition, group = condition)) +
    rain_layers() +
    stat_summary(fun = mean, geom = "point", shape = 18, size = 2, color = "black",
                 position = position_nudge(x = -0.4)) +
    scale_fill_manual(values = COND_COLORS, guide = "none") +
    scale_color_manual(values = COND_COLORS, guide = "none") +
    scale_x_continuous(breaks = (1:4) * 2, labels = COND_LABELS, expand = expansion(add = c(0.65, 1.00))) +
    labs(title = cat_label, y = "Mean SampEn", x = NULL) + panel_theme()
}
out_png("SampEn_overview_raincloud_4conditions.png",
        assemble(lapply(CAT_ORDER, make_panel),
                 sprintf("SampEn per condition and category (%s)", ATLAS_LABEL)), 10, 8)

# ── Figure: delta plot (post − pre per drug group; no bracket) ───────────────────
pre_df  <- subj_means[subj_means$session == "ses-01", c("category", "subject", "group", "mean_sampen")]
post_df <- subj_means[subj_means$session == "ses-02", c("category", "subject", "group", "mean_sampen")]
names(pre_df)[4] <- "pre_sampen"; names(post_df)[4] <- "post_sampen"
delta_df <- merge(pre_df, post_df, by = c("category", "subject", "group"))
delta_df$delta <- delta_df$post_sampen - delta_df$pre_sampen
delta_df$group <- factor(delta_df$group, levels = c("placebo", "verum"))

make_delta_panel <- function(cat_label) {
  d <- add_x(delta_df[delta_df$category == cat_label, ], "group")
  ggplot(d, aes(x = x_cond, y = delta, fill = group, color = group, group = group)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
    rain_layers() +
    scale_fill_manual(values = DELTA_COLORS, guide = "none") +
    scale_color_manual(values = DELTA_COLORS, guide = "none") +
    scale_x_continuous(breaks = (1:2) * 2, labels = c("Placebo", "Verum"), expand = expansion(add = c(0.65, 1.00))) +
    labs(title = cat_label, y = "ΔSampEn (post − pre)", x = NULL) + panel_theme()
}
out_png("SampEn_drug_delta_per_category.png",
        assemble(lapply(CAT_ORDER, make_delta_panel),
                 sprintf("Post − Pre SampEn change per category (%s)", ATLAS_LABEL)), 10, 8)

# ── Figure: delta plot with the DRUG EFFECT (group x session) significance ────────
# One bracket per panel between placebo and verum, labelled from the per-category
# drug-effect emmeans contrast (01_lmm_category_avg.R -> catavg_drug_effect.csv).
# The LMM is parcel-level (1|subject)+(1|parcel); emmeans marginalises over parcels.
# Run 01 first.
LMM_DRUG_CSV <- file.path(TABLES_DIR, "catavg_drug_effect.csv")
if (file.exists(LMM_DRUG_CSV)) {
  drug_tbl <- read.csv(LMM_DRUG_CSV, stringsAsFactors = FALSE)
  drug_tbl$category <- CAT_MAP[as.character(drug_tbl$self_layer)]
  # raw p and FDR-corrected q (p_fdr written by 01_lmm_category_avg.R, family = 4 layers);
  # fall back to raw only if an older table without p_fdr is present.
  drug_pq <- function(cat_label) {
    r <- drug_tbl[drug_tbl$category == cat_label, ]
    if (!nrow(r)) return(c(p = NA_real_, q = NA_real_))
    c(p = r$p[1], q = if ("p_fdr" %in% names(r)) r$p_fdr[1] else NA_real_)
  }
  make_delta_sig_panel <- function(cat_label) {
    d <- add_x(delta_df[delta_df$category == cat_label, ], "group")
    p <- ggplot(d, aes(x = x_cond, y = delta, fill = group, color = group, group = group)) +
      geom_hline(yintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.5) +
      rain_layers() +
      scale_fill_manual(values = DELTA_COLORS, guide = "none") +
      scale_color_manual(values = DELTA_COLORS, guide = "none") +
      scale_x_continuous(breaks = (1:2) * 2, labels = c("Placebo", "Verum"), expand = expansion(add = c(0.65, 1.00))) +
      labs(title = cat_label, y = "ΔSampEn (post − pre)", x = NULL) + panel_theme()
    pq <- drug_pq(cat_label); star <- sig_star(pq[["q"]])   # star reflects FDR q, not raw p
    lab <- if (is.na(pq[["p"]])) "n/a"
           else sprintf("%sp=%.3f (q=%.3f)", if (star != "") paste0(star, " ") else "",
                        pq[["p"]], pq[["q"]])
    sig_bracket(p, 1.6, 3.6, d$delta, lab)   # placebo (box ~1.6) vs verum (box ~3.6)
  }
  out_png("SampEn_drug_catavg_significance.png",
          assemble(lapply(CAT_ORDER, make_delta_sig_panel),
                   sprintf("Drug effect (group × session) per category — q = FDR across 4 layers (%s)", ATLAS_LABEL)), 10, 8)
} else {
  cat(sprintf("Skipped drug-effect plot - run 01_lmm_category_avg.R first (missing %s)\n", LMM_DRUG_CSV))
}

# ── Figure: retreat effect (placebo group, pre vs post; paired-t bracket) ────────
plac_means <- subj_means[subj_means$group == "placebo", ]
make_retreat_panel <- function(cat_label) {
  d <- plac_means[plac_means$category == cat_label, ]
  d$session_f <- factor(as.character(d$session), levels = c("ses-01", "ses-02"))
  d <- add_x(d, "session_f")
  p <- ggplot(d, aes(x = x_cond, y = mean_sampen, fill = session_f, color = session_f, group = session_f)) +
    rain_layers() +
    scale_fill_manual(values = SESSION_COLORS, guide = "none") +
    scale_color_manual(values = SESSION_COLORS, guide = "none") +
    scale_x_continuous(breaks = (1:2) * 2, labels = c("Pre", "Post"), expand = expansion(add = c(0.65, 1.00))) +
    labs(title = cat_label, y = "Mean SampEn", x = NULL) + panel_theme()
  sig_bracket(p, 1.6, 3.6, d$mean_sampen, fmt_p(pp(cat_label, "placebo")))
}
out_png("SampEn_retreat_per_category.png",
        assemble(lapply(CAT_ORDER, make_retreat_panel),
                 "Meditation retreat effect (placebo group)"), 10, 8)

# ── Figure: baseline balance (placebo vs verum at ses-01) ────────────────────────
make_base_panel <- function(d_in, panel_title) {
  d    <- data.frame(group = factor(as.character(d_in$group), levels = c("placebo", "verum")),
                     value = d_in$value)
  s    <- base_test(d$value[d$group == "placebo"], d$value[d$group == "verum"])
  stats_txt <- sprintf("t=%.3f (df=%.1f), p=%.3f\nU=%.0f, p=%.3f\nd=%+.3f",
                       s$t, s$df, s$p_t, s$U, s$p_mw, s$d)
  d <- add_x(d, "group")
  ggplot(d, aes(x = x_cond, y = value, fill = group, color = group, group = group)) +
    rain_layers() +
    annotate("label", x = Inf, y = -Inf, label = stats_txt, hjust = 1.05, vjust = -0.08,
             size = 2.8, family = "serif", fill = "white", color = "gray30",
             label.padding = unit(0.15, "cm")) +
    scale_fill_manual(values = BASE_COLORS, guide = "none") +
    scale_color_manual(values = BASE_COLORS, guide = "none") +
    scale_x_continuous(breaks = (1:2) * 2, labels = c("Placebo", "Verum"), expand = expansion(add = c(0.65, 1.00))) +
    labs(title = panel_title, y = "Mean SampEn", x = NULL) + panel_theme()
}
out_png("SampEn_QC_baseline_balance_pooled.png",
        make_base_panel(ses01_subj(), "Baseline balance — all regions pooled (ses-01)"), 5, 6)
out_png("SampEn_QC_baseline_balance_per_category.png",
        assemble(lapply(CAT_ORDER, function(cat) make_base_panel(ses01_subj(cat), cat)),
                 "Baseline balance by category (ses-01)"), 10, 8)

# ── Figure: paired pre→post lines per category (within-group paired-t brackets) ──
XPOS_PAIRED <- c("Placebo Pre" = 1.0, "Placebo Post" = 2.0, "Verum Pre" = 3.2, "Verum Post" = 4.2)
make_paired_panel <- function(cat_label) {
  d <- subj_means[subj_means$category == cat_label, ]
  d$x_pos <- XPOS_PAIRED[as.character(d$condition)]
  seg <- function(grp) merge(
    d[d$group == grp & d$session == "ses-01", c("subject", "mean_sampen", "x_pos")],
    d[d$group == grp & d$session == "ses-02", c("subject", "mean_sampen", "x_pos")],
    by = "subject", suffixes = c("_pre", "_post"))
  plac_segs <- seg("placebo"); verm_segs <- seg("verum")
  y_max <- max(d$mean_sampen, na.rm = TRUE); y_span <- diff(range(d$mean_sampen, na.rm = TRUE))
  p <- ggplot() +
    geom_segment(data = plac_segs, aes(x = x_pos_pre, xend = x_pos_post, y = mean_sampen_pre, yend = mean_sampen_post),
                 color = "#CD5C5C", alpha = 0.30, linewidth = 0.5) +
    geom_segment(data = verm_segs, aes(x = x_pos_pre, xend = x_pos_post, y = mean_sampen_pre, yend = mean_sampen_post),
                 color = "#4682B4", alpha = 0.30, linewidth = 0.5) +
    geom_point(data = d[d$group == "placebo", ], aes(x = x_pos, y = mean_sampen),
               color = "#CD5C5C", alpha = 0.60, size = 2, shape = 16) +
    geom_point(data = d[d$group == "verum", ], aes(x = x_pos, y = mean_sampen),
               color = "#4682B4", alpha = 0.60, size = 2, shape = 16) +
    geom_boxplot(data = d[d$group == "placebo", ], aes(x = x_pos, y = mean_sampen, group = condition),
                 fill = NA, color = "#CD5C5C", width = 0.22, outlier.shape = NA, linewidth = 0.60) +
    geom_boxplot(data = d[d$group == "verum", ], aes(x = x_pos, y = mean_sampen, group = condition),
                 fill = NA, color = "#4682B4", width = 0.22, outlier.shape = NA, linewidth = 0.60) +
    scale_x_continuous(breaks = unname(XPOS_PAIRED),
                       labels = c("Placebo\nPre", "Placebo\nPost", "Verum\nPre", "Verum\nPost"),
                       expand = expansion(add = c(0.5, 0.5))) +
    labs(title = cat_label, y = "Mean SampEn", x = NULL) + panel_theme(title_face = "bold")
  y_br1 <- y_max + y_span * 0.05; y_t1 <- y_br1 - y_span * 0.02
  y_br2 <- y_br1 + y_span * 0.05; y_t2 <- y_br2 - y_span * 0.02
  p +
    annotate("segment", x = 1.0, xend = 2.0, y = y_br1, yend = y_br1, color = "#CD5C5C", linewidth = 0.5) +
    annotate("segment", x = 1.0, xend = 1.0, y = y_br1, yend = y_t1,  color = "#CD5C5C", linewidth = 0.5) +
    annotate("segment", x = 2.0, xend = 2.0, y = y_br1, yend = y_t1,  color = "#CD5C5C", linewidth = 0.5) +
    annotate("text", x = 1.5, y = y_br1 + y_span * 0.02, label = fmt_p(pp(cat_label, "placebo")),
             size = 3, hjust = 0.5, color = "#CD5C5C") +
    annotate("segment", x = 3.2, xend = 4.2, y = y_br2, yend = y_br2, color = "#4682B4", linewidth = 0.5) +
    annotate("segment", x = 3.2, xend = 3.2, y = y_br2, yend = y_t2,  color = "#4682B4", linewidth = 0.5) +
    annotate("segment", x = 4.2, xend = 4.2, y = y_br2, yend = y_t2,  color = "#4682B4", linewidth = 0.5) +
    annotate("text", x = 3.7, y = y_br2 + y_span * 0.02, label = fmt_p(pp(cat_label, "verum")),
             size = 3, hjust = 0.5, color = "#4682B4") +
    coord_cartesian(ylim = c(NA, y_br2 + y_span * 0.10))
}
out_png("SampEn_retreat_paired_prepost.png",
        assemble(lapply(CAT_ORDER, make_paired_panel), "Pre → Post SampEn change by drug group"), 10, 10)

# ══ SampEn distribution QC ═══════════════════════════════════════════════════════
skewness <- function(x) { x <- x[is.finite(x)]; n <- length(x)
  if (n < 3) return(NA); s <- sd(x); if (s == 0) NA else (sum((x - mean(x))^3) / n) / s^3 }
kurt_excess <- function(x) { x <- x[is.finite(x)]; n <- length(x)
  if (n < 4) return(NA); s <- sd(x); if (s == 0) NA else (sum((x - mean(x))^4) / n) / s^4 - 3 }
print_summary <- function(label, vals) {
  finite <- vals[is.finite(vals)]; qs <- quantile(finite, c(0, .25, .5, .75, 1))
  cat(sprintf("  %s  N=%d  Missing/NaN=%d\n", label, length(finite), sum(is.na(vals) | is.nan(vals))))
  cat(sprintf("    Median=%.4f  Mean=%.4f SD=%.4f  Skew=%.4f Kurt=%.4f\n",
              qs[3], mean(finite), sd(finite), skewness(finite), kurt_excess(finite)))
}
dist_theme <- function() theme_minimal(base_size = 12, base_family = "serif") +
  theme(panel.background = element_rect(fill = "white", color = NA),
        plot.background  = element_rect(fill = "white", color = NA),
        panel.grid.minor = element_blank(),
        plot.title = element_text(face = "plain", size = 13, hjust = 0.5))

self_v <- df_raw$sampen[df_raw$atlas == "self"]; nonself_v <- df_raw$sampen[df_raw$atlas == "nonself"]
print_summary("Self", self_v); print_summary("Nonself", nonself_v)
plot_df1 <- data.frame(
  sampen = c(self_v[is.finite(self_v)], nonself_v[is.finite(nonself_v)]),
  atlas = factor(c(rep("Self", sum(is.finite(self_v))), rep("Nonself", sum(is.finite(nonself_v)))),
                 levels = c("Self", "Nonself")))
COLORS1 <- c(Self = "#123434", Nonself = "#2E8B8B")
LABELS1 <- c(Self = sprintf("Self (N=%d)", sum(is.finite(self_v))),
             Nonself = sprintf("Nonself (N=%d)", sum(is.finite(nonself_v))))
out_png("SampEn_selfVSnonself_density.png",
  ggplot(plot_df1, aes(x = sampen, fill = atlas, color = atlas)) +
    geom_density(alpha = 0.20, adjust = 0.8, linewidth = 1.0) +
    scale_fill_manual(values = COLORS1, labels = LABELS1, name = NULL) +
    scale_color_manual(values = COLORS1, labels = LABELS1, name = NULL) +
    labs(title = sprintf("SampEn distribution — self vs nonself (%s)", ATLAS_LABEL),
         x = "SampEn", y = "Density") +
    dist_theme() +
    theme(legend.position = c(0.82, 0.85),
          legend.background = element_rect(fill = "white", color = "gray80", linewidth = 0.3)),
  8, 6)

sampen_all <- df_raw$sampen[is.finite(df_raw$sampen)]
print_summary("All parcels (pooled)", df_raw$sampen)
out_png("SampEn_selfVSnonself_pooled_density.png",
  ggplot(data.frame(sampen = sampen_all), aes(x = sampen)) +
    geom_density(fill = "#2E8B8B", color = "#2E8B8B", alpha = 0.50, adjust = 0.8, linewidth = 0.7) +
    geom_rug(data = data.frame(sampen = sample(sampen_all, min(500, length(sampen_all)))),
             color = "#2E8B8B", alpha = 0.25, linewidth = 0.3) +
    labs(title = sprintf("SampEn distribution — all %s (pooled)", ATLAS_LABEL),
         x = "SampEn", y = "Density") + dist_theme(),
  8, 6)

GROUP_COLORS3 <- c(placebo = "#CD5C5C", verum = "#4682B4")
make_subject_panel <- function(atlas_val, session_val, panel_title, show_legend = FALSE) {
  d <- df_raw[df_raw$atlas == atlas_val & as.character(df_raw$session) == session_val & is.finite(df_raw$sampen), ]
  d$subject_f  <- factor(d$subject, levels = names(sort(tapply(d$sampen, d$subject, median, na.rm = TRUE))))
  d$drug_group <- factor(d$drug_group, levels = c("placebo", "verum"))
  ggplot(d, aes(x = subject_f, y = sampen, fill = drug_group)) +
    geom_hline(yintercept = median(d$sampen, na.rm = TRUE), linetype = "dashed",
               color = "gray50", linewidth = 0.5) +
    geom_boxplot(outlier.size = 0.6, outlier.alpha = 0.4, linewidth = 0.35, width = 0.75) +
    scale_fill_manual(values = GROUP_COLORS3, name = NULL,
                      labels = c(placebo = "Placebo", verum = "Verum")) +
    scale_x_discrete(labels = function(x) sub("sub-0?", "", x)) +
    labs(title = panel_title, y = "SampEn", x = NULL) +
    theme_minimal(base_size = 9, base_family = "serif") +
    theme(panel.background = element_rect(fill = "white", color = NA),
          plot.background  = element_rect(fill = "white", color = NA),
          panel.grid.minor = element_blank(), panel.grid.major.x = element_blank(),
          axis.text.x  = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 7),
          axis.title.y = element_text(size = 9),
          plot.title   = element_text(face = "plain", hjust = 0.5, size = 10, family = "serif"),
          legend.position = if (show_legend) "bottom" else "none")
}
subj_panels <- list(
  make_subject_panel("self",    "ses-01", "Self — Pre (ses-01)"),
  make_subject_panel("self",    "ses-02", "Self — Post (ses-02)", show_legend = TRUE),
  make_subject_panel("nonself", "ses-01", "Nonself — Pre (ses-01)"),
  make_subject_panel("nonself", "ses-02", "Nonself — Post (ses-02)"))
out_png("SampEn_selfVSnonself_per_subject.png",
        assemble(subj_panels, "Per-subject SampEn distribution by atlas × session"), 12, 8)

# ── Figure: per-parcel Group × Session drug effect (forest plot) ─────────────
PARCEL_GXS_CSV <- file.path(TABLES_DIR, "perparcel_drug_effect.csv")
if (file.exists(PARCEL_GXS_CSV)) {
  pp_res <- read.csv(PARCEL_GXS_CSV, stringsAsFactors = FALSE)
  # nonself parcels use "nonself" as self_layer key, not "somatomotor"
  PARCEL_LAYER_MAP <- c(Interoception = "Interoceptive Self", Exteroception = "Exteroceptive Self",
                        Cognition = "Mental Self", nonself = "Sensory-Motor",
                        somatomotor = "Sensory-Motor")
  pp_res$category <- PARCEL_LAYER_MAP[pp_res$self_layer]
  pp_res <- pp_res[!is.na(pp_res$category) & pp_res$status %in% c("ok", "singular", "warning"), ]
  pp_res$ci_lo <- pp_res$estimate - 1.96 * pp_res$SE
  pp_res$ci_hi <- pp_res$estimate + 1.96 * pp_res$SE
  # Significance tiers: raw p<0.05 and Benjamini-Hochberg FDR q<0.05 (p_fdr from 02_lmm_per_parcel.R)
  pp_res$sig_raw <- !is.na(pp_res$p_value) & pp_res$p_value < 0.05
  pp_res$sig_fdr <- if ("p_fdr" %in% names(pp_res)) (!is.na(pp_res$p_fdr) & pp_res$p_fdr < 0.05) else FALSE
  pp_res$sig_level <- factor(
    ifelse(pp_res$sig_fdr, "FDR q<0.05", ifelse(pp_res$sig_raw, "raw p<0.05", "n.s.")),
    levels = c("n.s.", "raw p<0.05", "FDR q<0.05"))

  n_sig_tbl <- pp_res %>%
    group_by(category) %>%
    summarise(n_raw = sum(sig_raw, na.rm = TRUE), n_fdr = sum(sig_fdr, na.rm = TRUE),
              n_tot = n(), .groups = "drop")
  cat("\nPer-parcel drug effect — significant parcels (raw p<0.05 / FDR q<0.05):\n")
  print(n_sig_tbl, row.names = FALSE)

  pp_res <- left_join(pp_res, n_sig_tbl, by = "category")
  flab <- function(cat, nr, nf, nt) sprintf("%s  (raw %d, FDR %d / %d)", cat, nr, nf, nt)
  pp_res$facet_label <- flab(pp_res$category, pp_res$n_raw, pp_res$n_fdr, pp_res$n_tot)
  facet_levels <- sapply(CAT_ORDER, function(cat) {
    r <- n_sig_tbl[n_sig_tbl$category == cat, ]
    if (nrow(r) == 0) return(NA_character_)
    flab(cat, r$n_raw, r$n_fdr, r$n_tot)
  })
  pp_res$facet_label <- factor(pp_res$facet_label, levels = facet_levels[!is.na(facet_levels)])

  # Sort parcels by estimate within each category for the y-axis
  # order by (category, estimate); make.unique guards against repeated parcel names
  pp_res <- pp_res[order(pp_res$category, pp_res$estimate), ]
  pp_res$roi_label <- factor(make.unique(pp_res$roi_name), levels = make.unique(pp_res$roi_name))

  p_parcel <- ggplot(pp_res, aes(x = estimate, y = roi_label, color = sig_level)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray50", linewidth = 0.4) +
    geom_errorbarh(aes(xmin = ci_lo, xmax = ci_hi), height = 0.3, linewidth = 0.35, alpha = 0.7) +
    geom_point(size = 1.5) +
    scale_color_manual(values = c("n.s." = "gray70", "raw p<0.05" = "#E8963E", "FDR q<0.05" = "#B22222"),
                       name = NULL, drop = FALSE) +
    facet_wrap(~ facet_label, ncol = 2, scales = "free_y") +
    labs(title = "Per-parcel drug effect (Group × Session, verum vs placebo)",
         subtitle = "Orange = raw p<0.05; red = Benjamini-Hochberg FDR q<0.05; error bars = 95% CI",
         x = "Estimate (verum − placebo, post − pre difference)", y = NULL) +
    theme_minimal(base_size = 8, base_family = "serif") +
    theme(panel.background   = element_rect(fill = "white", color = NA),
          plot.background    = element_rect(fill = "white", color = NA),
          panel.grid.major.y = element_blank(),
          panel.grid.major.x = element_line(color = "#e8e8e8", linewidth = 0.35),
          panel.grid.minor   = element_blank(),
          strip.text   = element_text(face = "plain", size = 9, family = "serif"),
          axis.text.y  = element_text(size = 6),
          legend.position = "top",
          plot.title   = element_text(face = "plain", size = 11, hjust = 0.5, family = "serif"),
          plot.subtitle = element_text(size = 8, hjust = 0.5, color = "gray40"))
  out_png("SampEn_drug_perparcel_forest.png", p_parcel, 12, 16)
} else {
  cat(sprintf("Skipped per-parcel drug-effect plot — run 02_lmm_per_parcel.R first\n"))
}

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
