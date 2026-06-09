# 04_qc_auc_distribution.R — AUC distribution plots
#
# Plot 1: self vs nonself density overlay
# Plot 2: all Glasser parcels pooled (single density + rug)
# Plot 3: per-subject AUC boxplots (2×2: atlas × session)
#
# Run from repo root:
#   Rscript 04_statistics/scripts/04_qc_auc_distribution.R

suppressPackageStartupMessages(library(ggplot2))
if (!requireNamespace("RcppTOML", quietly = TRUE))
  install.packages("RcppTOML", repos = "https://cloud.r-project.org", quiet = TRUE)
suppressPackageStartupMessages(library(RcppTOML))

# ── Paths ──────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
  REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
} else {
  REPO_ROOT  <- normalizePath(".")
}

# ── Config ─────────────────────────────────────────────────────────────────────
cfg              <- parseTOML(file.path(REPO_ROOT, "config.toml"))
ATLAS_METHOD     <- cfg$active$atlas_method
DENOISING_METHOD <- cfg$active$denoising_method

if (ATLAS_METHOD == "spheres")
  stop("Sphere pipeline not yet implemented in refactored statistics.")

TAG <- paste0(ATLAS_METHOD, "_", DENOISING_METHOD)

TABLES_DIR <- file.path(REPO_ROOT, "04_statistics", "results", TAG, "tables")
FIGS_DIR   <- file.path(REPO_ROOT, "04_statistics", "results", TAG, "figures")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

CSV1 <- file.path(TABLES_DIR, "analysis_long_format_auc.csv")
CSV2 <- file.path(TABLES_DIR, "glasser_self_nonself_model_ready.csv")
OUT1 <- file.path(FIGS_DIR,   "fig_auc_distribution_self_nonself.png")
OUT2 <- file.path(FIGS_DIR,   "fig_auc_distribution_all_glasser.png")
OUT3 <- file.path(FIGS_DIR,   "fig_auc_per_subject_distribution.png")

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n04_qc_auc_distribution.R — config: ", TAG, "\n", SEP, "\n\n", sep = "")

if (!file.exists(CSV1))
  stop("Input not found — run 01_build_dataframe.jl first: ", CSV1)
if (!file.exists(CSV2))
  stop("Input not found — run 02_lmm.R first: ", CSV2)

# ── Helpers ────────────────────────────────────────────────────────────────────
skewness <- function(x) {
  x <- x[is.finite(x)]; n <- length(x)
  if (n < 3) return(NA)
  m <- mean(x); s <- sd(x)
  if (s == 0) return(NA)
  (sum((x - m)^3) / n) / s^3
}
kurt_excess <- function(x) {
  x <- x[is.finite(x)]; n <- length(x)
  if (n < 4) return(NA)
  m <- mean(x); s <- sd(x)
  if (s == 0) return(NA)
  (sum((x - m)^4) / n) / s^4 - 3
}
print_summary <- function(label, vals) {
  n_miss <- sum(is.na(vals) | is.nan(vals))
  finite <- vals[is.finite(vals)]; n <- length(finite)
  qs     <- quantile(finite, c(0, 0.25, 0.5, 0.75, 1))
  cat(sprintf("  %s\n    N=%-8d  Missing/NaN=%d\n", label, n, n_miss))
  cat(sprintf("    Min=%.4f  Q1=%.4f  Median=%.4f  Q3=%.4f  Max=%.4f\n",
              qs[1], qs[2], qs[3], qs[4], qs[5]))
  cat(sprintf("    Mean=%.4f  SD=%.4f\n", mean(finite), sd(finite)))
  cat(sprintf("    Skewness=%.4f  Excess kurtosis=%.4f\n\n",
              skewness(finite), kurt_excess(finite)))
}
base_theme <- function() {
  theme_minimal(base_size = 12, base_family = "serif") +
    theme(panel.background = element_rect(fill = "white", color = NA),
          plot.background  = element_rect(fill = "white", color = NA),
          panel.grid.minor = element_blank(),
          plot.title = element_text(face = "plain", size = 13, hjust = 0.5))
}

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Self vs Nonself
# ══════════════════════════════════════════════════════════════════════════════
cat(SEP, "\nPLOT 1 — Self vs Nonself\n", SEP, "\n", sep = "")

df1 <- read.csv(CSV1, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n\n", nrow(df1)))

self_vals    <- df1$auc[df1$atlas == "self"]
nonself_vals <- df1$auc[df1$atlas == "nonself"]
print_summary("Self (atlas == 'self')",       self_vals)
print_summary("Nonself (atlas == 'nonself')", nonself_vals)

plot_df1 <- data.frame(
  auc   = c(self_vals[is.finite(self_vals)],
             nonself_vals[is.finite(nonself_vals)]),
  atlas = c(rep("Self",    sum(is.finite(self_vals))),
             rep("Nonself", sum(is.finite(nonself_vals))))
)
plot_df1$atlas <- factor(plot_df1$atlas, levels = c("Self", "Nonself"))
COLORS1 <- c(Self = "#123434", Nonself = "#2E8B8B")
LABELS1 <- c(
  Self    = sprintf("Self (N=%d)",    sum(is.finite(self_vals))),
  Nonself = sprintf("Nonself (N=%d)", sum(is.finite(nonself_vals)))
)

p1 <- ggplot(plot_df1, aes(x = auc, fill = atlas, color = atlas)) +
  geom_density(alpha = 0.20, adjust = 0.8, linewidth = 1.0) +
  scale_fill_manual(values = COLORS1, labels = LABELS1, name = NULL) +
  scale_color_manual(values = COLORS1, labels = LABELS1, name = NULL) +
  labs(title = "AUC distribution — self vs nonself (Glasser parcels)",
       x = "AUC (seconds)", y = "Density") +
  base_theme() +
  theme(legend.position = c(0.82, 0.85),
        legend.background = element_rect(fill = "white", color = "gray80",
                                         linewidth = 0.3))
ggsave(OUT1, p1, width = 8, height = 6, dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n\n", OUT1))

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — All Glasser parcels pooled
# ══════════════════════════════════════════════════════════════════════════════
cat(SEP, "\nPLOT 2 — All Glasser parcels pooled\n", SEP, "\n", sep = "")

df2 <- read.csv(CSV2, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n\n", nrow(df2)))
print_summary("All Glasser parcels (pooled)", df2$auc)

auc2 <- df2$auc[is.finite(df2$auc)]
p2 <- ggplot(data.frame(auc = auc2), aes(x = auc)) +
  geom_density(fill = "#2E8B8B", color = "#2E8B8B",
               alpha = 0.50, adjust = 0.8, linewidth = 0.7) +
  geom_rug(data = data.frame(auc = sample(auc2, min(500, length(auc2)))),
           color = "#2E8B8B", alpha = 0.25, linewidth = 0.3) +
  labs(title = "AUC distribution — all Glasser parcels (pooled)",
       x = "AUC (seconds)", y = "Density") +
  base_theme() + theme(legend.position = "none")
ggsave(OUT2, p2, width = 8, height = 6, dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n\n", OUT2))

# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Per-subject AUC boxplots
# ══════════════════════════════════════════════════════════════════════════════
if (!requireNamespace("patchwork", quietly = TRUE))
  install.packages("patchwork", repos = "https://cloud.r-project.org", quiet = TRUE)
suppressPackageStartupMessages(library(patchwork))

cat(SEP, "\nPLOT 3 — Per-subject AUC distribution\n", SEP, "\n", sep = "")

PANELS <- list(
  list(atlas = "self",    session = "ses-01", title = "Self — Pre (ses-01)"),
  list(atlas = "self",    session = "ses-02", title = "Self — Post (ses-02)"),
  list(atlas = "nonself", session = "ses-01", title = "Nonself — Pre (ses-01)"),
  list(atlas = "nonself", session = "ses-02", title = "Nonself — Post (ses-02)")
)
GROUP_COLORS3 <- c(placebo = "#CD5C5C", verum = "#4682B4")

for (pan in PANELS) {
  d <- df1[df1$atlas == pan$atlas & df1$session == pan$session &
             is.finite(df1$auc), ]
  subj_med  <- tapply(d$auc, d$subject, median, na.rm = TRUE)
  grand_med <- median(d$auc, na.rm = TRUE)
  grand_sd  <- sd(subj_med)
  lo <- grand_med - 2 * grand_sd; hi <- grand_med + 2 * grand_sd
  outliers <- names(subj_med)[subj_med < lo | subj_med > hi]
  cat(sprintf("  [%s]\n    Grand median: %.4f\n", pan$title, grand_med))
  cat(sprintf("    Subject median range: %.4f – %.4f\n",
              min(subj_med), max(subj_med)))
  if (length(outliers) > 0) {
    cat(sprintf("    Outliers (>2 SD): %s\n\n", paste(outliers, collapse = ", ")))
  } else cat("    Outliers (>2 SD): none\n\n")
}

make_subject_panel <- function(atlas_val, session_val, panel_title,
                               show_legend = FALSE) {
  d <- df1[df1$atlas == atlas_val & df1$session == session_val &
             is.finite(df1$auc), ]
  subj_med   <- tapply(d$auc, d$subject, median, na.rm = TRUE)
  subj_order <- names(sort(subj_med))
  d$subject_f  <- factor(d$subject, levels = subj_order)
  d$drug_group <- factor(d$drug_group, levels = c("placebo", "verum"))
  grand_med    <- median(d$auc, na.rm = TRUE)
  ggplot(d, aes(x = subject_f, y = auc, fill = drug_group)) +
    geom_hline(yintercept = grand_med, linetype = "dashed",
               color = "gray50", linewidth = 0.5) +
    geom_boxplot(outlier.size = 0.6, outlier.alpha = 0.4,
                 linewidth = 0.35, width = 0.75) +
    scale_fill_manual(values = GROUP_COLORS3, name = NULL,
                      labels = c(placebo = "Placebo", verum = "Verum")) +
    scale_x_discrete(labels = function(x) sub("sub-0?", "", x)) +
    labs(title = panel_title, y = "AUC (s)", x = NULL) +
    theme_minimal(base_size = 9, base_family = "serif") +
    theme(panel.background   = element_rect(fill = "white", color = NA),
          plot.background    = element_rect(fill = "white", color = NA),
          panel.grid.minor   = element_blank(),
          panel.grid.major.x = element_blank(),
          axis.text.x  = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 7),
          axis.title.y = element_text(size = 9),
          plot.title   = element_text(face = "plain", hjust = 0.5, size = 10,
                                      family = "serif"),
          legend.position = if (show_legend) "bottom" else "none")
}

subj_panels <- list(
  make_subject_panel("self",    "ses-01", "Self — Pre (ses-01)"),
  make_subject_panel("self",    "ses-02", "Self — Post (ses-02)", show_legend = TRUE),
  make_subject_panel("nonself", "ses-01", "Nonself — Pre (ses-01)"),
  make_subject_panel("nonself", "ses-02", "Nonself — Post (ses-02)")
)
fig3 <- (subj_panels[[1]] | subj_panels[[2]]) /
         (subj_panels[[3]] | subj_panels[[4]]) +
  plot_annotation(
    title = "Per-subject AUC distribution by atlas × session",
    theme = theme(plot.title = element_text(face = "plain", hjust = 0.5,
                                            size = 12, family = "serif"))
  )
ggsave(OUT3, fig3, width = 12, height = 8, dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n\n", OUT3))

cat(SEP, "\nDONE\n", SEP, "\n", sep = "")
