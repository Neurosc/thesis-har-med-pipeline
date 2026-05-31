# 07_auc_distribution.R — AUC distribution plots
#
# Plot 1: self (spheres) vs nonself density overlay
#   Input:  04_statistics/results/analysis_long_format_auc.csv
#   Output: 04_statistics/figures/fig_auc_distribution_self_nonself.png
#
# Plot 2: all Glasser parcels pooled (single density + rug)
#   Input:  04_statistics/results/glasser_self_nonself_model_ready.csv
#   Output: 04_statistics/figures/fig_auc_distribution_all_glasser.png
#
# Run from repo root:
#   Rscript 04_statistics/scripts/07_auc_distribution.R

suppressPackageStartupMessages(library(ggplot2))

# ── Paths ──────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
  REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
} else {
  REPO_ROOT <- normalizePath(".")
}

RES_DIR  <- file.path(REPO_ROOT, "04_statistics", "results")
FIG_DIR  <- file.path(REPO_ROOT, "04_statistics", "figures")
dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

CSV1 <- file.path(RES_DIR, "analysis_long_format_auc.csv")
CSV2 <- file.path(RES_DIR, "glasser_self_nonself_model_ready.csv")
OUT1 <- file.path(FIG_DIR, "fig_auc_distribution_self_nonself.png")
OUT2 <- file.path(FIG_DIR, "fig_auc_distribution_all_glasser.png")

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n07_auc_distribution.R\n", SEP, "\n\n", sep = "")

# ── Helper: console summary ────────────────────────────────────────────────────
skewness <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 3) return(NA)
  m <- mean(x); s <- sd(x)
  if (s == 0) return(NA)
  (sum((x - m)^3) / n) / s^3
}
kurt_excess <- function(x) {
  x <- x[is.finite(x)]
  n <- length(x)
  if (n < 4) return(NA)
  m <- mean(x); s <- sd(x)
  if (s == 0) return(NA)
  (sum((x - m)^4) / n) / s^4 - 3
}

print_summary <- function(label, vals) {
  n_miss   <- sum(is.na(vals) | is.nan(vals))
  finite   <- vals[is.finite(vals)]
  n        <- length(finite)
  qs       <- quantile(finite, c(0, 0.25, 0.5, 0.75, 1))
  cat(sprintf("  %s\n", label))
  cat(sprintf("    N=%-8d  Missing/NaN=%d\n", n, n_miss))
  cat(sprintf("    Min=%.4f  Q1=%.4f  Median=%.4f  Q3=%.4f  Max=%.4f\n",
              qs[1], qs[2], qs[3], qs[4], qs[5]))
  cat(sprintf("    Mean=%.4f  SD=%.4f\n", mean(finite), sd(finite)))
  cat(sprintf("    Skewness=%.4f  Excess kurtosis=%.4f\n\n",
              skewness(finite), kurt_excess(finite)))
}

# ── Shared theme ───────────────────────────────────────────────────────────────
base_theme <- function() {
  theme_minimal(base_size = 12, base_family = "serif") +
    theme(
      panel.background = element_rect(fill = "white", color = NA),
      plot.background  = element_rect(fill = "white", color = NA),
      panel.grid.minor = element_blank(),
      plot.title = element_text(face = "plain", size = 13, hjust = 0.5)
    )
}

# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Self (spheres) vs Nonself
# ══════════════════════════════════════════════════════════════════════════════
cat(SEP, "\nPLOT 1 — Self vs Nonself (analysis_long_format_auc.csv)\n",
    SEP, "\n", sep = "")

df1 <- read.csv(CSV1, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n\n", nrow(df1)))

self_vals    <- df1$auc[df1$atlas == "self"]
nonself_vals <- df1$auc[df1$atlas == "nonself"]

print_summary("Self (atlas == 'self')",    self_vals)
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
  labs(
    title = "AUC distribution — self (spheres) vs nonself",
    x = "AUC (seconds)",
    y = "Density"
  ) +
  base_theme() +
  theme(legend.position = c(0.82, 0.85),
        legend.background = element_rect(fill = "white", color = "gray80",
                                         linewidth = 0.3))

ggsave(OUT1, p1, width = 8, height = 6, dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n\n", OUT1))

# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — All Glasser parcels pooled
# ══════════════════════════════════════════════════════════════════════════════
cat(SEP, "\nPLOT 2 — All Glasser parcels pooled (glasser_self_nonself_model_ready.csv)\n",
    SEP, "\n", sep = "")

df2 <- read.csv(CSV2, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n\n", nrow(df2)))

print_summary("All Glasser parcels (pooled)", df2$auc)

auc2 <- df2$auc[is.finite(df2$auc)]

p2 <- ggplot(data.frame(auc = auc2), aes(x = auc)) +
  geom_density(fill = "#2E8B8B", color = "#2E8B8B",
               alpha = 0.50, adjust = 0.8, linewidth = 0.7) +
  geom_rug(data = data.frame(auc = sample(auc2, min(500, length(auc2)))),
           color = "#2E8B8B", alpha = 0.25, linewidth = 0.3) +
  labs(
    title = "AUC distribution — all Glasser parcels (pooled)",
    x = "AUC (seconds)",
    y = "Density"
  ) +
  base_theme() +
  theme(legend.position = "none")

ggsave(OUT2, p2, width = 8, height = 6, dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n\n", OUT2))

cat(SEP, "\nDONE\n", SEP, "\n", sep = "")
