# 07_auc_distribution.R — AUC distribution by atlas (denoisedNoGSR)
# Loads AUC long-format CSV and checks distribution: self vs nonself.
# Reports NaN/Inf counts, distinct values, pile-ups, summary statistics.
# Produces overlaid histogram + density plot.
#
# Input:  04_statistics/results/analysis_long_format_auc.csv
# Output: 04_statistics/figures/fig07_auc_distribution.png
#
# Run from repo root:
#   Rscript 04_statistics/scripts/07_auc_distribution.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
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

DATA_CSV <- file.path(REPO_ROOT, "04_statistics", "results",
                       "analysis_long_format_auc.csv")
FIG_DIR  <- file.path(REPO_ROOT, "04_statistics", "figures")
FIG_PATH <- file.path(FIG_DIR, "fig07_auc_distribution.png")

dir.create(FIG_DIR, showWarnings = FALSE, recursive = TRUE)

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n07_auc_distribution.R — AUC distribution by atlas\n",
    SEP, "\n\n", sep = "")

# ── Load ───────────────────────────────────────────────────────────────────────
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows, %d columns\n", nrow(df), ncol(df)))
cat(sprintf("Atlases:  %s\n", paste(unique(df$atlas), collapse = ", ")))
cat(sprintf("Sessions: %s\n", paste(sort(unique(df$session)), collapse = ", ")))
cat(sprintf("Subjects: %d\n\n", length(unique(df$subject))))

# ── Console summary per atlas ──────────────────────────────────────────────────
atlas_summary <- function(label, vals) {
  n_total    <- length(vals)
  n_nan      <- sum(is.nan(vals))
  n_inf      <- sum(is.infinite(vals))
  finite     <- vals[is.finite(vals)]
  n_finite   <- length(finite)
  n_distinct <- length(unique(finite))

  cat(sprintf("%s  (N=%d)\n", label, n_total))
  cat(sprintf("  NaN: %d   Inf: %d   Finite: %d   Distinct: %d\n",
              n_nan, n_inf, n_finite, n_distinct))
  if (n_finite > 0) {
    cat(sprintf("  min=%.3f   median=%.3f   max=%.3f\n",
                min(finite), median(finite), max(finite)))
    cat(sprintf("  mean=%.3f   SD=%.3f\n", mean(finite), sd(finite)))
  }

  # Top 5 most common finite values (pile-up check)
  tbl    <- sort(table(round(finite, 6)), decreasing = TRUE)
  top_n  <- min(5, length(tbl))
  cat("  Top 5 most common AUC values:\n")
  for (i in seq_len(top_n)) {
    cat(sprintf("    %.6f  →  %d occurrences\n",
                as.numeric(names(tbl)[i]), tbl[i]))
  }
  cat("\n")

  invisible(finite)
}

self_auc    <- df$auc[df$atlas == "self"]
nonself_auc <- df$auc[df$atlas == "nonself"]

self_finite    <- atlas_summary("Self",    self_auc)
nonself_finite <- atlas_summary("Nonself", nonself_auc)

# ── Figure: overlaid histogram + density ──────────────────────────────────────
COLOR_SELF    <- "#3D5A6C"
COLOR_NONSELF <- "#8B7355"

plot_df <- data.frame(
  auc   = c(self_finite, nonself_finite),
  atlas = c(rep("Self",    length(self_finite)),
            rep("Nonself", length(nonself_finite)))
)
plot_df$atlas <- factor(plot_df$atlas, levels = c("Self", "Nonself"))

n_self    <- length(self_finite)
n_nonself <- length(nonself_finite)

ATLAS_COLORS <- c(Self = COLOR_SELF, Nonself = COLOR_NONSELF)
ATLAS_LABELS <- c(
  Self    = sprintf("Self (N=%d)",    n_self),
  Nonself = sprintf("Nonself (N=%d)", n_nonself)
)

# Panel A: overlaid histogram
p_hist <- ggplot(plot_df, aes(x = auc, fill = atlas, color = atlas)) +
  geom_histogram(aes(y = after_stat(count)),
                 binwidth = 0.1, alpha = 0.55, position = "identity") +
  scale_fill_manual(values = ATLAS_COLORS, labels = ATLAS_LABELS,
                    name = NULL) +
  scale_color_manual(values = ATLAS_COLORS, labels = ATLAS_LABELS,
                     name = NULL) +
  labs(title = "AUC distribution by atlas — denoisedNoGSR",
       subtitle = "All subjects and sessions pooled",
       x = "AUC (seconds)", y = "Count") +
  theme_minimal(base_size = 12, base_family = "serif") +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    panel.grid.minor = element_blank(),
    legend.position  = c(0.80, 0.85),
    legend.background = element_rect(fill = "white", color = "gray80"),
    plot.title    = element_text(face = "bold", size = 13),
    plot.subtitle = element_text(size = 10, color = "gray40")
  )

# Panel B: density
p_dens <- ggplot(plot_df, aes(x = auc, fill = atlas, color = atlas)) +
  geom_density(alpha = 0.40, adjust = 0.8) +
  scale_fill_manual(values = ATLAS_COLORS, labels = ATLAS_LABELS,
                    name = NULL) +
  scale_color_manual(values = ATLAS_COLORS, labels = ATLAS_LABELS,
                     name = NULL) +
  labs(title = "AUC density by atlas",
       x = "AUC (seconds)", y = "Density") +
  theme_minimal(base_size = 12, base_family = "serif") +
  theme(
    panel.background = element_rect(fill = "white", color = NA),
    plot.background  = element_rect(fill = "white", color = NA),
    panel.grid.minor = element_blank(),
    legend.position  = c(0.80, 0.85),
    legend.background = element_rect(fill = "white", color = "gray80"),
    plot.title = element_text(face = "bold", size = 13)
  )

# Combine with patchwork-style using cowplot or just stack with ggsave
# Use base R graphics to stack two ggplots
if (!requireNamespace("patchwork", quietly = TRUE))
  install.packages("patchwork", repos = "https://cloud.r-project.org", quiet = TRUE)
library(patchwork)

fig <- p_hist / p_dens

ggsave(FIG_PATH, fig, width = 10, height = 9, dpi = 300, bg = "white")
cat(sprintf("[done] Figure → %s\n\n", FIG_PATH))

cat(SEP, "\nDONE\n", SEP, "\n", sep = "")
