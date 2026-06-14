# exploratory/ie_diff_global_scatter.R
#
# Cross-sectional scatter: IE AUC difference vs leave-out global AUC
#
# y  = IE_diff   : mean(AUC | Interoception) − mean(AUC | Exteroception)
# x  = leave_out_global : mean(AUC) over nonself + Cognition ONLY
#      (everything except the two layers in this difference)
#
# One dot per subject per session (35 × 2 = 70 points).
# Faceted by session: ses-01 = Pre, ses-02 = Post.
# One pooled regression line per facet, Pearson r/p annotated top-right.
# Descriptive only — no inferential drug claim.
#
# Run from repo root:
#   Rscript 04_statistics/scripts/exploratory/ie_diff_global_scatter.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(RcppTOML)
})

# ── Paths ────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
  REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", "..", ".."))
} else {
  REPO_ROOT <- normalizePath(".")
}

cfg              <- parseTOML(file.path(REPO_ROOT, "config.toml"))
ATLAS_METHOD     <- cfg$active$atlas_method
DENOISING_METHOD <- cfg$active$denoising_method
TAG              <- paste0(ATLAS_METHOD, "_", DENOISING_METHOD)

TABLES_DIR <- file.path(REPO_ROOT, "04_statistics", "results", TAG, "tables")
FIGS_DIR   <- file.path(REPO_ROOT, "04_statistics", "results", TAG, "figures")
EXPLOR_DIR <- file.path(REPO_ROOT, "04_statistics", "scripts", "exploratory")
dir.create(FIGS_DIR,   showWarnings = FALSE, recursive = TRUE)
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)

DATA_CSV <- file.path(TABLES_DIR, "model_ready.csv")
if (!file.exists(DATA_CSV))
  stop("Input not found — run 02_lmm.R first: ", DATA_CSV)

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\nie_diff_global_scatter.R — config: ", TAG, "\n", SEP, "\n\n", sep = "")

# ── Load ─────────────────────────────────────────────────────────────────────
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n", nrow(df)))

# ── Per-subject × session means per layer ────────────────────────────────────
# Average AUC across ROIs within each (subject, session, self_layer)
subj_layer <- df %>%
  group_by(subject, group, session, self_layer) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop")

# y: IE_diff = mean(Interoception) − mean(Exteroception)
ie <- subj_layer %>%
  filter(self_layer %in% c("Interoception", "Exteroception")) %>%
  pivot_wider(names_from = self_layer, values_from = mean_auc) %>%
  mutate(IE_diff = Interoception - Exteroception) %>%
  select(subject, group, session, IE_diff)

# x: leave_out_global = mean AUC over nonself + Cognition (exclude the two IE layers)
global <- subj_layer %>%
  filter(self_layer %in% c("nonself", "Cognition")) %>%
  group_by(subject, group, session) %>%
  summarise(leave_out_global = mean(mean_auc, na.rm = TRUE), .groups = "drop")

scatter_df <- ie %>%
  inner_join(global, by = c("subject", "group", "session"))

cat(sprintf("Scatter rows: %d (expect 70)\n\n", nrow(scatter_df)))

# ── Pearson r / p per session ─────────────────────────────────────────────────
# ses-01 = Pre, ses-02 = Post (confirmed from CLAUDE.md: ses-01 is baseline/pre-retreat)
corr_stats <- scatter_df %>%
  group_by(session) %>%
  summarise(
    r   = cor(leave_out_global, IE_diff, use = "complete.obs"),
    p   = cor.test(leave_out_global, IE_diff)$p.value,
    .groups = "drop"
  ) %>%
  mutate(
    label = sprintf("r = %.2f, p = %.2f", r, p),
    # place annotation at top-right of each facet
    x_pos = Inf,
    y_pos = Inf
  )

# Session facet labels: ses-01 = Pre, ses-02 = Post
session_labels <- c("ses-01" = "Pre (ses-01)", "ses-02" = "Post (ses-02)")

# ── Color palette — group-level colors (from 06_figures.R Pre colors) ────────
# placebo = #CD5C5C (Placebo Pre), verum = #4682B4 (Verum Pre)
GROUP_COLORS <- c("placebo" = "#CD5C5C", "verum" = "#4682B4")
GROUP_LABELS <- c("placebo" = "Placebo", "verum" = "Verum")

scatter_df$group   <- factor(scatter_df$group,   levels = c("placebo", "verum"))
scatter_df$session <- factor(scatter_df$session, levels = c("ses-01", "ses-02"))
corr_stats$session <- factor(corr_stats$session, levels = c("ses-01", "ses-02"))

# ── Figure ────────────────────────────────────────────────────────────────────
p <- ggplot(scatter_df, aes(x = leave_out_global, y = IE_diff)) +
  geom_smooth(method = "lm", se = TRUE, colour = "black", linewidth = 0.8,
              fill = "grey80", alpha = 0.3) +
  geom_point(aes(colour = group), size = 2.2, alpha = 0.85) +
  geom_text(
    data = corr_stats,
    aes(x = x_pos, y = y_pos, label = label),
    hjust = 1.05, vjust = 1.5,
    size = 3.2, colour = "black", inherit.aes = FALSE
  ) +
  facet_wrap(~ session, labeller = labeller(session = session_labels)) +
  scale_colour_manual(values = GROUP_COLORS, labels = GROUP_LABELS, name = "Group") +
  labs(
    x = "Global AUC (leave-out mean)",
    y = "Internal–External AUC difference",
    title = "IE AUC difference vs global timescale level"
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid.minor  = element_blank(),
    strip.background  = element_rect(fill = "grey95", colour = "grey70"),
    strip.text        = element_text(size = 10, face = "bold"),
    legend.position   = "bottom",
    plot.title        = element_text(size = 12, face = "bold", hjust = 0.5)
  )

OUT_FIG <- file.path(FIGS_DIR, "fig_explor_ie_global_scatter.png")
ggsave(OUT_FIG, p, width = 8, height = 4.5, dpi = 300)
cat(sprintf("Figure saved: %s\n", OUT_FIG))

# ── Per-subject table ─────────────────────────────────────────────────────────
OUT_CSV <- file.path(TABLES_DIR, "ie_diff_global_scatter_data.csv")
write.csv(
  scatter_df[, c("subject", "group", "session", "IE_diff", "leave_out_global")],
  OUT_CSV, row.names = FALSE
)
cat(sprintf("Table  saved: %s\n", OUT_CSV))

cat("\n=== Pearson correlations ===\n")
print(corr_stats[, c("session", "r", "p", "label")])
cat("\n")
