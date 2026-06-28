# statistics_resid.R — Drug-effect tests on quality-residualized AUC_diff
#
# Identical test design to statistics.R but the per-subject delta is the
# OLS residual from qc_residualize_auc.R (AUC_diff partialled for
# pcf_diff, pcf_sq_diff, mean_fd_retained_diff) rather than the raw delta.
#
# Test 1 (parametric with switch):
#   Shapiro-Wilk per group (alpha = .05).
#   Both pass → Welch two-sample t-test; otherwise → Mann-Whitney U.
#
# Test 2 (permutation):
#   Observed statistic: mean(verum residual) − mean(placebo residual).
#   Shuffle group labels 10,000 times; two-tailed p.  Seed = 42.
#
# Multiplicity: BH FDR across 6 layers, separately for each test.
#
# Outlier sensitivity:
#   (a) no exclusions
#   (b) ±2.5 SD from group mean, per layer per group
#   (c) 1.5 × IQR (Tukey fence), per layer across subjects
#
# Inputs
#   04_statistics/results/qinspheres/tables/auc_diff_quality_residuals.csv
#   04_statistics/results/qinspheres/tables/qinspheres_auc.csv   (drug_group)
#
# Outputs (same directory)
#   layer_drug_effect_resid.csv
#   layer_drug_effect_resid_sd25.csv
#   layer_drug_effect_resid_iqr.csv
#   layer_outlier_comparison_resid.csv
#
# Figures (residual analogs of 02_scatter.R, on the quality-residualized
# AUC_diff; one point = one subject):
#   qinspheres_layer_scatter_res.png         residual by layer, both groups pooled
#   qinspheres_prepost_delta_placebo_res.png residual by layer, placebo only
#   qinspheres_prepost_delta_verum_res.png   residual by layer, verum only
#   qinspheres_drug_diff_res.png             verum vs placebo residual per layer
#
# Run from repo root:
#   Rscript 04_statistics/scripts/qinspheres/statistics_resid.R

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

PIPELINE  <- { a <- commandArgs(trailingOnly = TRUE); if (length(a) >= 1) a[1] else "maximal" }
cat(sprintf("Pipeline: %s\n", PIPELINE))
QIN_DIR   <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres", PIPELINE)
RESID_CSV <- file.path(QIN_DIR, "tables", "auc_diff_quality_residuals.csv")
AUC_CSV   <- file.path(QIN_DIR, "tables", "qinspheres_auc.csv")
OUT_DIR   <- file.path(QIN_DIR, "tables")
FIGS_DIR  <- file.path(QIN_DIR, "figures")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

LAYER_ORDER <- c("visual", "auditory", "motor", "extero", "intero", "mental")
PERM_SEED   <- 42L
N_PERM      <- 10000L
SW_ALPHA    <- 0.05

# ── Plot styling (shared with 02_scatter.R) ───────────────────────────────────
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
GROUP_COLORS <- c(placebo = "#CD5C5C", verum = "#4682B4")

base_theme <- theme_bw(base_size = 11) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major = element_line(color = "#e8e8e8", linewidth = 0.35),
    axis.title       = element_text(size = 10),
    axis.text.x      = element_text(size = 10.5),
    plot.title       = element_text(size = 12, face = "bold", hjust = 0.5),
    plot.subtitle    = element_text(size = 9, hjust = 0.5, color = "grey40")
  )

# ── Load & validate ───────────────────────────────────────────────────────────
if (!file.exists(RESID_CSV))
  stop("Residuals file not found — run qc_residualize_auc.R first:\n  ", RESID_CSV)
if (!file.exists(AUC_CSV))
  stop("AUC file not found:\n  ", AUC_CSV)

resid_raw <- read.csv(RESID_CSV, stringsAsFactors = FALSE)
auc_raw   <- read.csv(AUC_CSV,   stringsAsFactors = FALSE)

cat(sprintf("Loaded residuals: %d subjects\n", nrow(resid_raw)))
cat(sprintf("Loaded AUC data:  %d rows (%d subjects)\n",
            nrow(auc_raw), n_distinct(auc_raw$subject)))

# Validate residual columns
resid_cols <- paste0(LAYER_ORDER, "_resid")
missing_resid <- setdiff(resid_cols, names(resid_raw))
if (length(missing_resid) > 0)
  stop("Missing residual columns: ", paste(missing_resid, collapse = ", "),
       "\n  Re-run qc_residualize_auc.R.")

# Validate AUC columns
if (!all(c("subject", "drug_group") %in% names(auc_raw)))
  stop("AUC file must contain 'subject' and 'drug_group' columns.")

# ── Drug group lookup ─────────────────────────────────────────────────────────
drug_map <- auc_raw %>%
  distinct(subject, drug_group)

dupe_grp <- drug_map %>% group_by(subject) %>% filter(n() > 1)
if (nrow(dupe_grp) > 0)
  stop("Subjects with conflicting drug_group labels:\n",
       paste(unique(dupe_grp$subject), collapse = ", "))

grps <- sort(unique(drug_map$drug_group))
if (!identical(grps, c("placebo", "verum")))
  stop("Unexpected drug_group labels: ", paste(grps, collapse = ", "))

# ── Join residuals with drug group ────────────────────────────────────────────
only_resid <- setdiff(resid_raw$subject, drug_map$subject)
only_drug  <- setdiff(drug_map$subject, resid_raw$subject)

if (length(only_resid) > 0)
  cat(sprintf("WARNING: %d subjects in residuals but not AUC (no group): %s\n",
              length(only_resid), paste(only_resid, collapse = ", ")))
if (length(only_drug) > 0)
  cat(sprintf("WARNING: %d subjects in AUC but not residuals (dropped): %s\n",
              length(only_drug), paste(only_drug, collapse = ", ")))

df_wide <- resid_raw %>%
  select(-any_of("drug_group")) %>%
  inner_join(drug_map, by = "subject")
cat(sprintf("Subjects after join (have both residuals and group): %d\n\n", nrow(df_wide)))

# Pivot to long for downstream outlier functions
df_long <- df_wide %>%
  pivot_longer(cols = all_of(resid_cols),
               names_to  = "layer",
               values_to = "resid") %>%
  mutate(layer = sub("_resid$", "", layer)) %>%
  filter(layer %in% LAYER_ORDER)

# Check sessions both present per layer
cat("Subjects per group per layer (no exclusions):\n")
df_long %>%
  filter(!is.na(resid)) %>%
  group_by(layer, drug_group) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = drug_group, values_from = n) %>%
  arrange(match(layer, LAYER_ORDER)) %>%
  print()

small <- df_long %>%
  filter(!is.na(resid)) %>%
  group_by(layer, drug_group) %>%
  summarise(n = n(), .groups = "drop") %>%
  filter(n < 5)
if (nrow(small) > 0) {
  cat("\nFLAG: n < 5 in at least one layer/group — permutation null unreliable:\n")
  print(small)
} else {
  cat("Group size check OK: all layers n >= 5 in both groups.\n\n")
}

# ── Outlier exclusion functions ───────────────────────────────────────────────
exclude_sd25 <- function(d) {
  d %>%
    group_by(layer, drug_group) %>%
    mutate(z = (resid - mean(resid, na.rm = TRUE)) / sd(resid, na.rm = TRUE),
           outlier = !is.na(z) & abs(z) > 2.5) %>%
    ungroup() %>%
    filter(!outlier) %>%
    select(-z, -outlier)
}

exclude_iqr <- function(d) {
  d %>%
    group_by(layer) %>%
    mutate(q1  = quantile(resid, 0.25, na.rm = TRUE),
           q3  = quantile(resid, 0.75, na.rm = TRUE),
           iqr = q3 - q1,
           outlier = !is.na(resid) &
             (resid < (q1 - 1.5 * iqr) | resid > (q3 + 1.5 * iqr))) %>%
    ungroup() %>%
    filter(!outlier) %>%
    select(-q1, -q3, -iqr, -outlier)
}

report_exclusions <- function(d_full, d_trimmed, label) {
  n_removed <- nrow(d_full) - nrow(d_trimmed)
  cat(sprintf("\n%s: removed %d subject-layer rows\n", label, n_removed))
  if (n_removed > 0) {
    removed <- anti_join(d_full, d_trimmed,
                         by = c("subject", "layer", "drug_group"))
    print(removed[, c("subject", "drug_group", "layer", "resid")])
  }
}

# ── Core test function ────────────────────────────────────────────────────────
run_tests <- function(data, label) {
  cat(sprintf("\n══ %s ══\n", label))

  set.seed(PERM_SEED)
  rows <- lapply(LAYER_ORDER, function(lyr) {
    d        <- data %>% filter(layer == lyr, !is.na(resid))
    v_vals   <- d %>% filter(drug_group == "verum")   %>% pull(resid)
    p_vals   <- d %>% filter(drug_group == "placebo") %>% pull(resid)
    n_v      <- length(v_vals)
    n_p      <- length(p_vals)
    obs_diff <- mean(v_vals) - mean(p_vals)

    if (n_v < 3 || n_p < 3) {
      cat(sprintf("FLAG %-10s: n too small (nV=%d nP=%d)\n", lyr, n_v, n_p))
      return(data.frame(layer=lyr, n_verum=n_v, n_placebo=n_p,
                        observed_diff=obs_diff,
                        sw_p_verum=NA, sw_p_placebo=NA,
                        test1_method=NA, test1_stat=NA,
                        test1_p_raw=NA, test1_p_fdr=NA,
                        perm_p_raw=NA,  perm_p_fdr=NA,
                        stringsAsFactors=FALSE))
    }

    sw_v <- shapiro.test(v_vals)$p.value
    sw_p <- shapiro.test(p_vals)$p.value

    if (sw_v > SW_ALPHA && sw_p > SW_ALPHA) {
      tt           <- t.test(v_vals, p_vals, var.equal = FALSE)
      test1_method <- "Welch t-test"
      test1_stat   <- unname(tt$statistic)
      test1_p_raw  <- tt$p.value
    } else {
      wt           <- wilcox.test(v_vals, p_vals, exact = FALSE)
      test1_method <- "Mann-Whitney U"
      test1_stat   <- unname(wt$statistic)
      test1_p_raw  <- wt$p.value
    }

    all_vals  <- c(v_vals, p_vals)
    n_total   <- length(all_vals)
    perm_null <- replicate(N_PERM, {
      idx <- sample(n_total, n_v, replace = FALSE)
      mean(all_vals[idx]) - mean(all_vals[-idx])
    })
    perm_p_raw <- (sum(abs(perm_null) >= abs(obs_diff)) + 1) / (N_PERM + 1)

    cat(sprintf(
      "  %-10s  nV=%d nP=%d  diff=%+.4f  [%s: stat=%.3f p=%.4f]  [perm p=%.4f]  [SW: V=%.3f P=%.3f]\n",
      lyr, n_v, n_p, obs_diff,
      test1_method, test1_stat, test1_p_raw,
      perm_p_raw, sw_v, sw_p
    ))

    data.frame(layer=lyr, n_verum=n_v, n_placebo=n_p,
               observed_diff=obs_diff,
               sw_p_verum=sw_v, sw_p_placebo=sw_p,
               test1_method=test1_method, test1_stat=test1_stat,
               test1_p_raw=test1_p_raw, test1_p_fdr=NA_real_,
               perm_p_raw=perm_p_raw,   perm_p_fdr=NA_real_,
               stringsAsFactors=FALSE)
  })

  out <- do.call(rbind, rows)
  out$test1_p_fdr <- p.adjust(out$test1_p_raw, method = "BH")
  out$perm_p_fdr  <- p.adjust(out$perm_p_raw,  method = "BH")
  out
}

# ── Run three outlier variants ────────────────────────────────────────────────
df_clean <- df_long %>% filter(!is.na(resid))

df_sd25 <- exclude_sd25(df_clean)
df_iqr  <- exclude_iqr(df_clean)

report_exclusions(df_clean, df_sd25, "±2.5 SD (per layer per group)")
report_exclusions(df_clean, df_iqr,  "1.5×IQR (per layer across groups)")

res_full <- run_tests(df_clean, "No exclusions")
res_sd25 <- run_tests(df_sd25,  "±2.5 SD exclusions")
res_iqr  <- run_tests(df_iqr,   "1.5×IQR exclusions")

# ── Save individual result tables ─────────────────────────────────────────────
write.csv(res_full, file.path(OUT_DIR, "layer_drug_effect_resid.csv"),      row.names = FALSE)
write.csv(res_sd25, file.path(OUT_DIR, "layer_drug_effect_resid_sd25.csv"), row.names = FALSE)
write.csv(res_iqr,  file.path(OUT_DIR, "layer_drug_effect_resid_iqr.csv"),  row.names = FALSE)

# ── Side-by-side comparison ───────────────────────────────────────────────────
fmt_p <- function(p) ifelse(is.na(p), "NA", sprintf("%.4f", p))

comparison <- data.frame(
  layer              = LAYER_ORDER,
  obs_diff_full      = res_full$observed_diff,
  test1_p_full       = res_full$test1_p_raw,
  test1_pfdr_full    = res_full$test1_p_fdr,
  perm_p_full        = res_full$perm_p_raw,
  perm_pfdr_full     = res_full$perm_p_fdr,
  obs_diff_sd25      = res_sd25$observed_diff,
  test1_p_sd25       = res_sd25$test1_p_raw,
  test1_pfdr_sd25    = res_sd25$test1_p_fdr,
  perm_p_sd25        = res_sd25$perm_p_raw,
  perm_pfdr_sd25     = res_sd25$perm_p_fdr,
  obs_diff_iqr       = res_iqr$observed_diff,
  test1_p_iqr        = res_iqr$test1_p_raw,
  test1_pfdr_iqr     = res_iqr$test1_p_fdr,
  perm_p_iqr         = res_iqr$perm_p_raw,
  perm_pfdr_iqr      = res_iqr$perm_p_fdr,
  stringsAsFactors   = FALSE
)

write.csv(comparison, file.path(OUT_DIR, "layer_outlier_comparison_resid.csv"),
          row.names = FALSE)

cat("\n══ Side-by-side perm p-values (raw) ══\n")
cat(sprintf("  %-10s  %8s  %8s  %8s\n", "layer", "full", "SD2.5", "IQR"))
for (i in seq_len(nrow(comparison))) {
  cat(sprintf("  %-10s  %8s  %8s  %8s\n",
              comparison$layer[i],
              fmt_p(comparison$perm_p_full[i]),
              fmt_p(comparison$perm_p_sd25[i]),
              fmt_p(comparison$perm_p_iqr[i])))
}

# ── Figures (residual analogs of 02_scatter.R) ────────────────────────────────
# All plots use the no-exclusion residuals (df_clean), one point = one subject.
# Residual = quality-residualized AUC_diff (partialled for pcf_diff, pcf_sq_diff,
# mean_fd_retained_diff), so it already encodes the post−pre change.
plot_df <- df_clean %>%
  mutate(layer = factor(layer, levels = LAYER_ORDER))

resid_ylim <- range(plot_df$resid, na.rm = TRUE)
cat(sprintf("\nResidual shared y range: [%.3f, %.3f]\n",
            resid_ylim[1], resid_ylim[2]))

# Per-layer residual scatter for one subset (group filter optional)
make_resid_scatter <- function(d, title, subtitle) {
  med_df <- d %>%
    group_by(layer) %>%
    summarise(med = median(resid), .groups = "drop")

  ggplot(d, aes(x = layer, y = resid, color = layer)) +
    geom_hline(yintercept = 0, linetype = "dashed",
               color = "grey50", linewidth = 0.5) +
    geom_jitter(size = 2.8, alpha = 0.8, width = 0.18, height = 0) +
    geom_crossbar(data = med_df,
                  aes(x = layer, y = med, ymin = med, ymax = med),
                  width = 0.45, linewidth = 0.8, color = "black",
                  inherit.aes = FALSE) +
    scale_x_discrete(labels = LAYER_LABELS) +
    scale_color_manual(values = LAYER_COLORS, guide = "none") +
    scale_y_continuous(limits = resid_ylim) +
    labs(x = NULL, y = "Quality-adjusted ΔAUC (residual)",
         title = title, subtitle = subtitle) +
    base_theme
}

# 1) Both groups pooled
p_all <- make_resid_scatter(
  plot_df,
  "Residualized AUC change per layer — all subjects",
  "Both groups pooled. One point = one subject. Black bar = median.")
ggsave(file.path(FIGS_DIR, "qinspheres_layer_scatter_res.png"),
       p_all, width = 8, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_layer_scatter_res.png")))

# 2) Placebo only
p_plac <- make_resid_scatter(
  plot_df %>% filter(drug_group == "placebo"),
  "Residualized AUC change per layer — Placebo",
  "Placebo only. One point = one subject. Black bar = median.")
ggsave(file.path(FIGS_DIR, "qinspheres_prepost_delta_placebo_res.png"),
       p_plac, width = 8, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_prepost_delta_placebo_res.png")))

# 3) Verum only
p_verm <- make_resid_scatter(
  plot_df %>% filter(drug_group == "verum"),
  "Residualized AUC change per layer — Verum (DMT)",
  "Verum only. One point = one subject. Black bar = median.")
ggsave(file.path(FIGS_DIR, "qinspheres_prepost_delta_verum_res.png"),
       p_verm, width = 8, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_prepost_delta_verum_res.png")))

# 4) Drug effect: verum vs placebo residual per layer (dodged), with group medians.
#    Visualizes mean(verum) − mean(placebo), the statistic tested per layer.
med_grp <- plot_df %>%
  group_by(layer, drug_group) %>%
  summarise(med = median(resid), .groups = "drop")

dodge <- position_dodge(width = 0.55)
p_diff <- ggplot(plot_df,
                 aes(x = layer, y = resid, color = drug_group, group = drug_group)) +
  geom_hline(yintercept = 0, linetype = "dashed",
             color = "grey50", linewidth = 0.5) +
  geom_point(position = position_jitterdodge(jitter.width = 0.12,
                                             dodge.width = 0.55),
             size = 2.4, alpha = 0.75) +
  geom_crossbar(data = med_grp,
                aes(x = layer, y = med, ymin = med, ymax = med, group = drug_group),
                width = 0.45, linewidth = 0.7, color = "black",
                position = dodge, inherit.aes = FALSE) +
  scale_x_discrete(labels = LAYER_LABELS) +
  scale_color_manual(values = GROUP_COLORS,
                     labels = c(placebo = "Placebo", verum = "Verum (DMT)"),
                     name = NULL) +
  labs(x = NULL, y = "Quality-adjusted ΔAUC (residual)",
       title = "Drug effect: Verum vs Placebo residual per layer",
       subtitle = "One point = one subject. Black bars = group medians.") +
  base_theme +
  theme(legend.position = "top")
ggsave(file.path(FIGS_DIR, "qinspheres_drug_diff_res.png"),
       p_diff, width = 8, height = 5.5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n",
            file.path(FIGS_DIR, "qinspheres_drug_diff_res.png")))

cat(sprintf("\nSaved to %s:\n", OUT_DIR))
cat("  layer_drug_effect_resid.csv\n")
cat("  layer_drug_effect_resid_sd25.csv\n")
cat("  layer_drug_effect_resid_iqr.csv\n")
cat("  layer_outlier_comparison_resid.csv\n")
cat(sprintf("\nFigures saved to %s:\n", FIGS_DIR))
cat("  qinspheres_layer_scatter_res.png\n")
cat("  qinspheres_prepost_delta_placebo_res.png\n")
cat("  qinspheres_prepost_delta_verum_res.png\n")
cat("  qinspheres_drug_diff_res.png\n")
