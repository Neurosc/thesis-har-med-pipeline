# statistics.R — Subject-level drug-effect tests per layer (qinspheres AUC)
#
# Design: between-group comparison of post-pre deltas (verum vs placebo).
# Statistic: mean(verum delta) - mean(placebo delta).
#
# Test 1 (parametric with switch):
#   Shapiro-Wilk on each group's deltas (alpha = .05).
#   Both pass -> Welch two-sample t-test.
#   Otherwise -> Mann-Whitney U (Wilcoxon rank-sum).
#
# Test 2 (permutation):
#   Same statistic; shuffle group labels 10,000 times.
#   Two-tailed p = (|null| >= |obs| + 1) / (n_perm + 1). Seed = 42.
#
# Multiplicity: BH FDR across 6 layers, separately for each test.
#
# Outlier sensitivity:
#   Run analysis three times on the same delta data:
#     (a) no exclusions
#     (b) ±2.5 SD from group mean, per layer per group
#     (c) 1.5 × IQR (Tukey fence), per layer across all subjects
#   Outputs: layer_drug_effect.csv          (a)
#            layer_drug_effect_sd25.csv     (b)
#            layer_drug_effect_iqr.csv      (c)
#            layer_outlier_comparison.csv   side-by-side p-values for all three
#
# Input:  04_statistics/results/qinspheres/tables/qinspheres_auc.csv
# Output: 04_statistics/results/qinspheres/tables/
#
# Run from repo root:
#   Rscript 04_statistics/scripts/qinspheres/statistics.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
args      <- commandArgs(trailingOnly = FALSE)
file_arg  <- args[grep("--file=", args)]
REPO_ROOT <- if (length(file_arg) > 0) {
  normalizePath(file.path(dirname(normalizePath(sub("--file=", "", file_arg))),
                          "..", "..", ".."))
} else {
  normalizePath(".")
}

DATA_CSV <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres",
                      "tables", "qinspheres_auc.csv")
OUT_DIR  <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres", "tables")

if (!file.exists(DATA_CSV))
  stop("Input not found — run 01_build_df.jl first:\n  ", DATA_CSV)

# ── Constants ─────────────────────────────────────────────────────────────────
LAYER_ORDER <- c("visual", "auditory", "motor", "extero", "intero", "mental")
PERM_SEED   <- 42L
N_PERM      <- 10000L
SW_ALPHA    <- 0.05

# ── Load and verify ───────────────────────────────────────────────────────────
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n", nrow(df)))

sessions <- sort(unique(df$session))
if (!identical(sessions, c("ses-01", "ses-02")))
  stop("FLAG: unexpected session labels: ", paste(sessions, collapse = ", "))
cat("Session mapping confirmed: ses-01 = pre, ses-02 = post\n")

grps <- sort(unique(df$drug_group))
if (!identical(grps, c("placebo", "verum")))
  stop("FLAG: unexpected drug_group labels: ", paste(grps, collapse = ", "))

missing_cats <- setdiff(LAYER_ORDER, unique(df$category))
if (length(missing_cats) > 0)
  stop("FLAG: missing categories in data: ", paste(missing_cats, collapse = ", "))

# ── Aggregate to subject × session × layer ────────────────────────────────────
subj_layer <- df %>%
  filter(is.finite(auc)) %>%
  group_by(subject, session, drug_group, category) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop")

# Delta = post - pre per subject × layer
deltas_full <- subj_layer %>%
  pivot_wider(names_from = session, values_from = mean_auc) %>%
  rename(pre = `ses-01`, post = `ses-02`) %>%
  filter(!is.na(pre), !is.na(post)) %>%
  mutate(delta = post - pre)

cat("\nSubjects per group per layer (no exclusions):\n")
deltas_full %>%
  group_by(category, drug_group) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = drug_group, values_from = n) %>%
  arrange(match(category, LAYER_ORDER)) %>%
  print()

small <- deltas_full %>%
  group_by(category, drug_group) %>%
  summarise(n = n(), .groups = "drop") %>%
  filter(n < 5)
if (nrow(small) > 0) {
  cat("\nFLAG: n < 5 in at least one layer/group — permutation null unreliable:\n")
  print(small)
} else {
  cat("Group size check OK: all layers n >= 5 in both groups.\n")
}

# ── Outlier exclusion functions ───────────────────────────────────────────────
exclude_sd25 <- function(d) {
  # Per layer per group: remove subjects > 2.5 SD from their group mean
  d %>%
    group_by(category, drug_group) %>%
    mutate(z = (delta - mean(delta)) / sd(delta),
           outlier = abs(z) > 2.5) %>%
    ungroup() %>%
    filter(!outlier) %>%
    select(-z, -outlier)
}

exclude_iqr <- function(d) {
  # Per layer (across groups): remove subjects outside Q1 - 1.5*IQR or Q3 + 1.5*IQR
  d %>%
    group_by(category) %>%
    mutate(q1  = quantile(delta, 0.25),
           q3  = quantile(delta, 0.75),
           iqr = q3 - q1,
           outlier = delta < (q1 - 1.5 * iqr) | delta > (q3 + 1.5 * iqr)) %>%
    ungroup() %>%
    filter(!outlier) %>%
    select(-q1, -q3, -iqr, -outlier)
}

report_exclusions <- function(d_full, d_trimmed, label) {
  n_removed <- nrow(d_full) - nrow(d_trimmed)
  cat(sprintf("\n%s: removed %d subject-layer rows\n", label, n_removed))
  if (n_removed > 0) {
    removed <- anti_join(d_full, d_trimmed,
                         by = c("subject", "category", "drug_group"))
    print(removed[, c("subject", "drug_group", "category", "delta")])
  }
}

# ── Core test function ────────────────────────────────────────────────────────
run_tests <- function(deltas, label) {
  cat(sprintf("\n══ %s ══\n", label))

  set.seed(PERM_SEED)
  rows <- lapply(LAYER_ORDER, function(cat_name) {
    d        <- deltas %>% filter(category == cat_name)
    v_deltas <- d %>% filter(drug_group == "verum")   %>% pull(delta)
    p_deltas <- d %>% filter(drug_group == "placebo") %>% pull(delta)
    n_v      <- length(v_deltas)
    n_p      <- length(p_deltas)
    obs_diff <- mean(v_deltas) - mean(p_deltas)

    if (n_v < 3 || n_p < 3) {
      cat(sprintf("FLAG %-10s: n too small for tests (nV=%d nP=%d)\n",
                  cat_name, n_v, n_p))
      return(data.frame(layer=cat_name, n_verum=n_v, n_placebo=n_p,
                        observed_diff=obs_diff, sw_p_verum=NA, sw_p_placebo=NA,
                        test1_method=NA, test1_stat=NA,
                        test1_p_raw=NA, test1_p_fdr=NA,
                        perm_p_raw=NA, perm_p_fdr=NA,
                        stringsAsFactors=FALSE))
    }

    sw_v <- shapiro.test(v_deltas)$p.value
    sw_p <- shapiro.test(p_deltas)$p.value

    if (sw_v > SW_ALPHA && sw_p > SW_ALPHA) {
      tt           <- t.test(v_deltas, p_deltas, var.equal = FALSE)
      test1_method <- "Welch t-test"
      test1_stat   <- unname(tt$statistic)
      test1_p_raw  <- tt$p.value
    } else {
      wt           <- wilcox.test(v_deltas, p_deltas, exact = FALSE)
      test1_method <- "Mann-Whitney U"
      test1_stat   <- unname(wt$statistic)
      test1_p_raw  <- wt$p.value
    }

    all_d     <- c(v_deltas, p_deltas)
    n_total   <- length(all_d)
    perm_null <- replicate(N_PERM, {
      idx <- sample(n_total, n_v, replace = FALSE)
      mean(all_d[idx]) - mean(all_d[-idx])
    })
    perm_p_raw <- (sum(abs(perm_null) >= abs(obs_diff)) + 1) / (N_PERM + 1)

    cat(sprintf(
      "  %-10s  nV=%d nP=%d  diff=%+.4f  [%s: stat=%.3f p=%.4f]  [perm p=%.4f]  [SW: V=%.3f P=%.3f]\n",
      cat_name, n_v, n_p, obs_diff,
      test1_method, test1_stat, test1_p_raw,
      perm_p_raw, sw_v, sw_p
    ))

    data.frame(layer=cat_name, n_verum=n_v, n_placebo=n_p,
               observed_diff=obs_diff, sw_p_verum=sw_v, sw_p_placebo=sw_p,
               test1_method=test1_method, test1_stat=test1_stat,
               test1_p_raw=test1_p_raw, test1_p_fdr=NA_real_,
               perm_p_raw=perm_p_raw, perm_p_fdr=NA_real_,
               stringsAsFactors=FALSE)
  })

  out <- do.call(rbind, rows)
  out$test1_p_fdr <- p.adjust(out$test1_p_raw, method = "BH")
  out$perm_p_fdr  <- p.adjust(out$perm_p_raw,  method = "BH")
  out
}

# ── Run three analyses ────────────────────────────────────────────────────────
deltas_sd25 <- exclude_sd25(deltas_full)
deltas_iqr  <- exclude_iqr(deltas_full)

report_exclusions(deltas_full, deltas_sd25, "±2.5 SD (per layer per group)")
report_exclusions(deltas_full, deltas_iqr,  "1.5×IQR (per layer across groups)")

res_full <- run_tests(deltas_full, "No exclusions")
res_sd25 <- run_tests(deltas_sd25, "±2.5 SD exclusions")
res_iqr  <- run_tests(deltas_iqr,  "1.5×IQR exclusions")

# ── Save individual tables ────────────────────────────────────────────────────
write.csv(res_full, file.path(OUT_DIR, "layer_drug_effect.csv"),     row.names = FALSE)
write.csv(res_sd25, file.path(OUT_DIR, "layer_drug_effect_sd25.csv"), row.names = FALSE)
write.csv(res_iqr,  file.path(OUT_DIR, "layer_drug_effect_iqr.csv"),  row.names = FALSE)

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

write.csv(comparison, file.path(OUT_DIR, "layer_outlier_comparison.csv"),
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

cat(sprintf("\nSaved to %s:\n", OUT_DIR))
cat("  layer_drug_effect.csv\n")
cat("  layer_drug_effect_sd25.csv\n")
cat("  layer_drug_effect_iqr.csv\n")
cat("  layer_outlier_comparison.csv\n")
