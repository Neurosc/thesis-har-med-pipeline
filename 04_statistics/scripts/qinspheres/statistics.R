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
# Input:  04_statistics/results/qinspheres/tables/qinspheres_auc.csv
# Output: 04_statistics/results/qinspheres/tables/layer_drug_effect.csv
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
OUT_CSV  <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres",
                      "tables", "layer_drug_effect.csv")

if (!file.exists(DATA_CSV))
  stop("Input not found — run 01_build_df.jl first:\n  ", DATA_CSV)

# ── Constants ─────────────────────────────────────────────────────────────────
LAYER_ORDER <- c("visual", "auditory", "motor", "extero", "intero", "mental")
PERM_SEED   <- 42L
N_PERM      <- 10000L
SW_ALPHA    <- 0.05

# ── Step 1: Load and verify ───────────────────────────────────────────────────
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

# ── Step 1: Aggregate to subject × session × layer ───────────────────────────
subj_layer <- df %>%
  filter(is.finite(auc)) %>%
  group_by(subject, session, drug_group, category) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop")

# Delta = post - pre per subject × layer
deltas <- subj_layer %>%
  pivot_wider(names_from = session, values_from = mean_auc) %>%
  rename(pre = `ses-01`, post = `ses-02`) %>%
  filter(!is.na(pre), !is.na(post)) %>%
  mutate(delta = post - pre)

# Report group sizes per layer
cat("\nSubjects per group per layer:\n")
deltas %>%
  group_by(category, drug_group) %>%
  summarise(n = n(), .groups = "drop") %>%
  pivot_wider(names_from = drug_group, values_from = n) %>%
  arrange(match(category, LAYER_ORDER)) %>%
  print()

# Flag any layer where a group has n < 5 (degenerate permutation null)
small <- deltas %>%
  group_by(category, drug_group) %>%
  summarise(n = n(), .groups = "drop") %>%
  filter(n < 5)
if (nrow(small) > 0) {
  cat("\nFLAG: following groups have n < 5 — permutation null unreliable:\n")
  print(small)
} else {
  cat("Group size check OK: all layers have n >= 5 in both groups.\n")
}

# ── Steps 2 & 3: Per-layer tests ──────────────────────────────────────────────
set.seed(PERM_SEED)
cat(sprintf("\nPermutation seed: %d,  n_perm: %d\n\n", PERM_SEED, N_PERM))

results <- lapply(LAYER_ORDER, function(cat_name) {
  d <- deltas %>% filter(category == cat_name)

  v_deltas <- d %>% filter(drug_group == "verum")   %>% pull(delta)
  p_deltas <- d %>% filter(drug_group == "placebo") %>% pull(delta)

  n_v      <- length(v_deltas)
  n_p      <- length(p_deltas)
  obs_diff <- mean(v_deltas) - mean(p_deltas)

  # ── Test 1: Shapiro-Wilk -> choose test ────────────────────────────────────
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

  # ── Permutation test ────────────────────────────────────────────────────────
  all_d     <- c(v_deltas, p_deltas)
  n_total   <- length(all_d)
  perm_null <- replicate(N_PERM, {
    idx <- sample(n_total, n_v, replace = FALSE)
    mean(all_d[idx]) - mean(all_d[-idx])
  })
  perm_p_raw <- (sum(abs(perm_null) >= abs(obs_diff)) + 1) / (N_PERM + 1)

  cat(sprintf(
    "%-10s  nV=%d nP=%d  obs_diff=%+.4f  [%s: stat=%.3f p=%.4f]  [perm p=%.4f]  [SW: V=%.3f P=%.3f]\n",
    cat_name, n_v, n_p, obs_diff,
    test1_method, test1_stat, test1_p_raw,
    perm_p_raw, sw_v, sw_p
  ))

  data.frame(
    layer         = cat_name,
    n_verum       = n_v,
    n_placebo     = n_p,
    observed_diff = obs_diff,
    sw_p_verum    = sw_v,
    sw_p_placebo  = sw_p,
    test1_method  = test1_method,
    test1_stat    = test1_stat,
    test1_p_raw   = test1_p_raw,
    test1_p_fdr   = NA_real_,
    perm_p_raw    = perm_p_raw,
    perm_p_fdr    = NA_real_,
    stringsAsFactors = FALSE
  )
})

out <- do.call(rbind, results)

# ── BH FDR across 6 layers ────────────────────────────────────────────────────
out$test1_p_fdr <- p.adjust(out$test1_p_raw, method = "BH")
out$perm_p_fdr  <- p.adjust(out$perm_p_raw,  method = "BH")

# ── Final report ──────────────────────────────────────────────────────────────
cat("\n── Results table ──\n")
print(
  out[, c("layer", "n_verum", "n_placebo", "test1_method",
          "test1_stat", "test1_p_raw", "test1_p_fdr",
          "perm_p_raw", "perm_p_fdr", "observed_diff")],
  digits = 4, row.names = FALSE
)

# ── Save ──────────────────────────────────────────────────────────────────────
write.csv(out, OUT_CSV, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", OUT_CSV))
