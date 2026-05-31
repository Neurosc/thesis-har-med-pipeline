# 10_baseline_drug_balance.R
# Baseline balance check: placebo vs verum at ses-01
#
# Test A — Pooled: grand mean AUC per subject across all ROIs (ses-01)
# Test B — Per-category: mean AUC per subject × category (ses-01)
#
# Input:  04_statistics/results/glasser_self_nonself_model_ready.csv
# Output: 04_statistics/results/baseline_balance_pooled.csv
#         04_statistics/results/baseline_balance_per_category.csv
#
# Run from repo root:
#   Rscript 04_statistics/scripts/10_baseline_drug_balance.R

suppressPackageStartupMessages(library(dplyr))

# ── Paths ──────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
  REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
} else {
  REPO_ROOT <- normalizePath(".")
}

DATA_CSV   <- file.path(REPO_ROOT, "04_statistics", "results",
                         "glasser_self_nonself_model_ready.csv")
OUT_POOLED <- file.path(REPO_ROOT, "04_statistics", "results",
                         "baseline_balance_pooled.csv")
OUT_CAT    <- file.path(REPO_ROOT, "04_statistics", "results",
                         "baseline_balance_per_category.csv")

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n10_baseline_drug_balance.R\n", SEP, "\n\n", sep = "")

# ── Load & prepare ─────────────────────────────────────────────────────────────
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n", nrow(df)))

ses01 <- df[df$session == "ses-01", ]
cat(sprintf("ses-01 subset: %d rows\n\n", nrow(ses01)))

CAT_MAP <- c(
  Interoception = "Interoceptive Self",
  Exteroception = "Exteroceptive Self",
  Cognition     = "Mental Self",
  nonself       = "Sensory-Motor"
)
ses01$category <- CAT_MAP[ses01$self_layer]

CAT_ORDER <- c("Interoceptive Self", "Exteroceptive Self",
               "Mental Self", "Sensory-Motor")

# ── Helpers ────────────────────────────────────────────────────────────────────
cohens_d <- function(x, y) {
  sp <- sqrt(((length(x) - 1) * var(x) + (length(y) - 1) * var(y)) /
               (length(x) + length(y) - 2))
  if (sp == 0) return(0)
  (mean(x) - mean(y)) / sp
}

d_label <- function(d) {
  a <- abs(d)
  if (a < 0.2) "negligible" else if (a < 0.5) "small" else
  if (a < 0.8) "medium"     else "large"
}

run_tests <- function(plac, verm) {
  t_r  <- t.test(plac, verm, var.equal = FALSE)
  mw_r <- wilcox.test(plac, verm, exact = FALSE)
  dv   <- cohens_d(plac, verm)
  list(t    = unname(t_r$statistic),
       df   = unname(t_r$parameter),
       t_p  = t_r$p.value,
       U    = unname(mw_r$statistic),
       U_p  = mw_r$p.value,
       d    = dv,
       d_lab = d_label(dv))
}

print_tests <- function(st, n_p, n_v) {
  cat(sprintf("    n_placebo=%d  n_verum=%d\n", n_p, n_v))
  cat(sprintf("    Welch t=%.4f (df=%.2f), p=%.4f\n", st$t, st$df, st$t_p))
  cat(sprintf("    Mann-Whitney U=%.0f, p=%.4f\n", st$U, st$U_p))
  cat(sprintf("    Cohen's d=%+.4f (%s)\n\n", st$d, st$d_lab))
}

# ── Test A: Pooled (all ROIs combined) ─────────────────────────────────────────
cat(SEP, "\nTEST A — Pooled baseline (all ROIs)\n", SEP, "\n", sep = "")

pooled <- ses01 %>%
  group_by(subject, group) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop")

plac_p <- pooled$mean_auc[pooled$group == "placebo"]
verm_p <- pooled$mean_auc[pooled$group == "verum"]
st_p   <- run_tests(plac_p, verm_p)
print_tests(st_p, length(plac_p), length(verm_p))

pooled_row <- data.frame(
  t = st_p$t, df = st_p$df, t_p = st_p$t_p,
  U = st_p$U, U_p = st_p$U_p,
  d = st_p$d, d_lab = st_p$d_lab,
  stringsAsFactors = FALSE
)
write.csv(pooled_row, OUT_POOLED, row.names = FALSE)
cat(sprintf("Saved: %s\n\n", OUT_POOLED))

# ── Test B: Per-category ───────────────────────────────────────────────────────
cat(SEP, "\nTEST B — Per-category baseline\n", SEP, "\n", sep = "")

cat_rows <- list()
for (cat in CAT_ORDER) {
  sub <- ses01[ses01$category == cat, ]
  subj_means <- sub %>%
    group_by(subject, group) %>%
    summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop")

  plac <- subj_means$mean_auc[subj_means$group == "placebo"]
  verm <- subj_means$mean_auc[subj_means$group == "verum"]
  st   <- run_tests(plac, verm)

  cat(sprintf("  [%s]\n", cat))
  print_tests(st, length(plac), length(verm))

  cat_rows[[cat]] <- data.frame(
    category = cat,
    t = st$t, df = st$df, t_p = st$t_p,
    U = st$U, U_p = st$U_p,
    d = st$d, d_lab = st$d_lab,
    stringsAsFactors = FALSE
  )
}

cat_tbl <- do.call(rbind, cat_rows)
rownames(cat_tbl) <- NULL
write.csv(cat_tbl, OUT_CAT, row.names = FALSE)
cat(sprintf("Saved: %s\n\n", OUT_CAT))

cat(SEP, "\nDONE\n", SEP, "\n", sep = "")
