# exploratory/per_category_subjectlevel_tests.R
#
# df CHECK: confirm the per-category drug×session / pre→post p-values (especially
# Sensory-Motor) are computed at the SUBJECT level, not with inflated ROI-level df.
#
# Reuses 05_per_category_glasser.R's exact ROI→subject collapse:
#   group_by(subject, session, group, self_layer) %>% summarise(mean(auc))
# applied to all four categories. 05 itself only collapses three layers
# (Interoception, Exteroception, nonself) to build the IE_diffs; here the IDENTICAL
# collapse mechanism is extended to the 4th layer (Cognition / Mental Self), which
# is present in model_ready.csv (12 parcels).
#
# Category labels (self_layer -> label):
#   Interoception -> Interoception
#   Exteroception -> Exteroception
#   nonself       -> Sensory-Motor        (matches 05's "Sensory-Motor" usage)
#   Cognition     -> Mental Self          (no 05 precedent; CAT_MAP label from 06)
#
# Per subject × session: cat_mean = mean(AUC over that category's ROIs).
# Per category:
#   placebo pre→post : paired t.test(post, pre, paired = TRUE)   -> est = mean(post−pre)
#   verum   pre→post : paired t.test(post, pre, paired = TRUE)   -> est = mean(post−pre)
#   drug × session   : Welch t.test(verum Δ, placebo Δ)          -> est = mean(verum Δ) − mean(placebo Δ)
#                      (Δ = per-subject post − pre)
#
# Expected df (35 subjects: 18 placebo, 17 verum):
#   within-group paired = group_n − 1  (~17 placebo, ~16 verum)  — subject-level
#   drug × session Welch ≈ 30–33 (≤ n − 2)                       — subject-level
# Any test with df in the hundreds/thousands signals ROI-level pseudoreplication;
# the script then ABORTS (guard: df > 100).
#
# Subjects come from model_ready.csv, already filtered to the canonical 35 included
# subjects upstream (01_build_dataframe.jl) — no subject list hardcoded.
#
# Run from repo root:
#   Rscript 04_statistics/scripts/exploratory/per_category_subjectlevel_tests.R

suppressPackageStartupMessages({
  library(dplyr)
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
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)

DATA_CSV <- file.path(TABLES_DIR, "model_ready.csv")
if (!file.exists(DATA_CSV))
  stop("Input not found — run 02_lmm.R first: ", DATA_CSV)

SEP <- paste(rep("=", 78), collapse = "")
cat(SEP, "\nper_category_subjectlevel_tests.R — config: ", TAG, "\n", SEP, "\n\n", sep = "")

# ── ROI→subject collapse (identical mechanism to 05) ─────────────────────────
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n", nrow(df)))

CAT_ORDER <- c("Interoception", "Exteroception", "nonself", "Cognition")
CAT_LABEL <- c(Interoception = "Interoception", Exteroception = "Exteroception",
               nonself = "Sensory-Motor", Cognition = "Mental Self")

cat_means <- df %>%
  filter(self_layer %in% CAT_ORDER) %>%
  group_by(subject, session, group, self_layer) %>%
  summarise(cat_mean = mean(auc, na.rm = TRUE), .groups = "drop") %>%
  as.data.frame()

n_subj <- length(unique(cat_means$subject))
grp_n  <- tapply(unique(cat_means[, c("subject", "group")])$group,
                 unique(cat_means[, c("subject", "group")])$group, length)
cat(sprintf("Subjects: %d   (placebo %d, verum %d)\n\n",
            n_subj, grp_n[["placebo"]], grp_n[["verum"]]))

# ── Subject-level tests per category ─────────────────────────────────────────
DF_GUARD <- 100   # subject-level df is ≤ ~33; ROI-level would be in the thousands

run_cat <- function(layer) {
  d    <- cat_means[cat_means$self_layer == layer, ]
  pre  <- d[d$session == "ses-01", c("subject", "group", "cat_mean")]
  post <- d[d$session == "ses-02", c("subject", "cat_mean")]
  names(pre)[3] <- "val_pre"; names(post)[2] <- "val_post"
  both <- merge(pre, post, by = "subject")
  both$delta <- both$val_post - both$val_pre

  bp <- both[both$group == "placebo", ]
  bv <- both[both$group == "verum", ]
  t_plac <- t.test(bp$val_post, bp$val_pre, paired = TRUE)     # placebo pre→post
  t_verm <- t.test(bv$val_post, bv$val_pre, paired = TRUE)     # verum   pre→post
  t_int  <- t.test(both$delta[both$group == "verum"],          # drug × session (Welch)
                   both$delta[both$group == "placebo"], var.equal = FALSE)

  data.frame(
    category = unname(CAT_LABEL[layer]),
    test     = c("placebo_prepost", "verum_prepost", "drug_x_session"),
    estimate = c(unname(t_plac$estimate),
                 unname(t_verm$estimate),
                 unname(t_int$estimate[1] - t_int$estimate[2])),
    t        = c(unname(t_plac$statistic), unname(t_verm$statistic), unname(t_int$statistic)),
    df       = c(unname(t_plac$parameter), unname(t_verm$parameter), unname(t_int$parameter)),
    p        = c(t_plac$p.value, t_verm$p.value, t_int$p.value),
    row.names = NULL, stringsAsFactors = FALSE)
}

res <- do.call(rbind, lapply(CAT_ORDER, run_cat))

# ── df sanity guard ──────────────────────────────────────────────────────────
if (any(res$df > DF_GUARD)) {
  bad <- res[res$df > DF_GUARD, ]
  print(bad, row.names = FALSE)
  stop(sprintf(paste0("df above %d detected — these tests are NOT subject-level ",
                      "(ROI-level pseudoreplication). Aborting."), DF_GUARD))
}
cat(sprintf("df sanity check passed: all df ≤ %d (subject-level).\n", DF_GUARD))
cat("  within-group paired df = group_n − 1 (~17 placebo, ~16 verum);\n")
cat("  drug×session Welch df ≈ 30–33.  None inflated.\n\n")

# ── Write + print ────────────────────────────────────────────────────────────
OUT_CSV <- file.path(TABLES_DIR, "per_category_subjectlevel_tests.csv")
write.csv(res, OUT_CSV, row.names = FALSE)
cat(sprintf("Table saved: %s\n\n", OUT_CSV))

cat(SEP, "\nPer-category subject-level tests (estimate in seconds)\n", SEP, "\n", sep = "")
cat(sprintf("  %-15s %-16s %10s %8s %7s %10s\n",
            "Category", "Test", "Estimate", "t", "df", "p"))
cat(strrep("-", 72), "\n")
for (i in seq_len(nrow(res))) {
  r <- res[i, ]
  cat(sprintf("  %-15s %-16s %+10.4f %8.3f %7.1f %10.4f\n",
              r$category, r$test, r$estimate, r$t, r$df, r$p))
}
cat(strrep("-", 72), "\n")

cat("\n", SEP, "\nDONE (subject-level df confirmed)\n", SEP, "\n", sep = "")
