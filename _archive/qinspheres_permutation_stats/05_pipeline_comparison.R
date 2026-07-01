# 05_pipeline_comparison.R — Cross-pipeline robustness summary (detrend/glm/maximal).
# Collects per-layer drug-effect permutation p + BH-FDR q from each pipeline's tables,
# for the no-exclusion ("full") AND the Tukey 1.5xIQR outlier-trim variants, raw and
# motion-residualized (residualized = maximal only).
#
# Input : 04_statistics/results/qinspheres[/{metric}]/{pipeline}/tables/layer_drug_effect{,_iqr,_resid,_resid_iqr}.csv
# Output: .../pipeline_comparison.csv  + console tables
#
# Run from repo root:
#   Rscript 04_statistics/scripts/qinspheres/05_pipeline_comparison.R [metric]

suppressPackageStartupMessages(library(dplyr))

args     <- commandArgs(trailingOnly = FALSE)
file_arg <- grep("^--file=", args, value = TRUE)
REPO_ROOT <- if (length(file_arg))
  normalizePath(file.path(dirname(normalizePath(sub("^--file=", "", file_arg[1]))),
                          "..", "..", "..")) else normalizePath(".")

METRIC    <- { a <- commandArgs(trailingOnly = TRUE); if (length(a) >= 1) a[1] else "auc" }
cat(sprintf("Metric: %s\n", METRIC))
ATLAS     <- { .aa <- commandArgs(trailingOnly = TRUE); if (length(.aa) >= 2) .aa[2] else "qinspheres" }
QBASE     <- file.path(REPO_ROOT, "04_statistics", "results", ATLAS)
QROOT     <- if (METRIC == "auc") QBASE else file.path(QBASE, METRIC)
PIPELINES <- c("detrend", "glm", "maximal")
LAYERS    <- c("visual", "auditory", "motor", "extero", "intero", "mental")

# perm p + BH-FDR q for one (pipeline, variant); NA-filled if the file is absent
# (residualization is maximal-only, so resid variants are missing for detrend/glm).
col <- function(pipeline, suffix, what) {
  f <- file.path(QROOT, pipeline, "tables", paste0("layer_drug_effect", suffix, ".csv"))
  if (!file.exists(f)) return(rep(NA_real_, length(LAYERS)))
  d <- read.csv(f, stringsAsFactors = FALSE)
  v <- if (what == "p") d$perm_p_raw else d$perm_p_fdr
  round(v[match(LAYERS, d$layer)], 4)
}

cmp <- data.frame(layer = LAYERS)
for (pl in PIPELINES) {
  cmp[[paste0(pl, "_raw_full_p")]]   <- col(pl, "",           "p")
  cmp[[paste0(pl, "_raw_full_q")]]   <- col(pl, "",           "q")
  cmp[[paste0(pl, "_raw_iqr_p")]]    <- col(pl, "_iqr",       "p")
  cmp[[paste0(pl, "_raw_iqr_q")]]    <- col(pl, "_iqr",       "q")
  cmp[[paste0(pl, "_resid_full_p")]] <- col(pl, "_resid",     "p")
  cmp[[paste0(pl, "_resid_full_q")]] <- col(pl, "_resid",     "q")
  cmp[[paste0(pl, "_resid_iqr_p")]]  <- col(pl, "_resid_iqr", "p")
  cmp[[paste0(pl, "_resid_iqr_q")]]  <- col(pl, "_resid_iqr", "q")
}

out_csv <- file.path(QROOT, "pipeline_comparison.csv")
write.csv(cmp, out_csv, row.names = FALSE)

p3 <- function(title, key) {  # key e.g. "raw_full_q" -> prints detrend/glm/maximal
  cat(sprintf("\n== %s ==\n", title))
  print(data.frame(layer   = LAYERS,
                   detrend = cmp[[paste0("detrend_", key)]],
                   glm     = cmp[[paste0("glm_",     key)]],
                   maximal = cmp[[paste0("maximal_", key)]]), row.names = FALSE)
}
p3("RAW   perm p (full)", "raw_full_p")
p3("RAW   FDR q  (full)", "raw_full_q")
p3("RESID perm p (full)", "resid_full_p")
p3("RESID FDR q  (full)", "resid_full_q")
cat(sprintf("\n(Tukey-IQR variants for raw and resid are in %s)\n", out_csv))
