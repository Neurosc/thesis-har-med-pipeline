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
QBASE     <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres")
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
  cmp[[paste0(pl, "_raw_full_p")]] <- col(pl, "",     "p")
  cmp[[paste0(pl, "_raw_full_q")]] <- col(pl, "",     "q")
  cmp[[paste0(pl, "_raw_iqr_p")]]  <- col(pl, "_iqr", "p")
  cmp[[paste0(pl, "_raw_iqr_q")]]  <- col(pl, "_iqr", "q")
}
cmp[["maximal_resid_full_p"]] <- col("maximal", "_resid",     "p")
cmp[["maximal_resid_full_q"]] <- col("maximal", "_resid",     "q")
cmp[["maximal_resid_iqr_p"]]  <- col("maximal", "_resid_iqr", "p")
cmp[["maximal_resid_iqr_q"]]  <- col("maximal", "_resid_iqr", "q")

out_csv <- file.path(QROOT, "pipeline_comparison.csv")
write.csv(cmp, out_csv, row.names = FALSE)

show <- function(title, cols) {
  cat(sprintf("\n== %s ==\n", title))
  d <- cmp[, c("layer", cols)]
  names(d) <- c("layer", sub("^[a-z]+_(raw|resid)_", "", cols))
  print(d, row.names = FALSE)
}
show("RAW perm p  (full vs Tukey-IQR)",
     c("detrend_raw_full_p","detrend_raw_iqr_p","glm_raw_full_p","glm_raw_iqr_p",
       "maximal_raw_full_p","maximal_raw_iqr_p"))
show("RAW FDR q   (full vs Tukey-IQR)",
     c("detrend_raw_full_q","detrend_raw_iqr_q","glm_raw_full_q","glm_raw_iqr_q",
       "maximal_raw_full_q","maximal_raw_iqr_q"))
cat("\n== MAXIMAL motion-residualized (full vs Tukey-IQR) ==\n")
print(data.frame(layer = LAYERS,
                 full_p = cmp$maximal_resid_full_p, full_q = cmp$maximal_resid_full_q,
                 iqr_p  = cmp$maximal_resid_iqr_p,  iqr_q  = cmp$maximal_resid_iqr_q),
      row.names = FALSE)
cat(sprintf("\nSaved: %s\n", out_csv))
