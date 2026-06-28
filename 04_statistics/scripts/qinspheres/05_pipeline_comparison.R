# 05_pipeline_comparison.R — Cross-pipeline robustness summary (detrend/glm/maximal).
# Collects the per-layer drug-effect permutation p (raw + motion-residualized,
# no-exclusion variant) from each pipeline's tables and writes a single comparison
# table + a focused console summary.
#
# Input : 04_statistics/results/qinspheres/{pipeline}/tables/layer_drug_effect.csv
#         04_statistics/results/qinspheres/{pipeline}/tables/layer_drug_effect_resid.csv
# Output: 04_statistics/results/qinspheres/pipeline_comparison.csv
#
# Run from repo root:
#   Rscript 04_statistics/scripts/qinspheres/05_pipeline_comparison.R

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

grab <- function(pipeline, kind) {   # kind = "" (raw) or "_resid"
  f <- file.path(QROOT, pipeline, "tables", paste0("layer_drug_effect", kind, ".csv"))
  if (!file.exists(f)) { warning("missing: ", f); return(NULL) }
  d <- read.csv(f, stringsAsFactors = FALSE)
  data.frame(layer = d$layer, diff = d$observed_diff,
             p = d$perm_p_raw, q = d$perm_p_fdr, stringsAsFactors = FALSE)
}

cmp <- data.frame(layer = LAYERS, stringsAsFactors = FALSE)
for (pl in PIPELINES) {
  raw <- grab(pl, "");      res <- grab(pl, "_resid")
  cmp[[paste0(pl, "_raw_p")]]   <- round(raw$p[match(LAYERS, raw$layer)], 4)
  cmp[[paste0(pl, "_raw_q")]]   <- round(raw$q[match(LAYERS, raw$layer)], 3)
  cmp[[paste0(pl, "_resid_p")]] <- round(res$p[match(LAYERS, res$layer)], 4)
  cmp[[paste0(pl, "_resid_q")]] <- round(res$q[match(LAYERS, res$layer)], 3)
}

out_csv <- file.path(QROOT, "pipeline_comparison.csv")
write.csv(cmp, out_csv, row.names = FALSE)

cat("\n==================== RAW ΔAUC drug-effect (perm p) ====================\n")
raw_tab <- cmp[, c("layer", "detrend_raw_p", "glm_raw_p", "maximal_raw_p")]
names(raw_tab) <- c("layer", "detrend", "glm", "maximal")
print(raw_tab, row.names = FALSE)

cat("\n============ MOTION-RESIDUALIZED ΔAUC drug-effect (perm p) ============\n")
res_tab <- cmp[, c("layer", "detrend_resid_p", "glm_resid_p", "maximal_resid_p")]
names(res_tab) <- c("layer", "detrend", "glm", "maximal")
print(res_tab, row.names = FALSE)

cat(sprintf("\nSaved: %s\n", out_csv))
