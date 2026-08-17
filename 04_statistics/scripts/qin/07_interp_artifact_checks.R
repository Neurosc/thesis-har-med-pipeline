# 07_interp_artifact_checks.R — interpolation-artifact robustness checks for the AUC result.
#
#   CHECK A — Does censoring inflate AUC? (does the artifact exist / did covariates handle it)
#     A1 raw     : cor(pcf, region-mean AUC) across the 70 runs. A POSITIVE r would be the
#                  fingerprint of interpolation manufacturing autocorrelation -> longer AUC.
#     A2 residual: cor(pcf, residuals) from the AUC model with pcf/pcf_sq DROPPED
#                  (AUC ~ session*arm + mean_fd + (1|subject)). This is the pcf-AUC
#                  relationship that the model's OTHER terms do NOT absorb — i.e. the
#                  variance pcf/pcf_sq are in the model to handle. Raw = "does it exist";
#                  residual = "do the covariates catch what's left".
#
#   CHECK B — Does the pipeline disagreement track motion? (needs the maximal_nocensor recompute)
#     Per subject, dAUC_pipe = AUC(maximal) - AUC(maximal_nocensor); cor(dAUC_pipe, mean FD).
#     The claim being tested is that the SIZE of the disagreement grows with motion: censoring
#     can only act where there are frames to censor, so low movers should sit near dAUC_pipe = 0
#     and high movers should diverge. If so, the pipelines differ for the expected (motion)
#     reason and the censored version is the trustworthy one.
#
#     READ THE SIGN CAREFULLY. dAUC_pipe is NEGATIVE in every region: the UNcensored run gives
#     the LONGER timescale, because retained motion contributes slow, strongly autocorrelated
#     signal that inflates AUC. (Note this is the opposite of the original worry, which was that
#     LS interpolation would manufacture autocorrelation; empirically the inflating force is the
#     motion that censoring removes, not the interpolation that replaces it.) Since dAUC_pipe is
#     negative throughout, "diverges more" means MORE negative — so r_vs_fd < 0 is the PASSING
#     result, not a failure. The test is about |dAUC_pipe| growing with FD; the sign of r merely
#     follows the sign of dAUC_pipe. An earlier version of this header stated the criterion as
#     r_vs_fd > 0, which would invert the conclusion.
#
#     Skipped (with a note) until the control pipeline's AUC table exists.
#
# Usage : Rscript 04_statistics/scripts/qin/07_interp_artifact_checks.R [pipeline] [atlas]
#         defaults: maximal qinspheres
# Output: results/acw/{spheres|parcels/self_regions}/{pipeline}/tables/robust_censoring_vs_auc.csv
#         (+ robust_pipeline_diff.csv when Check B runs)
#         results/acw/{spheres|parcels/self_regions}/{pipeline}/figures/qin_robust_censoring_vs_auc.png

needed <- c("lme4", "dplyr", "tidyr")
for (pkg in needed)
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
suppressPackageStartupMessages({ library(lme4); library(dplyr); library(tidyr) })

# --- settings & paths -------------------------------------------------------
args     <- commandArgs(trailingOnly = TRUE)
PIPELINE <- if (length(args) >= 1) args[1] else "maximal"
ATLAS    <- if (length(args) >= 2) args[2] else "qinspheres"
CONTROL  <- "maximal_nocensor"                          # Check B comparison pipeline

find_script_dir <- function() {
  fa <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  if (length(fa)) return(dirname(normalizePath(sub("^--file=", "", fa[1]))))
  for (i in rev(seq_len(sys.nframe())))
    if (!is.null(sys.frame(i)$ofile)) return(dirname(normalizePath(sys.frame(i)$ofile)))
  getwd()
}
REPO      <- normalizePath(file.path(find_script_dir(), "..", "..", ".."))
# AUC-only script; results tree = results/acw/{spheres|parcels/self_regions}/{pipeline}/...
ATLAS_DIR <- if (ATLAS == "qinspheres") "spheres" else file.path("parcels", "self_regions")
ACW_BASE  <- file.path(REPO, "04_statistics", "results", "acw", ATLAS_DIR)
auc_path  <- function(pl) file.path(ACW_BASE, pl, "tables", "auc.csv")
COV_CSV   <- file.path(REPO, "99_QC", "01_motion_qc", "results", "fd_covariates_long_thresh03.csv")
TABLE_DIR <- file.path(ACW_BASE, PIPELINE, "tables")
FIG_DIR   <- file.path(ACW_BASE, PIPELINE, "figures")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

CATEGORY_TO_REGION <- c(visual = "visual", intero = "intero", extero = "extero",
                        mental = "cognition", auditory = "auditory")
REGIONS <- unname(CATEGORY_TO_REGION)

# --- run-level region-mean AUC + per-run motion (pcf, mean_fd) --------------
region_run_means <- function(pl) {
  read.csv(auc_path(pl)) %>%
    filter(is.finite(auc), auc > 0, category %in% names(CATEGORY_TO_REGION)) %>%
    group_by(subject, session, drug_group, category) %>%
    summarise(AUC = mean(auc), .groups = "drop") %>%
    mutate(region = CATEGORY_TO_REGION[category])
}
cov <- read.csv(COV_CSV) %>% select(subject, session, pcf, mean_fd = mean_fd_retained)

dat <- region_run_means(PIPELINE) %>%
  left_join(cov, by = c("subject", "session")) %>%
  mutate(subject = factor(subject),
         session = factor(session), arm = factor(drug_group))

# ===========================================================================
# CHECK A — censoring (pcf) vs AUC: raw + residual (pcf/pcf_sq dropped)
# ===========================================================================
checkA_one <- function(region_name) {
  d <- dat %>% filter(region == region_name)
  raw <- cor.test(d$pcf, d$AUC)                                  # A1: does the artifact exist?
  m   <- suppressWarnings(suppressMessages(                      # A2: model WITHOUT pcf/pcf_sq
           lmer(AUC ~ session * arm + mean_fd + (1 | subject), data = d, REML = TRUE)))
  res <- cor.test(d$pcf, resid(m))                              # leftover pcf-AUC the model misses
  data.frame(region = region_name, n = nrow(d),
             raw_r = round(unname(raw$estimate), 3), raw_p = round(raw$p.value, 4),
             resid_r = round(unname(res$estimate), 3), resid_p = round(res$p.value, 4))
}
checkA <- do.call(rbind, lapply(REGIONS, checkA_one))
checkA$raw_q   <- round(p.adjust(checkA$raw_p, "BH"), 4)         # FDR across 5 regions
checkA$resid_q <- round(p.adjust(checkA$resid_p, "BH"), 4)
write.csv(checkA, file.path(TABLE_DIR, "robust_censoring_vs_auc.csv"), row.names = FALSE)

cat(sprintf("\n######## CHECK A — censoring (pcf) vs AUC — %s/%s (70 runs) ########\n", ATLAS, PIPELINE))
cat("raw_r  = cor(pcf, raw AUC): +ve => interpolation inflating timescales (the artifact).\n")
cat("resid_r= cor(pcf, resid | pcf/pcf_sq dropped): pcf-AUC link the other covariates miss.\n\n")
print(checkA[, c("region","n","raw_r","raw_p","raw_q","resid_r","resid_p","resid_q")], row.names = FALSE)

# figure: pcf vs AUC per region, with raw + residual r annotated
png(file.path(FIG_DIR, "qin_robust_censoring_vs_auc.png"), width = 1650, height = 1050, res = 140)
old <- par(mfrow = c(2, 3), mar = c(4, 4, 3, 1), oma = c(0, 0, 2.2, 0))
for (rg in REGIONS) {
  d <- dat %>% filter(region == rg); a <- checkA[checkA$region == rg, ]
  plot(d$pcf, d$AUC, pch = 16, col = rgb(0, 0, 0, 0.45),
       xlab = "% censored (pcf)", ylab = "region-mean AUC (s)",
       main = sprintf("%s\nraw r=%.2f (p=%.3f) | resid r=%.2f (p=%.3f)",
                      rg, a$raw_r, a$raw_p, a$resid_r, a$resid_p), cex.main = 0.85)
  abline(lm(AUC ~ pcf, d), col = "red", lwd = 2)
}
plot.new(); mtext(sprintf("Censoring vs AUC — %s/%s", ATLAS, PIPELINE), outer = TRUE, cex = 1)
par(old); dev.off()

# ===========================================================================
# CHECK B — between-pipeline AUC difference vs motion (needs maximal_nocensor)
# ===========================================================================
if (file.exists(auc_path(CONTROL))) {
  ctrl <- region_run_means(CONTROL) %>% select(subject, session, region, AUC_ctrl = AUC)
  diff <- region_run_means(PIPELINE) %>% select(subject, session, region, AUC_max = AUC) %>%
    inner_join(ctrl, by = c("subject", "session", "region")) %>%
    left_join(cov, by = c("subject", "session")) %>%
    mutate(dAUC_pipe = AUC_max - AUC_ctrl)
  checkB <- do.call(rbind, lapply(REGIONS, function(rg) {
    d <- diff %>% filter(region == rg); ct <- cor.test(d$mean_fd, d$dAUC_pipe)
    data.frame(region = rg, n = nrow(d),
               mean_dAUC_pipe = round(mean(d$dAUC_pipe), 4),
               r_vs_fd = round(unname(ct$estimate), 3), p = round(ct$p.value, 4))
  }))
  checkB$q <- round(p.adjust(checkB$p, "BH"), 4)
  write.csv(checkB, file.path(TABLE_DIR, "robust_pipeline_diff.csv"), row.names = FALSE)
  cat(sprintf("\n######## CHECK B — (maximal - %s) AUC vs mean FD — %s ########\n", CONTROL, ATLAS))
  cat("dAUC_pipe < 0 => the UNcensored run gives the longer timescale (retained motion adds slow,\n")
  cat("autocorrelated signal that inflates AUC). Because dAUC_pipe is negative throughout, the\n")
  cat("disagreement widening with motion appears as r_vs_fd < 0 — that is the PASS: the pipelines\n")
  cat("differ for the expected (motion) reason and the censored version is the trustworthy one.\n")
  cat("Do NOT read r_vs_fd < 0 as a failure; the test is |dAUC_pipe| growing with FD.\n\n")
  print(checkB, row.names = FALSE)
} else {
  cat(sprintf("\n[CHECK B skipped] control AUC not found: %s\n", auc_path(CONTROL)))
  cat("  Build it first (server denoise+extract, local ACW+build):\n")
  cat("    python 01_denoising/scripts/01_denoise_all.py --pipeline maximal_nocensor       # server\n")
  cat("    PIPELINES=maximal_nocensor python 02_timeseries_extraction/scripts/01_extract_sphere_timeseries.py  # server\n")
  cat("    PIPELINES=maximal_nocensor julia 03_intrinsic_neural_metrics/scripts/acw/01_spheres.jl              # local\n")
  cat("    PIPELINES=maximal_nocensor julia 04_statistics/scripts/qin/01_build_auc.jl qinspheres              # local\n")
  cat("  then re-run this script.\n")
}

cat(sprintf("\nWrote: robust_censoring_vs_auc.csv, qin_robust_censoring_vs_auc.png -> %s\n", TABLE_DIR))
