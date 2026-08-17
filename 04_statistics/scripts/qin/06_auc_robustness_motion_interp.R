# 06_auc_robustness_motion_interp.R — two robustness checks for the exteroception
# AUC (intrinsic-timescale) result. Descriptive, on the primary config (maximal / qinspheres, n=35).
#
#   CHECK 1 — Does the effect track motion?
#     Per region, regress each subject's dAUC (= mean AUC post - pre) on their
#     d(mean FD) (= mean_fd post - pre). If the timescale change correlates with a
#     motion change, it is a motion artifact; if not, the covariate adjustment held.
#
#   CHECK 2 — Does it survive without interpolation?
#     Re-test exteroception in the CLEANEST subset: subjects with <20% censored frames
#     in BOTH sessions (pcf_pre<20 & pcf_post<20), where L-S interpolation touched few
#     frames, so it cannot be inventing the effect. Same within-arm / DiD contrasts as
#     the full sample. (A true no-interpolation recompute needs re-denoising on the
#     server; this subset is the feasible local proxy — interpolation acts minimally.)
#
# Usage : Rscript 04_statistics/scripts/qin/06_auc_robustness_motion_interp.R [pipeline] [atlas]
#         defaults: maximal qinspheres
# Output: results/acw/{spheres|parcels/self_regions}/{pipeline}/tables/robust_dauc_vs_motion.csv
#         results/acw/{spheres|parcels/self_regions}/{pipeline}/tables/robust_lowcens_extero.csv
#         results/acw/{spheres|parcels/self_regions}/{pipeline}/figures/qin_robust_dauc_vs_dfd.png

suppressPackageStartupMessages({ library(dplyr); library(tidyr) })

# --- settings & paths -------------------------------------------------------
args     <- commandArgs(trailingOnly = TRUE)
PIPELINE <- if (length(args) >= 1) args[1] else "maximal"
ATLAS    <- if (length(args) >= 2) args[2] else "qinspheres"
PCF_CUT  <- 20                                         # "low censoring" = pcf < 20% both sessions

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
CONFIG_DIR<- file.path(REPO, "04_statistics", "results", "acw", ATLAS_DIR, PIPELINE)
AUC_CSV   <- file.path(CONFIG_DIR, "tables", "auc.csv")
COV_CSV   <- file.path(REPO, "99_QC", "01_motion_qc", "results", "fd_covariates_wide_thresh03.csv")
TABLE_DIR <- file.path(CONFIG_DIR, "tables")
FIG_DIR   <- file.path(CONFIG_DIR, "figures")
dir.create(FIG_DIR, recursive = TRUE, showWarnings = FALSE)

# 5 analysed regions (motor excluded), display order.
CATEGORY_TO_REGION <- c(visual = "visual", intero = "intero", extero = "extero",
                        mental = "cognition", auditory = "auditory")
REGIONS <- unname(CATEGORY_TO_REGION)

# --- build per-subject x region dAUC + per-subject motion deltas ------------
raw <- read.csv(AUC_CSV)
raw <- raw[is.finite(raw$auc) & raw$auc > 0, ]         # same filter as the primary analysis

region_wide <- raw %>%
  group_by(subject, session, drug_group, category) %>%
  summarise(AUC = mean(auc), .groups = "drop") %>%
  filter(category %in% names(CATEGORY_TO_REGION)) %>%
  mutate(region = CATEGORY_TO_REGION[category],
         session = ifelse(session == "ses-01", "pre", "post")) %>%
  select(subject, arm = drug_group, region, session, AUC) %>%
  pivot_wider(names_from = session, values_from = AUC) %>%
  mutate(dAUC = post - pre)

cov <- read.csv(COV_CSV) %>%
  transmute(subject,
            dfd     = mean_fd_retained_post - mean_fd_retained_pre,
            pcf_pre, pcf_post,
            low_cens = pcf_pre < PCF_CUT & pcf_post < PCF_CUT)

dat <- left_join(region_wide, cov, by = "subject")
dat$region <- factor(dat$region, levels = REGIONS)

# ===========================================================================
# CHECK 1 — dAUC vs d(mean FD), per region (and extero split by arm)
# ===========================================================================
cor_row <- function(d, label) {
  ct <- suppressWarnings(cor.test(d$dAUC, d$dfd))       # Pearson r + p
  sl <- coef(lm(dAUC ~ dfd, d))[["dfd"]]                # regression slope (AUC per mm FD)
  data.frame(region = label, n = nrow(d),
             slope = round(sl, 3), r = round(unname(ct$estimate), 3),
             p = round(ct$p.value, 4))
}
check1 <- do.call(rbind, lapply(REGIONS, function(rg) cor_row(dat[dat$region == rg, ], rg)))
# extero, split by arm (the placebo lengthening is the specific worry)
ex <- dat[dat$region == "extero", ]
check1_extero_arm <- do.call(rbind, lapply(c("placebo", "verum"),
  function(a) cor_row(ex[ex$arm == a, ], paste0("extero_", a))))

check1$q <- round(p.adjust(check1$p, "BH"), 4)          # FDR across the 5 regions
write.csv(rbind(check1[c("region","n","slope","r","p")], check1_extero_arm),
          file.path(TABLE_DIR, "robust_dauc_vs_motion.csv"), row.names = FALSE)

cat(sprintf("\n######## CHECK 1 — dAUC vs d(mean FD) — %s/%s (n=%d) ########\n",
            ATLAS, PIPELINE, length(unique(dat$subject))))
cat("slope = dAUC per +1mm dFD; r/p = Pearson; a positive r near extero would flag a motion artifact\n\n")
print(check1, row.names = FALSE)
cat("\n-- exteroception split by arm --\n"); print(check1_extero_arm, row.names = FALSE)

# ===========================================================================
# CHECK 2 — exteroception in the low-censoring (minimal-interpolation) subset
# ===========================================================================
# Descriptive per-arm dAUC (mean, one-sample t vs 0) + between-arm DiD (Welch t),
# for the FULL sample and the low-censoring SUBSET.
arm_summary <- function(d) {
  do.call(rbind, lapply(c("placebo", "verum"), function(a) {
    x <- d$dAUC[d$arm == a]; tt <- t.test(x)
    data.frame(arm = a, n = length(x), mean_dAUC = round(mean(x), 4),
               t = round(unname(tt$statistic), 3), p = round(tt$p.value, 4))
  }))
}
did_row <- function(d) {                                # verum - placebo difference in dAUC
  tt <- t.test(dAUC ~ arm, d)                           # Welch; groups placebo vs verum
  data.frame(arm = "DiD (verum-placebo)", n = nrow(d),
             mean_dAUC = round(diff(rev(tapply(d$dAUC, d$arm, mean))), 4),
             t = round(unname(tt$statistic), 3), p = round(tt$p.value, 4))
}
ex_full <- rbind(arm_summary(ex),            did_row(ex))
ex_low  <- rbind(arm_summary(ex[ex$low_cens, ]), did_row(ex[ex$low_cens, ]))
ex_full$sample <- sprintf("full (n=%d)", nrow(ex))
ex_low$sample  <- sprintf("pcf<%d%% (n=%d)", PCF_CUT, sum(ex$low_cens))
check2 <- rbind(ex_full, ex_low)[, c("sample", "arm", "n", "mean_dAUC", "t", "p")]
write.csv(check2, file.path(TABLE_DIR, "robust_lowcens_extero.csv"), row.names = FALSE)

cat(sprintf("\n######## CHECK 2 — exteroception dAUC, full vs low-censoring subset — %s/%s ########\n",
            ATLAS, PIPELINE))
cat(sprintf("low-censoring = pcf<%d%% in both sessions (interpolation acts on few frames)\n", PCF_CUT))
cat("mean_dAUC>0 in placebo = the timescale 'lengthening'; DiD = verum-placebo\n\n")
print(check2, row.names = FALSE)

# ===========================================================================
# Figure — dAUC vs d(mean FD), one panel per region, coloured by arm
# ===========================================================================
cols <- c(placebo = "#1b7837", verum = "#762a83")
png(file.path(FIG_DIR, "qin_robust_dauc_vs_dfd.png"), width = 1650, height = 1050, res = 140)
old <- par(mfrow = c(2, 3), mar = c(4, 4, 2.6, 1), oma = c(0, 0, 2.4, 0))
for (rg in REGIONS) {
  d <- dat[dat$region == rg, ]
  plot(d$dfd, d$dAUC, pch = 16, col = cols[as.character(d$arm)],
       main = sprintf("%s (r=%.2f, p=%.3f)", rg, check1$r[check1$region == rg], check1$p[check1$region == rg]),
       xlab = "d(mean FD)  post-pre (mm)", ylab = "dAUC  post-pre (s)", cex.main = 0.95)
  abline(h = 0, col = "grey60", lty = 3); abline(v = 0, col = "grey60", lty = 3)
  abline(lm(dAUC ~ dfd, d), col = "black", lwd = 2)
}
plot.new(); legend("center", legend = names(cols), col = cols, pch = 16, bty = "n",
                   title = "arm", cex = 1.2)
mtext(sprintf("dAUC vs motion change — %s/%s (n=%d)", ATLAS, PIPELINE, length(unique(dat$subject))),
      outer = TRUE, cex = 1)
par(old); dev.off()

cat(sprintf("\nWrote: robust_dauc_vs_motion.csv, robust_lowcens_extero.csv -> %s\n", TABLE_DIR))
cat(sprintf("Wrote: qin_robust_dauc_vs_dfd.png -> %s\n", FIG_DIR))
