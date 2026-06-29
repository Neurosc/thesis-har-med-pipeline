# 06_network_interaction.R — does the drug effect differ ACROSS the 5 networks, and is DMN
# different from the others? Residualized ΔAUC (motion-corrected, mandatory).
#
# (A) Omnibus: lmer(resid ~ network * arm + (1|subject)), Type III network:arm interaction F/p
#     + subject-level permutation of arm labels.
# (B) DMN-vs-others (the focus): per subject, dmn_dev = resid[DMN] - mean(resid[other 4]),
#     then verum vs placebo on dmn_dev (10k permutation). Positive interaction => DMN's drug
#     response differs from the rest.
#
# In : 04_statistics/results/glasser_g1/tables/glasser_g1_auc_tidy.csv
# Out: 04_statistics/results/glasser_g1/tables/glasser_network_interaction.csv
# Run: Rscript 04_statistics/scripts/glasser_g1/06_network_interaction.R

suppressPackageStartupMessages({
  library(dplyr); library(tidyr); library(lme4); library(lmerTest)
})

REPO <- normalizePath(file.path(dirname(sub("^--file=", "",
        grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "..", "..", ".."))
TIDY <- file.path(REPO, "04_statistics", "results", "glasser_g1", "tables", "glasser_g1_auc_tidy.csv")
FD   <- file.path(REPO, "99_QC", "01_motion_qc", "results", "fd_covariates_wide_thresh03.csv")
TAB  <- file.path(REPO, "04_statistics", "results", "glasser_g1", "tables")

NET5   <- c(Somatomotor = "Sensorimotor", `Cingulo-Opercular` = "Salience",
            `Dorsal-Attention` = "Dorsal Attention", Frontoparietal = "Central Executive",
            Default = "Default Mode")
ORDER  <- c("Sensorimotor", "Salience", "Dorsal Attention", "Central Executive", "Default Mode")
OTHERS <- setdiff(ORDER, "Default Mode")
NPERM_LMM <- 2000L; NPERM <- 10000L
set.seed(42)

df  <- read.csv(TIDY, stringsAsFactors = FALSE)
cov <- read.csv(FD, stringsAsFactors = FALSE)[, c("subject", "pcf_diff", "pcf_sq_diff", "mean_fd_retained_diff")]

resid_deltas <- function(pipeline) {
  df %>% filter(pipeline == !!pipeline, network %in% names(NET5), auc > 0) %>%
    mutate(net5 = NET5[network]) %>%
    group_by(subject, arm, net5, session) %>% summarise(m = mean(auc), .groups = "drop") %>%
    pivot_wider(names_from = session, values_from = m) %>%
    filter(!is.na(Pre), !is.na(Post)) %>% mutate(delta = Post - Pre) %>%
    left_join(cov, by = "subject") %>%
    group_by(net5) %>%
    mutate(resid = residuals(lm(delta ~ pcf_diff + pcf_sq_diff + mean_fd_retained_diff,
                                na.action = na.exclude))) %>%
    ungroup() %>% filter(!is.na(resid))
}

perm_p <- function(v, p) {
  obs <- mean(v) - mean(p); all <- c(v, p); nv <- length(v)
  null <- replicate(NPERM, { i <- sample(length(all), nv); mean(all[i]) - mean(all[-i]) })
  c(effect = obs, p = (sum(abs(null) >= abs(obs)) + 1) / (NPERM + 1))
}

out <- list()
for (pipeline in c("maximal", "detrend", "glm")) {
  d <- resid_deltas(pipeline) %>%
    mutate(net5 = factor(net5, levels = ORDER), arm = factor(arm, levels = c("placebo", "verum")))

  # (A) omnibus network x arm interaction
  m  <- lmer(resid ~ net5 * arm + (1 | subject), data = d,
             control = lmerControl(check.conv.singular = .makeCC("ignore", tol = 1e-4)))
  aov <- anova(m, type = 3); ix <- "net5:arm"
  Fobs <- aov[ix, "F value"]; pA_param <- aov[ix, "Pr(>F)"]
  sd <- d %>% distinct(subject, arm)
  permF <- replicate(NPERM_LMM, {
    sp <- sd; sp$arm <- sample(sp$arm)
    dp <- d; dp$arm <- sp$arm[match(dp$subject, sp$subject)]
    mp <- suppressMessages(suppressWarnings(lmer(resid ~ net5 * arm + (1 | subject), data = dp,
          control = lmerControl(check.conv.singular = .makeCC("ignore", tol = 1e-4)))))
    suppressWarnings(anova(mp, type = 3)[ix, "F value"])
  })
  pA_perm <- (sum(permF >= Fobs) + 1) / (NPERM_LMM + 1)

  # (B) DMN vs others
  w <- d %>% select(subject, arm, net5, resid) %>%
    pivot_wider(names_from = net5, values_from = resid)
  w$dmn_dev <- w[["Default Mode"]] - rowMeans(w[, OTHERS])
  vv <- w$dmn_dev[w$arm == "verum"]; pp <- w$dmn_dev[w$arm == "placebo"]
  prB <- perm_p(vv, pp)
  # raw drug effects for context
  dmn_eff   <- mean(d$resid[d$net5 == "Default Mode" & d$arm == "verum"]) -
               mean(d$resid[d$net5 == "Default Mode" & d$arm == "placebo"])
  other_eff <- mean(d$resid[d$net5 %in% OTHERS & d$arm == "verum"]) -
               mean(d$resid[d$net5 %in% OTHERS & d$arm == "placebo"])

  out[[pipeline]] <- data.frame(
    pipeline = pipeline,
    omnibus_F = round(Fobs, 3), omnibus_p_param = round(pA_param, 4), omnibus_p_perm = round(pA_perm, 4),
    dmn_drug_effect = round(dmn_eff, 3), others_drug_effect = round(other_eff, 3),
    dmn_vs_others_diff = round(unname(prB["effect"]), 3), dmn_vs_others_perm_p = round(unname(prB["p"]), 4))
  cat(sprintf("\n== %s ==\n  (A) network x arm interaction: F=%.2f  p_param=%.4f  p_perm=%.4f\n",
              pipeline, Fobs, pA_param, pA_perm))
  cat(sprintf("  (B) DMN drug-effect=%.3f vs Others=%.3f ; DMN-minus-others diff=%.3f  perm_p=%.4f\n",
              dmn_eff, other_eff, unname(prB["effect"]), unname(prB["p"])))
}
res <- do.call(rbind, out)
write.csv(res, file.path(TAB, "glasser_network_interaction.csv"), row.names = FALSE)
cat(sprintf("\nsaved -> %s\n", file.path(TAB, "glasser_network_interaction.csv")))
