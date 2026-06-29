# 05_network_drugeffect_test.R — per-network drug effect (Verum vs Placebo), residualized.
# For each of the 5 focus networks: per-subject motion-residualized ΔAUC, then
# effect = mean(verum) − mean(placebo), 10k subject-label permutation (seed 42, two-tailed),
# BH-FDR across the 5 networks. Mirrors the qinspheres Result-1 test.
#
# In : 04_statistics/results/glasser_g1/tables/glasser_g1_auc_tidy.csv
# Out: 04_statistics/results/glasser_g1/tables/glasser_network_drugeffect_{pipeline}.csv
# Run: Rscript 04_statistics/scripts/glasser_g1/05_network_drugeffect_test.R

suppressPackageStartupMessages({library(dplyr); library(tidyr)})

REPO <- normalizePath(file.path(dirname(sub("^--file=", "",
        grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "..", "..", ".."))
TIDY <- file.path(REPO, "04_statistics", "results", "glasser_g1", "tables", "glasser_g1_auc_tidy.csv")
FD   <- file.path(REPO, "99_QC", "01_motion_qc", "results", "fd_covariates_wide_thresh03.csv")
TAB  <- file.path(REPO, "04_statistics", "results", "glasser_g1", "tables")

NET5  <- c(Somatomotor = "Sensorimotor", `Cingulo-Opercular` = "Salience",
           `Dorsal-Attention` = "Dorsal Attention", Frontoparietal = "Central Executive",
           Default = "Default Mode")
ORDER <- c("Sensorimotor", "Salience", "Dorsal Attention", "Central Executive", "Default Mode")
NPERM <- 10000L

df  <- read.csv(TIDY, stringsAsFactors = FALSE)
cov <- read.csv(FD, stringsAsFactors = FALSE)[, c("subject", "pcf_diff", "pcf_sq_diff", "mean_fd_retained_diff")]

resid_deltas <- function(pipeline) {
  d5 <- df %>% filter(pipeline == !!pipeline, network %in% names(NET5), auc > 0) %>%
    mutate(net5 = NET5[network])
  d5 %>% group_by(subject, arm, net5, session) %>% summarise(m = mean(auc), .groups = "drop") %>%
    pivot_wider(names_from = session, values_from = m) %>%
    filter(!is.na(Pre), !is.na(Post)) %>% mutate(delta = Post - Pre) %>%
    left_join(cov, by = "subject") %>%
    group_by(net5) %>%
    mutate(resid = residuals(lm(delta ~ pcf_diff + pcf_sq_diff + mean_fd_retained_diff,
                                na.action = na.exclude))) %>%
    ungroup() %>% filter(!is.na(resid))
}

perm_p <- function(v, p) {
  obs <- mean(v) - mean(p); all <- c(v, p); nv <- length(v); set.seed(42)
  null <- replicate(NPERM, { i <- sample(length(all), nv); mean(all[i]) - mean(all[-i]) })
  c(effect = obs, p = (sum(abs(null) >= abs(obs)) + 1) / (NPERM + 1))
}

for (pipeline in c("maximal", "detrend", "glm")) {
  d <- resid_deltas(pipeline)
  rows <- lapply(ORDER, function(net) {
    dd <- d[d$net5 == net, ]
    v  <- dd$resid[dd$arm == "verum"]; p <- dd$resid[dd$arm == "placebo"]
    pr <- perm_p(v, p)
    data.frame(network = net, n_verum = length(v), n_placebo = length(p),
               verum_med = median(v), placebo_med = median(p),
               effect_verum_minus_placebo = unname(pr["effect"]), perm_p = unname(pr["p"]))
  })
  res <- do.call(rbind, rows)
  res$perm_q <- p.adjust(res$perm_p, "BH")
  write.csv(res, file.path(TAB, sprintf("glasser_network_drugeffect_%s.csv", pipeline)), row.names = FALSE)
  cat(sprintf("\n===== %s (residualized, n=%d/%d) =====\n",
              pipeline, res$n_verum[1], res$n_placebo[1]))
  print(transform(res,
        verum_med = round(verum_med, 3), placebo_med = round(placebo_med, 3),
        effect_verum_minus_placebo = round(effect_verum_minus_placebo, 3),
        perm_p = round(perm_p, 4), perm_q = round(perm_q, 4))[
        , c("network", "placebo_med", "verum_med", "effect_verum_minus_placebo", "perm_p", "perm_q")],
        row.names = FALSE)
}
