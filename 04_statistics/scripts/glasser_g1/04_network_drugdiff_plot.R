# 04_network_drugdiff_plot.R — first-glance drug-effect plot for the 5 chosen networks.
# Mirrors qinspheres_drug_diff_res.png: per network, Placebo vs Verum jittered points
# (one per subject), black median bars, dashed zero line.
#
# MOTION-RESIDUALIZED ΔAUC (mandatory preprocessing — never raw): per network,
# residuals(lm(delta ~ pcf_diff + pcf_sq_diff + mean_fd_retained_diff)) from
# 99_QC/01_motion_qc/results/fd_covariates_wide_thresh03.csv.
#
# Networks (Cole-Anticevic mapping): Sensorimotor=Somatomotor, Salience=Cingulo-Opercular,
# Dorsal Attention=Dorsal-Attention, Central Executive=Frontoparietal, Default Mode=Default.
#
# In : 04_statistics/results/glasser_g1/tables/glasser_g1_auc_tidy.csv
# Out: 04_statistics/results/glasser_g1/figures/glasser_network_drugdiff_{pipeline}.png
#      04_statistics/results/glasser_g1/tables/glasser_network_delta_resid_{pipeline}.csv
# Run: Rscript 04_statistics/scripts/glasser_g1/04_network_drugdiff_plot.R [pipeline]

suppressPackageStartupMessages({library(dplyr); library(tidyr); library(ggplot2)})

args <- commandArgs(trailingOnly = TRUE)
PIPELINE <- if (length(args) >= 1) args[1] else "maximal"

REPO <- normalizePath(file.path(dirname(sub("^--file=", "",
        grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "..", "..", ".."))
TIDY <- file.path(REPO, "04_statistics", "results", "glasser_g1", "tables", "glasser_g1_auc_tidy.csv")
FD   <- file.path(REPO, "99_QC", "01_motion_qc", "results", "fd_covariates_wide_thresh03.csv")
FIG  <- file.path(REPO, "04_statistics", "results", "glasser_g1", "figures")
TAB  <- file.path(REPO, "04_statistics", "results", "glasser_g1", "tables")

NET5 <- c(Somatomotor = "Sensorimotor", `Cingulo-Opercular` = "Salience",
          `Dorsal-Attention` = "Dorsal Attention", Frontoparietal = "Central Executive",
          Default = "Default Mode")
ORDER <- c("Sensorimotor", "Salience", "Dorsal Attention", "Central Executive", "Default Mode")
COLS  <- c(placebo = "#C0504D", verum = "#4F81BD")

df  <- read.csv(TIDY, stringsAsFactors = FALSE)
cov <- read.csv(FD, stringsAsFactors = FALSE)[, c("subject", "pcf_diff", "pcf_sq_diff", "mean_fd_retained_diff")]

d5 <- df %>%
  filter(pipeline == PIPELINE, network %in% names(NET5), auc > 0) %>%
  mutate(net5 = NET5[network])

# per-subject raw ΔAUC per network, then motion-residualize within each network
delta <- d5 %>%
  group_by(subject, arm, net5, session) %>% summarise(m = mean(auc), .groups = "drop") %>%
  pivot_wider(names_from = session, values_from = m) %>%
  filter(!is.na(Pre), !is.na(Post)) %>%
  mutate(delta = Post - Pre) %>%
  left_join(cov, by = "subject")

delta <- delta %>% group_by(net5) %>%
  mutate(resid = residuals(lm(delta ~ pcf_diff + pcf_sq_diff + mean_fd_retained_diff,
                              na.action = na.exclude))) %>%
  ungroup() %>%
  filter(!is.na(resid)) %>%
  mutate(net5 = factor(net5, levels = ORDER),
         arm  = factor(arm, levels = c("placebo", "verum")))

write.csv(delta, file.path(TAB, sprintf("glasser_network_delta_resid_%s.csv", PIPELINE)), row.names = FALSE)

p <- ggplot(delta, aes(net5, resid, color = arm)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(position = position_jitterdodge(jitter.width = 0.18, dodge.width = 0.75),
             alpha = 0.6, size = 2.6) +
  stat_summary(aes(group = arm), fun = median, geom = "crossbar", width = 0.55,
               color = "black", linewidth = 0.45, position = position_dodge(width = 0.75)) +
  scale_color_manual(values = COLS, labels = c("Placebo", "Verum (DMT)"), name = NULL) +
  labs(title = sprintf("Drug effect: Verum vs Placebo per network (%s, motion-residualized)", PIPELINE),
       subtitle = "One point = one subject. Black bars = group medians.",
       x = NULL, y = "Quality-adjusted ΔAUC (residual)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIG, sprintf("glasser_network_drugdiff_%s.png", PIPELINE)), p,
       width = 11, height = 7, dpi = 150, bg = "white")
cat(sprintf("saved glasser_network_drugdiff_%s.png  (residualized; n=%d subject-network deltas)\n",
            PIPELINE, nrow(delta)))
delta %>% group_by(net5, arm) %>%
  summarise(median_resid = round(median(resid), 3), n = n(), .groups = "drop") %>%
  arrange(net5, arm) %>% as.data.frame() %>% print(row.names = FALSE)
