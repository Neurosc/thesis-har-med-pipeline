# 04_network_drugdiff_plot.R — first-glance drug-effect plot for the 5 chosen networks.
# Mirrors qinspheres_drug_diff_res.png: per network, Placebo vs Verum jittered points
# (one per subject), black median bars, dashed zero line. RAW dAUC (Post-Pre) for now.
#
# Networks (Cole-Anticevic mapping): Sensorimotor=Somatomotor, Salience=Cingulo-Opercular,
# Dorsal Attention=Dorsal-Attention, Central Executive=Frontoparietal, Default Mode=Default.
#
# In : 04_statistics/results/glasser_g1/tables/glasser_g1_auc_tidy.csv
# Out: 04_statistics/results/glasser_g1/figures/glasser_network_drugdiff_{pipeline}.png
# Run: Rscript 04_statistics/scripts/glasser_g1/04_network_drugdiff_plot.R [pipeline]

suppressPackageStartupMessages({library(dplyr); library(tidyr); library(ggplot2)})

args <- commandArgs(trailingOnly = TRUE)
PIPELINE <- if (length(args) >= 1) args[1] else "maximal"

REPO <- normalizePath(file.path(dirname(sub("^--file=", "",
        grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "..", "..", ".."))
TIDY <- file.path(REPO, "04_statistics", "results", "glasser_g1", "tables", "glasser_g1_auc_tidy.csv")
FIG  <- file.path(REPO, "04_statistics", "results", "glasser_g1", "figures")

NET5 <- c(Somatomotor = "Sensorimotor", `Cingulo-Opercular` = "Salience",
          `Dorsal-Attention` = "Dorsal Attention", Frontoparietal = "Central Executive",
          Default = "Default Mode")
ORDER <- c("Sensorimotor", "Salience", "Dorsal Attention", "Central Executive", "Default Mode")
COLS  <- c(placebo = "#C0504D", verum = "#4F81BD")

df <- read.csv(TIDY, stringsAsFactors = FALSE)
d5 <- df %>%
  filter(pipeline == PIPELINE, network %in% names(NET5), auc > 0) %>%
  mutate(net5 = NET5[network])

delta <- d5 %>%
  group_by(subject, arm, net5, session) %>% summarise(m = mean(auc), .groups = "drop") %>%
  pivot_wider(names_from = session, values_from = m) %>%
  filter(!is.na(Pre), !is.na(Post)) %>%
  mutate(delta = Post - Pre,
         net5 = factor(net5, levels = ORDER),
         arm  = factor(arm, levels = c("placebo", "verum")))

p <- ggplot(delta, aes(net5, delta, color = arm)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  geom_point(position = position_jitterdodge(jitter.width = 0.18, dodge.width = 0.75),
             alpha = 0.6, size = 2.6) +
  stat_summary(aes(group = arm), fun = median, geom = "crossbar", width = 0.55,
               color = "black", linewidth = 0.45, position = position_dodge(width = 0.75)) +
  scale_color_manual(values = COLS, labels = c("Placebo", "Verum (DMT)"), name = NULL) +
  labs(title = sprintf("Drug effect: Verum vs Placebo per network (%s, raw ΔAUC)", PIPELINE),
       subtitle = "One point = one subject. Black bars = group medians.",
       x = NULL, y = "ΔAUC (Post − Pre)") +
  theme_minimal(base_size = 13) +
  theme(legend.position = "top", panel.grid.minor = element_blank(),
        plot.title = element_text(face = "bold"))

ggsave(file.path(FIG, sprintf("glasser_network_drugdiff_%s.png", PIPELINE)), p,
       width = 11, height = 7, dpi = 150, bg = "white")
cat(sprintf("saved glasser_network_drugdiff_%s.png  (n=%d subject-network deltas)\n",
            PIPELINE, nrow(delta)))
# quick numeric peek: median delta per arm x network
delta %>% group_by(net5, arm) %>%
  summarise(median_delta = round(median(delta), 3), n = n(), .groups = "drop") %>%
  arrange(net5, arm) %>% as.data.frame() %>% print(row.names = FALSE)
