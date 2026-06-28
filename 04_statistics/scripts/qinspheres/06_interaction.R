# 06_interaction.R — Category × Drug interaction on ΔAUC.
# Question (result 2): do the 6 networks change DIFFERENTLY from each other under the drug?
# DV = ΔAUC (post − pre) per subject × category; category (6, within) × drug_group (2, between).
# Mixed model lmer(ΔAUC ~ category * drug + (1|subject)); Type III interaction F/p (Satterthwaite)
# + subject-level permutation of drug labels. Run for raw and motion-residualized ΔAUC.
#
# Run from repo root:  Rscript 04_statistics/scripts/qinspheres/06_interaction.R [pipeline] [metric]
#   defaults: maximal auc

for (pkg in c("lme4", "lmerTest", "dplyr", "tidyr", "ggplot2", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
}
suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(dplyr); library(tidyr)
  library(ggplot2); library(patchwork)
})

args      <- commandArgs(trailingOnly = FALSE)
file_arg  <- grep("^--file=", args, value = TRUE)
REPO_ROOT <- if (length(file_arg))
  normalizePath(file.path(dirname(normalizePath(sub("^--file=", "", file_arg[1]))), "..", "..", "..")) else normalizePath(".")
ta        <- commandArgs(trailingOnly = TRUE)
PIPELINE  <- if (length(ta) >= 1) ta[1] else "maximal"
METRIC    <- if (length(ta) >= 2) ta[2] else "auc"
NPERM     <- 2000L
set.seed(42)

QBASE   <- file.path(REPO_ROOT, "04_statistics", "results", "qinspheres")
QDIR    <- if (METRIC == "auc") file.path(QBASE, PIPELINE) else file.path(QBASE, METRIC, PIPELINE)
TABLES  <- file.path(QDIR, "tables"); FIGS <- file.path(QDIR, "figures")
dir.create(FIGS, recursive = TRUE, showWarnings = FALSE)
MLAB    <- if (METRIC == "acw50") "ACW-50" else "AUC"
LAYERS  <- c("intero","extero","mental","auditory","motor","visual")
LABS    <- c(intero="Interoception", extero="Exteroception", mental="Cognition",
             auditory="Auditory", motor="Motor", visual="Visual")
COLS    <- c(placebo = "#C0504D", verum = "#4F81BD")

cat(sprintf("\n==== Category × Drug interaction — %s / %s ====\n", PIPELINE, MLAB))

# ── raw ΔAUC (post − pre per subject × category) ────────────────────────────────
raw_delta <- read.csv(file.path(TABLES, paste0("qinspheres_", METRIC, ".csv")), stringsAsFactors = FALSE) %>%
  filter(is.finite(auc)) %>%
  group_by(subject, session, drug_group, category) %>% summarise(m = mean(auc), .groups = "drop") %>%
  pivot_wider(names_from = session, values_from = m) %>%
  rename(pre = `ses-01`, post = `ses-02`) %>%
  filter(!is.na(pre), !is.na(post)) %>%
  transmute(subject, drug_group, category, delta = post - pre)

# ── residualized ΔAUC (wide {cat}_resid -> long) ────────────────────────────────
resid_file <- file.path(TABLES, "auc_diff_quality_residuals.csv")
res_delta  <- NULL
if (file.exists(resid_file)) {
  rw    <- read.csv(resid_file, stringsAsFactors = FALSE)
  rcols <- intersect(paste0(LAYERS, "_resid"), names(rw))
  res_delta <- rw %>% select(subject, drug_group, all_of(rcols)) %>%
    pivot_longer(all_of(rcols), names_to = "category", values_to = "delta") %>%
    mutate(category = sub("_resid$", "", category)) %>% filter(!is.na(delta))
}

fit_interaction <- function(d, label, cats = LAYERS, do_perm = TRUE) {
  d <- d %>% filter(category %in% cats) %>%
    mutate(category = factor(category, levels = cats),
           drug_group = factor(drug_group, levels = c("placebo","verum")))
  m   <- lmer(delta ~ category * drug_group + (1 | subject), data = d,
              control = lmerControl(check.conv.singular = .makeCC("ignore", tol = 1e-4)))
  aov <- anova(m, type = 3)                       # Satterthwaite
  ix  <- "category:drug_group"
  obsF <- aov[ix, "F value"]; df1 <- aov[ix, "NumDF"]; df2 <- aov[ix, "DenDF"]; pParam <- aov[ix, "Pr(>F)"]
  pPerm <- NA_real_
  if (do_perm) {                                  # subject-level permutation of drug labels
    sd  <- d %>% distinct(subject, drug_group)
    permF <- replicate(NPERM, {
      sp <- sd; sp$drug_group <- sample(sp$drug_group)
      dp <- d; dp$drug_group <- sp$drug_group[match(dp$subject, sp$subject)]
      mp <- suppressMessages(suppressWarnings(
        lmer(delta ~ category * drug_group + (1 | subject), data = dp,
             control = lmerControl(check.conv.singular = .makeCC("ignore", tol = 1e-4)))))
      suppressWarnings(anova(mp, type = 3)[ix, "F value"])
    })
    pPerm <- (sum(permF >= obsF) + 1) / (NPERM + 1)
  }
  cat(sprintf("  %-14s  interaction F(%.0f,%.1f)=%.3f  p_param=%.4f  p_perm=%s\n",
              label, df1, df2, obsF, pParam, ifelse(is.na(pPerm), "  -  ", sprintf("%.4f", pPerm))))
  data.frame(model = label, term = ix, F = round(obsF, 3), df1 = round(df1, 0),
             df2 = round(df2, 1), p_param = round(pParam, 4), p_perm = round(pPerm, 4))
}

NOVIS <- setdiff(LAYERS, "visual")               # visual = known maximal denoising artifact
out <- fit_interaction(raw_delta, "raw")
out <- rbind(out, fit_interaction(raw_delta, "raw_novisual", NOVIS, do_perm = FALSE))
if (!is.null(res_delta)) {
  out <- rbind(out, fit_interaction(res_delta, "resid"))
  out <- rbind(out, fit_interaction(res_delta, "resid_novisual", NOVIS, do_perm = FALSE))
}
write.csv(out, file.path(TABLES, "interaction_category_by_drug.csv"), row.names = FALSE)

# ── interaction plot: mean ΔAUC ± SE per category × group (raw + resid) ──────────
ip <- function(d, ttl) {
  s <- d %>% mutate(category = factor(category, levels = LAYERS)) %>%
    group_by(category, drug_group) %>%
    summarise(m = mean(delta), se = sd(delta)/sqrt(n()), .groups = "drop")
  ggplot(s, aes(category, m, color = drug_group, group = drug_group)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray60") +
    geom_line(linewidth = 0.9, position = position_dodge(0.25)) +
    geom_point(size = 2.6, position = position_dodge(0.25)) +
    geom_errorbar(aes(ymin = m-se, ymax = m+se), width = 0.18, linewidth = 0.7,
                  position = position_dodge(0.25)) +
    scale_color_manual(values = COLS, name = NULL) +
    scale_x_discrete(labels = LABS[LAYERS]) +
    labs(title = ttl, x = NULL, y = sprintf("Mean Δ%s ± SE", MLAB)) +
    theme_minimal(base_size = 12, base_family = "serif") +
    theme(panel.background = element_rect(fill="white", color=NA),
          plot.background = element_rect(fill="white", color=NA),
          axis.text.x = element_text(angle = 30, hjust = 1),
          plot.title = element_text(hjust = 0.5), legend.position = "top")
}
pr <- ip(raw_delta, sprintf("Raw (interaction p_perm=%.3f)", out$p_perm[out$model=="raw"]))
plt <- if (!is.null(res_delta))
  pr + ip(res_delta, sprintf("Motion-residualized (p_perm=%.3f)", out$p_perm[out$model=="resid"])) else pr
plt <- plt + plot_annotation(
  title = sprintf("Category × Drug interaction on Δ%s — %s", MLAB, PIPELINE),
  theme = theme(plot.title = element_text(hjust = 0.5, family = "serif", size = 14)))
ggsave(file.path(FIGS, "qin_interaction_category_by_drug.png"), plt,
       width = if (!is.null(res_delta)) 13 else 7, height = 5.2, dpi = 150, bg = "white")
cat(sprintf("Saved: interaction_category_by_drug.csv + qin_interaction_category_by_drug.png\n"))
