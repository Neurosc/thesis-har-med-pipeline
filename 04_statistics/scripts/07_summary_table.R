# 07_summary_table.R — one tidy results table + tidy per-subject deltas + distribution plots,
# spanning metric (AUC, SampEn) × atlas (qinspheres, qinparcels) × pipeline (3) × layer (6) = 72 rows.
#
# effect / p_raw / p_fdr are read from the committed RAW per-layer tables (layer_drug_effect.csv)
# so they match the analysis exactly; p_*_resid are the motion-residualized p (AUC only).
# CI = 10k bootstrap percentile on the RAW per-subject Δ; n_dropped = ROI-observations dropped
# for that cell (AUC: auc≤0 broadband-collapse drops; SampEn: non-finite).
#
# Units: AUC in SECONDS (∫ ACF over lag-time, IntrinsicTimescales acw with fs=1/TR);
#        SampEn unitless (log base 2). Δ = post − pre.
#
# Out: 04_statistics/results/summary/results_long.csv, deltas_long.csv, figures/dist_{metric}_{atlas}.png
# Run: Rscript 04_statistics/scripts/07_summary_table.R

suppressPackageStartupMessages({library(dplyr); library(tidyr); library(ggplot2)})
REPO <- normalizePath(file.path(dirname(sub("^--file=", "",
        grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "..", ".."))
RES  <- file.path(REPO, "04_statistics", "results")
OUT  <- file.path(RES, "summary"); FIG <- file.path(OUT, "figures")
dir.create(FIG, recursive = TRUE, showWarnings = FALSE)
NBOOT <- 10000
LAYERS <- c("intero", "extero", "mental", "visual", "motor", "auditory")
LL <- c(intero="intero", extero="extero", mental="cognition",
        visual="visual", motor="motor", auditory="auditory")

subj_delta <- function(df, valid) {
  df[valid, ] %>% group_by(subject, drug_group, category, session) %>%
    summarise(m = mean(auc), .groups = "drop") %>%
    pivot_wider(names_from = session, values_from = m) %>%
    filter(!is.na(`ses-01`), !is.na(`ses-02`)) %>%
    transmute(subject, drug_group, category, delta = `ses-02` - `ses-01`)
}
bootci <- function(v, p) {
  if (length(v) < 2 || length(p) < 2) return(c(NA, NA))
  set.seed(42)
  b <- replicate(NBOOT, mean(sample(v, replace = TRUE)) - mean(sample(p, replace = TRUE)))
  unname(quantile(b, c(.025, .975)))
}

rows <- list(); dl <- list()
for (atlas in c("qinspheres", "qinparcels")) {
  for (pl in c("detrend", "glm", "maximal")) {
    auc  <- read.csv(file.path(RES, atlas, pl, "tables", "qinspheres_auc.csv"))
    samp <- read.csv(file.path(RES, atlas, "sampen", pl, "tables", "qinspheres_sampen.csv"))
    lde  <- list(AUC  = read.csv(file.path(RES, atlas, pl, "tables", "layer_drug_effect.csv")),
                 SampEn = read.csv(file.path(RES, atlas, "sampen", pl, "tables", "layer_drug_effect.csv")))
    rde_p <- file.path(RES, atlas, pl, "tables", "layer_drug_effect_resid.csv")
    rde   <- if (file.exists(rde_p)) read.csv(rde_p) else NULL
    for (met in c("AUC", "SampEn")) {
      df    <- if (met == "AUC") auc else samp
      valid <- if (met == "AUC") (df$auc > 0 & is.finite(df$auc)) else is.finite(df$auc)
      d     <- subj_delta(df, valid)
      for (lay in LAYERS) {
        r  <- lde[[met]][lde[[met]]$layer == lay, ]
        dd <- d[d$category == lay, ]
        v  <- dd$delta[dd$drug_group == "verum"]; p <- dd$delta[dd$drug_group == "placebo"]
        ci <- bootci(v, p)
        full    <- sum(samp$category == lay)            # all ROI-observations (SampEn unfiltered)
        present <- sum(valid & df$category == lay)
        prr <- pqr <- NA
        if (met == "AUC" && !is.null(rde)) { rr <- rde[rde$layer == lay, ]
          if (nrow(rr)) { prr <- rr$perm_p_raw[1]; pqr <- rr$perm_p_fdr[1] } }
        rows[[length(rows) + 1]] <- data.frame(
          metric = met, atlas = atlas, pipeline = pl, layer = unname(LL[lay]),
          effect = round(r$observed_diff[1], 4), ci_low = round(ci[1], 4), ci_high = round(ci[2], 4),
          sd_verum = round(sd(v), 4), sd_placebo = round(sd(p), 4),
          p_raw = round(r$perm_p_raw[1], 4), p_fdr = round(r$perm_p_fdr[1], 4),
          p_raw_resid = round(prr, 4), p_fdr_resid = round(pqr, 4),
          n_verum = r$n_verum[1], n_placebo = r$n_placebo[1], n_dropped = full - present)
        dl[[length(dl) + 1]] <- data.frame(subject_id = dd$subject, drug = dd$drug_group,
          metric = met, atlas = atlas, pipeline = pl, layer = unname(LL[lay]), delta = dd$delta)
      }
    }
  }
}
results <- do.call(rbind, rows); deltas <- do.call(rbind, dl)
write.csv(results, file.path(OUT, "results_long.csv"), row.names = FALSE)
write.csv(deltas,  file.path(OUT, "deltas_long.csv"),  row.names = FALSE)
cat(sprintf("results_long.csv: %d rows ; deltas_long.csv: %d rows\n", nrow(results), nrow(deltas)))

# ── subject-level distribution plots (one per metric × atlas, faceted by pipeline) ──
COLS <- c(placebo = "#C0504D", verum = "#4F81BD")
deltas$layer <- factor(deltas$layer, levels = c("intero","extero","cognition","visual","motor","auditory"))
deltas$drug  <- factor(deltas$drug,  levels = c("placebo","verum"))
deltas$pipeline <- factor(deltas$pipeline, levels = c("detrend","glm","maximal"))
for (met in c("AUC","SampEn")) for (at in c("qinspheres","qinparcels")) {
  dsub <- deltas[deltas$metric == met & deltas$atlas == at, ]
  ylab <- if (met == "AUC") "Δ AUC (post − pre, s)" else "Δ SampEn (post − pre)"
  g <- ggplot(dsub, aes(layer, delta, color = drug)) +
    geom_hline(yintercept = 0, linetype = "dashed", color = "gray60") +
    geom_boxplot(outlier.shape = NA, position = position_dodge(0.8), width = 0.6, alpha = 0.25) +
    geom_point(position = position_jitterdodge(jitter.width = 0.15, dodge.width = 0.8),
               size = 1.2, alpha = 0.5) +
    facet_wrap(~pipeline, ncol = 1, scales = "free_y") +
    scale_color_manual(values = COLS, labels = c("Placebo","Verum"), name = NULL) +
    labs(title = sprintf("Subject-level Δ distributions — %s, %s (n=35)", met, at),
         x = NULL, y = ylab) +
    theme_minimal(base_size = 12) +
    theme(legend.position = "top", panel.grid.minor = element_blank(),
          axis.text.x = element_text(angle = 20, hjust = 1))
  ggsave(file.path(FIG, sprintf("dist_%s_%s.png", met, at)), g, width = 10, height = 9, dpi = 150, bg = "white")
}
cat("figures: dist_{AUC,SampEn}_{qinspheres,qinparcels}.png\n")
