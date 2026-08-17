# 04c_placebo_forest.R — placebo-only version of qin_time_effect_forest.png.
#
# Per-region PLACEBO pre->post time effect (the meditation-retreat effect, no drug), with 95% CI,
# ordered by magnitude, FDR survivors marked. Same forest style as 04_figures.R's arm-averaged
# forest, but fit on placebo subjects only:  AUC ~ session + pcf + pcf_sq + mean_fd + (1 | subject).
# (The main pipeline uses one arm-averaged model; this is a separate figure, not a core stats table.)
#
# Args : [pipeline] [metric] [atlas]   (default maximal auc qinspheres)
# Reads: results/mixed_models/per_config/{atlas}_{pipeline}{_metric}_longtable.csv  (from 03_mixed_models.R)
# Out  : results/{acw|sampen}/{spheres|parcels/self_regions}/{pipeline}/figures/qin_time_effect_forest_placebo.png

for (p in c("lme4","lmerTest","ggplot2"))
  if (!requireNamespace(p, quietly=TRUE)) install.packages(p, repos="https://cloud.r-project.org", quiet=TRUE)
suppressPackageStartupMessages({library(lmerTest); library(ggplot2)})

a <- commandArgs(trailingOnly=TRUE)
PIPELINE <- if (length(a)>=1) a[1] else "maximal"
METRIC   <- if (length(a)>=2) a[2] else "auc"
ATLAS    <- if (length(a)>=3) a[3] else "qinspheres"
MSUF  <- if (METRIC=="auc") "" else paste0("_",METRIC)
MLAB  <- if (METRIC=="sampen") "SampEn" else "AUC"
DUNIT <- if (METRIC=="sampen") "" else ", s"
REPO <- local({
  m <- grep("^--file=", commandArgs(FALSE), value=TRUE)
  f <- if (length(m)) sub("^--file=","",m[1]) else NULL
  if (is.null(f)) normalizePath(getwd())
  else normalizePath(file.path(dirname(normalizePath(f)), "..","..","..")) })
PC   <- file.path(REPO,"04_statistics","results","mixed_models","per_config")
METRIC_DIR <- if (METRIC == "auc") "acw" else METRIC
ATLAS_DIR  <- if (ATLAS == "qinspheres") "spheres" else file.path("parcels", "self_regions")
ATLAS_TAG  <- gsub("/", "_", ATLAS_DIR)   # per_config filename token: spheres | parcels_self_regions
LONG <- read.csv(file.path(PC, sprintf("%s_%s%s_longtable.csv",ATLAS_TAG,PIPELINE,MSUF)), stringsAsFactors=FALSE)
FIG  <- file.path(REPO,"04_statistics","results",METRIC_DIR,ATLAS_DIR,PIPELINE,"figures")
dir.create(FIG, recursive=TRUE, showWarnings=FALSE)

REGIONS <- c("intero","extero","cognition","visual","auditory")
RLAB <- c(intero="Interoception", extero="Exteroception", cognition="Cognition",
          visual="Visual", auditory="Auditory")
sig_star <- function(q) if (is.na(q)) "" else if (q<0.001) "***" else if (q<0.01) "**" else if (q<0.05) "*" else ""

# ── placebo-only per-region fit: session coefficient = post - pre ──
placebo <- LONG[LONG$arm=="placebo", ]
placebo$session <- factor(placebo$session, levels=c("pre","post"))
n_placebo <- length(unique(placebo$subject))
fit_region <- function(rg){
  d <- placebo[placebo$region==rg, ]
  m <- suppressWarnings(suppressMessages(lmer(AUC ~ session + pcf + pcf_sq + mean_fd + (1|subject), data=d, REML=TRUE)))
  co <- tryCatch(summary(m, ddf="Kenward-Roger")$coefficients, error=function(e) summary(m)$coefficients)
  est <- co["sessionpost",1]; se <- co["sessionpost",2]
  df  <- if (ncol(co)>=5) co["sessionpost",3] else NA; p <- co["sessionpost", ncol(co)]
  crit <- if (!is.na(df)) qt(.975, df) else 1.96
  data.frame(region=rg, session_est=est, session_ci_low=est-crit*se,
             session_ci_high=est+crit*se, session_p=p, singular=isSingular(m))
}
res <- do.call(rbind, lapply(REGIONS, fit_region))
res$session_q <- p.adjust(res$session_p, "BH")   # FDR across the 5 regions
cat(sprintf("Placebo-only session effect (n=%d placebo subjects) — %s/%s/%s\n", n_placebo, ATLAS, PIPELINE, METRIC))
print(res[order(-res$session_est), c("region","session_est","session_ci_low","session_ci_high","session_p","session_q")], row.names=FALSE)

# ── forest plot (same style as qin_time_effect_forest.png) ──
res$label    <- RLAB[res$region]
res$survives <- !is.na(res$session_q) & res$session_q < 0.05
res$star     <- vapply(res$session_q, sig_star, character(1))
res$region_f <- reorder(factor(res$label), res$session_est)   # ascending est -> largest region at top
xr  <- range(c(res$session_ci_low, res$session_ci_high, 0)); pad <- diff(xr)*0.14
forest <- ggplot(res, aes(x=session_est, y=region_f)) +
  geom_vline(xintercept=0, linetype="dashed", color="gray55", linewidth=0.5) +
  geom_errorbar(aes(xmin=session_ci_low, xmax=session_ci_high, color=survives),
                orientation="y", width=0.20, linewidth=0.75) +
  geom_point(aes(color=survives, shape=survives, size=survives)) +
  geom_text(aes(x=session_ci_high, label=star), hjust=-0.35, vjust=0.78, size=5, color="gray20", family="serif") +
  scale_color_manual(values=c(`FALSE`="gray55", `TRUE`="#1B5E20"), guide="none") +
  scale_shape_manual(values=c(`FALSE`=16, `TRUE`=18), guide="none") +
  scale_size_manual(values =c(`FALSE`=2.6, `TRUE`=4.0), guide="none") +
  scale_x_continuous(limits=c(xr[1]-pad*0.3, xr[2]+pad), expand=expansion(mult=c(0.02,0.02))) +
  labs(title=sprintf("Per-region time effect (post - pre), PLACEBO only - %s, %s pipeline", MLAB, PIPELINE),
       subtitle=sprintf("Placebo-arm meditation-retreat effect (n=%d) - LMM session coef - 95%% CI - green diamond + star = survives BH-FDR (q<0.05)", n_placebo),
       x=sprintf("Δ%s (post - pre%s)", MLAB, DUNIT), y=NULL) +
  theme_minimal(base_size=11, base_family="serif") +
  theme(panel.background=element_rect(fill="white",color=NA), plot.background=element_rect(fill="white",color=NA),
        panel.grid.major.y=element_blank(), panel.grid.minor=element_blank(),
        panel.grid.major.x=element_line(color="#e8e8e8", linewidth=0.4),
        plot.title=element_text(hjust=0.5, size=12), plot.subtitle=element_text(hjust=0.5, size=7.5, color="gray35"),
        axis.text.y=element_text(size=11))
ggsave(file.path(FIG,"qin_time_effect_forest_placebo.png"), forest, width=7.5, height=4, dpi=300, bg="white")
cat(sprintf("Saved: %s\n", file.path(FIG,"qin_time_effect_forest_placebo.png")))
