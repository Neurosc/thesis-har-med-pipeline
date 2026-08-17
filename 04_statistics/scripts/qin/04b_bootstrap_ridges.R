# 04b_bootstrap_ridges.R — subject-level cluster-bootstrap ridgeline of the per-region
# post−pre time effect (the "real blobs" version of qin_time_effect_forest.png).
#
# HEAVY: B×5 lmer refits. Run for the primary config only (default maximal auc qinspheres);
# it is NOT part of the fast 04_figures.R sweep.
#
# Recipe:
#   1. For each region, standardize AUC (→ standardized coefficient), then bootstrap B times:
#      resample the ~35 subjects with replacement (BOTH sessions kept together, duplicates
#      relabelled as distinct clusters), refit AUC~session*arm+covars+(1|subject) with sum
#      contrasts, extract post−pre = −2·session1 (averaged over arm). B values = one blob.
#   2. Long: region × value (~B×5 rows).
#   3. Order regions by median draw (visual top → auditory bottom); tag FDR survivors in labels.
#   4. ggridges density ridges, red line at 0, sequential fill by order.
#
# Args : [pipeline] [metric] [atlas]   (default maximal auc qinspheres)  ·  [B] via env BOOT_B (default 2000)
# Reads: results/mixed_models/per_config/{atlas}_{pipeline}{_metric}_{longtable,overall}.csv
# Out  : results/{acw|sampen}/{spheres|parcels/self_regions}/{pipeline}/figures/qin_time_effect_bootstrap_ridges.png
#        results/{acw|sampen}/{spheres|parcels/self_regions}/{pipeline}/tables/qin_time_effect_bootstrap_draws.csv

for (p in c("lme4","ggplot2","ggridges","dplyr"))
  if (!requireNamespace(p, quietly=TRUE)) install.packages(p, repos="https://cloud.r-project.org", quiet=TRUE)
suppressPackageStartupMessages({library(lme4); library(ggplot2); library(ggridges); library(dplyr)})

a <- commandArgs(trailingOnly=TRUE)
PIPELINE <- if (length(a)>=1) a[1] else "maximal"
METRIC   <- if (length(a)>=2) a[2] else "auc"
ATLAS    <- if (length(a)>=3) a[3] else "qinspheres"
B        <- as.integer(Sys.getenv("BOOT_B", "2000"))
MSUF <- if (METRIC=="auc") "" else paste0("_",METRIC)
MLAB <- if (METRIC=="sampen") "SampEn" else "AUC"
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
OVR  <- read.csv(file.path(PC, sprintf("%s_%s%s_overall.csv",ATLAS_TAG,PIPELINE,MSUF)),   stringsAsFactors=FALSE)
QDIR <- file.path(REPO,"04_statistics","results",METRIC_DIR,ATLAS_DIR,PIPELINE)
FIG  <- file.path(QDIR,"figures"); TAB <- file.path(QDIR,"tables")
dir.create(FIG,showWarnings=FALSE,recursive=TRUE); dir.create(TAB,showWarnings=FALSE,recursive=TRUE)

REGS   <- c("intero","extero","cognition","visual","auditory")
RLAB   <- c(intero="Interoception", extero="Exteroception", cognition="Cognition",
            visual="Visual", auditory="Auditory")

# ── 1. bootstrap draws per region ──
boot_region <- function(rg){
  d <- LONG[LONG$region==rg, ]
  d$AUC <- as.numeric(scale(d$AUC))                                       # standardized coefficient
  sp   <- split(d, d$subject); subs <- names(sp)
  outv <- numeric(B)
  for (b in seq_len(B)){
    samp <- sample(subs, length(subs), replace=TRUE)
    rb   <- do.call(rbind, Map(function(s,i){ r<-sp[[s]]; r$subject<-paste0(s,"_",i); r }, samp, seq_along(samp)))
    # sum contrasts on the reassembled frame (rbind drops the attribute) → post−pre = −2·session1
    rb$session <- C(factor(rb$session, levels=c("pre","post")),   "contr.sum")
    rb$arm     <- C(factor(rb$arm,     levels=c("placebo","verum")),"contr.sum")
    m <- tryCatch(suppressWarnings(suppressMessages(
           lmer(AUC~session*arm+pcf+pcf_sq+mean_fd+(1|subject), data=rb, REML=TRUE))), error=function(e) NULL)
    outv[b] <- if (is.null(m) || is.na(fixef(m)["session1"])) NA_real_ else -2*fixef(m)[["session1"]]  # post−pre, avg arm
  }
  cat(sprintf("  %-13s %d draws (%.1f%% ok)  median=%+.3f\n", rg, B, 100*mean(is.finite(outv)), median(outv,na.rm=TRUE)))
  data.frame(region=rg, value=outv[is.finite(outv)], stringsAsFactors=FALSE)
}
set.seed(42)
cat(sprintf("Cluster bootstrap: B=%d × %d regions — %s / %s / %s\n", B, length(REGS), ATLAS, PIPELINE, METRIC))
draws <- do.call(rbind, lapply(REGS, boot_region))
write.csv(draws, file.path(TAB,"qin_time_effect_bootstrap_draws.csv"), row.names=FALSE)

# ── 2-3. order by median draw; tag FDR survivors ──
med  <- tapply(draws$value, draws$region, median)
ord  <- names(sort(med))                                                   # ascending → visual (largest) last = top
surv <- OVR$region[!is.na(OVR$session_q) & OVR$session_q < 0.05]
mklab <- function(r) if (r %in% surv) paste0(RLAB[[r]], " *") else RLAB[[r]]
lev  <- vapply(ord, mklab, character(1))
draws$region_lab <- factor(vapply(draws$region, mklab, character(1)), levels=lev)

# ── 4. ridgeline ──
p <- ggplot(draws, aes(x=value, y=region_lab, fill=region_lab)) +
  geom_vline(xintercept=0, color="red", linewidth=0.7) +
  geom_density_ridges(scale=1.7, rel_min_height=0.012, color="white", linewidth=0.3, alpha=0.92) +
  scale_fill_viridis_d(option="mako", begin=0.18, end=0.9, direction=-1, guide="none") +
  scale_x_continuous(expand=expansion(mult=c(0.02,0.03))) +
  labs(title=sprintf("Per-region time effect (post − pre) — %s, %s pipeline", MLAB, PIPELINE),
       subtitle="* = survives BH-FDR (q<0.05, across 5 regions)",
       x="Standardized post − pre coefficient", y=NULL,
       caption=sprintf("Bootstrap distribution of the post−pre coefficient (session main effect, averaged over arm);\nsubject-level cluster resampling, N=%d. Not the spread of subject values.", B)) +
  theme_minimal(base_size=11, base_family="serif") +
  theme(panel.background=element_rect(fill="white",color=NA), plot.background=element_rect(fill="white",color=NA),
        panel.grid.major.y=element_blank(), panel.grid.minor=element_blank(),
        panel.grid.major.x=element_line(color="#e8e8e8", linewidth=0.4),
        plot.title=element_text(hjust=0.5, size=12), plot.subtitle=element_text(hjust=0.5, size=9, color="gray35"),
        plot.caption=element_text(hjust=0, size=7.5, color="gray40"),
        axis.text.y=element_text(size=11))
ggsave(file.path(FIG,"qin_time_effect_bootstrap_ridges.png"), p, width=7.5, height=4.6, dpi=300, bg="white")
cat(sprintf("Saved: %s\n", file.path(FIG,"qin_time_effect_bootstrap_ridges.png")))
