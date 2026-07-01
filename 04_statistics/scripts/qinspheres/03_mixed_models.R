# 09_mixed_retreat_did.R — mixed-model retreat + DiD drug effect, per region.
# Long table: subject × session × region. AUC = mean over region ROIs (auc>0).
# Covariates (session-specific): pcf (% censored), pcf_sq=pcf^2, mean_fd (mean FD of retained frames).
#   Retreat (placebo only): AUC ~ session + pcf + pcf_sq + mean_fd + (1|subject); pull `sessionpost`.
#   DiD (both arms):        AUC ~ session*arm + pcf + pcf_sq + mean_fd + (1|subject); pull `sessionpost:armverum`.
# REML, Kenward-Roger df (lmerTest+pbkrtest). Singular fit → note + paired/label permutation fallback.
# FDR (BH) across the 6 regions, per analysis. Sign kept visible.
#
# Args: [pipeline] [atlas]  (default maximal qinspheres)
# Out : 04_statistics/results/mixed_models/{atlas}_{pipeline}_{longtable,retreat,did}.csv
# Run : Rscript 04_statistics/scripts/09_mixed_retreat_did.R [pipeline] [atlas]

for (p in c("lme4","lmerTest","pbkrtest","dplyr","tidyr"))
  if (!requireNamespace(p, quietly=TRUE)) install.packages(p, repos="https://cloud.r-project.org", quiet=TRUE)
suppressPackageStartupMessages({library(lmerTest); library(pbkrtest); library(dplyr); library(tidyr)})

a <- commandArgs(trailingOnly=TRUE)
PIPELINE <- if (length(a)>=1) a[1] else "maximal"
ATLAS    <- if (length(a)>=2) a[2] else "qinspheres"
REPO <- normalizePath(file.path(dirname(sub("^--file=","",grep("^--file=",commandArgs(FALSE),value=TRUE)[1])),"..","..",".."))
TBL  <- file.path(REPO,"04_statistics","results",ATLAS,PIPELINE,"tables","qinspheres_auc.csv")
QC   <- file.path(REPO,"99_QC","01_motion_qc","results","fd_covariates_wide_thresh03.csv")
OUT  <- file.path(REPO,"04_statistics","results","mixed_models"); dir.create(OUT,recursive=TRUE,showWarnings=FALSE)
LAB  <- c(intero="intero",extero="extero",mental="cognition",visual="visual",motor="motor",auditory="auditory")
REGS <- unname(LAB)

# ── 1. long table ──
auc <- read.csv(TBL) %>% filter(auc>0, is.finite(auc)) %>%
  group_by(subject, session, drug_group, category) %>% summarise(AUC=mean(auc), .groups="drop")
cov <- read.csv(QC)
cl <- cov %>% transmute(subject, pre.pcf=pcf_pre, post.pcf=pcf_post,
                        pre.mean_fd=mean_fd_retained_pre, post.mean_fd=mean_fd_retained_post) %>%
  pivot_longer(-subject, names_to=c("session","var"), names_sep="\\.") %>%
  pivot_wider(names_from=var, values_from=value)
dat <- auc %>% mutate(session=ifelse(session=="ses-01","pre","post"), region=LAB[category]) %>%
  left_join(cl, by=c("subject","session")) %>%
  mutate(pcf_sq=pcf^2,
         subject=factor(subject),
         session=factor(session, levels=c("pre","post")),
         arm=factor(drug_group, levels=c("placebo","verum")),
         region=factor(region, levels=REGS)) %>%
  select(AUC, subject, session, arm, region, pcf, pcf_sq, mean_fd)
write.csv(dat, file.path(OUT, sprintf("%s_%s_longtable.csv",ATLAS,PIPELINE)), row.names=FALSE)
cat(sprintf("long table: %d rows | factors: subject=%s session=%s(%s) arm=%s(%s) region=%s\n",
    nrow(dat), is.factor(dat$subject), is.factor(dat$session), paste(levels(dat$session),collapse="/"),
    is.factor(dat$arm), paste(levels(dat$arm),collapse="/"), is.factor(dat$region)))

permp <- function(x, mode, g=NULL){  # mode: "signflip" (retreat Δ) | "label" (DiD Δ diff)
  set.seed(42)
  if (mode=="signflip"){ obs<-mean(x); nul<-replicate(10000, mean(x*sample(c(-1,1),length(x),TRUE))) }
  else { v<-x[g=="verum"]; p<-x[g=="placebo"]; obs<-mean(v)-mean(p); a<-c(v,p); nv<-length(v)
         nul<-replicate(10000,{i<-sample(length(a),nv); mean(a[i])-mean(a[-i])}) }
  (sum(abs(nul)>=abs(obs))+1)/10001
}

fit_one <- function(reg, kind){
  d <- dat %>% filter(region==reg); if (kind=="retreat") d <- d %>% filter(arm=="placebo")
  form <- if (kind=="retreat") AUC ~ session + pcf + pcf_sq + mean_fd + (1|subject) else
                               AUC ~ session*arm + pcf + pcf_sq + mean_fd + (1|subject)
  m <- suppressWarnings(suppressMessages(lmer(form, data=d, REML=TRUE)))
  term <- if (kind=="retreat") "sessionpost" else "sessionpost:armverum"
  co <- tryCatch(summary(m, ddf="Kenward-Roger")$coefficients, error=function(e) summary(m)$coefficients)
  est<-co[term,1]; se<-co[term,2]; df<-if(ncol(co)>=5) co[term,3] else NA; p<-co[term,ncol(co)]
  ci<-if(!is.na(df)) est+c(-1,1)*qt(.975,df)*se else est+c(-1,1)*1.96*se
  sing <- isSingular(m)
  # per-subject Δ (post−pre) for std effect + fallback
  w <- d %>% select(subject, arm, session, AUC) %>% pivot_wider(names_from=session, values_from=AUC) %>%
       filter(!is.na(pre),!is.na(post)) %>% mutate(D=post-pre)
  if (kind=="retreat"){ dz<-mean(w$D)/sd(w$D); nsub<-nrow(w)
    if (sing) p <- permp(w$D,"signflip")
  } else { dv<-w$D[w$arm=="verum"]; dp<-w$D[w$arm=="placebo"]
    dz<-(mean(dv)-mean(dp))/sqrt(((length(dv)-1)*var(dv)+(length(dp)-1)*var(dp))/(length(dv)+length(dp)-2))
    nsub<-nrow(w); if (sing) p <- permp(w$D,"label",w$arm) }
  data.frame(region=reg, effect=round(est,4), ci_low=round(ci[1],4), ci_high=round(ci[2],4),
             dz=round(dz,3), KR_df=round(df,1), p=round(p,4), n=nsub, singular=sing)
}

run <- function(kind){
  r <- do.call(rbind, lapply(REGS, fit_one, kind=kind)); r$q <- round(p.adjust(r$p,"BH"),4)
  r$survives <- ifelse(r$q<0.05,"yes","no"); r <- r[,c("region","effect","dz","ci_low","ci_high","p","q","survives","n","singular","KR_df")]
  write.csv(r, file.path(OUT, sprintf("%s_%s_%s.csv",ATLAS,PIPELINE,kind)), row.names=FALSE)
  cat(sprintf("\n==== %s — %s/%s (n=%d) ====\n", toupper(kind), ATLAS, PIPELINE, r$n[1])); print(r, row.names=FALSE); r
}
cat(sprintf("\n######## %s / %s ########\n", ATLAS, PIPELINE))
invisible(run("retreat")); invisible(run("did"))
