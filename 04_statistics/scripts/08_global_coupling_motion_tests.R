# 08_global_coupling_motion_tests.R
# Three follow-up tests on the maximal pipeline (n=35), per metric × atlas where relevant:
#   Test 3 — motion confound (do FIRST): arm difference in post/Δ raw mean FD (t-test) +
#            ΔFD ↔ ΔSampEn correlation (n=35). SampEn is unresidualized → load-bearing check.
#   Test 1 — global/diffuse effect: per subject = mean Δ across the 6 layers; verum vs placebo
#            10k permutation (raw + motion-residualized). One test per cell, NO FDR.
#   Test 2 — AUC × SampEn coupling: per subject mean ΔAUC vs mean ΔSampEn; Pearson within
#            verum (n=17) and all 35. Expect negative (shorter timescale ↔ higher entropy).
# Units: AUC seconds, SampEn unitless. Δ = post − pre.
#
# In : 04_statistics/results/summary/deltas_long.csv ; 99_QC/01_motion_qc/results/{subject_level_fd_summary.tsv,
#      fd_covariates_wide_thresh03.csv} ; participants.tsv
# Out: 04_statistics/results/summary/{test1_global,test2_coupling,test3_motion}.csv + global_coupling_motion.txt
# Run: Rscript 04_statistics/scripts/08_global_coupling_motion_tests.R

suppressPackageStartupMessages({library(dplyr); library(tidyr)})
REPO <- normalizePath(file.path(dirname(sub("^--file=", "",
        grep("^--file=", commandArgs(FALSE), value = TRUE)[1])), "..", ".."))
SUM  <- file.path(REPO, "04_statistics", "results", "summary")
QC   <- file.path(REPO, "99_QC", "01_motion_qc", "results")
EXCL <- c("sub-06","sub-08","sub-12","sub-26","sub-36"); NPERM <- 10000
LOG  <- c()
say  <- function(...) { line <- sprintf(...); LOG[[length(LOG)+1]] <<- line; cat(line, "\n") }

del <- read.csv(file.path(SUM, "deltas_long.csv")) %>% filter(pipeline == "maximal")
sm  <- del %>% group_by(subject_id, drug, metric, atlas) %>% summarise(mD = mean(delta), .groups = "drop")

fd  <- read.delim(file.path(QC, "subject_level_fd_summary.tsv"))
fd$dFD <- fd$mean_fd_ses02 - fd$mean_fd_ses01; fd$postFD <- fd$mean_fd_ses02
part <- read.delim(file.path(REPO, "participants.tsv")); arm <- setNames(part$condition, part$participant_id)
fd <- fd %>% filter(!subject %in% EXCL) %>% mutate(arm = arm[subject]) %>%
  select(subject, mean_fd_ses01, mean_fd_ses02, dFD, postFD, arm)
cov <- read.csv(file.path(QC, "fd_covariates_wide_thresh03.csv"))[, c("subject","pcf_diff","pcf_sq_diff","mean_fd_retained_diff")]

permp  <- function(v,p){obs<-mean(v)-mean(p);a<-c(v,p);nv<-length(v);set.seed(42)
  n<-replicate(NPERM,{i<-sample(length(a),nv);mean(a[i])-mean(a[-i])});(sum(abs(n)>=abs(obs))+1)/(NPERM+1)}
cohend <- function(v,p){(mean(v)-mean(p))/sqrt(((length(v)-1)*var(v)+(length(p)-1)*var(p))/(length(v)+length(p)-2))}

# ── Test 3 ──
say("###### TEST 3 — MOTION CONFOUND (do first) ######")
t3 <- list()
for (meas in c("postFD","dFD")) {
  v<-fd[[meas]][fd$arm=="verum"]; p<-fd[[meas]][fd$arm=="placebo"]; tt<-t.test(v,p)
  say("  %-7s verum=%.4f placebo=%.4f diff=%+.4f  t=%.2f p=%.3f d=%.2f",
      meas, mean(v), mean(p), mean(v)-mean(p), tt$statistic, tt$p.value, cohend(v,p))
  t3[[length(t3)+1]] <- data.frame(check=paste0("armdiff_",meas), atlas=NA, verum=mean(v),
      placebo=mean(p), diff=mean(v)-mean(p), stat=unname(tt$statistic), p=tt$p.value,
      d_or_r=cohend(v,p), n_verum=length(v), n_placebo=length(p))
}
for (at in c("qinspheres","qinparcels")) {
  m<-merge(sm[sm$metric=="SampEn"&sm$atlas==at,c("subject_id","mD")], fd, by.x="subject_id", by.y="subject")
  ct<-cor.test(m$dFD, m$mD)
  say("  dFD vs dSampEn (%s): r=%+.3f p=%.3f n=%d", at, ct$estimate, ct$p.value, nrow(m))
  t3[[length(t3)+1]] <- data.frame(check="dFD_vs_dSampEn", atlas=at, verum=NA, placebo=NA, diff=NA,
      stat=NA, p=ct$p.value, d_or_r=unname(ct$estimate), n_verum=NA, n_placebo=nrow(m))
}

# ── Test 1 ──
say("\n###### TEST 1 — GLOBAL EFFECT (maximal, no FDR) ######")
t1 <- list()
for (met in c("AUC","SampEn")) for (at in c("qinspheres","qinparcels")) {
  d<-sm[sm$metric==met&sm$atlas==at,]; v<-d$mD[d$drug=="verum"]; p<-d$mD[d$drug=="placebo"]
  dm<-merge(d,cov,by.x="subject_id",by.y="subject")
  dm$res<-residuals(lm(mD~pcf_diff+pcf_sq_diff+mean_fd_retained_diff,data=dm,na.action=na.exclude))
  rv<-dm$res[dm$drug=="verum"]; rp<-dm$res[dm$drug=="placebo"]; prr<-permp(rv,rp)
  say("  %-6s %-11s verum=%+.3f placebo=%+.3f diff=%+.3f p_raw=%.3f d=%.2f | resid_diff=%+.3f p=%.3f",
      met,at,mean(v),mean(p),mean(v)-mean(p),permp(v,p),cohend(v,p),mean(rv)-mean(rp),prr)
  t1[[length(t1)+1]] <- data.frame(metric=met, atlas=at, verum_mean=mean(v), placebo_mean=mean(p),
      diff=mean(v)-mean(p), p_raw=permp(v,p), cohen_d=cohend(v,p),
      resid_diff=mean(rv)-mean(rp), resid_p=prr, n_verum=length(v), n_placebo=length(p))
}

# ── Test 2 ──
say("\n###### TEST 2 — AUC x SampEn COUPLING (maximal) ######")
t2 <- list()
for (at in c("qinspheres","qinparcels")) {
  w<-sm[sm$atlas==at,]%>%pivot_wider(names_from=metric,values_from=mD)
  wv<-w[w$drug=="verum",]; cv<-cor.test(wv$AUC,wv$SampEn); ca<-cor.test(w$AUC,w$SampEn)
  say("  %s: verum r=%+.3f p=%.3f (n=%d) | all35 r=%+.3f p=%.3f (n=%d)",
      at, cv$estimate, cv$p.value, nrow(wv), ca$estimate, ca$p.value, nrow(w))
  t2[[length(t2)+1]] <- data.frame(atlas=at, group="verum", r=unname(cv$estimate), p=cv$p.value, n=nrow(wv))
  t2[[length(t2)+1]] <- data.frame(atlas=at, group="all35", r=unname(ca$estimate), p=ca$p.value, n=nrow(w))
}

say("\nPOWER: SampEn global d≈0.31 → ~16%% power at n=17/18 (α=.05). 80%% power needs d≈0.97, or ~165/arm at d=0.31.")

write.csv(do.call(rbind,t1), file.path(SUM,"test1_global.csv"),  row.names=FALSE)
write.csv(do.call(rbind,t2), file.path(SUM,"test2_coupling.csv"),row.names=FALSE)
write.csv(do.call(rbind,t3), file.path(SUM,"test3_motion.csv"),  row.names=FALSE)
writeLines(unlist(LOG), file.path(SUM,"global_coupling_motion.txt"))
cat("\nsaved: test1_global.csv, test2_coupling.csv, test3_motion.csv, global_coupling_motion.txt\n")
