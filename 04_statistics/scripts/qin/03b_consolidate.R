# 03b_consolidate.R — collapse the per-config mixed-model outputs into one tidy CSV
# per (atlas × metric), so results are readable at a glance.
#
# Reads : 04_statistics/results/mixed_models/per_config/{atlas_tag}_{pipeline}{_metric}_{did,overall,overall_global}.csv
#         atlas_tag = spheres | parcels_self_regions   (matches the results tree)
#         pipeline  = maximal  (detrend excluded — broadband-degraded; glm removed from the project)
# Writes: 04_statistics/results/mixed_models/{atlas_tag}_{metric}_summary.csv   (4 files)
#         Long/tidy: one row per (pipeline × analysis × region × term).
#           analysis = did      → term "session:arm (DiD)"          (drug modulation of post−pre)
#           analysis = overall  → terms "session (post−pre)", "arm (verum−placebo)"  (per-region main effects)
#           analysis = global   → region "ALL": overall session / arm / DiD pooled over the 5 regions
# Run   : Rscript 04_statistics/scripts/qin/03b_consolidate.R

REPO <- local({
  m <- grep("^--file=", commandArgs(FALSE), value=TRUE)
  f <- if (length(m)) sub("^--file=","",m[1]) else NULL
  if (is.null(f)) normalizePath(getwd())
  else normalizePath(file.path(dirname(normalizePath(f)), "..","..","..")) })
BASE     <- file.path(REPO,"04_statistics","results","mixed_models")
PC       <- file.path(BASE,"per_config")
PIPES    <- c("maximal")                  # detrend broadband-degraded; glm removed from the project
# Filename tokens, matching the results tree (03_mixed_models.R writes these via ATLAS_TAG).
# The scripts' CLI argument is still qinspheres|qinparcels; only the filenames use these.
ATLASES  <- c("spheres","parcels_self_regions")
METRICS  <- c("auc","sampen")
COLS     <- c("pipeline","analysis","region","term","estimate","dz","ci_low","ci_high","p","q","KR_df","n","singular")

blank <- function(n) { d <- as.data.frame(matrix(NA, n, length(COLS))); names(d) <- COLS; d }
rd <- function(f) if (file.exists(f)) read.csv(f, stringsAsFactors=FALSE) else NULL

collect <- function(atlas, metric){
  msuf <- if (metric=="auc") "" else paste0("_",metric)
  out <- list()
  for (pipe in PIPES){
    stem <- file.path(PC, sprintf("%s_%s%s", atlas, pipe, msuf))
    did <- rd(paste0(stem,"_did.csv")); ov <- rd(paste0(stem,"_overall.csv")); gl <- rd(paste0(stem,"_overall_global.csv"))
    if (is.null(did) && is.null(ov) && is.null(gl)){
      cat(sprintf("  [skip] %s / %s — no per-config files\n", atlas, pipe)); next }
    if (!is.null(did)){
      d <- blank(nrow(did)); d$pipeline<-pipe; d$analysis<-"did"; d$region<-did$region
      d$term<-"session:arm (DiD)"; d$estimate<-did$effect; d$dz<-did$dz
      d$ci_low<-did$ci_low; d$ci_high<-did$ci_high; d$p<-did$p; d$q<-did$q
      d$KR_df<-did$KR_df; d$n<-did$n; d$singular<-did$singular; out[[length(out)+1]]<-d }
    if (!is.null(ov)){
      s <- blank(nrow(ov)); s$pipeline<-pipe; s$analysis<-"overall"; s$region<-ov$region
      s$term<-"session (post-pre)"; s$estimate<-ov$session_est; s$ci_low<-ov$session_ci_low; s$ci_high<-ov$session_ci_high
      s$p<-ov$session_p; s$q<-ov$session_q; s$singular<-ov$singular
      a <- blank(nrow(ov)); a$pipeline<-pipe; a$analysis<-"overall"; a$region<-ov$region
      a$term<-"arm (verum-placebo)"; a$estimate<-ov$arm_est; a$ci_low<-ov$arm_ci_low; a$ci_high<-ov$arm_ci_high
      a$p<-ov$arm_p; a$q<-ov$arm_q; a$singular<-ov$singular
      out[[length(out)+1]]<-s; out[[length(out)+1]]<-a }
    if (!is.null(gl)){
      g <- blank(nrow(gl)); g$pipeline<-pipe; g$analysis<-"global"; g$region<-"ALL"
      g$term<-gl$term; g$estimate<-gl$est; g$p<-gl$p; g$singular<-gl$singular; out[[length(out)+1]]<-g }
  }
  if (!length(out)) return(NULL)
  df <- do.call(rbind, out)
  df[order(match(df$pipeline,PIPES), match(df$analysis,c("did","overall","global")), df$region, df$term), ]
}

for (atlas in ATLASES) for (metric in METRICS){
  df <- collect(atlas, metric)
  if (is.null(df)){ cat(sprintf("[none] %s / %s\n", atlas, metric)); next }
  f <- file.path(BASE, sprintf("%s_%s_summary.csv", atlas, metric))
  write.csv(df, f, row.names=FALSE)
  cat(sprintf("[write] %s  (%d rows: %s)\n", basename(f), nrow(df), paste(PIPES,collapse="+")))
}
cat("done.\n")
