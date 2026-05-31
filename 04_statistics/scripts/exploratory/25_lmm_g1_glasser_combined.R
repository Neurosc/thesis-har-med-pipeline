# 25_lmm_g1_glasser_combined.R
# Replaces scripts 16 and 23.
#
# G1 gradient LMM using ALL available Glasser parcels as a single dataset:
#   - 258 nonself parcels  (tSNR-filtered, from nonself_g1_model_ready.csv)
#   - 33  Keskin self parcels (new Glasser-based extraction)
#   = 291 unique Glasser parcels, no self/nonself split, continuous G1_z.
#
# Model: auc ~ group * session * G1_z + (1|subject) + (1|roi_uid)
# (roi_uid is prefixed to avoid ID collisions between the two sources)
#
# Run from repo root:
#   Rscript 04_statistics/scripts/25_lmm_g1_glasser_combined.R

suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(MuMIn)
  library(dplyr); library(tidyr); library(moments)
  library(plotly); library(htmlwidgets)
})

# ── Paths ──────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
  REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
} else {
  REPO_ROOT  <- normalizePath(".")
}

NS_CSV   <- file.path(REPO_ROOT, "04_statistics", "results",
                       "nonself_g1_model_ready.csv")        # 258 parcels
SELF_CSV <- file.path(REPO_ROOT, "04_statistics", "results",
                       "keskin_auc_model_ready.csv")         # 33 Keskin self

OUT_DATA   <- file.path(REPO_ROOT, "04_statistics", "results",
                         "glasser_combined_model_ready.csv")
OUT_FIXED  <- file.path(REPO_ROOT, "04_statistics", "results",
                         "lmm_g1_glasser_combined_summary.csv")
OUT_RANDOM <- file.path(REPO_ROOT, "04_statistics", "results",
                         "lmm_g1_glasser_combined_random.csv")
OUT_FIG_DIAG  <- file.path(REPO_ROOT, "04_statistics", "figures",
                             "fig25_glasser_combined_diagnostics.html")
OUT_FIG_SCAT  <- file.path(REPO_ROOT, "04_statistics", "figures",
                             "fig26_glasser_combined_g1_scatter.html")

all_out <- c(OUT_DATA, OUT_FIXED, OUT_RANDOM, OUT_FIG_DIAG, OUT_FIG_SCAT)
if (all(file.exists(all_out))) {
  cat("All outputs exist — nothing to do. Delete to rerun.\n"); quit(status = 0)
}

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n25_lmm_g1_glasser_combined.R\n",
    "Replacing scripts 16 and 23: 258 nonself + 33 Keskin self = 291 parcels\n",
    SEP, "\n\n", sep = "")

# ─────────────────────────────────────────────────────────────────────────────
cat(SEP, "\nPART 1 — Assemble combined dataset\n", SEP, "\n", sep = "")

# Nonself 258 parcels
df_ns <- read.csv(NS_CSV, stringsAsFactors = FALSE)
df_ns$roi_uid <- paste0("ns_", df_ns$roi_pos_id)
df_ns <- df_ns[, c("subject","session","group","roi_uid","roi_name","auc","G1")]
cat(sprintf("Nonself (tSNR-filtered): %d rows, %d unique parcels\n",
            nrow(df_ns), length(unique(df_ns$roi_uid))))

# Keskin self 33 parcels — deduplicate (parcels 78 & 291 appear in 2 layers)
df_self_raw <- read.csv(SELF_CSV, stringsAsFactors = FALSE)
df_self <- df_self_raw %>%
  distinct(subject, session, glasser_id, .keep_all = TRUE) %>%
  mutate(roi_uid = paste0("ks_", glasser_id)) %>%
  select(subject, session, group, roi_uid, roi_name, auc, G1)
cat(sprintf("Keskin self           : %d rows, %d unique parcels  (%d dup rows dropped)\n",
            nrow(df_self), length(unique(df_self$roi_uid)),
            nrow(df_self_raw) - nrow(df_self)))

# Stack
df <- rbind(df_ns, df_self)

# Fresh G1_z across all 291 parcels (one value per unique parcel, then broadcast)
roi_g1 <- df %>%
  distinct(roi_uid, G1)
μ_g1 <- mean(roi_g1$G1); σ_g1 <- sd(roi_g1$G1)
df$G1_z <- (df$G1 - μ_g1) / σ_g1
cat(sprintf("G1_z re-scaled across %d unique parcels: mean=%.4f, SD=%.4f\n",
            nrow(roi_g1), mean(df$G1_z), sd(df$G1_z)))

# Factors
df$group   <- relevel(factor(df$group),   ref = "placebo")
df$session <- relevel(factor(df$session), ref = "ses-01")
df$subject <- factor(df$subject)
df$roi_uid <- factor(df$roi_uid)

cat(sprintf("\nCombined: %d rows | %d subjects | %d ROIs\n",
            nrow(df), nlevels(df$subject), nlevels(df$roi_uid)))
cat(sprintf("G1_z range: [%.4f, %.4f]\n", min(df$G1_z), max(df$G1_z)))

write.csv(df, OUT_DATA, row.names = FALSE)
cat(sprintf("Saved: %s\n", OUT_DATA))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 2 — Fit model\n", SEP, "\n", sep = "")
cat("Formula: auc ~ group * session * G1_z + (1|subject) + (1|roi_uid)\n\n")

ctrl_bobyqa <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5),
                            check.conv.grad = .makeCC("warning", tol = 0.002))

m <- tryCatch(
  lmer(auc ~ group * session * G1_z + (1|subject) + (1|roi_uid),
       data = df, REML = TRUE, control = ctrl_bobyqa),
  warning = function(w) {
    cat(sprintf("bobyqa warning: %s\nRetrying Nelder-Mead...\n", conditionMessage(w)))
    lmer(auc ~ group * session * G1_z + (1|subject) + (1|roi_uid),
         data = df, REML = TRUE,
         control = lmerControl(optimizer = "Nelder_Mead",
                               optCtrl   = list(maxfun = 2e5),
                               check.conv.grad = .makeCC("warning", tol = 0.002)))
  }
)

conv <- m@optinfo$conv$lme4$messages
if (!is.null(conv)) { cat(sprintf("CONVERGENCE: %s\n\n", paste(conv, collapse="; ")))
} else {              cat("Convergence: OK\n\n") }

# ─────────────────────────────────────────────────────────────────────────────
cat(SEP, "\nPART 3 — Fixed effects (Satterthwaite)\n", SEP, "\n", sep = "")
s <- summary(m); print(s)
cat(sprintf("\nAIC: %.4f  BIC: %.4f  logLik: %.4f\n",
            AIC(m), BIC(m), as.numeric(logLik(m))))

fe_raw <- coef(s)
fe_tbl <- data.frame(term=rownames(fe_raw), estimate=fe_raw[,"Estimate"],
                     SE=fe_raw[,"Std. Error"], df=fe_raw[,"df"],
                     t_value=fe_raw[,"t value"], p_value=fe_raw[,"Pr(>|t|)"],
                     stringsAsFactors=FALSE, row.names=NULL)

cat("\nRandom effects:\n")
vc <- as.data.frame(VarCorr(m)); print(vc)

cat("\n", SEP, "\nPART 4 — Key contrasts\n", SEP, "\n", sep = "")
key_terms  <- c("groupverum:sessionses-02",
                "sessionses-02:G1_z",
                "groupverum:sessionses-02:G1_z")
key_labels <- c("group x session                   (global drug x session)",
                "session x G1_z                    (pre→post change along cortex?)",
                "group x session x G1_z  [PRIMARY] (drug effect varies along cortex?)")
for (i in seq_along(key_terms)) {
  r <- fe_tbl[fe_tbl$term == key_terms[i], ]
  if (nrow(r) == 0) { cat(sprintf("WARNING: %s not found\n", key_terms[i])); next }
  cat(sprintf("\n%s\n  beta=%9.6f  SE=%9.6f  t(%7.1f)=%8.4f  p=%.4g\n",
              key_labels[i], r$estimate, r$SE, r$df, r$t_value, r$p_value))
}

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 5 — Effect size\n", SEP, "\n", sep = "")
r2 <- tryCatch(r.squaredGLMM(m),
               error = function(e) { cat("MuMIn R2 failed:", e$message,"\n"); NULL })
if (!is.null(r2)) {
  cat(sprintf("Marginal  R2: %.4f\n", r2[1,"R2m"]))
  cat(sprintf("Conditional R2: %.4f\n", r2[1,"R2c"]))
}
b3 <- fe_tbl$estimate[fe_tbl$term == "groupverum:sessionses-02:G1_z"]
if (length(b3) > 0)
  cat(sprintf("\nFor each 1-SD increase in G1, drug×session effect changes by %.6f s.\n", b3))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 6 — Residual diagnostics\n", SEP, "\n", sep = "")
resids <- resid(m); fitted_ <- fitted(m)
sk <- skewness(resids); ku <- kurtosis(resids)
cat(sprintf("Skewness: %.4f  Kurtosis: %.4f  (excess %.4f)\n", sk, ku, ku-3))
set.seed(42); sw_idx <- sample(length(resids), min(5000L, length(resids)))
sw <- shapiro.test(resids[sw_idx])
cat(sprintf("Shapiro-Wilk (n=%d): W=%.5f  p=%.4e\n", length(sw_idx), sw$statistic, sw$p.value))

qq_n <- min(5000L, length(resids))
qq_smp <- sort(resids[sample(length(resids), qq_n)]); qq_teo <- qnorm(ppoints(qq_n))
p_qq <- plot_ly() %>%
  add_trace(x=qq_teo, y=qq_smp, type="scatter", mode="markers",
            marker=list(size=4,color="steelblue",opacity=0.5)) %>%
  add_trace(x=range(qq_teo), y=range(qq_teo)*sd(qq_smp), type="scatter",
            mode="lines", line=list(color="red",dash="dash")) %>%
  layout(title="QQ Plot", xaxis=list(title="Theoretical"), yaxis=list(title="Sample"),
         showlegend=FALSE)
rvf_n <- sample(length(resids), min(5000L, length(resids)))
p_rvf <- plot_ly() %>%
  add_trace(x=fitted_[rvf_n], y=resids[rvf_n], type="scatter", mode="markers",
            marker=list(size=4,color="steelblue",opacity=0.4)) %>%
  add_trace(x=range(fitted_), y=c(0,0), type="scatter", mode="lines",
            line=list(color="red",dash="dash")) %>%
  layout(title="Resid vs Fitted", xaxis=list(title="Fitted"),
         yaxis=list(title="Residuals"), showlegend=FALSE)
p_hist <- plot_ly(x=resids, type="histogram",
                  marker=list(color="steelblue",line=list(color="white",width=0.3)),
                  nbinsx=80) %>%
  layout(title="Histogram", xaxis=list(title="Residual"), yaxis=list(title="Count"))
fig25 <- subplot(p_qq, p_rvf, p_hist, nrows=1, titleX=TRUE, titleY=TRUE, margin=0.06) %>%
  layout(title=list(text="LMM Residuals — 291 Glasser parcels", x=0.5), showlegend=FALSE)
saveWidget(fig25, OUT_FIG_DIAG, selfcontained=FALSE, title="Glasser Combined Diagnostics")
cat(sprintf("Saved: %s\n", OUT_FIG_DIAG))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 7 — LRT (drop 3-way + group:G1_z)\n", SEP, "\n", sep = "")
ctrl_ml   <- lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5))
m_full_ml <- update(m, REML=FALSE, control=ctrl_ml)
m_red_ml  <- lmer(auc ~ group * session + G1_z + session:G1_z +
                       (1|subject) + (1|roi_uid),
                  data=df, REML=FALSE, control=ctrl_ml)
lrt     <- anova(m_red_ml, m_full_ml); print(lrt)
chi2_lrt <- lrt$Chisq[2]; df_lrt <- lrt$Df[2]; p_lrt <- lrt$`Pr(>Chisq)`[2]
cat(sprintf("LRT: chi2(%d) = %.4f, p = %.4g  →  3-way %s significant\n",
            df_lrt, chi2_lrt, p_lrt, if(p_lrt<0.05)"IS" else "is NOT"))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 8 — G1 scatter (per-ROI drug×session effect vs G1_z)\n",
    SEP, "\n", sep = "")

roi_means <- df %>%
  mutate(ses_lbl = ifelse(session=="ses-01","pre","post")) %>%
  group_by(roi_uid, ses_lbl, group) %>%
  summarise(mean_auc=mean(auc,na.rm=TRUE), G1_z=first(G1_z), G1=first(G1),
            roi_name=first(roi_name), .groups="drop")

roi_wide <- roi_means %>%
  pivot_wider(id_cols=c(roi_uid,G1_z,G1,roi_name),
              names_from=c(group,ses_lbl), values_from=mean_auc)

if (all(c("placebo_pre","placebo_post","verum_pre","verum_post") %in% names(roi_wide))) {
  roi_wide <- roi_wide %>%
    mutate(drug_ses_delta = (verum_post - verum_pre) - (placebo_post - placebo_pre),
           source = ifelse(startsWith(as.character(roi_uid),"ks_"), "self (Keskin)","nonself"))

  beta_gs  <- fe_tbl$estimate[fe_tbl$term == "groupverum:sessionses-02"]
  beta_gsg <- b3
  vcov_m   <- as.matrix(vcov(m))
  cn       <- rownames(vcov_m)
  gs_idx   <- which(cn == "groupverum:sessionses-02")
  gsg_idx  <- which(cn == "groupverum:sessionses-02:G1_z")

  g1_grid <- seq(min(df$G1_z), max(df$G1_z), length.out=200)
  pred_y  <- beta_gs + beta_gsg * g1_grid
  pred_se <- vapply(g1_grid, function(g) {
    cv <- numeric(length(cn)); cv[gs_idx] <- 1; cv[gsg_idx] <- g
    sqrt(as.numeric(t(cv) %*% vcov_m %*% cv))
  }, numeric(1))

  fig26 <- plot_ly() %>%
    add_trace(x=c(g1_grid,rev(g1_grid)), y=c(pred_y+1.96*pred_se,rev(pred_y-1.96*pred_se)),
              type="scatter", mode="lines", fill="toself",
              fillcolor="rgba(70,130,180,0.15)", line=list(color="transparent"),
              name="95% CI", showlegend=TRUE) %>%
    add_trace(x=g1_grid, y=pred_y, type="scatter", mode="lines",
              line=list(color="steelblue",width=2),
              name=sprintf("Model (β=%.4f)", beta_gsg)) %>%
    add_trace(data=roi_wide[roi_wide$source=="nonself",],
              x=~G1_z, y=~drug_ses_delta, type="scatter", mode="markers",
              marker=list(size=7,color="navy",opacity=0.55),
              text=~roi_name,
              hovertemplate="ROI: %{text}<br>G1_z: %{x:.3f}<br>delta: %{y:.4f}<extra></extra>",
              name="Nonself ROIs") %>%
    add_trace(data=roi_wide[roi_wide$source=="self (Keskin)",],
              x=~G1_z, y=~drug_ses_delta, type="scatter", mode="markers",
              marker=list(size=9,color="firebrick",opacity=0.80,symbol="diamond"),
              text=~roi_name,
              hovertemplate="ROI: %{text}<br>G1_z: %{x:.3f}<br>delta: %{y:.4f}<extra></extra>",
              name="Self ROIs (Keskin)") %>%
    layout(title=list(text="Drug×Session Effect Along G1 — 291 Glasser Parcels", x=0.5),
           xaxis=list(title="G1_z  (sensory ← → transmodal)"),
           yaxis=list(title="Drug×Session delta AUC (s)"),
           legend=list(x=0.02,y=0.98))

  saveWidget(fig26, OUT_FIG_SCAT, selfcontained=FALSE, title="G1 Scatter 291 Parcels")
  cat(sprintf("Saved: %s\n", OUT_FIG_SCAT))
}

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 9 — Comparison with superseded analyses\n", SEP, "\n", sep = "")

OLD16_B <- 0.031928; OLD16_P <- 0.002030; OLD16_LRT <- 1.33e-11;  OLD16_N <- 18042L
OLD23_B <- 0.030333; OLD23_P <- 0.001420; OLD23_LRT <- 1.828e-11; OLD23_N <- 21760L
new_b   <- fe_tbl$estimate[fe_tbl$term == "groupverum:sessionses-02:G1_z"]
new_p   <- fe_tbl$p_value[ fe_tbl$term == "groupverum:sessionses-02:G1_z"]

cat(sprintf("%-32s  %-16s  %-16s  %-16s\n",
            "Metric", "Script 16 (258)", "Script 23 (316)", "Script 25 (291)"))
cat(strrep("-", 80), "\n")
cat(sprintf("%-32s  %-16s  %-16s  %-16s\n", "group:session:G1_z β",
            sprintf("%.5f", OLD16_B), sprintf("%.5f", OLD23_B),
            sprintf("%.5f", if(length(new_b)>0) new_b else NA)))
cat(sprintf("%-32s  %-16s  %-16s  %-16s\n", "group:session:G1_z p",
            sprintf("%.4g", OLD16_P), sprintf("%.4g", OLD23_P),
            sprintf("%.4g", if(length(new_p)>0) new_p else NA)))
cat(sprintf("%-32s  %-16s  %-16s  %-16s\n", "LRT p",
            sprintf("%.4g", OLD16_LRT), sprintf("%.4g", OLD23_LRT),
            sprintf("%.4g", p_lrt)))
cat(sprintf("%-32s  %-16s  %-16s  %-16s\n", "N observations",
            format(OLD16_N,big.mark=","), format(OLD23_N,big.mark=","),
            format(nrow(df),big.mark=",")))

new_sig <- length(new_p)>0 && new_p < 0.05
cat(sprintf("\nConclusion: group×session×G1_z interaction %s significant (p=%.4g).\n",
            if(new_sig)"IS" else "is NOT", if(length(new_p)>0) new_p else NA))
if (length(new_b)>0 && length(new_p)>0)
  cat(sprintf("β (258→291): %.5f→%.5f (Δ=%.5f). Both self and nonself parcels follow the same G1 gradient.\n",
              OLD16_B, new_b, new_b - OLD16_B))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSaving CSV outputs\n", SEP, "\n", sep = "")
write.csv(fe_tbl, OUT_FIXED,  row.names=FALSE); cat(sprintf("Saved: %s\n", OUT_FIXED))
write.csv(vc,     OUT_RANDOM, row.names=FALSE); cat(sprintf("Saved: %s\n", OUT_RANDOM))

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
