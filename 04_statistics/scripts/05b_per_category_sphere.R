# 05b_per_category_sphere.R
# Per-category drug × session tests (sphere-based data)
#
# Part 1: Per-category LMMs (auc ~ group * session + RE)
# Part A: Exteroception vs nonself formal contrast using full 04b model
#
# Input:  04_statistics/results/sphere_lmm_model_ready.csv
# Output: 04_statistics/results/sphere_per_category_drug_session.csv
#         04_statistics/results/sphere_per_category_simple_effects.csv
#
# Run from repo root:
#   Rscript 04_statistics/scripts/05b_per_category_sphere.R

if (!requireNamespace("emmeans", quietly = TRUE))
  install.packages("emmeans", repos = "https://cloud.r-project.org", quiet = TRUE)

suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(emmeans)
  library(dplyr)
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

DATA_CSV <- file.path(REPO_ROOT, "04_statistics", "results",
                       "sphere_lmm_model_ready.csv")
OUT_INT  <- file.path(REPO_ROOT, "04_statistics", "results",
                       "sphere_per_category_drug_session.csv")
OUT_SFX  <- file.path(REPO_ROOT, "04_statistics", "results",
                       "sphere_per_category_simple_effects.csv")

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n05b_per_category_sphere.R\n", SEP, "\n\n", sep = "")

# ── Load data ──────────────────────────────────────────────────────────────────
df_raw <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n\n", nrow(df_raw)))

df_raw$category  <- ifelse(df_raw$self_layer == "nonself",
                            "Sensory-Motor", df_raw$self_layer)
df_raw$self_layer <- relevel(factor(df_raw$self_layer), ref = "nonself")
df_raw$group      <- relevel(factor(df_raw$group),      ref = "placebo")
df_raw$session    <- relevel(factor(df_raw$session),    ref = "ses-01")
df_raw$subject    <- factor(df_raw$subject)
df_raw$roi_uid    <- factor(df_raw$roi_uid)

CAT_ORDER <- c("Sensory-Motor", "Interoception", "Exteroception", "Cognition")
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5),
                    check.conv.grad = .makeCC("warning", tol = 0.002))

# ── Part 1: Per-category LMMs ─────────────────────────────────────────────────
cat(SEP, "\nPART 1 — Per-category drug × session tests\n", SEP, "\n", sep = "")

int_rows <- list()
sfx_rows <- list()

for (cat in CAT_ORDER) {
  sub <- df_raw[df_raw$category == cat, ]
  sub$roi_uid <- droplevels(sub$roi_uid)
  n_roi <- nlevels(sub$roi_uid)

  m_cat <- tryCatch(
    suppressWarnings(
      lmer(auc ~ group * session + (1|subject) + (1|roi_uid),
           data = sub, REML = TRUE, control = ctrl)
    ),
    error = function(e) { cat(sprintf("  [%s] ERROR: %s\n", cat, e$message)); NULL }
  )
  if (is.null(m_cat)) next

  fe    <- coef(summary(m_cat))
  b_int <- fe["groupverum:sessionses-02", "Estimate"]
  se_i  <- fe["groupverum:sessionses-02", "Std. Error"]
  t_i   <- fe["groupverum:sessionses-02", "t value"]
  p_i   <- fe["groupverum:sessionses-02", "Pr(>|t|)"]

  int_rows[[cat]] <- data.frame(category = cat, n_roi = n_roi,
    drug_session_b = b_int, drug_session_SE = se_i,
    drug_session_t = t_i,   drug_session_p  = p_i,
    stringsAsFactors = FALSE)

  b_pl  <- fe["sessionses-02", "Estimate"]
  se_pl <- fe["sessionses-02", "Std. Error"]
  t_pl  <- fe["sessionses-02", "t value"]
  p_pl  <- fe["sessionses-02", "Pr(>|t|)"]
  b_ve  <- b_pl + b_int
  vcv   <- as.matrix(vcov(m_cat))
  se_ve <- sqrt(vcv["sessionses-02","sessionses-02"] +
                vcv["groupverum:sessionses-02","groupverum:sessionses-02"] +
                2 * vcv["sessionses-02","groupverum:sessionses-02"])
  df_ve <- fe["groupverum:sessionses-02", "df"]
  t_ve  <- b_ve / se_ve
  p_ve  <- 2 * pt(-abs(t_ve), df = df_ve)

  sfx_rows[[cat]] <- rbind(
    data.frame(category = cat, group = "placebo",
               preto_post_b = b_pl, SE = se_pl, t = t_pl, p = p_pl,
               stringsAsFactors = FALSE),
    data.frame(category = cat, group = "verum",
               preto_post_b = b_ve, SE = se_ve, t = t_ve, p = p_ve,
               stringsAsFactors = FALSE))
}

int_tbl <- do.call(rbind, int_rows); rownames(int_tbl) <- NULL
sfx_tbl <- do.call(rbind, sfx_rows); rownames(sfx_tbl) <- NULL

cat("\nDrug × Session interaction per category:\n")
cat(sprintf("  %-16s %5s %10s %8s %6s %8s\n", "Category","N_ROI","β","SE","t","p"))
cat(strrep("-", 58), "\n")
for (i in seq_len(nrow(int_tbl))) {
  r <- int_tbl[i,]
  cat(sprintf("  %-16s %5d %10.6f %8.6f %6.3f %8.4g%s\n",
              r$category, r$n_roi, r$drug_session_b, r$drug_session_SE,
              r$drug_session_t, r$drug_session_p,
              if (!is.na(r$drug_session_p) && r$drug_session_p < 0.05) " *" else ""))
}

cat("\nSimple effects:\n")
cat(sprintf("  %-16s %-8s %10s %8s %6s %8s\n", "Category","Group","β","SE","t","p"))
cat(strrep("-", 58), "\n")
for (i in seq_len(nrow(sfx_tbl))) {
  r <- sfx_tbl[i,]
  cat(sprintf("  %-16s %-8s %10.6f %8.6f %6.3f %8.4g%s\n",
              r$category, r$group, r$preto_post_b, r$SE, r$t, r$p,
              if (!is.na(r$p) && r$p < 0.05) " *" else ""))
}

sig_cats <- int_tbl$category[!is.na(int_tbl$drug_session_p) & int_tbl$drug_session_p < 0.05]
cat(sprintf("\nSignificant: %s\n",
            if (length(sig_cats) > 0) paste(sig_cats, collapse = ", ") else "none"))

write.csv(int_tbl, OUT_INT, row.names = FALSE); cat(sprintf("Saved: %s\n", OUT_INT))
write.csv(sfx_tbl, OUT_SFX, row.names = FALSE); cat(sprintf("Saved: %s\n", OUT_SFX))

# ── Part A: Exteroception vs Sensory-Motor formal contrast ─────────────────────
cat("\n", SEP, "\nPART A — Exteroception vs Sensory-Motor (nonself) formal contrast\n",
    SEP, "\n", sep = "")

m_full <- tryCatch(
  suppressWarnings(
    lmer(auc ~ group * session * self_layer +
           (1 + session | subject) + (1 | roi_uid),
         data = df_raw, REML = TRUE, control = ctrl)
  ),
  error = function(e) {
    cat(sprintf("  Full RE failed: %s\nFalling back...\n", e$message))
    suppressWarnings(
      lmer(auc ~ group * session * self_layer + (1|subject) + (1|roi_uid),
           data = df_raw, REML = TRUE, control = ctrl))
  }
)

if (!is.null(m_full) && isSingular(m_full)) {
  cat("  Singular — refitting with (1|subject)+(1|roi_uid)...\n")
  m_full <- suppressWarnings(
    lmer(auc ~ group * session * self_layer + (1|subject) + (1|roi_uid),
         data = df_raw, REML = TRUE, control = ctrl))
}

conv <- m_full@optinfo$conv$lme4$messages
cat(if (!is.null(conv)) sprintf("  Convergence: %s\n", paste(conv, collapse = "; "))
    else "  Convergence: OK\n")

# self_layer factor levels: nonself(1), Cognition(2), Exteroception(3), Interoception(4)
emm_options(lmerTest.limit = nrow(df_raw), pbkrtest.limit = nrow(df_raw))
emm_by      <- emmeans(m_full, ~ self_layer | group + session)
gaps_ext    <- contrast(emm_by,
  list("Exteroception - nonself" = c(-1, 0, 1, 0)), adjust = "none")
gap_ses_ext <- contrast(gaps_ext, "consec",
                        by = c("contrast", "group"), adjust = "none")
flat_ext    <- contrast(gap_ses_ext, "consec", by = "contrast", adjust = "none")
flat_ext_df <- as.data.frame(summary(flat_ext, infer = TRUE))
if (!"lower.CL" %in% names(flat_ext_df)) {
  flat_ext_df$lower.CL <- flat_ext_df$estimate - 1.96 * flat_ext_df$SE
  flat_ext_df$upper.CL <- flat_ext_df$estimate + 1.96 * flat_ext_df$SE
}

ext_b <- flat_ext_df$estimate[1]; ext_se <- flat_ext_df$SE[1]
ext_df_val <- flat_ext_df$df[1];  ext_t <- flat_ext_df$t.ratio[1]
ext_p <- flat_ext_df$p.value[1]
ext_lo <- flat_ext_df$lower.CL[1]; ext_hi <- flat_ext_df$upper.CL[1]

cat(sprintf("\n%s\n", strrep("─", 70)))
cat("EXTEROCEPTION vs NONSELF CONTRAST\n")
cat(sprintf("Estimate: %.6f\nSE      : %.6f\nt(%6.1f): %.4f\np       : %.4g\n95%% CI  : [%.6f, %.6f]\n",
            ext_b, ext_se, ext_df_val, ext_t, ext_p, ext_lo, ext_hi))
sig_str <- if (!is.na(ext_p) && ext_p < 0.05) "significantly" else "NOT significantly"
cat(sprintf("\nInterpretation: The drug×session effect in Exteroception is %s\n",
            sig_str))
cat(sprintf("different from Sensory-Motor (β = %.5f, p = %.4g).\n", ext_b, ext_p))
cat(sprintf("%s\n", strrep("─", 70)))

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
