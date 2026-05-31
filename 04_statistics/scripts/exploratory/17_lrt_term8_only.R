# 17_lrt_term8_only.R
# Targeted LRT: isolate term 8 (group:session:G1_z) contribution
#
# Full    : auc ~ group * session * G1_z + (1|subject) + (1|roi_pos_id)
# Reduced : auc ~ group + session + G1_z + group:session + group:G1_z +
#                 session:G1_z + (1|subject) + (1|roi_pos_id)
#   => drops ONLY the 3-way interaction (term 8), keeps group:G1_z (term 6)
#   => 1 df test, isolating term 8 alone
#
# Context: original LRT in Step 12 dropped both term 6 AND term 8 (2 df),
# chi2(2) = 50.08, p = 1.33e-11.  This script separates that signal.
#
# Run from repo root:
#   Rscript 04_statistics/scripts/17_lrt_term8_only.R

suppressPackageStartupMessages({
  library(lme4)
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

DATA_CSV <- file.path(REPO_ROOT, "04_statistics", "results", "nonself_g1_model_ready.csv")

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n17_lrt_term8_only.R — Targeted LRT for term 8 (group:session:G1_z)\n",
    SEP, "\n\n", sep = "")

# ── Load data ──────────────────────────────────────────────────────────────────
cat("Loading data...\n")
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Data: %d rows x %d cols\n\n", nrow(df), ncol(df)))
stopifnot(nrow(df) == 18042)

df$group      <- relevel(factor(df$group),   ref = "placebo")
df$session    <- relevel(factor(df$session), ref = "ses-01")
df$subject    <- factor(df$subject)
df$roi_pos_id <- factor(df$roi_pos_id)

# ── Model formulas ─────────────────────────────────────────────────────────────
FORMULA_FULL <- auc ~ group * session * G1_z + (1 | subject) + (1 | roi_pos_id)
FORMULA_RED  <- auc ~ group + session + G1_z +
                      group:session + group:G1_z + session:G1_z +
                      (1 | subject) + (1 | roi_pos_id)

cat("Full model formula:\n ")
cat(deparse(FORMULA_FULL), "\n\n")
cat("Reduced model formula (drops ONLY group:session:G1_z, keeps group:G1_z):\n ")
cat(paste(deparse(FORMULA_RED), collapse = " "), "\n\n")

# ── Fit both models with ML (required for LRT) ─────────────────────────────────
ctrl_ml <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

cat("Fitting full model (ML)...\n")
m_full <- lmer(FORMULA_FULL, data = df, REML = FALSE, control = ctrl_ml)

cat("Fitting reduced model (ML)...\n")
m_red  <- lmer(FORMULA_RED,  data = df, REML = FALSE, control = ctrl_ml)

# ── LRT ───────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nLikelihood Ratio Test\n", SEP, "\n", sep = "")
lrt <- anova(m_red, m_full)
print(lrt)

chi2   <- lrt$Chisq[2]
df_lrt <- lrt$Df[2]
pval   <- lrt$`Pr(>Chisq)`[2]

cat(sprintf("\nLRT result: chi2(%d) = %.4f, p = %.4g\n", df_lrt, chi2, pval))

# ── One-line interpretation ────────────────────────────────────────────────────
sig_str <- if (pval < 0.05) "IS significant" else "is NOT significant"
cat(sprintf(
  "\nDropping only group:session:G1_z: chi2(%d) = %.4f, p = %.4g\n=> term 8 %s on its own, independent of term 6.\n",
  df_lrt, chi2, pval, sig_str
))

# ── Comparison with original 2-df LRT ─────────────────────────────────────────
CHI2_ORIG <- 50.08
PVAL_ORIG <- 1.33e-11

cat(sprintf("\n%s\nComparison with original 2-df LRT (Step 12 / script 16)\n%s\n", SEP, SEP))
cat(sprintf("  Original  (drops term 6 + term 8, 2 df): chi2(2) = %.4f, p = %.4g\n",
            CHI2_ORIG, PVAL_ORIG))
cat(sprintf("  This test (drops term 8 only,    1 df): chi2(1) = %.4f, p = %.4g\n",
            chi2, pval))
cat(sprintf("  Difference (attributable to term 6)   : chi2    = %.4f\n",
            CHI2_ORIG - chi2))
cat(sprintf(
  "\nInterpretation: term 8 alone accounts for %.1f%% of the original 2-df chi2 signal.\n",
  100 * chi2 / CHI2_ORIG
))

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
