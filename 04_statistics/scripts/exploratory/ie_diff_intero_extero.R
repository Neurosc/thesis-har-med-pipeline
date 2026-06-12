# exploratory/ie_diff_intero_extero.R
#
# Interoception − Exteroception AUC difference: drug × session LMM
# No global covariate (Cognition is a self-layer; nonself option deferred).
#
# self_layer values in model_ready.csv:
#   "nonself"       → atlas_src = nonself_glasser
#   "Interoception" → atlas_src = self_glasser
#   "Exteroception" → atlas_src = self_glasser
#   "Cognition"     → atlas_src = self_glasser
# group levels   : "placebo" (ref), "verum"
# session levels : "ses-01" (ref = pre), "ses-02" (= post)
# → group:session interaction = (verum post−pre) − (placebo post−pre)
#
# Run from repo root:
#   Rscript 04_statistics/scripts/exploratory/ie_diff_intero_extero.R

suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(emmeans)
  library(dplyr)
  library(RcppTOML)
})

# ── Paths ───────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
  REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", "..", ".."))
} else {
  REPO_ROOT  <- normalizePath(".")
}

cfg              <- parseTOML(file.path(REPO_ROOT, "config.toml"))
ATLAS_METHOD     <- cfg$active$atlas_method
DENOISING_METHOD <- cfg$active$denoising_method
TAG              <- paste0(ATLAS_METHOD, "_", DENOISING_METHOD)

TABLES_DIR <- file.path(REPO_ROOT, "04_statistics", "results", TAG, "tables")
DATA_CSV   <- file.path(TABLES_DIR, "model_ready.csv")
OUT_CSV    <- file.path(TABLES_DIR, "ie_diff_intero_extero_models.csv")

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\nie_diff_intero_extero.R — config: ", TAG, "\n", SEP, "\n\n", sep = "")

if (!file.exists(DATA_CSV))
  stop("Input not found — run 01_build_dataframe.jl + 02_lmm.R first: ", DATA_CSV)

if (ATLAS_METHOD != "parcels")
  stop("This script requires atlas_method = 'parcels' (Glasser self-layer assignment needed).")

# ── Included subjects (canonical list from subject_filter logic) ────────────────
EXCLUDED <- c("sub-06", "sub-08", "sub-12", "sub-26", "sub-36")

# ── Load & validate ─────────────────────────────────────────────────────────────
df_raw <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
df_raw <- df_raw[!df_raw$subject %in% EXCLUDED, ]
cat(sprintf("Loaded: %d rows, %d subjects\n\n",
            nrow(df_raw), length(unique(df_raw$subject))))

# ── Per subject × session means ──────────────────────────────────────────────────
subj_ses <- df_raw %>%
  group_by(subject, session, group) %>%
  summarise(
    intero_mean = mean(auc[self_layer == "Interoception"], na.rm = TRUE),
    extero_mean = mean(auc[self_layer == "Exteroception"], na.rm = TRUE),
    n_intero    = sum(!is.na(auc[self_layer == "Interoception"])),
    n_extero    = sum(!is.na(auc[self_layer == "Exteroception"])),
    .groups = "drop"
  ) %>%
  mutate(IE_diff = intero_mean - extero_mean)

# ── Global reference check ───────────────────────────────────────────────────────
remaining <- setdiff(unique(df_raw$self_layer),
                     c("Interoception", "Exteroception", "nonself"))
cat("Categories feeding a potential global covariate (excl. Intero, Extero, nonself):\n")
cat(sprintf("  %s\n", paste(remaining, collapse = ", ")))
cat("→ Only Cognition remains (a self-layer); global covariate deferred.\n\n")

missing_subj <- subj_ses[is.na(subj_ses$IE_diff), ]
if (nrow(missing_subj) > 0) {
  cat("WARNING — NA IE_diff for:\n")
  print(missing_subj[, c("subject", "session", "n_intero", "n_extero")])
}

cat(sprintf("IE_diff summary (n=%d obs):\n", nrow(subj_ses)))
print(summary(subj_ses$IE_diff))
cat("\n")

# ── Factor levels ────────────────────────────────────────────────────────────────
subj_ses$group   <- relevel(factor(subj_ses$group),   ref = "placebo")
subj_ses$session <- relevel(factor(subj_ses$session), ref = "ses-01")
subj_ses$subject <- factor(subj_ses$subject)

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5),
                    check.conv.grad = .makeCC("warning", tol = 0.002))

# ── m1: IE_diff ~ group * session + (1 | subject) ───────────────────────────────
cat(SEP, "\nm1: IE_diff ~ group * session + (1 | subject)\n", SEP, "\n", sep = "")

m1 <- tryCatch(
  suppressWarnings(
    lmer(IE_diff ~ group * session + (1 | subject),
         data = subj_ses, REML = TRUE, control = ctrl)
  ),
  error = function(e) { cat("ERROR:", e$message, "\n"); NULL }
)

if (is.null(m1)) stop("m1 failed to fit.")

conv <- m1@optinfo$conv$lme4$messages
cat(if (!is.null(conv)) sprintf("Convergence: %s\n", paste(conv, collapse = "; "))
    else "Convergence: OK\n")
if (isSingular(m1)) cat("WARNING: singular fit\n")

fe1 <- coef(summary(m1))
print(fe1)

# ── emmeans simple effects ────────────────────────────────────────────────────────
cat("\nSimple effects (post − pre within each group):\n")
emm_options(lmerTest.limit = nrow(subj_ses), pbkrtest.limit = nrow(subj_ses))
emm1    <- emmeans(m1, ~ session | group)
sfx1    <- contrast(emm1, list("post-pre" = c(-1, 1)), adjust = "none")
sfx1_df <- as.data.frame(summary(sfx1, infer = TRUE))
print(sfx1_df[, c("group", "contrast", "estimate", "SE", "df", "t.ratio", "p.value")])

# ── Tidy output table ────────────────────────────────────────────────────────────
interaction_row <- function(model_name, fe) {
  rn <- "groupverum:sessionses-02"
  data.frame(model   = model_name,
             term    = "groupverum:sessionses-02",
             estimate = fe[rn, "Estimate"],
             SE       = fe[rn, "Std. Error"],
             df       = fe[rn, "df"],
             t        = fe[rn, "t value"],
             p        = fe[rn, "Pr(>|t|)"],
             stringsAsFactors = FALSE)
}

sfx_rows <- function(model_name, sfx_df) {
  data.frame(model    = model_name,
             term     = paste0("post-pre_", sfx_df$group),
             estimate = sfx_df$estimate,
             SE       = sfx_df$SE,
             df       = sfx_df$df,
             t        = sfx_df$t.ratio,
             p        = sfx_df$p.value,
             stringsAsFactors = FALSE)
}

out_tbl <- rbind(
  interaction_row("m1", fe1),
  sfx_rows("m1_simple", sfx1_df)
)
rownames(out_tbl) <- NULL

write.csv(out_tbl, OUT_CSV, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", OUT_CSV))

# ── Console summary ───────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSUMMARY — m1: IE_diff ~ group * session + (1|subject)\n", SEP, "\n",
    sep = "")
cat(sprintf("Drug × Session (groupverum:sessionses-02):\n"))
cat(sprintf("  β = %.6f  SE = %.6f  df = %.1f  t = %.4f  p = %.4g\n",
            out_tbl$estimate[1], out_tbl$SE[1], out_tbl$df[1],
            out_tbl$t[1], out_tbl$p[1]))
cat(sprintf("  → %s\n", if (out_tbl$p[1] < 0.05) "SIGNIFICANT *" else "not significant"))
cat("\nSimple effects:\n")
for (i in which(out_tbl$model == "m1_simple")) {
  cat(sprintf("  %-22s β = %.6f  SE = %.6f  p = %.4g\n",
              out_tbl$term[i], out_tbl$estimate[i], out_tbl$SE[i], out_tbl$p[i]))
}
cat("\nNote: global covariate not included — Cognition is a self-layer, not a neutral\n")
cat("signal. Nonself-as-baseline or GSR-based global option to be added separately.\n")
cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
