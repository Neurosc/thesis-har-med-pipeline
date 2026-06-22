# 03_LMM.R — Pre vs post LMM on category-averaged AUC (parcels, NoGSR; self-contained).
# Reads glasser_full_dataframe.csv (output of 01), drops the 58 low-tSNR parcels,
# AVERAGES AUC across parcels within each ROI category (per subject per session), then
# fits the LMM and reports the PRE->POST change (Session) and Group x Session.
#
# Model (one row per subject x session x category):
#   auc ~ session * group * self_layer + (1 | subject)
#   auc = mean AUC across parcels in the category; subject = random intercept
#   self_layer uses sum-to-zero contrasts (no reference level)
#   self_layer: Interoception, Exteroception, Cognition (Mental Self), Sensory-Motor
# Effects reported (via emmeans on the one combined model):
#   - pre -> post change per arm (group x layer)
#   - drug effect = (verum pre->post) - (placebo pre->post), per layer
#   - 3-way F-test: does the drug effect differ across layers?
#
# Run from repo root:  Rscript 04_statistics/scripts/parcels_NoGSR/03_LMM.R

for (pkg in c("lme4", "lmerTest", "emmeans", "moments", "dplyr", "ggplot2", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
}
suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(emmeans)
  library(moments); library(dplyr); library(ggplot2); library(patchwork)
})

# ── Paths (repo root from this script's location: works for Rscript and source()) ──
script_dir <- function() {
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa)) return(dirname(normalizePath(sub("^--file=", "", fa[1]), winslash = "/")))
  for (i in rev(seq_len(sys.nframe()))) {
    of <- get0("ofile", envir = sys.frame(i), inherits = FALSE)
    if (!is.null(of)) return(dirname(normalizePath(of, winslash = "/")))
  }
  normalizePath(getwd(), winslash = "/")
}
REPO_ROOT  <- normalizePath(file.path(script_dir(), "..", "..", ".."), winslash = "/", mustWork = FALSE)
TABLES_DIR <- file.path(REPO_ROOT, "04_statistics", "results", "parcels_NoGSR", "tables")
FIGS_DIR   <- file.path(REPO_ROOT, "04_statistics", "results", "parcels_NoGSR", "figures")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

DATA_CSV    <- file.path(TABLES_DIR, "glasser_full_dataframe.csv")
TSNR_TSV    <- file.path(REPO_ROOT, "excluded_rois_low_tsnr.tsv")
OUT_FE      <- file.path(TABLES_DIR, "lmm_fixed_effects.csv")
OUT_PREPOST <- file.path(TABLES_DIR, "lmm_prepost_by_group_category.csv")
OUT_GXS     <- file.path(TABLES_DIR, "lmm_group_x_session.csv")
OUT_DIAG    <- file.path(FIGS_DIR,   "lmm_diagnostics.png")
if (!file.exists(DATA_CSV)) stop("Input not found — run 01_build_dataframe.jl first: ", DATA_CSV)

SEP <- strrep("=", 70)
cat(SEP, "\n03_LMM.R — parcels_NoGSR (pre vs post)\n", SEP, "\n\n", sep = "")

# ── Shared helpers ───────────────────────────────────────────────────────────────
diag_theme <- theme_minimal(base_family = "serif") +
  theme(panel.background = element_rect(fill = "white", color = NA),
        plot.background  = element_rect(fill = "white", color = NA),
        plot.title = element_text(face = "plain", size = 11, hjust = 0.5))

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5),
                    check.conv.grad = .makeCC("warning", tol = 0.002))

try_fit <- function(formula, label, data) {
  tryCatch(
    withCallingHandlers(
      lmer(formula, data = data, REML = TRUE, control = ctrl),
      warning = function(w) {
        cat(sprintf("  [%s] warning: %s\n", label, conditionMessage(w)))
        invokeRestart("muffleWarning")
      }),
    error = function(e) { cat(sprintf("  [%s] ERROR: %s\n", label, e$message)); NULL })
}

save_diag_fig <- function(path, resids, fitted_v) {
  qq_n   <- min(5000L, length(resids))
  qq_smp <- sort(sample(resids, qq_n)); qq_teo <- qnorm(ppoints(qq_n))
  rvf_i  <- sample(length(resids), min(5000L, length(resids)))
  p_qq <- ggplot(data.frame(theoretical = qq_teo, sample = qq_smp),
                 aes(x = theoretical, y = sample)) +
    geom_point(size = 0.8, alpha = 0.35, color = "steelblue") +
    geom_abline(color = "red", linetype = "dashed") +
    labs(title = "QQ Plot", x = "Theoretical", y = "Sample") + diag_theme
  p_rvf <- ggplot(data.frame(fitted = fitted_v[rvf_i], resid = resids[rvf_i]),
                  aes(x = fitted, y = resid)) +
    geom_point(size = 0.8, alpha = 0.25, color = "steelblue") +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    labs(title = "Resid vs Fitted", x = "Fitted", y = "Residuals") + diag_theme
  p_hist <- ggplot(data.frame(resid = resids), aes(x = resid)) +
    geom_histogram(bins = 80, fill = "steelblue", color = "white", linewidth = 0.2) +
    labs(title = "Histogram", x = "Residual", y = "Count") + diag_theme
  ggsave(path, p_qq | p_rvf | p_hist, width = 12, height = 4, dpi = 300, bg = "white")
  cat(sprintf("Saved: %s\n", path))
}

# ── Part 1: Load, drop low-tSNR parcels, average to subject x session x category ──
cat(SEP, "\nPART 1 — Load and prepare\n", SEP, "\n", sep = "")
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded glasser_full_dataframe: %d rows\n", nrow(df)))

excl <- read.delim(TSNR_TSV, stringsAsFactors = FALSE)$roi_id
n0 <- nrow(df); df <- df[!(df$roi_pos_id %in% excl), ]
cat(sprintf("Dropped %d rows from %d low-tSNR parcels (%d -> %d)\n",
            n0 - nrow(df), length(excl), n0, nrow(df)))

df$group <- df$drug_group

# average AUC across parcels within each ROI category, per subject x session
df <- as.data.frame(
  df %>% group_by(subject, session, group, self_layer) %>%
    summarise(auc = mean(auc, na.rm = TRUE), n_parcels = dplyr::n(), .groups = "drop"))
cat(sprintf("Averaged across parcels -> %d rows (subject x session x category)\n", nrow(df)))

LAYER_LEVELS  <- c("nonself", "Interoception", "Exteroception", "Cognition")
df$self_layer <- factor(df$self_layer, levels = LAYER_LEVELS)
df$group      <- relevel(factor(df$group),   ref = "placebo")
df$session    <- relevel(factor(df$session), ref = "ses-01")
df$subject    <- factor(df$subject)

cat(sprintf("Subjects: %d  |  rows: %d  |  parcels averaged per cell: %d-%d\n",
            nlevels(df$subject), nrow(df), min(df$n_parcels), max(df$n_parcels)))

emm_options(lmerTest.limit = nrow(df), pbkrtest.limit = nrow(df), lmer.df = "satterthwaite")

# ── Part 2: Fit the combined model ────────────────────────────────────────────────
# self_layer uses sum-to-zero contrasts (no reference level; each layer vs the average).
cat("\n", SEP, "\nPART 2 — Combined model: auc ~ session * group * self_layer + (1 | subject)\n", SEP, "\n", sep = "")
contrasts(df$self_layer) <- contr.sum(nlevels(df$self_layer))
m <- try_fit(auc ~ session * group * self_layer + (1 | subject), "combined", df)
if (is.null(m)) stop("Model fit failed.")
if (isSingular(m)) cat("NOTE: fit is singular.\n") else cat("Convergence: OK\n")

fe <- coef(summary(m))
write.csv(data.frame(term = rownames(fe), estimate = fe[, "Estimate"], SE = fe[, "Std. Error"],
                     df = fe[, "df"], t_value = fe[, "t value"], p_value = fe[, "Pr(>|t|)"],
                     row.names = NULL), OUT_FE, row.names = FALSE)

# ── Part 3: pre->post change and drug effect (emmeans) ────────────────────────────
cat("\n", SEP, "\nPART 3 — Pre->post change and drug effect (emmeans)\n", SEP, "\n", sep = "")
emm <- emmeans(m, ~ session | group * self_layer)

# pre -> post change within each arm (group x layer)
chg    <- contrast(emm, "revpairwise", by = c("group", "self_layer"), adjust = "none")
chg_df <- as.data.frame(summary(chg, infer = TRUE))
write.csv(chg_df, OUT_PREPOST, row.names = FALSE)

# drug effect = difference of those changes (verum vs placebo), per layer
drug_df  <- as.data.frame(summary(
  contrast(chg, "revpairwise", by = "self_layer", adjust = "none"), infer = TRUE))
drug_out <- data.frame(self_layer = drug_df$self_layer, estimate = drug_df$estimate, SE = drug_df$SE,
                       df = drug_df$df, t = drug_df$t.ratio, p = drug_df$p.value, stringsAsFactors = FALSE)
cat("\nDrug effect (verum vs placebo difference in the pre->post change) per layer:\n")
print(drug_out, row.names = FALSE, digits = 3)
write.csv(drug_out, OUT_GXS, row.names = FALSE)
cat(sprintf("  saved: %s (drug effect) + %s (pre-post per arm)\n", basename(OUT_GXS), basename(OUT_PREPOST)))

# omnibus: does the drug effect differ across layers? (3-way interaction, Type III)
av <- as.data.frame(anova(m)); av$term <- rownames(av); rownames(av) <- NULL
OUT_OMNI <- file.path(TABLES_DIR, "lmm_interaction_test.csv")
write.csv(av, OUT_OMNI, row.names = FALSE)
key <- av[grepl("session", av$term) & grepl("group", av$term) & grepl("self_layer", av$term), ]
cat("\nDoes the drug effect differ across layers? (Type III F-test)\n")
if (nrow(key)) cat(sprintf("  %s:  F(%.0f, %.1f) = %.3f,  p = %.4g\n",
                           key$term[1], key[["NumDF"]][1], key[["DenDF"]][1],
                           key[["F value"]][1], key[["Pr(>F)"]][1]))
cat(sprintf("  saved: %s\n", basename(OUT_OMNI)))

# ── Part 4: Residual diagnostics ─────────────────────────────────────────────────
cat("\n", SEP, "\nPART 4 — Residual diagnostics\n", SEP, "\n", sep = "")
resids <- resid(m); fitted_ <- fitted(m)
sk <- skewness(resids); ku <- kurtosis(resids)
cat(sprintf("Skewness: %.4f   Kurtosis: %.4f  (excess %.4f)\n", sk, ku, ku - 3))
n_sw <- min(5000L, length(resids)); set.seed(42)
sw <- shapiro.test(resids[sample(length(resids), n_sw)])
cat(sprintf("Shapiro-Wilk (n=%d): W=%.5f  p=%.4e\n", n_sw, sw$statistic, sw$p.value))
save_diag_fig(OUT_DIAG, resids, fitted_)

cat("\n", SEP, "\nSUMMARY\n", SEP, "\n", sep = "")
cat("Drug effect per layer (verum vs placebo, in pre->post)  -> lmm_group_x_session.csv\n")
cat("Pre-post change per arm                                 -> lmm_prepost_by_group_category.csv\n")
cat("Drug effect differs across layers? (3-way F-test)       -> lmm_interaction_test.csv\n")
cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
