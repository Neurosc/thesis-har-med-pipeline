# 02_lmm_category_avg.R — Pre vs post LMM on AUC (parcels, NoGSR; self-contained).
# Reads glasser_full_dataframe.csv (output of 01), drops the 58 low-tSNR parcels,
# and fits the LMM at the parcel level (no averaging).
#
# Model (one row per subject x session x parcel):
#   AUC ~ Group * Time * Category + (1 | Subject) + (1 | Parcel)
#   Subject = random intercept; Parcel = random intercept
#   Category uses sum-to-zero contrasts (no reference level)
#   Category: Interoception, Exteroception, Cognition (Mental Self), Sensory-Motor
# Effects reported (via emmeans on the one combined model):
#   - pre -> post change per arm (Group x Category)
#   - drug effect = (verum pre->post) - (placebo pre->post), per Category
#   - 3-way F-test: does the drug effect differ across categories?
#
# Run from repo root:  Rscript 04_statistics/scripts/parcels_NoGSR/intrinsic_timescale/02_lmm_category_avg.R

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
REPO_ROOT  <- normalizePath(file.path(script_dir(), "..", "..", "..", ".."), winslash = "/", mustWork = FALSE)
TABLES_DIR <- file.path(REPO_ROOT, "04_statistics", "results", "parcels_NoGSR", "int_tables")
FIGS_DIR   <- file.path(REPO_ROOT, "04_statistics", "results", "parcels_NoGSR", "int_figures")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

DATA_CSV    <- file.path(TABLES_DIR, "glasser_full_dataframe.csv")
TSNR_TSV    <- file.path(REPO_ROOT, "excluded_rois_low_tsnr.tsv")
OUT_FE      <- file.path(TABLES_DIR, "catavg_fixed_effects.csv")
OUT_PREPOST <- file.path(TABLES_DIR, "catavg_prepost_by_arm.csv")
OUT_GXS     <- file.path(TABLES_DIR, "catavg_drug_effect.csv")
OUT_DIAG    <- file.path(FIGS_DIR,   "INT_QC_lmm_diagnostics.png")
if (!file.exists(DATA_CSV)) stop("Input not found — run 01_build_dataframe.jl first: ", DATA_CSV)

SEP <- strrep("=", 70)
cat(SEP, "\n02_lmm_category_avg.R — parcels_NoGSR (pre vs post)\n", SEP, "\n\n", sep = "")

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

# ── Part 1: Load, drop low-tSNR parcels, rename columns ─────────────────────────
cat(SEP, "\nPART 1 — Load and prepare\n", SEP, "\n", sep = "")
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded glasser_full_dataframe: %d rows\n", nrow(df)))

excl <- read.delim(TSNR_TSV, stringsAsFactors = FALSE)$roi_id
n0 <- nrow(df); df <- df[!(df$roi_pos_id %in% excl), ]
cat(sprintf("Dropped %d rows from %d low-tSNR parcels (%d -> %d)\n",
            n0 - nrow(df), length(excl), n0, nrow(df)))

df <- df %>% rename(AUC = auc, Time = session, Group = drug_group,
                    Category = self_layer, Subject = subject, Parcel = roi_pos_id)

LAYER_LEVELS  <- c("nonself", "Interoception", "Exteroception", "Cognition")
df$Category <- factor(df$Category, levels = LAYER_LEVELS)
df$Group    <- relevel(factor(df$Group),   ref = "placebo")
df$Time     <- relevel(factor(df$Time),    ref = "ses-01")
df$Subject  <- factor(df$Subject)
df$Parcel   <- factor(df$Parcel)

cat(sprintf("Subjects: %d  |  Parcels: %d  |  rows: %d\n",
            nlevels(df$Subject), nlevels(df$Parcel), nrow(df)))

emm_options(lmerTest.limit = nrow(df), pbkrtest.limit = nrow(df), lmer.df = "satterthwaite")

# ── Part 2: Fit the combined model ────────────────────────────────────────────────
cat("\n", SEP, "\nPART 2 — Combined model: AUC ~ Group * Time * Category + (1 | Subject) + (1 | Parcel)\n", SEP, "\n", sep = "")
contrasts(df$Category) <- contr.sum(nlevels(df$Category))
m <- try_fit(AUC ~ Group * Time * Category + (1 | Subject) + (1 | Parcel), "combined", df)
if (is.null(m)) stop("Model fit failed.")
if (isSingular(m)) cat("NOTE: fit is singular.\n") else cat("Convergence: OK\n")

fe <- coef(summary(m))
write.csv(data.frame(term = rownames(fe), estimate = fe[, "Estimate"], SE = fe[, "Std. Error"],
                     df = fe[, "df"], t_value = fe[, "t value"], p_value = fe[, "Pr(>|t|)"],
                     row.names = NULL), OUT_FE, row.names = FALSE)

# ── Part 3: pre->post change and drug effect (emmeans) ────────────────────────────
cat("\n", SEP, "\nPART 3 — Pre->post change and drug effect (emmeans)\n", SEP, "\n", sep = "")
emm <- emmeans(m, ~ Time | Group * Category)

# pre -> post change within each arm (Group x Category)
chg    <- contrast(emm, "revpairwise", by = c("Group", "Category"), adjust = "none")
chg_df <- as.data.frame(summary(chg, infer = TRUE))
write.csv(chg_df, OUT_PREPOST, row.names = FALSE)

# drug effect = difference of those changes (verum vs placebo), per Category
drug_df  <- as.data.frame(summary(
  contrast(chg, "revpairwise", by = "Category", adjust = "none"), infer = TRUE))
drug_out <- data.frame(Category = drug_df$Category, estimate = drug_df$estimate, SE = drug_df$SE,
                       df = drug_df$df, t = drug_df$t.ratio, p = drug_df$p.value, stringsAsFactors = FALSE)
# Multiple-comparison correction across the 4 per-category drug-effect contrasts (one family).
# p_holm = Holm-Bonferroni (FWER control); p_fdr = Benjamini-Hochberg (FDR). The omnibus
# Group:Time F-test (catavg_3way_interaction_Ftest.csv) is a single test, NOT in this family.
drug_out$p_holm <- p.adjust(drug_out$p, method = "holm")
drug_out$p_fdr  <- p.adjust(drug_out$p, method = "BH")
cat("\nDrug effect (verum vs placebo difference in the pre->post change) per category:\n")
cat("  p = raw | p_holm = Holm-Bonferroni (FWER) | p_fdr = Benjamini-Hochberg (FDR); family = 4 categories\n")
print(drug_out, row.names = FALSE, digits = 3)
write.csv(drug_out, OUT_GXS, row.names = FALSE)
cat(sprintf("  saved: %s (drug effect, raw + Holm + FDR) + %s (pre-post per arm)\n", basename(OUT_GXS), basename(OUT_PREPOST)))

# omnibus: does the drug effect differ across categories? (3-way interaction, Type III)
av <- as.data.frame(anova(m)); av$term <- rownames(av); rownames(av) <- NULL
OUT_OMNI <- file.path(TABLES_DIR, "catavg_3way_interaction_Ftest.csv")
write.csv(av, OUT_OMNI, row.names = FALSE)
key <- av[grepl("Time", av$term) & grepl("Group", av$term) & grepl("Category", av$term), ]
cat("\nDoes the drug effect differ across categories? (Type III F-test)\n")
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

# ── Extended diagnostics: colored by Category and Subject ─────────────────────
cat("Building extended residual diagnostics...\n")
diag_df            <- df
diag_df$resid      <- resid(m)
diag_df$fitted_val <- fitted(m)
diag_df$abs_resid  <- abs(diag_df$resid)
# QQ quantiles computed within each category so each facet has its own proper QQ line
diag_df <- diag_df %>%
  group_by(Category) %>%
  mutate(qq_theo = qnorm(rank(resid, ties.method = "average") / (n() + 1))) %>%
  ungroup()

CAT_COLORS <- c(nonself       = "#2E8B8B",
                Interoception  = "#4682B4",
                Exteroception  = "#E8963E",
                Cognition      = "#8B4682")

facet_theme <- diag_theme +
  theme(strip.text       = element_text(size = 9, face = "plain"),
        strip.background = element_rect(fill = "#f0f0f0", color = NA),
        legend.position  = "none")

# ① QQ + resid-vs-fitted, faceted by Category (4 panels each)
p_qq_cat <- ggplot(diag_df, aes(x = qq_theo, y = resid, color = Category)) +
  geom_point(size = 0.7, alpha = 0.35) +
  geom_abline(color = "black", linetype = "dashed", linewidth = 0.5) +
  scale_color_manual(values = CAT_COLORS) +
  facet_wrap(~ Category, nrow = 1, scales = "free") +
  labs(title = "QQ — by Category", x = "Theoretical quantile", y = "Residual") +
  facet_theme

p_rvf_cat <- ggplot(diag_df, aes(x = fitted_val, y = resid, color = Category)) +
  geom_point(size = 0.7, alpha = 0.35) +
  geom_hline(yintercept = 0, color = "black", linetype = "dashed", linewidth = 0.5) +
  scale_color_manual(values = CAT_COLORS) +
  facet_wrap(~ Category, nrow = 1, scales = "free") +
  labs(title = "Resid vs Fitted — by Category", x = "Fitted", y = "Residual") +
  facet_theme

ggsave(file.path(FIGS_DIR, "INT_QC_residuals_by_category.png"),
       p_qq_cat / p_rvf_cat, width = 14, height = 8, dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n", file.path(FIGS_DIR, "INT_QC_residuals_by_category.png")))

# ② Residuals per Subject — boxplot sorted by median |resid|, colored by Group
subj_order        <- names(sort(tapply(diag_df$abs_resid, diag_df$Subject, median, na.rm = TRUE)))
diag_df$Subject_f <- factor(diag_df$Subject, levels = subj_order)
p_subj <- ggplot(diag_df, aes(x = Subject_f, y = resid, fill = Group)) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray60", linewidth = 0.4) +
  geom_boxplot(outlier.size = 0.5, outlier.alpha = 0.4, linewidth = 0.35, width = 0.75) +
  scale_fill_manual(values = c(placebo = "#CD5C5C", verum = "#4682B4"), name = NULL,
                    labels = c(placebo = "Placebo", verum = "Verum")) +
  scale_x_discrete(labels = function(x) sub("sub-0?", "", x)) +
  labs(title = "Residuals by Subject (sorted by median |residual|)", x = NULL, y = "Residual") +
  diag_theme +
  theme(axis.text.x   = element_text(angle = 45, hjust = 1, size = 8),
        legend.position = "bottom", legend.text = element_text(size = 9))
ggsave(file.path(FIGS_DIR, "INT_QC_residuals_by_subject.png"),
       p_subj, width = 14, height = 5, dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n", file.path(FIGS_DIR, "INT_QC_residuals_by_subject.png")))

# ③ Top-30 extreme residuals — which parcels / subjects drive the tails?
extreme <- diag_df[order(-diag_df$abs_resid),
                   c("Subject", "Group", "Time", "Category", "Parcel", "roi_name",
                     "AUC", "fitted_val", "resid", "abs_resid")]
extreme <- head(extreme, 30); rownames(extreme) <- NULL
write.csv(extreme, file.path(TABLES_DIR, "INT_QC_extreme_residuals.csv"), row.names = FALSE)
cat(sprintf("Saved: %s\n", file.path(TABLES_DIR, "INT_QC_extreme_residuals.csv")))
cat("\nTop-30 extreme residuals (|resid| descending):\n")
print(extreme[, c("Subject", "Category", "roi_name", "resid")], row.names = FALSE, digits = 3)
cat("\nExtreme residuals by Category:\n");  print(table(extreme$Category))
cat("Extreme residuals by Subject:\n");    print(sort(table(extreme$Subject), decreasing = TRUE))

# ── Part 5 (exploratory): (nonself − Interoception) gap × Drug × Time ───────────
# Q: does the somatomotor − internal-self AUC gap change differently pre→post under DMT?
# "Internal Self" = Interoceptive Self (Interoception category only).
# Grid restricted to these two categories; pairwise = second − first, so placing
# Interoception first gives: nonself − Interoception for the Category contrast.
# Time pairwise: ses-02 − ses-01 (post − pre). Group pairwise: verum − placebo.
# Single contrast, no correction applied (exploratory).
cat("\n", SEP, "\nPART 5 (EXPLORATORY) — Somatomotor−Internal Self gap × pre→post × drug\n", SEP, "\n", sep = "")
emm_gap <- emmeans(m, ~ Time * Group * Category,
                   at = list(Category = c("Interoception", "nonself")))
gap_df  <- as.data.frame(summary(
  contrast(emm_gap, interaction = c("pairwise", "pairwise", "pairwise")),
  infer = TRUE))
cat("Gap: nonself (Somatomotor) − Interoception (Internal Self)\n")
cat("(post − pre) × (verum − placebo)\n")
cat("Positive = somatomotor−internal-self gap grows more pre→post under DMT than placebo\n\n")
print(gap_df, row.names = FALSE, digits = 3)
OUT_GAP <- file.path(TABLES_DIR, "exploratory_somato_intrinsic_gap.csv")
write.csv(gap_df, OUT_GAP, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", OUT_GAP))

cat("\n", SEP, "\nSUMMARY\n", SEP, "\n", sep = "")
cat("Drug effect per category (verum vs placebo, in pre->post) -> catavg_drug_effect.csv\n")
cat("Pre-post change per arm                                   -> catavg_prepost_by_arm.csv\n")
cat("Drug effect differs across categories? (3-way F-test)     -> catavg_3way_interaction_Ftest.csv\n")
cat("[exploratory] Somatomotor−InternalSelf gap × Drug × Time   -> exploratory_somato_intrinsic_gap.csv\n")
cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
