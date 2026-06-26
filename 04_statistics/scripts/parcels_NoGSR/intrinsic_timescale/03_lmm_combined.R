# 03_lmm_combined.R — parcel-level LMM on AUC (parcels_NoGSR, self-contained)
#
# Model: auc ~ group * session * category + (1 + session | subject) + (1 | subject:parcel)
#   group ref = placebo | session ref = ses-01 | category: sum-to-zero contrasts
#   58 low-tSNR parcels dropped before fitting
#
# Outputs (int_tables/):
#   combined_lmm_fixed_effects.csv, combined_lmm_prepost_by_arm.csv,
#   combined_lmm_drug_effect.csv, combined_lmm_3way_ftest.csv,
#   combined_lmm_somato_internal_gap.csv
#
# Run from repo root:
#   Rscript 04_statistics/scripts/parcels_NoGSR/intrinsic_timescale/03_lmm_combined.R

# ── Packages ──────────────────────────────────────────────────────────────────
for (pkg in c("lme4", "lmerTest", "emmeans", "pbkrtest", "moments", "dplyr", "ggplot2", "patchwork")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
}
suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(emmeans); library(pbkrtest)
  library(moments); library(dplyr); library(ggplot2); library(patchwork)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
# Works both via Rscript (--file=) and via source() in RStudio (ofile in sys.frames).
# Returns NULL if neither is found, in which case getwd() is used as REPO_ROOT directly.
.script_dir <- function() {
  fa <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
  if (length(fa)) return(dirname(normalizePath(sub("^--file=", "", fa[1]), winslash = "/")))
  for (i in rev(seq_len(sys.nframe()))) {
    of <- get0("ofile", envir = sys.frame(i), inherits = FALSE)
    if (!is.null(of)) return(dirname(normalizePath(of, winslash = "/")))
  }
  NULL
}
.sd       <- .script_dir()
REPO_ROOT <- if (!is.null(.sd)) {
  normalizePath(file.path(.sd, "..", "..", "..", ".."), winslash = "/", mustWork = FALSE)
} else {
  normalizePath(getwd(), winslash = "/")  # assume already at repo root
}
TABLES_DIR <- file.path(REPO_ROOT, "04_statistics", "results", "parcels_NoGSR", "int_tables")
FIGS_DIR   <- file.path(REPO_ROOT, "04_statistics", "results", "parcels_NoGSR", "int_figures")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

DATA_CSV    <- file.path(TABLES_DIR, "glasser_full_dataframe.csv")
TSNR_TSV    <- file.path(REPO_ROOT,  "excluded_rois_low_tsnr.tsv")
OUT_FE      <- file.path(TABLES_DIR, "combined_lmm_fixed_effects.csv")
OUT_PREPOST <- file.path(TABLES_DIR, "combined_lmm_prepost_by_arm.csv")
OUT_DRUG    <- file.path(TABLES_DIR, "combined_lmm_drug_effect.csv")
OUT_OMNI    <- file.path(TABLES_DIR, "combined_lmm_3way_ftest.csv")
OUT_GAP     <- file.path(TABLES_DIR, "combined_lmm_somato_internal_gap.csv")
OUT_DIAG    <- file.path(FIGS_DIR,   "INT_QC_combined_lmm_diagnostics.png")
OUT_FIG     <- file.path(FIGS_DIR,   "INT_intero_somato_trajectories.png")

if (!file.exists(DATA_CSV))
  stop("Input not found — run 01_build_dataframe.jl first: ", DATA_CSV)

# ── Helpers ───────────────────────────────────────────────────────────────────
SEP     <- strrep("=", 70)
section <- function(x) cat("\n", SEP, "\n", x, "\n", SEP, "\n\n", sep = "")

diag_theme <- theme_minimal(base_family = "serif") +
  theme(panel.background = element_rect(fill = "white", color = NA),
        plot.background  = element_rect(fill = "white", color = NA),
        plot.title       = element_text(face = "plain", size = 11, hjust = 0.5))

save_diag_fig <- function(path, resids, fitted_v) {
  set.seed(42)
  n    <- min(5000L, length(resids))
  samp <- sample(length(resids), n)
  p_qq <- ggplot(data.frame(theoretical = qnorm(ppoints(n)), sample = sort(resids[samp])),
                 aes(x = theoretical, y = sample)) +
    geom_point(size = 0.8, alpha = 0.35, color = "steelblue") +
    geom_abline(color = "red", linetype = "dashed") +
    labs(title = "QQ Plot", x = "Theoretical", y = "Sample") + diag_theme
  p_rvf <- ggplot(data.frame(fitted = fitted_v[samp], resid = resids[samp]),
                  aes(x = fitted, y = resid)) +
    geom_point(size = 0.8, alpha = 0.25, color = "steelblue") +
    geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
    labs(title = "Resid vs Fitted", x = "Fitted", y = "Residuals") + diag_theme
  p_hist <- ggplot(data.frame(resid = resids), aes(x = resid)) +
    geom_histogram(bins = 80, fill = "steelblue", color = "white", linewidth = 0.2) +
    labs(title = "Residual distribution", x = "Residual", y = "Count") + diag_theme
  ggsave(path, p_qq | p_rvf | p_hist, width = 12, height = 4, dpi = 300, bg = "white")
  cat("Saved:", path, "\n")
}

# ── 1. Load and prepare ───────────────────────────────────────────────────────
section("1. Load and prepare")
df   <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
excl <- read.delim(TSNR_TSV, stringsAsFactors = FALSE)$roi_id
n0   <- nrow(df)
df   <- df[!(df$roi_pos_id %in% excl) & is.finite(df$auc), ]
cat(sprintf("Rows after tSNR exclusion + finite AUC: %d -> %d\n", n0, nrow(df)))

df$category <- factor(df$self_layer,
                      levels = c("nonself", "Interoception", "Exteroception", "Cognition"))
df$parcel   <- factor(df$roi_pos_id)
df$group    <- relevel(factor(df$drug_group), ref = "placebo")
df$session  <- relevel(factor(df$session),    ref = "ses-01")
df$subject  <- factor(df$subject)
cat(sprintf("Subjects: %d  Parcels: %d\n", nlevels(df$subject), nlevels(df$parcel)))

emm_options(lmerTest.limit = nrow(df), pbkrtest.limit = nrow(df), lmer.df = "kenward-roger")

# ── 2. Model fit ──────────────────────────────────────────────────────────────
section("2. Model fit")

# Sum-to-zero contrasts for category: each category effect is a deviation from
# the grand mean rather than a comparison to a single reference level.
contrasts(df$category) <- contr.sum(nlevels(df$category))

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 4e5),
                    check.conv.grad = .makeCC("warning", tol = 0.002))
m <- lmer(auc ~ group * session * category + (1 + session | subject) + (1 | subject:parcel),
          data = df, REML = TRUE, control = ctrl)

cat(if (isSingular(m)) "NOTE: singular fit.\n" else "Convergence: OK\n")
cat("\nRandom effects:\n"); print(VarCorr(m))

fe <- coef(summary(m))
write.csv(data.frame(term = rownames(fe), estimate = fe[, "Estimate"], SE = fe[, "Std. Error"],
                     df = fe[, "df"], t_value = fe[, "t value"], p_value = fe[, "Pr(>|t|)"],
                     row.names = NULL), OUT_FE, row.names = FALSE)
cat("Fixed effects ->", basename(OUT_FE), "\n")

# ── 3. Pre→post change and drug effect (emmeans) ──────────────────────────────
section("3. Pre→post change and drug effect")

# Step 1: pre→post change within each group × category cell
emm <- emmeans(m, ~ session | group * category)
chg <- contrast(emm, "revpairwise", by = c("group", "category"), adjust = "none")
write.csv(as.data.frame(summary(chg, infer = TRUE)), OUT_PREPOST, row.names = FALSE)
cat("Pre→post by arm ->", basename(OUT_PREPOST), "\n")

# Step 2: drug effect = (verum post-pre) − (placebo post-pre), per category
drug_out <- as.data.frame(summary(
    contrast(chg, "revpairwise", by = "category", adjust = "none"), infer = TRUE)) %>%
  select(category, estimate, SE, df, t = t.ratio, p = p.value) %>%
  mutate(p_holm = p.adjust(p, method = "holm"),   # Holm FWER, family = 4 categories
         p_fdr  = p.adjust(p, method = "BH"))      # BH FDR,   family = 4 categories
cat("\nDrug effect per category:\n")
print(drug_out, row.names = FALSE, digits = 3)
write.csv(drug_out, OUT_DRUG, row.names = FALSE)
cat("Drug effect ->", basename(OUT_DRUG), "\n")

# Step 3: omnibus F-test — does the drug effect size differ across categories?
av      <- as.data.frame(anova(m))
av$term <- rownames(av); rownames(av) <- NULL
key     <- av[grepl("session", av$term) & grepl("group", av$term) & grepl("category", av$term), ]
if (nrow(key))
  cat(sprintf("\n3-way F-test: %s  F(%.0f, %.1f) = %.3f  p = %.4g\n",
              key$term[1], key$NumDF[1], key$DenDF[1], key[["F value"]][1], key[["Pr(>F)"]][1]))
write.csv(av, OUT_OMNI, row.names = FALSE)
cat("3-way F-test ->", basename(OUT_OMNI), "\n")

# ── 4. Somatomotor − Interoceptive Self gap (3-way contrast, KR df) ───────────
section("4. Somatomotor − Interoceptive Self gap")
# Direction: (nonself − Interoception) × (post − pre) × (verum − placebo)
# Positive = the somatomotor–interoceptive AUC gap widens more under DMT than placebo.
emm_gap <- emmeans(m, ~ category * session * group,
                   at = list(category = c("Interoception", "nonself")))
gap_df  <- as.data.frame(summary(
  contrast(emm_gap, interaction = c("revpairwise", "revpairwise", "revpairwise")),
  infer = TRUE))
print(gap_df, row.names = FALSE, digits = 3)
write.csv(gap_df, OUT_GAP, row.names = FALSE)
cat("Gap contrast ->", basename(OUT_GAP), "\n")

# ── 5. Residual diagnostics ───────────────────────────────────────────────────
section("5. Residual diagnostics")
resids <- resid(m)
cat(sprintf("Skewness: %.3f  Excess kurtosis: %.3f\n",
            skewness(resids), kurtosis(resids) - 3))
set.seed(42)
n_sw <- min(5000L, length(resids))
sw   <- shapiro.test(resids[sample(length(resids), n_sw)])
cat(sprintf("Shapiro-Wilk (n=%d): W = %.4f  p = %.3e\n", n_sw, sw$statistic, sw$p.value))
save_diag_fig(OUT_DIAG, resids, fitted(m))

# ── 6. Figure: Interoceptive Self vs Sensory-Motor trajectories ───────────────
section("6. Trajectory figure: Interoceptive Self vs Sensory-Motor")

fig_df <- df %>%
  filter(category %in% c("Interoception", "nonself")) %>%
  group_by(subject, group, session, category) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    session_num = ifelse(session == "ses-01", 1, 2),
    group_label = factor(ifelse(group == "placebo", "Placebo", "DMT (Verum)"),
                         levels = c("Placebo", "DMT (Verum)")),
    cat_label   = factor(ifelse(category == "Interoception", "Interoceptive Self", "Sensory-Motor"),
                         levels = c("Interoceptive Self", "Sensory-Motor"))
  )

gmeans <- fig_df %>%
  group_by(group_label, session_num, cat_label) %>%
  summarise(m = mean(mean_auc), se = sd(mean_auc) / sqrt(n()), .groups = "drop")

CAT_COLORS <- c("Interoceptive Self" = "#4682B4", "Sensory-Motor" = "#2E8B8B")
fig_theme  <- theme_minimal(base_size = 11, base_family = "serif") +
  theme(panel.background   = element_rect(fill = "white", color = NA),
        plot.background    = element_rect(fill = "white", color = NA),
        panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(color = "#e8e8e8", linewidth = 0.4),
        strip.text         = element_text(size = 11, face = "plain"),
        legend.position    = "bottom", legend.title = element_blank())

p_traj <- ggplot() +
  geom_line(data = fig_df,
            aes(x = session_num, y = mean_auc, group = interaction(subject, cat_label),
                color = cat_label),
            alpha = 0.25, linewidth = 0.4) +
  geom_ribbon(data = gmeans,
              aes(x = session_num, ymin = m - se, ymax = m + se,
                  fill = cat_label, group = cat_label),
              alpha = 0.20) +
  geom_line(data = gmeans,
            aes(x = session_num, y = m, color = cat_label, group = cat_label),
            linewidth = 1.2) +
  geom_point(data = gmeans, aes(x = session_num, y = m, color = cat_label), size = 3) +
  scale_color_manual(values = CAT_COLORS) +
  scale_fill_manual(values  = CAT_COLORS) +
  scale_x_continuous(breaks = c(1, 2), labels = c("Pre", "Post"),
                     expand = expansion(add = 0.3)) +
  facet_wrap(~ group_label) +
  labs(title = "Interoceptive Self vs Sensory-Motor AUC — pre→post by group",
       x = NULL, y = "Mean AUC (s)") +
  fig_theme

ggsave(OUT_FIG, p_traj, width = 8, height = 5, dpi = 300, bg = "white")
cat("Saved:", OUT_FIG, "\n")

# ── Summary ───────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSUMMARY\n", SEP, "\n", sep = "")
for (f in c(OUT_FE, OUT_PREPOST, OUT_DRUG, OUT_OMNI, OUT_GAP, OUT_DIAG, OUT_FIG))
  cat(" ", basename(f), "\n")
cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
