# 04a_lmm_glasser.R
# Step 16: Glasser-based self vs nonself LMM (no sphere confound)
#
# Combines Keskin et al. self parcels (Glasser-based, 33 unique parcels → 35
# layer-parcel combos after handling duplicates) with the 316 nonself Glasser
# parcels. Both sets are Glasser parcels, eliminating the sphere-vs-parcel
# confound from the earlier Qin-sphere analysis.
#
# Layer variable (self_layer):
#   "Interoception"  — 10 parcels
#   "Exteroception"  — 14 parcels
#   "Cognition"      — 11 parcels  (incl. 2 shared with other layers)
#   "nonself"        — 316 parcels
#
# Reference: "nonself"  (all coefficients relative to nonself)
#
# Run from repo root:
#   Rscript 04_statistics/scripts/04a_lmm_glasser.R

if (!requireNamespace("emmeans", quietly = TRUE))
  install.packages("emmeans", repos = "https://cloud.r-project.org", quiet = TRUE)

suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(emmeans)
  library(moments); library(dplyr); library(tidyr)
  library(ggplot2)
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

KESKIN_CSV  <- file.path(REPO_ROOT, "04_statistics", "results",
                          "keskin_auc_model_ready.csv")
NONSELF_CSV <- file.path(REPO_ROOT, "04_statistics", "results",
                          "nonself_g1_model_ready_316.csv")
LOOKUP_CSV  <- file.path(REPO_ROOT, "04_statistics", "results",
                          "glasser_g1_lookup.csv")

OUT_DATA    <- file.path(REPO_ROOT, "04_statistics", "results",
                          "glasser_self_nonself_model_ready.csv")
OUT_FE      <- file.path(REPO_ROOT, "04_statistics", "results",
                          "glasser_self_lmm_fixed_effects.csv")
OUT_EMM     <- file.path(REPO_ROOT, "04_statistics", "results",
                          "glasser_self_lmm_emm_contrasts.csv")
OUT_FLAT    <- file.path(REPO_ROOT, "04_statistics", "results",
                          "glasser_self_lmm_flattening.csv")
OUT_FIG24   <- file.path(REPO_ROOT, "04_statistics", "figures",
                          "fig24_glasser_self_lmm_diagnostics.png")

all_out <- c(OUT_DATA, OUT_FE, OUT_EMM, OUT_FLAT, OUT_FIG24)
if (all(file.exists(all_out))) {
  cat("All outputs exist — nothing to do. Delete to rerun.\n"); quit(status = 0)
}

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n04a_lmm_glasser.R — Step 16\n", SEP, "\n\n", sep = "")

# ── Part 1: Load and combine ───────────────────────────────────────────────────
cat(SEP, "\nPART 1 — Assemble combined Glasser self + nonself dataset\n",
    SEP, "\n", sep = "")

# Keskin self
df_self <- read.csv(KESKIN_CSV, stringsAsFactors = FALSE)
df_self$self_layer  <- df_self$layer
df_self$roi_uid     <- paste0("self_", df_self$glasser_id)
df_self$atlas_src   <- "self_glasser"
cat(sprintf("Keskin self rows : %d  (%d unique parcels)\n",
            nrow(df_self), length(unique(df_self$glasser_id))))

# Nonself 316: add Glasser IDs via lookup, set self_layer = "nonself"
df_ns     <- read.csv(NONSELF_CSV, stringsAsFactors = FALSE)
lookup    <- read.csv(LOOKUP_CSV,  stringsAsFactors = FALSE)
# lookup: nonself_pos_id, roi_name, G1, G1_z, glasser_row_index
names(lookup)[names(lookup) == "nonself_pos_id"]    <- "roi_pos_id"
names(lookup)[names(lookup) == "glasser_row_index"] <- "glasser_id"

df_ns <- merge(df_ns, lookup[, c("roi_pos_id", "glasser_id")],
               by = "roi_pos_id", all.x = TRUE)
df_ns$self_layer <- "nonself"
df_ns$roi_uid    <- paste0("nonself_", df_ns$roi_pos_id)
df_ns$atlas_src  <- "nonself_glasser"
cat(sprintf("Nonself rows     : %d  (%d unique parcels)\n",
            nrow(df_ns), length(unique(df_ns$roi_pos_id))))

# Harmonise columns and stack
keep <- c("subject", "session", "group", "glasser_id", "roi_uid",
          "auc", "G1", "self_layer", "atlas_src")

df_self$roi_pos_id <- df_self$glasser_id
df_self_k  <- df_self[, c("subject","session","group","glasser_id","roi_uid",
                           "auc","G1","self_layer","atlas_src")]
df_ns_k    <- df_ns[,   c("subject","session","group","glasser_id","roi_uid",
                           "auc","G1","self_layer","atlas_src")]

df <- rbind(df_self_k, df_ns_k)

# Re-scale G1_z across the full combined dataset
μ_g1 <- mean(df$G1, na.rm = TRUE)
σ_g1 <- sd(df$G1,   na.rm = TRUE)
df$G1_z <- (df$G1 - μ_g1) / σ_g1
cat(sprintf("Combined G1_z: mean = %.4f (SD = %.4f, expected ~0 and ~1)\n",
            mean(df$G1_z), sd(df$G1_z)))

# Factor setup
LAYER_LEVELS <- c("nonself","Interoception","Exteroception","Cognition")
df$self_layer <- factor(df$self_layer, levels = LAYER_LEVELS)
df$group      <- relevel(factor(df$group),   ref = "placebo")
df$session    <- relevel(factor(df$session), ref = "ses-01")
df$subject    <- factor(df$subject)
df$roi_uid    <- factor(df$roi_uid)

cat(sprintf("\nself_layer levels (ref='nonself'): %s\n",
            paste(levels(df$self_layer), collapse = ", ")))
cat(sprintf("Subjects: %d  |  ROI UIDs: %d  |  Total rows: %d\n",
            nlevels(df$subject), nlevels(df$roi_uid), nrow(df)))
cat("\nRows per layer:\n")
print(table(df$self_layer))

write.csv(df, OUT_DATA, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", OUT_DATA))

# ── Part 2: Baseline gradient ─────────────────────────────────────────────────
cat("\n", SEP, "\nPART 2 — Baseline gradient check (placebo, ses-01)\n",
    SEP, "\n", sep = "")
df_base <- df[df$group == "placebo" & df$session == "ses-01", ]
lm_means <- tapply(df_base$auc, df_base$self_layer, mean, na.rm = TRUE)
cat("Mean AUC per layer at baseline:\n")
for (l in LAYER_LEVELS) cat(sprintf("  %-15s: %.4f\n", l, lm_means[l]))

aov_base <- aov(auc ~ self_layer, data = df_base)
aov_s    <- summary(aov_base)[[1]]
cat(sprintf("\nANOVA: F(%d,%d) = %.4f, p = %.4g\n",
            aov_s["self_layer","Df"], aov_s["Residuals","Df"],
            aov_s["self_layer","F value"], aov_s["self_layer","Pr(>F)"]))
cat("\nTukey HSD:\n"); print(round(TukeyHSD(aov_base,"self_layer")$self_layer, 4))

# ── Part 3: Fit primary model ──────────────────────────────────────────────────
cat("\n", SEP, "\nPART 3 — Fit primary model\n", SEP, "\n", sep = "")

FORMULA_FULL <- auc ~ group * session * self_layer +
                      (1 + session | subject) + (1 | roi_uid)
FORMULA_SIMP <- auc ~ group * session * self_layer +
                      (1 | subject) + (1 | roi_uid)

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5),
                    check.conv.grad = .makeCC("warning", tol = 0.002))

try_fit <- function(formula, label) {
  tryCatch(
    withCallingHandlers(
      lmer(formula, data = df, REML = TRUE, control = ctrl),
      warning = function(w) {
        cat(sprintf("  [%s] warning: %s\n", label, conditionMessage(w)))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) { cat(sprintf("  [%s] ERROR: %s\n", label, e$message)); NULL }
  )
}

cat("Fitting (1+session|subject) + (1|roi_uid)...\n")
m <- try_fit(FORMULA_FULL, "full-RE/bobyqa")
re_label <- "(1 + session | subject) + (1 | roi_uid)"

if (!is.null(m) && isSingular(m)) {
  cat("Singular. Falling back to (1|subject) + (1|roi_uid).\n")
  m <- NULL
}
if (is.null(m)) {
  cat("Fallback: (1|subject) + (1|roi_uid)...\n")
  m <- try_fit(FORMULA_SIMP, "simple-RE/bobyqa")
  re_label <- "(1 | subject) + (1 | roi_uid)"
  FORMULA_FULL <- FORMULA_SIMP
}
if (is.null(m)) stop("All model fits failed.")

conv <- m@optinfo$conv$lme4$messages
if (!is.null(conv)) {
  cat(sprintf("CONVERGENCE: %s\n", paste(conv, collapse="; ")))
} else {
  cat(sprintf("Convergence: OK  |  RE: %s\n", re_label))
}

s <- summary(m); print(s)
cat(sprintf("\nAIC: %.2f  BIC: %.2f  logLik: %.4f\n",
            AIC(m), BIC(m), as.numeric(logLik(m))))

fe_raw <- coef(s)
fe_tbl <- data.frame(term=rownames(fe_raw), estimate=fe_raw[,"Estimate"],
                     SE=fe_raw[,"Std. Error"], df=fe_raw[,"df"],
                     t_value=fe_raw[,"t value"], p_value=fe_raw[,"Pr(>|t|)"],
                     stringsAsFactors=FALSE, row.names=NULL)
cat("\nRandom effects:\n"); print(as.data.frame(VarCorr(m)))

# ── Part 5: EMMs ──────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 5 — Estimated marginal means and contrasts\n", SEP, "\n", sep = "")

emm_options(lmerTest.limit = nrow(df), pbkrtest.limit = nrow(df))
emm_main <- emmeans(m, ~ group * session * self_layer)

cat("\n--- 5a: Cell means (16 cells) ---\n")
emm_df  <- as.data.frame(emm_main)
ci_cols <- intersect(c("lower.CL","upper.CL","asymp.LCL","asymp.UCL"), names(emm_df))
df_col  <- intersect("df", names(emm_df))
print(emm_df[, c("group","session","self_layer","emmean","SE", df_col, ci_cols)],
      row.names = FALSE)

cat("\n--- 5b: Self-nonself gap per group × session ---\n")
SELF_LAYERS <- c("Interoception","Exteroception","Cognition")
emm_by <- emmeans(m, ~ self_layer | group + session)
gaps <- contrast(emm_by,
  method = list(
    "Interoception - nonself" = c(-1, 1, 0, 0),
    "Exteroception - nonself" = c(-1, 0, 1, 0),
    "Cognition - nonself"     = c(-1, 0, 0, 1)
  ), adjust = "none")
gaps_df <- as.data.frame(summary(gaps, infer = TRUE))
if (!"lower.CL" %in% names(gaps_df)) {
  gaps_df$lower.CL <- gaps_df$estimate - 1.96 * gaps_df$SE
  gaps_df$upper.CL <- gaps_df$estimate + 1.96 * gaps_df$SE
}
print(gaps_df, row.names = FALSE)

cat("\n--- 5c: Flattening (Δgap_verum − Δgap_placebo) ---\n")
gap_ses <- contrast(gaps, "consec", by = c("contrast","group"), adjust = "none")
flat    <- contrast(gap_ses, "consec", by = "contrast", adjust = "none")
flat_df <- as.data.frame(summary(flat, infer = TRUE))
if (!"lower.CL" %in% names(flat_df)) {
  flat_df$lower.CL <- flat_df$estimate - 1.96 * flat_df$SE
  flat_df$upper.CL <- flat_df$estimate + 1.96 * flat_df$SE
}
print(flat_df, row.names = FALSE)

# Average flattening (16-cell contrast on emm_main)
flat_avg_vec <- c(-1, 1, 1, -1,
                   1/3, -1/3, -1/3, 1/3,
                   1/3, -1/3, -1/3, 1/3,
                   1/3, -1/3, -1/3, 1/3)
flat_avg    <- contrast(emm_main, method = list("avg_flattening" = flat_avg_vec))
flat_avg_df <- as.data.frame(summary(flat_avg, infer = TRUE))
if (!"lower.CL" %in% names(flat_avg_df)) {
  flat_avg_df$lower.CL <- flat_avg_df$estimate - 1.96 * flat_avg_df$SE
  flat_avg_df$upper.CL <- flat_avg_df$estimate + 1.96 * flat_avg_df$SE
}
cat("\nAverage flattening across 3 self layers:\n"); print(flat_avg_df, row.names = FALSE)
avg_flat_b <- flat_avg_df$estimate[1]; avg_flat_p <- flat_avg_df$p.value[1]

cat("\n--- 5d: Gap = 0 at verum post? ---\n")
meet_df <- gaps_df[gaps_df$group == "verum" & gaps_df$session == "ses-02", ]
print(meet_df[, intersect(c("contrast","estimate","SE","df","lower.CL","upper.CL",
                             "t.ratio","p.value"), names(meet_df))], row.names = FALSE)

# Save EMM outputs
write.csv(gaps_df, OUT_EMM,  row.names = FALSE); cat(sprintf("Saved: %s\n", OUT_EMM))

flat_save <- flat_df
flat_save$avg_flat_b <- NA; flat_save$avg_flat_p <- NA
flat_save$avg_flat_b[1] <- avg_flat_b; flat_save$avg_flat_p[1] <- avg_flat_p
write.csv(flat_save, OUT_FLAT, row.names = FALSE); cat(sprintf("Saved: %s\n", OUT_FLAT))

# ── Part 6: Residual diagnostics (fig24) ─────────────────────────────────────
cat("\n", SEP, "\nPART 6 — Residual diagnostics\n", SEP, "\n", sep = "")

resids  <- resid(m)
fitted_ <- fitted(m)
sk <- skewness(resids); ku <- kurtosis(resids)
cat(sprintf("Skewness: %.4f   Kurtosis: %.4f  (excess %.4f)\n", sk, ku, ku - 3))
set.seed(42); sw_idx <- sample(length(resids), min(5000L, length(resids)))
sw <- shapiro.test(resids[sw_idx])
cat(sprintf("Shapiro-Wilk (n=%d): W=%.5f  p=%.4e\n",
            length(sw_idx), sw$statistic, sw$p.value))

if (!requireNamespace("patchwork", quietly = TRUE))
  install.packages("patchwork", repos = "https://cloud.r-project.org", quiet = TRUE)
suppressPackageStartupMessages(library(patchwork))

qq_n   <- min(5000L, length(resids))
qq_smp <- sort(sample(resids, qq_n))
qq_teo <- qnorm(ppoints(qq_n))

diag_theme <- theme_minimal(base_family = "serif") +
  theme(panel.background = element_rect(fill = "white", color = NA),
        plot.background  = element_rect(fill = "white", color = NA),
        plot.title = element_text(face = "plain", size = 11, hjust = 0.5))

p_qq <- ggplot(data.frame(theoretical = qq_teo, sample = qq_smp),
               aes(x = theoretical, y = sample)) +
  geom_point(size = 0.8, alpha = 0.35, color = "steelblue") +
  geom_abline(color = "red", linetype = "dashed") +
  labs(title = "QQ Plot", x = "Theoretical", y = "Sample") +
  diag_theme

rvf_idx <- sample(length(resids), min(5000L, length(resids)))
p_rvf <- ggplot(data.frame(fitted = fitted_[rvf_idx], resid = resids[rvf_idx]),
                aes(x = fitted, y = resid)) +
  geom_point(size = 0.8, alpha = 0.25, color = "steelblue") +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  labs(title = "Resid vs Fitted", x = "Fitted", y = "Residuals") +
  diag_theme

p_hist <- ggplot(data.frame(resid = resids), aes(x = resid)) +
  geom_histogram(bins = 80, fill = "steelblue", color = "white", linewidth = 0.2) +
  labs(title = "Histogram", x = "Residual", y = "Count") +
  diag_theme

fig24 <- p_qq | p_rvf | p_hist
ggsave(OUT_FIG24, fig24, width = 12, height = 4, dpi = 300, bg = "white")
cat(sprintf("Saved: %s\n", OUT_FIG24))

# ── Save CSVs ─────────────────────────────────────────────────────────────────
write.csv(fe_tbl, OUT_FE, row.names = FALSE); cat(sprintf("Saved: %s\n", OUT_FE))

# ── Summary ───────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSUMMARY\n", SEP, "\n", sep = "")
cat(sprintf(
  "Average flattening (self-nonself gap change under DMT): β=%.5f, p=%.4g.\n",
  avg_flat_b, avg_flat_p))
cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
