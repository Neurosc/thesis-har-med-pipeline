# 02_lmm.R — Self vs nonself LMM (parcels and spheres)
#
# Both atlas methods:
#   auc ~ group * session * self_layer + (1+session|subject) + (1|roi_uid)
#   self_layer: nonself (ref), Interoception, Exteroception, Cognition
#   Sphere layers from Qin 2020 self_coordinates.txt; parcel layers from Keskin 2025
#
# Run from repo root:
#   Rscript 04_statistics/scripts/02_lmm.R

if (!requireNamespace("emmeans",  quietly = TRUE))
  install.packages("emmeans",  repos = "https://cloud.r-project.org", quiet = TRUE)
if (!requireNamespace("RcppTOML", quietly = TRUE))
  install.packages("RcppTOML", repos = "https://cloud.r-project.org", quiet = TRUE)

suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(emmeans)
  library(moments); library(dplyr)
  library(ggplot2); library(RcppTOML)
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

# ── Config ─────────────────────────────────────────────────────────────────────
cfg              <- parseTOML(file.path(REPO_ROOT, "config.toml"))
ATLAS_METHOD     <- cfg$active$atlas_method
DENOISING_METHOD <- cfg$active$denoising_method
TAG              <- paste0(ATLAS_METHOD, "_", DENOISING_METHOD)

TABLES_DIR <- file.path(REPO_ROOT, "04_statistics", "results", TAG, "tables")
FIGS_DIR   <- file.path(REPO_ROOT, "04_statistics", "results", TAG, "figures")
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGS_DIR,   showWarnings = FALSE, recursive = TRUE)

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n02_lmm.R — config: ", TAG, "\n", SEP, "\n\n", sep = "")

# ── Shared helpers ─────────────────────────────────────────────────────────────
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
      }
    ),
    error = function(e) {
      cat(sprintf("  [%s] ERROR: %s\n", label, e$message)); NULL
    }
  )
}

save_diag_fig <- function(path, resids, fitted_v) {
  if (!requireNamespace("patchwork", quietly = TRUE))
    install.packages("patchwork", repos = "https://cloud.r-project.org", quiet = TRUE)
  suppressPackageStartupMessages(library(patchwork))
  qq_n   <- min(5000L, length(resids))
  qq_smp <- sort(sample(resids, qq_n))
  qq_teo <- qnorm(ppoints(qq_n))
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
  ggsave(path, p_qq | p_rvf | p_hist, width = 12, height = 4,
         dpi = 300, bg = "white")
  cat(sprintf("Saved: %s\n", path))
}

add_CI <- function(d) {
  if (!"lower.CL" %in% names(d)) {
    d$lower.CL <- d$estimate - 1.96 * d$SE
    d$upper.CL <- d$estimate + 1.96 * d$SE
  }
  d
}

get_fe <- function(fe_mat, term) {
  if (term %in% rownames(fe_mat))
    fe_mat[term, c("Estimate", "Pr(>|t|)")]
  else c(Estimate = NA, `Pr(>|t|)` = NA)
}

# ══════════════════════════════════════════════════════════════════════════════
# PARCELS BRANCH
# ══════════════════════════════════════════════════════════════════════════════
if (ATLAS_METHOD == "parcels") {

  KESKIN_CSV  <- file.path(TABLES_DIR, "keskin_auc_model_ready.csv")
  NONSELF_CSV <- file.path(TABLES_DIR, "nonself_model_ready.csv")
  OUT_DATA    <- file.path(TABLES_DIR, "model_ready.csv")
  OUT_FE      <- file.path(TABLES_DIR, "lmm_fixed_effects.csv")
  OUT_EMM     <- file.path(TABLES_DIR, "lmm_emm_contrasts.csv")
  OUT_FLAT    <- file.path(TABLES_DIR, "lmm_flattening.csv")
  OUT_FIG24   <- file.path(FIGS_DIR,   "fig24_lmm_diagnostics.png")
  OUT_FIG_TR  <- file.path(FIGS_DIR,   "fig_lmm_diagnostics_trimmed.png")

  if (!file.exists(KESKIN_CSV))
    stop("Input not found — run 01_build_dataframe.jl first: ", KESKIN_CSV)
  if (!file.exists(NONSELF_CSV))
    stop("Input not found — run 01_build_dataframe.jl first: ", NONSELF_CSV)

  # ── Part 1: Assemble combined dataset ───────────────────────────────────────
  cat(SEP, "\nPART 1 — Assemble combined Glasser self + nonself dataset\n",
      SEP, "\n", sep = "")

  df_self <- read.csv(KESKIN_CSV, stringsAsFactors = FALSE)
  df_self$self_layer <- df_self$layer
  df_self$roi_uid    <- paste0("self_", df_self$glasser_id)
  df_self$atlas_src  <- "self_glasser"
  cat(sprintf("Keskin self rows : %d  (%d unique parcels)\n",
              nrow(df_self), length(unique(df_self$glasser_id))))

  df_ns <- read.csv(NONSELF_CSV, stringsAsFactors = FALSE)
  df_ns$self_layer <- "nonself"
  df_ns$roi_uid    <- paste0("nonself_", df_ns$glasser_id)
  df_ns$atlas_src  <- "nonself_glasser"
  cat(sprintf("Nonself rows     : %d  (%d unique parcels)\n",
              nrow(df_ns), length(unique(df_ns$glasser_id))))

  KEEP <- c("subject", "session", "group", "glasser_id", "roi_uid",
            "auc", "self_layer", "atlas_src")
  df   <- rbind(df_self[, KEEP], df_ns[, KEEP])

  LAYER_LEVELS  <- c("nonself", "Interoception", "Exteroception", "Cognition")
  df$self_layer <- factor(df$self_layer, levels = LAYER_LEVELS)
  df$group      <- relevel(factor(df$group),   ref = "placebo")
  df$session    <- relevel(factor(df$session), ref = "ses-01")
  df$subject    <- factor(df$subject)
  df$roi_uid    <- factor(df$roi_uid)

  cat(sprintf("\nself_layer levels (ref='nonself'): %s\n",
              paste(levels(df$self_layer), collapse = ", ")))
  cat(sprintf("Subjects: %d  |  ROI UIDs: %d  |  Total rows: %d\n",
              nlevels(df$subject), nlevels(df$roi_uid), nrow(df)))
  cat("\nRows per layer:\n"); print(table(df$self_layer))

  write.csv(df, OUT_DATA, row.names = FALSE)
  cat(sprintf("\nSaved: %s\n", OUT_DATA))

  # ── Part 2: Baseline gradient check ─────────────────────────────────────────
  cat("\n", SEP, "\nPART 2 — Baseline gradient check (placebo, ses-01)\n",
      SEP, "\n", sep = "")

  df_base  <- df[df$group == "placebo" & df$session == "ses-01", ]
  lm_means <- tapply(df_base$auc, df_base$self_layer, mean, na.rm = TRUE)
  cat("Mean AUC per layer at baseline:\n")
  for (l in LAYER_LEVELS) cat(sprintf("  %-15s: %.4f\n", l, lm_means[l]))

  aov_base <- aov(auc ~ self_layer, data = df_base)
  aov_s    <- summary(aov_base)[[1]]
  cat(sprintf("\nANOVA: F(%d,%d) = %.4f, p = %.4g\n",
              aov_s["self_layer", "Df"], aov_s["Residuals", "Df"],
              aov_s["self_layer", "F value"], aov_s["self_layer", "Pr(>F)"]))
  cat("\nTukey HSD:\n")
  print(round(TukeyHSD(aov_base, "self_layer")$self_layer, 4))

  # ── Part 3: Fit primary model ────────────────────────────────────────────────
  cat("\n", SEP, "\nPART 3 — Fit primary model\n", SEP, "\n", sep = "")

  FORMULA_FULL <- auc ~ group * session * self_layer +
                        (1 + session | subject) + (1 | roi_uid)
  FORMULA_SIMP <- auc ~ group * session * self_layer +
                        (1 | subject) + (1 | roi_uid)

  cat("Fitting (1+session|subject) + (1|roi_uid)...\n")
  m        <- try_fit(FORMULA_FULL, "full-RE/bobyqa", df)
  re_label <- "(1 + session | subject) + (1 | roi_uid)"

  if (!is.null(m) && isSingular(m)) {
    cat("Singular. Falling back to (1|subject) + (1|roi_uid).\n"); m <- NULL
  }
  if (is.null(m)) {
    cat("Fallback: (1|subject) + (1|roi_uid)...\n")
    m            <- try_fit(FORMULA_SIMP, "simple-RE/bobyqa", df)
    re_label     <- "(1 | subject) + (1 | roi_uid)"
    FORMULA_FULL <- FORMULA_SIMP
  }
  if (is.null(m)) stop("All model fits failed.")

  conv <- m@optinfo$conv$lme4$messages
  if (!is.null(conv)) cat(sprintf("CONVERGENCE: %s\n", paste(conv, collapse = "; ")))
  else cat(sprintf("Convergence: OK  |  RE: %s\n", re_label))

  s      <- summary(m); print(s)
  cat(sprintf("\nAIC: %.2f  BIC: %.2f  logLik: %.4f\n",
              AIC(m), BIC(m), as.numeric(logLik(m))))

  fe_raw <- coef(s)
  fe_tbl <- data.frame(term     = rownames(fe_raw),
                       estimate = fe_raw[, "Estimate"],
                       SE       = fe_raw[, "Std. Error"],
                       df       = fe_raw[, "df"],
                       t_value  = fe_raw[, "t value"],
                       p_value  = fe_raw[, "Pr(>|t|)"],
                       stringsAsFactors = FALSE, row.names = NULL)

  cat("\nRandom effects:\n"); print(as.data.frame(VarCorr(m)))

  # ── Part 4: EMMs ────────────────────────────────────────────────────────────
  cat("\n", SEP, "\nPART 4 — Estimated marginal means and contrasts\n",
      SEP, "\n", sep = "")

  emm_options(lmerTest.limit = nrow(df), pbkrtest.limit = nrow(df))
  emm_main <- emmeans(m, ~ group * session * self_layer)

  cat("\n--- 4a: Cell means (16 cells) ---\n")
  emm_df  <- as.data.frame(emm_main)
  ci_cols <- intersect(c("lower.CL", "upper.CL", "asymp.LCL", "asymp.UCL"), names(emm_df))
  df_col  <- intersect("df", names(emm_df))
  print(emm_df[, c("group", "session", "self_layer", "emmean", "SE", df_col, ci_cols)],
        row.names = FALSE)

  cat("\n--- 4b: Self-nonself gap per group × session ---\n")
  emm_by <- emmeans(m, ~ self_layer | group + session)
  gaps   <- contrast(emm_by,
    method = list(
      "Interoception - nonself" = c(-1, 1, 0, 0),
      "Exteroception - nonself" = c(-1, 0, 1, 0),
      "Cognition - nonself"     = c(-1, 0, 0, 1)
    ), adjust = "none")
  gaps_df <- add_CI(as.data.frame(summary(gaps, infer = TRUE)))
  print(gaps_df, row.names = FALSE)

  cat("\n--- 4c: Flattening (Δgap_verum − Δgap_placebo) ---\n")
  gap_ses <- contrast(gaps, "consec", by = c("contrast", "group"), adjust = "none")
  flat    <- contrast(gap_ses, "consec", by = "contrast", adjust = "none")
  flat_df <- add_CI(as.data.frame(summary(flat, infer = TRUE)))
  print(flat_df, row.names = FALSE)

  # emmeans cell order for ~ group * session * self_layer:
  # (group fastest, session next, self_layer slowest)
  # 1:(plac,s01,ns) 2:(ver,s01,ns) 3:(plac,s02,ns) 4:(ver,s02,ns)
  # 5:(plac,s01,In) 6:(ver,s01,In) 7:(plac,s02,In) 8:(ver,s02,In)
  # 9:(plac,s01,Ex) 10:(ver,s01,Ex) 11:(plac,s02,Ex) 12:(ver,s02,Ex)
  # 13:(plac,s01,Co) 14:(ver,s01,Co) 15:(plac,s02,Co) 16:(ver,s02,Co)
  flat_avg_vec <- c(-1, 1, 1, -1,
                     1/3, -1/3, -1/3, 1/3,
                     1/3, -1/3, -1/3, 1/3,
                     1/3, -1/3, -1/3, 1/3)
  flat_avg    <- contrast(emm_main, method = list("avg_flattening" = flat_avg_vec))
  flat_avg_df <- add_CI(as.data.frame(summary(flat_avg, infer = TRUE)))
  cat("\nAverage flattening across 3 self layers:\n")
  print(flat_avg_df, row.names = FALSE)
  avg_flat_b <- flat_avg_df$estimate[1]
  avg_flat_p <- flat_avg_df$p.value[1]

  cat("\n--- 4d: Gap = 0 at verum post? ---\n")
  meet_df <- gaps_df[gaps_df$group == "verum" & gaps_df$session == "ses-02", ]
  print(meet_df[, intersect(c("contrast", "estimate", "SE", "df",
                               "lower.CL", "upper.CL", "t.ratio", "p.value"),
                             names(meet_df))], row.names = FALSE)

  write.csv(gaps_df, OUT_EMM,  row.names = FALSE); cat(sprintf("Saved: %s\n", OUT_EMM))

  flat_save            <- flat_df
  flat_save$avg_flat_b <- NA; flat_save$avg_flat_p <- NA
  flat_save$avg_flat_b[1] <- avg_flat_b; flat_save$avg_flat_p[1] <- avg_flat_p
  write.csv(flat_save, OUT_FLAT, row.names = FALSE); cat(sprintf("Saved: %s\n", OUT_FLAT))

  # ── Part 5: Residual diagnostics ────────────────────────────────────────────
  cat("\n", SEP, "\nPART 5 — Residual diagnostics\n", SEP, "\n", sep = "")

  resids  <- resid(m)
  fitted_ <- fitted(m)
  sk <- skewness(resids); ku <- kurtosis(resids)
  cat(sprintf("Skewness: %.4f   Kurtosis: %.4f  (excess %.4f)\n", sk, ku, ku - 3))
  set.seed(42); sw_idx <- sample(length(resids), min(5000L, length(resids)))
  sw <- shapiro.test(resids[sw_idx])
  cat(sprintf("Shapiro-Wilk (n=%d): W=%.5f  p=%.4e\n",
              length(sw_idx), sw$statistic, sw$p.value))
  save_diag_fig(OUT_FIG24, resids, fitted_)

  # ── Part 6: Sensitivity — residual trimming ──────────────────────────────────
  cat("\n", SEP, "\nPART 6 — Sensitivity: residual trimming (|resid| > 2.5 SD)\n",
      SEP, "\n", sep = "")

  resid_sd  <- sd(resids)
  resid_thr <- 2.5 * resid_sd
  flagged   <- abs(resids) > resid_thr
  n_flagged <- sum(flagged); n_total <- length(resids)
  cat(sprintf("Residual SD = %.4f   Threshold = ±%.4f (2.5 × SD)\n",
              resid_sd, resid_thr))
  cat(sprintf("Flagged: %d / %d observations (%.2f%%)\n\n",
              n_flagged, n_total, 100 * n_flagged / n_total))

  df_trim  <- df[!flagged, ]
  m_trim   <- try_fit(FORMULA_FULL, "trim/full-RE", df_trim)
  if (!is.null(m_trim) && isSingular(m_trim)) {
    cat("  Singular — retrying with (1|subject)+(1|roi_uid)...\n")
    m_trim <- try_fit(FORMULA_SIMP, "trim/simple-RE", df_trim)
  }
  if (is.null(m_trim)) stop("Trimmed model fit failed.")
  conv_trim <- m_trim@optinfo$conv$lme4$messages
  cat(if (!is.null(conv_trim))
        sprintf("  Convergence: %s\n", paste(conv_trim, collapse = "; "))
      else "  Convergence: OK\n")

  TERMS <- c(
    "Exteroception" = "groupverum:sessionses-02:self_layerExteroception",
    "Interoception" = "groupverum:sessionses-02:self_layerInteroception",
    "Cognition"     = "groupverum:sessionses-02:self_layerCognition"
  )

  emm_options(lmerTest.limit = nrow(df_trim), pbkrtest.limit = nrow(df_trim))
  emm_trim         <- emmeans(m_trim, ~ group * session * self_layer)
  flat_avg_trim    <- contrast(emm_trim, method = list("avg_flattening" = flat_avg_vec))
  flat_avg_trim_df <- add_CI(as.data.frame(summary(flat_avg_trim, infer = TRUE)))
  avg_flat_b_trim  <- flat_avg_trim_df$estimate[1]
  avg_flat_p_trim  <- flat_avg_trim_df$p.value[1]

  fe_trim <- coef(summary(m_trim))
  cat(sprintf("\n%-30s  %-22s  %-22s\n", "",
              "Full data", sprintf("Trimmed (|e| < 2.5 SD)")))
  cat(sprintf("%-30s  %-22s  %-22s\n", "",
              sprintf("N = %d", n_total), sprintf("N = %d", nrow(df_trim))))
  cat(strrep("-", 78), "\n")
  for (nm in names(TERMS)) {
    full_v <- get_fe(as.matrix(coef(summary(m))), TERMS[[nm]])
    trim_v <- get_fe(as.matrix(fe_trim),          TERMS[[nm]])
    cat(sprintf("%-30s  β=%8.5f  p=%7.4g    β=%8.5f  p=%7.4g%s\n",
                paste0("group:session:", nm),
                full_v["Estimate"], full_v["Pr(>|t|)"],
                trim_v["Estimate"], trim_v["Pr(>|t|)"],
                if (!is.na(trim_v["Pr(>|t|)"]) && trim_v["Pr(>|t|)"] < 0.05) " *" else ""))
  }
  cat(sprintf("%-30s  β=%8.5f  p=%7.4g    β=%8.5f  p=%7.4g%s\n",
              "Average flattening",
              avg_flat_b, avg_flat_p,
              avg_flat_b_trim, avg_flat_p_trim,
              if (!is.na(avg_flat_p_trim) && avg_flat_p_trim < 0.05) " *" else ""))
  cat(strrep("-", 78), "\n")

  save_diag_fig(OUT_FIG_TR, resid(m_trim), fitted(m_trim))

  # ── Save FE table ────────────────────────────────────────────────────────────
  write.csv(fe_tbl, OUT_FE, row.names = FALSE); cat(sprintf("Saved: %s\n", OUT_FE))

  cat("\n", SEP, "\nSUMMARY\n", SEP, "\n", sep = "")
  cat(sprintf(
    "Average flattening (self-nonself gap change under DMT): β=%.5f, p=%.4g.\n",
    avg_flat_b, avg_flat_p))

# ══════════════════════════════════════════════════════════════════════════════
# SPHERE BRANCH
# ══════════════════════════════════════════════════════════════════════════════
} else {  # ATLAS_METHOD == "spheres"

  LONG_CSV   <- file.path(TABLES_DIR, "analysis_long_format_auc.csv")
  OUT_DATA   <- file.path(TABLES_DIR, "model_ready.csv")
  OUT_FE     <- file.path(TABLES_DIR, "lmm_fixed_effects.csv")
  OUT_EMM    <- file.path(TABLES_DIR, "lmm_emm_contrasts.csv")
  OUT_FLAT   <- file.path(TABLES_DIR, "lmm_flattening.csv")
  OUT_FIG24  <- file.path(FIGS_DIR,   "fig24_lmm_diagnostics.png")
  OUT_FIG_TR <- file.path(FIGS_DIR,   "fig_lmm_diagnostics_trimmed.png")

  if (!file.exists(LONG_CSV))
    stop("Input not found — run 01_build_dataframe.jl first: ", LONG_CSV)

  # ── Part 1: Assemble combined sphere dataset ─────────────────────────────────
  cat(SEP, "\nPART 1 — Assemble sphere self + nonself dataset\n",
      SEP, "\n", sep = "")

  raw <- read.csv(LONG_CSV, stringsAsFactors = FALSE)
  cat(sprintf("Loaded analysis_long_format_auc.csv: %d rows\n", nrow(raw)))

  df <- raw
  names(df)[names(df) == "drug_group"] <- "group"
  names(df)[names(df) == "atlas"]      <- "roi_category"
  df$roi_uid   <- paste0(df$roi_category, "_", df$roi_id)
  df$atlas_src <- paste0(df$roi_category, "_sphere")

  LAYER_LEVELS <- c("nonself", "Interoception", "Exteroception", "Cognition")
  df$self_layer <- factor(df$self_layer, levels = LAYER_LEVELS)
  df$group      <- relevel(factor(df$group),   ref = "placebo")
  df$session    <- relevel(factor(df$session), ref = "ses-01")
  df$subject    <- factor(df$subject)
  df$roi_uid    <- factor(df$roi_uid)

  cat(sprintf("self_layer levels (ref='nonself'): %s\n",
              paste(levels(df$self_layer), collapse = ", ")))
  cat(sprintf("Subjects: %d  |  ROI UIDs: %d  |  Total rows: %d\n",
              nlevels(df$subject), nlevels(df$roi_uid), nrow(df)))
  cat("\nRows per layer:\n"); print(table(df$self_layer))

  KEEP_SPH <- c("subject", "session", "group", "roi_id", "roi_uid",
                "auc", "self_layer", "atlas_src")
  df <- df[, KEEP_SPH]
  write.csv(df, OUT_DATA, row.names = FALSE)
  cat(sprintf("\nSaved: %s\n", OUT_DATA))

  # ── Part 2: Baseline gradient check ─────────────────────────────────────────
  cat("\n", SEP, "\nPART 2 — Baseline gradient check (placebo, ses-01)\n",
      SEP, "\n", sep = "")

  df_base  <- df[df$group == "placebo" & df$session == "ses-01", ]
  lm_means <- tapply(df_base$auc, df_base$self_layer, mean, na.rm = TRUE)
  cat("Mean AUC per layer at baseline:\n")
  for (l in LAYER_LEVELS) cat(sprintf("  %-15s: %.4f\n", l, lm_means[l]))

  aov_base <- aov(auc ~ self_layer, data = df_base)
  aov_s    <- summary(aov_base)[[1]]
  cat(sprintf("\nANOVA: F(%d,%d) = %.4f, p = %.4g\n",
              aov_s["self_layer", "Df"], aov_s["Residuals", "Df"],
              aov_s["self_layer", "F value"], aov_s["self_layer", "Pr(>F)"]))
  cat("\nTukey HSD:\n")
  print(round(TukeyHSD(aov_base, "self_layer")$self_layer, 4))

  # ── Part 3: Fit primary model ────────────────────────────────────────────────
  cat("\n", SEP, "\nPART 3 — Fit primary model\n", SEP, "\n", sep = "")

  FORMULA_FULL <- auc ~ group * session * self_layer +
                        (1 + session | subject) + (1 | roi_uid)
  FORMULA_SIMP <- auc ~ group * session * self_layer +
                        (1 | subject) + (1 | roi_uid)

  cat("Fitting (1+session|subject) + (1|roi_uid)...\n")
  m        <- try_fit(FORMULA_FULL, "full-RE/bobyqa", df)
  re_label <- "(1 + session | subject) + (1 | roi_uid)"

  if (!is.null(m) && isSingular(m)) {
    cat("Singular. Falling back to (1|subject) + (1|roi_uid).\n"); m <- NULL
  }
  if (is.null(m)) {
    cat("Fallback: (1|subject) + (1|roi_uid)...\n")
    m            <- try_fit(FORMULA_SIMP, "simple-RE/bobyqa", df)
    re_label     <- "(1 | subject) + (1 | roi_uid)"
    FORMULA_FULL <- FORMULA_SIMP
  }
  if (is.null(m)) stop("All model fits failed.")

  conv <- m@optinfo$conv$lme4$messages
  if (!is.null(conv)) cat(sprintf("CONVERGENCE: %s\n", paste(conv, collapse = "; ")))
  else cat(sprintf("Convergence: OK  |  RE: %s\n", re_label))

  s      <- summary(m); print(s)
  cat(sprintf("\nAIC: %.2f  BIC: %.2f  logLik: %.4f\n",
              AIC(m), BIC(m), as.numeric(logLik(m))))

  fe_raw <- coef(s)
  fe_tbl <- data.frame(term     = rownames(fe_raw),
                       estimate = fe_raw[, "Estimate"],
                       SE       = fe_raw[, "Std. Error"],
                       df       = fe_raw[, "df"],
                       t_value  = fe_raw[, "t value"],
                       p_value  = fe_raw[, "Pr(>|t|)"],
                       stringsAsFactors = FALSE, row.names = NULL)

  cat("\nRandom effects:\n"); print(as.data.frame(VarCorr(m)))

  # ── Part 4: EMMs ────────────────────────────────────────────────────────────
  cat("\n", SEP, "\nPART 4 — Estimated marginal means and contrasts\n",
      SEP, "\n", sep = "")

  emm_options(lmerTest.limit = nrow(df), pbkrtest.limit = nrow(df))
  emm_main <- emmeans(m, ~ group * session * self_layer)

  cat("\n--- 4a: Cell means (16 cells) ---\n")
  emm_df  <- as.data.frame(emm_main)
  ci_cols <- intersect(c("lower.CL", "upper.CL", "asymp.LCL", "asymp.UCL"), names(emm_df))
  df_col  <- intersect("df", names(emm_df))
  print(emm_df[, c("group", "session", "self_layer", "emmean", "SE", df_col, ci_cols)],
        row.names = FALSE)

  cat("\n--- 4b: Self-layer vs nonself gap per group × session ---\n")
  emm_by <- emmeans(m, ~ self_layer | group + session)
  gaps   <- contrast(emm_by,
    method = list(
      "Interoception - nonself" = c(-1, 1, 0, 0),
      "Exteroception - nonself" = c(-1, 0, 1, 0),
      "Cognition - nonself"     = c(-1, 0, 0, 1)
    ), adjust = "none")
  gaps_df <- add_CI(as.data.frame(summary(gaps, infer = TRUE)))
  print(gaps_df, row.names = FALSE)

  cat("\n--- 4c: Flattening (Δgap_verum − Δgap_placebo) ---\n")
  gap_ses <- contrast(gaps, "consec", by = c("contrast", "group"), adjust = "none")
  flat    <- contrast(gap_ses, "consec", by = "contrast", adjust = "none")
  flat_df <- add_CI(as.data.frame(summary(flat, infer = TRUE)))
  print(flat_df, row.names = FALSE)

  # emmeans cell order for ~ group * session * self_layer:
  # (group fastest, session next, self_layer slowest)
  # 1:(plac,s01,ns) 2:(ver,s01,ns) 3:(plac,s02,ns) 4:(ver,s02,ns)
  # 5:(plac,s01,In) 6:(ver,s01,In) 7:(plac,s02,In) 8:(ver,s02,In)
  # 9:(plac,s01,Ex) 10:(ver,s01,Ex) 11:(plac,s02,Ex) 12:(ver,s02,Ex)
  # 13:(plac,s01,Co) 14:(ver,s01,Co) 15:(plac,s02,Co) 16:(ver,s02,Co)
  flat_avg_vec <- c(-1, 1, 1, -1,
                     1/3, -1/3, -1/3, 1/3,
                     1/3, -1/3, -1/3, 1/3,
                     1/3, -1/3, -1/3, 1/3)
  flat_avg    <- contrast(emm_main, method = list("avg_flattening" = flat_avg_vec))
  flat_avg_df <- add_CI(as.data.frame(summary(flat_avg, infer = TRUE)))
  cat("\nAverage flattening across 3 self layers:\n")
  print(flat_avg_df, row.names = FALSE)
  avg_flat_b <- flat_avg_df$estimate[1]
  avg_flat_p <- flat_avg_df$p.value[1]

  cat("\n--- 4d: Gap = 0 at verum post? ---\n")
  meet_df <- gaps_df[gaps_df$group == "verum" & gaps_df$session == "ses-02", ]
  print(meet_df[, intersect(c("contrast", "estimate", "SE", "df",
                               "lower.CL", "upper.CL", "t.ratio", "p.value"),
                             names(meet_df))], row.names = FALSE)

  write.csv(gaps_df, OUT_EMM,  row.names = FALSE); cat(sprintf("Saved: %s\n", OUT_EMM))

  flat_save            <- flat_df
  flat_save$avg_flat_b <- NA; flat_save$avg_flat_p <- NA
  flat_save$avg_flat_b[1] <- avg_flat_b; flat_save$avg_flat_p[1] <- avg_flat_p
  write.csv(flat_save, OUT_FLAT, row.names = FALSE); cat(sprintf("Saved: %s\n", OUT_FLAT))

  # ── Part 5: Residual diagnostics ────────────────────────────────────────────
  cat("\n", SEP, "\nPART 5 — Residual diagnostics\n", SEP, "\n", sep = "")

  resids  <- resid(m)
  fitted_ <- fitted(m)
  sk <- skewness(resids); ku <- kurtosis(resids)
  cat(sprintf("Skewness: %.4f   Kurtosis: %.4f  (excess %.4f)\n", sk, ku, ku - 3))
  set.seed(42); sw_idx <- sample(length(resids), min(5000L, length(resids)))
  sw <- shapiro.test(resids[sw_idx])
  cat(sprintf("Shapiro-Wilk (n=%d): W=%.5f  p=%.4e\n",
              length(sw_idx), sw$statistic, sw$p.value))
  save_diag_fig(OUT_FIG24, resids, fitted_)

  # ── Part 6: Sensitivity — residual trimming ──────────────────────────────────
  cat("\n", SEP, "\nPART 6 — Sensitivity: residual trimming (|resid| > 2.5 SD)\n",
      SEP, "\n", sep = "")

  resid_sd  <- sd(resids)
  resid_thr <- 2.5 * resid_sd
  flagged   <- abs(resids) > resid_thr
  n_flagged <- sum(flagged); n_total <- length(resids)
  cat(sprintf("Residual SD = %.4f   Threshold = ±%.4f (2.5 × SD)\n",
              resid_sd, resid_thr))
  cat(sprintf("Flagged: %d / %d observations (%.2f%%)\n\n",
              n_flagged, n_total, 100 * n_flagged / n_total))

  df_trim  <- df[!flagged, ]
  m_trim   <- try_fit(FORMULA_FULL, "trim/full-RE", df_trim)
  if (!is.null(m_trim) && isSingular(m_trim)) {
    cat("  Singular — retrying with (1|subject)+(1|roi_uid)...\n")
    m_trim <- try_fit(FORMULA_SIMP, "trim/simple-RE", df_trim)
  }
  if (is.null(m_trim)) stop("Trimmed model fit failed.")
  conv_trim <- m_trim@optinfo$conv$lme4$messages
  cat(if (!is.null(conv_trim))
        sprintf("  Convergence: %s\n", paste(conv_trim, collapse = "; "))
      else "  Convergence: OK\n")

  TERMS <- c(
    "Exteroception" = "groupverum:sessionses-02:self_layerExteroception",
    "Interoception" = "groupverum:sessionses-02:self_layerInteroception",
    "Cognition"     = "groupverum:sessionses-02:self_layerCognition"
  )

  emm_options(lmerTest.limit = nrow(df_trim), pbkrtest.limit = nrow(df_trim))
  emm_trim         <- emmeans(m_trim, ~ group * session * self_layer)
  flat_avg_trim    <- contrast(emm_trim, method = list("avg_flattening" = flat_avg_vec))
  flat_avg_trim_df <- add_CI(as.data.frame(summary(flat_avg_trim, infer = TRUE)))
  avg_flat_b_trim  <- flat_avg_trim_df$estimate[1]
  avg_flat_p_trim  <- flat_avg_trim_df$p.value[1]

  fe_trim <- coef(summary(m_trim))
  cat(sprintf("\n%-30s  %-22s  %-22s\n", "",
              "Full data", sprintf("Trimmed (|e| < 2.5 SD)")))
  cat(sprintf("%-30s  %-22s  %-22s\n", "",
              sprintf("N = %d", n_total), sprintf("N = %d", nrow(df_trim))))
  cat(strrep("-", 78), "\n")
  for (nm in names(TERMS)) {
    full_v <- get_fe(as.matrix(coef(summary(m))), TERMS[[nm]])
    trim_v <- get_fe(as.matrix(fe_trim),          TERMS[[nm]])
    cat(sprintf("%-30s  β=%8.5f  p=%7.4g    β=%8.5f  p=%7.4g%s\n",
                paste0("group:session:", nm),
                full_v["Estimate"], full_v["Pr(>|t|)"],
                trim_v["Estimate"], trim_v["Pr(>|t|)"],
                if (!is.na(trim_v["Pr(>|t|)"]) && trim_v["Pr(>|t|)"] < 0.05) " *" else ""))
  }
  cat(sprintf("%-30s  β=%8.5f  p=%7.4g    β=%8.5f  p=%7.4g%s\n",
              "Average flattening",
              avg_flat_b, avg_flat_p,
              avg_flat_b_trim, avg_flat_p_trim,
              if (!is.na(avg_flat_p_trim) && avg_flat_p_trim < 0.05) " *" else ""))
  cat(strrep("-", 78), "\n")

  save_diag_fig(OUT_FIG_TR, resid(m_trim), fitted(m_trim))

  # ── Save FE table ────────────────────────────────────────────────────────────
  write.csv(fe_tbl, OUT_FE, row.names = FALSE); cat(sprintf("Saved: %s\n", OUT_FE))

  cat("\n", SEP, "\nSUMMARY\n", SEP, "\n", sep = "")
  cat(sprintf(
    "Average flattening (self-nonself gap change under DMT): β=%.5f, p=%.4g.\n",
    avg_flat_b, avg_flat_p))

}  # atlas_method branch

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
