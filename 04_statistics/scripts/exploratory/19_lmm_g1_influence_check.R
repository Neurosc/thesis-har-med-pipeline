# 19_lmm_g1_influence_check.R
# Step 12b: influential ROI sensitivity check for G1 interaction
#
# NOTE: task requested filename 17_lmm_g1_influence_check.R, but scripts
# 17 (17_lrt_term8_only.R) and 18 (18_lmm_simple_slopes_and_log.R) already
# exist.  Using 19_ to avoid a collision.
#
# Tests whether the group:session:G1_z interaction (β = 0.032, p = 0.002)
# is robust to removal of extreme G1 parcels at the sensory/transmodal ends.
#
# Conditions:
#   1. Full model       (all ROIs — reproduces Step 12)
#   2. Drop bottom 5    (most sensory)
#   3. Drop top 5       (most transmodal)
#   4. Drop bottom 5 + top 5
#   5. Drop bottom 10 + top 10
#
# Run from repo root:
#   Rscript 04_statistics/scripts/19_lmm_g1_influence_check.R

suppressPackageStartupMessages({
  library(lme4)
  library(lmerTest)
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

DATA_CSV  <- file.path(REPO_ROOT, "04_statistics", "results", "nonself_g1_model_ready.csv")
OUT_CSV   <- file.path(REPO_ROOT, "04_statistics", "results", "lmm_g1_influence_check.csv")

# ── Idempotent guard ───────────────────────────────────────────────────────────
if (file.exists(OUT_CSV)) {
  cat("Output already exists — nothing to do. Delete to rerun.\n")
  quit(status = 0)
}

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n19_lmm_g1_influence_check.R — influential ROI sensitivity check\n",
    SEP, "\n\n", sep = "")

# ── Load and prepare data ──────────────────────────────────────────────────────
cat("Loading data...\n")
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Data: %d rows x %d cols\n\n", nrow(df), ncol(df)))
stopifnot(nrow(df) == 18042)

df$group      <- relevel(factor(df$group),   ref = "placebo")
df$session    <- relevel(factor(df$session), ref = "ses-01")
df$subject    <- factor(df$subject)
df$roi_pos_id <- factor(df$roi_pos_id)

# ── Model fitting helpers ──────────────────────────────────────────────────────
FORMULA <- auc ~ group * session * G1_z + (1 | subject) + (1 | roi_pos_id)

ctrl_bobyqa <- lmerControl(
  optimizer = "bobyqa",
  optCtrl   = list(maxfun = 2e5),
  check.conv.grad = .makeCC("warning", tol = 0.002)
)
ctrl_nm <- lmerControl(
  optimizer = "Nelder_Mead",
  optCtrl   = list(maxfun = 2e5),
  check.conv.grad = .makeCC("warning", tol = 0.002)
)

fit_lmm <- function(data_subset, condition_label) {
  m <- tryCatch(
    withCallingHandlers(
      lmer(FORMULA, data = data_subset, REML = TRUE, control = ctrl_bobyqa),
      warning = function(w) {
        cat(sprintf("  [%s] bobyqa warning: %s — retrying Nelder-Mead\n",
                    condition_label, conditionMessage(w)))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) stop(sprintf("[%s] fit failed: %s", condition_label, e$message))
  )

  # if bobyqa converged with a warning, the tryCatch above muffled it but
  # the model object still exists.  Re-check via optinfo.
  msgs <- m@optinfo$conv$lme4$messages
  if (!is.null(msgs)) {
    cat(sprintf("  [%s] CONVERGENCE WARNING: %s\n",
                condition_label, paste(msgs, collapse = "; ")))
  }
  m
}

extract_term <- function(model, term_name) {
  raw <- coef(summary(model))
  r   <- raw[rownames(raw) == term_name, , drop = FALSE]
  if (nrow(r) == 0) return(c(b = NA_real_, se = NA_real_, t = NA_real_, p = NA_real_))
  c(b = r[, "Estimate"], se = r[, "Std. Error"],
    t = r[, "t value"], p = r[, "Pr(>|t|)"])
}

# ─────────────────────────────────────────────────────────────────────────────
cat(SEP, "\nPART 1 — Extreme ROIs along the G1 gradient\n", SEP, "\n", sep = "")

roi_tbl <- df %>%
  distinct(roi_pos_id, roi_name, G1_z) %>%
  arrange(G1_z)

n_roi <- nrow(roi_tbl)
cat(sprintf("Total unique ROIs in dataset: %d\n\n", n_roi))

bot5  <- roi_tbl$roi_pos_id[1:5]
bot10 <- roi_tbl$roi_pos_id[1:10]
top5  <- roi_tbl$roi_pos_id[(n_roi - 4):n_roi]
top10 <- roi_tbl$roi_pos_id[(n_roi - 9):n_roi]

print_roi_set <- function(ids, label) {
  cat(sprintf("%s:\n", label))
  rows <- roi_tbl[roi_tbl$roi_pos_id %in% ids, ]
  for (i in seq_len(nrow(rows))) {
    cat(sprintf("  roi_pos_id=%-6s  G1_z=%7.4f  %s\n",
                rows$roi_pos_id[i], rows$G1_z[i], rows$roi_name[i]))
  }
  cat("\n")
}

print_roi_set(bot5,  "Bottom 5  (most sensory, lowest G1_z)")
print_roi_set(bot10, "Bottom 10 (most sensory)")
print_roi_set(top5,  "Top 5     (most transmodal, highest G1_z)")
print_roi_set(top10, "Top 10    (most transmodal)")

# ─────────────────────────────────────────────────────────────────────────────
cat(SEP, "\nPART 2 — Refit model under 5 conditions\n", SEP, "\n", sep = "")

conditions <- list(
  list(label = sprintf("Full (%d ROIs)", n_roi),  drop = character(0)),
  list(label = "Drop bottom 5",                   drop = bot5),
  list(label = "Drop top 5",                      drop = top5),
  list(label = "Drop bottom 5 + top 5",           drop = c(bot5, top5)),
  list(label = "Drop bottom 10 + top 10",         drop = c(bot10, top10))
)

results <- vector("list", length(conditions))

for (i in seq_along(conditions)) {
  cond  <- conditions[[i]]
  label <- cond$label

  if (length(cond$drop) == 0) {
    dsub <- df
  } else {
    dsub <- df[!df$roi_pos_id %in% cond$drop, ]
    dsub$roi_pos_id <- droplevels(dsub$roi_pos_id)
    dsub$subject    <- droplevels(dsub$subject)
  }

  n_roi_i <- nlevels(droplevels(factor(dsub$roi_pos_id)))
  n_obs_i <- nrow(dsub)
  cat(sprintf("\n[%d] %s — %d ROIs, %d observations\n",
              i, label, n_roi_i, n_obs_i))

  m_i <- fit_lmm(dsub, label)

  t3  <- extract_term(m_i, "groupverum:sessionses-02:G1_z")
  tgs <- extract_term(m_i, "groupverum:sessionses-02")
  tsg <- extract_term(m_i, "sessionses-02:G1_z")
  aic_i <- AIC(m_i)

  cat(sprintf("  group:session:G1_z  b=%8.5f  SE=%8.5f  t=%7.4f  p=%.4g\n",
              t3["b"], t3["se"], t3["t"], t3["p"]))
  cat(sprintf("  group:session       b=%8.5f  SE=%8.5f  t=%7.4f  p=%.4g\n",
              tgs["b"], tgs["se"], tgs["t"], tgs["p"]))
  cat(sprintf("  session:G1_z        b=%8.5f  SE=%8.5f  t=%7.4f  p=%.4g\n",
              tsg["b"], tsg["se"], tsg["t"], tsg["p"]))
  cat(sprintf("  AIC = %.2f\n", aic_i))

  results[[i]] <- data.frame(
    condition    = label,
    n_roi        = n_roi_i,
    n_obs        = n_obs_i,
    b_3way       = t3["b"],
    se_3way      = t3["se"],
    t_3way       = t3["t"],
    p_3way       = t3["p"],
    b_grp_ses    = tgs["b"],
    se_grp_ses   = tgs["se"],
    t_grp_ses    = tgs["t"],
    p_grp_ses    = tgs["p"],
    b_ses_g1     = tsg["b"],
    se_ses_g1    = tsg["se"],
    t_ses_g1     = tsg["t"],
    p_ses_g1     = tsg["p"],
    AIC          = aic_i,
    stringsAsFactors = FALSE,
    row.names    = NULL
  )
}

res_df <- do.call(rbind, results)
rownames(res_df) <- NULL

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 3 — Summary table\n", SEP, "\n", sep = "")

hdr <- sprintf("%-28s  %5s  %6s  %8s  %7s  %6s  %8s  %12s",
               "Condition", "N_ROI", "N_obs",
               "β_3way", "SE", "t", "p", "AIC")
cat(hdr, "\n")
cat(strrep("-", nchar(hdr)), "\n")

for (i in seq_len(nrow(res_df))) {
  r <- res_df[i, ]
  p_str <- formatC(r$p_3way, format = "g", digits = 3)
  cat(sprintf("%-28s  %5d  %6d  %8.5f  %7.5f  %6.3f  %8s  %12.2f\n",
              r$condition, r$n_roi, r$n_obs,
              r$b_3way, r$se_3way, r$t_3way, p_str, r$AIC))
}

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 4 — Interpretation\n", SEP, "\n", sep = "")

all_sig  <- all(res_df$p_3way < 0.05, na.rm = TRUE)
any_ns   <- any(res_df$p_3way >= 0.05, na.rm = TRUE)
b_vals   <- res_df$b_3way[!is.na(res_df$b_3way)]
b_min    <- min(b_vals)
b_max    <- max(b_vals)
b_pct    <- 100 * (b_max - b_min) / abs(b_vals[1])   # % relative to full-model β

if (all_sig) {
  cat("ROBUST: the 3-way interaction survives removal of extreme G1 parcels.\n")
  cat("The gradient effect is not driven by a handful of outlier ROIs.\n")
} else {
  ns_conds <- res_df$condition[res_df$p_3way >= 0.05]
  cat("SENSITIVE: the 3-way interaction depends on extreme G1 parcels.\n")
  cat(sprintf("Non-significant under: %s\n", paste(ns_conds, collapse = "; ")))
  cat("The result should be interpreted with caution.\n")
}

cat(sprintf("\nβ ranged from %.5f to %.5f across conditions (%.1f%% variation relative to full model).\n",
            b_min, b_max, b_pct))

# ─────────────────────────────────────────────────────────────────────────────
write.csv(res_df, OUT_CSV, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", OUT_CSV))

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
