# 05_per_category_glasser.R — Internal–external ACW difference analysis
#
# Computes subject-level Interoception−Exteroception (IE_diff_1) and
# Interoception−Sensory-Motor (IE_diff_2) difference scores from the
# parcels_NoGSR model-ready data, then tests for session effects and
# drug × session interactions.
#
# Run from repo root:
#   Rscript 04_statistics/scripts/05_per_category_glasser.R

required_pkgs <- c("dplyr", "tidyr", "RcppTOML")
for (pkg in required_pkgs) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
}
suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(RcppTOML)
})

args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
  REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", ".."))
} else {
  REPO_ROOT <- normalizePath(".")
}

TAG       <- "parcels_NoGSR"
TABLES    <- file.path(REPO_ROOT, "04_statistics", "results", TAG, "tables")
MODEL_CSV <- file.path(TABLES, "model_ready.csv")
SEP       <- paste(rep("=", 72), collapse = "")

if (!file.exists(MODEL_CSV)) stop("Missing input: ", MODEL_CSV)

df <- read.csv(MODEL_CSV, stringsAsFactors = FALSE)
df$group   <- factor(df$group,   levels = c("placebo", "verum"))
df$session <- factor(df$session, levels = c("ses-01", "ses-02"))

cat(SEP, "\n05_per_category_glasser.R — Internal–External ACW Differences\n",
    SEP, "\n\n", sep = "")
cat(sprintf("Input: %s  (%d rows)\n", MODEL_CSV, nrow(df)))
cat(sprintf("Subjects: %d    Sessions: %s\n\n",
            length(unique(df$subject)), paste(levels(df$session), collapse = ", ")))

# ── Step 1: Subject × session means per layer ────────────────────────────────
layer_means <- df %>%
  filter(self_layer %in% c("Interoception", "Exteroception", "nonself")) %>%
  group_by(subject, session, group, self_layer) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = self_layer, values_from = mean_auc)

# Verify expected ROI counts
n_rois <- df %>%
  filter(self_layer %in% c("Interoception", "Exteroception", "nonself")) %>%
  group_by(self_layer) %>%
  summarise(n_roi = n_distinct(glasser_id), .groups = "drop")
cat("ROIs per layer (total across both sessions):\n")
for (i in seq_len(nrow(n_rois)))
  cat(sprintf("  %-16s %d\n", n_rois$self_layer[i], n_rois$n_roi[i]))
cat("\n")

# ── Step 2: Difference scores ────────────────────────────────────────────────
diff_df <- layer_means %>%
  mutate(
    IE_diff_1 = Interoception - Exteroception,
    IE_diff_2 = Interoception - nonself
  )

# ── Helpers ──────────────────────────────────────────────────────────────────
cohens_d_paired   <- function(x, y) { d <- x - y; mean(d, na.rm=TRUE) / sd(d, na.rm=TRUE) }
cohens_d_unpaired <- function(x, y) {
  n1 <- sum(!is.na(x)); n2 <- sum(!is.na(y))
  sp <- sqrt(((n1-1)*var(x, na.rm=TRUE) + (n2-1)*var(y, na.rm=TRUE)) / (n1+n2-2))
  (mean(x, na.rm=TRUE) - mean(y, na.rm=TRUE)) / sp
}
fmt_p <- function(p) {
  if (is.na(p)) return("n.a.")
  if (p < 0.001) return("< 0.001")
  sprintf("%.3f", p)
}
fmt_t <- function(t, df_val) sprintf("t(%g) = %.3f", round(df_val, 1), t)

# ── Step 3: Tests ────────────────────────────────────────────────────────────
run_tests <- function(diff_col) {
  pre  <- diff_df[diff_df$session == "ses-01", c("subject", "group", diff_col)]
  post <- diff_df[diff_df$session == "ses-02", c("subject", diff_col)]
  names(pre)[3] <- "val_pre"; names(post)[2] <- "val_post"
  both <- merge(pre, post, by = "subject")

  # a. Session effect (all subjects, paired)
  t_ses  <- t.test(both$val_post, both$val_pre, paired = TRUE)
  d_ses  <- cohens_d_paired(both$val_post, both$val_pre)

  # b. Drug × session: unpaired t-test on per-subject deltas (post − pre)
  both$delta    <- both$val_post - both$val_pre
  plac_delta    <- both$delta[both$group == "placebo"]
  verm_delta    <- both$delta[both$group == "verum"]
  t_inter       <- t.test(verm_delta, plac_delta, var.equal = FALSE)
  d_inter       <- cohens_d_unpaired(verm_delta, plac_delta)

  # b2. ANOVA: diff ~ group * session + Error(subject/session)
  aov_fit   <- tryCatch(
    aov(as.formula(paste0(diff_col, " ~ group * session + Error(subject/session)")),
        data = diff_df),
    error = function(e) NULL)
  aov_inter_p <- tryCatch({
    ws <- summary(aov_fit)[["Error: subject:session"]][[1]]
    ws["group:session", "Pr(>F)"]
  }, error = function(e) NA_real_)

  # c. Within-group pre→post (paired t)
  bp <- both[both$group == "placebo", ]
  bv <- both[both$group == "verum",   ]
  t_plac <- t.test(bp$val_post, bp$val_pre, paired = TRUE)
  t_verm <- t.test(bv$val_post, bv$val_pre, paired = TRUE)
  d_plac <- cohens_d_paired(bp$val_post, bp$val_pre)
  d_verm <- cohens_d_paired(bv$val_post, bv$val_pre)

  list(t_ses = t_ses, d_ses = d_ses,
       t_inter = t_inter, d_inter = d_inter, aov_inter_p = aov_inter_p,
       t_plac = t_plac, d_plac = d_plac,
       t_verm = t_verm, d_verm = d_verm)
}

res1 <- run_tests("IE_diff_1")
res2 <- run_tests("IE_diff_2")

# ── Print detailed tests ─────────────────────────────────────────────────────
print_tests <- function(label, r) {
  cat(SEP, "\n", label, "\n", SEP, "\n", sep = "")
  cat(sprintf("  a) Session effect (all subjects, paired t):\n"))
  cat(sprintf("     %s  p = %s  d = %.3f\n",
              fmt_t(r$t_ses$statistic, r$t_ses$parameter),
              fmt_p(r$t_ses$p.value), r$d_ses))
  cat(sprintf("  b) Drug × Session (unpaired t on Δ per subject):\n"))
  cat(sprintf("     %s  p = %s  d = %.3f\n",
              fmt_t(r$t_inter$statistic, r$t_inter$parameter),
              fmt_p(r$t_inter$p.value), r$d_inter))
  cat(sprintf("     ANOVA interaction p = %s\n", fmt_p(r$aov_inter_p)))
  cat(sprintf("  c) Placebo pre→post (paired t):\n"))
  cat(sprintf("     %s  p = %s  d = %.3f\n",
              fmt_t(r$t_plac$statistic, r$t_plac$parameter),
              fmt_p(r$t_plac$p.value), r$d_plac))
  cat(sprintf("  c) Verum pre→post (paired t):\n"))
  cat(sprintf("     %s  p = %s  d = %.3f\n\n",
              fmt_t(r$t_verm$statistic, r$t_verm$parameter),
              fmt_p(r$t_verm$p.value), r$d_verm))
}

print_tests("IE_diff_1 : Interoception − Exteroception", res1)
print_tests("IE_diff_2 : Interoception − Sensory-Motor",  res2)

# ── Step 4: Summary table ────────────────────────────────────────────────────
cell_stat <- function(col, grp, ses) {
  v <- diff_df[[col]][diff_df$group == grp & diff_df$session == ses]
  sprintf("%.3f ± %.3f", mean(v, na.rm=TRUE), sd(v, na.rm=TRUE))
}

cat(SEP, "\nSUMMARY TABLE\n", SEP, "\n", sep = "")
cat(sprintf("  %-28s  %-18s  %-18s\n",
            "", "IE_diff_1", "IE_diff_2"))
cat(sprintf("  %-28s  %-18s  %-18s\n",
            "", "(Intro − Extero)", "(Intro − SensMotor)"))
cat(strrep("-", 70), "\n")
for (r in list(
    list("Placebo pre  (mean±SD)", "placebo", "ses-01"),
    list("Placebo post (mean±SD)", "placebo", "ses-02"),
    list("Verum pre    (mean±SD)", "verum",   "ses-01"),
    list("Verum post   (mean±SD)", "verum",   "ses-02"))) {
  cat(sprintf("  %-28s  %-18s  %-18s\n", r[[1]],
              cell_stat("IE_diff_1", r[[2]], r[[3]]),
              cell_stat("IE_diff_2", r[[2]], r[[3]])))
}
cat(strrep("-", 70), "\n")
cat(sprintf("  %-28s  %-18s  %-18s\n",
            "Session effect p",
            fmt_p(res1$t_ses$p.value), fmt_p(res2$t_ses$p.value)))
cat(sprintf("  %-28s  %-18s  %-18s\n",
            "Drug × session p (t-test)",
            fmt_p(res1$t_inter$p.value), fmt_p(res2$t_inter$p.value)))
cat(sprintf("  %-28s  %-18s  %-18s\n",
            "Drug × session p (ANOVA)",
            fmt_p(res1$aov_inter_p), fmt_p(res2$aov_inter_p)))
cat(sprintf("  %-28s  %-18s  %-18s\n",
            "Placebo pre→post p",
            fmt_p(res1$t_plac$p.value), fmt_p(res2$t_plac$p.value)))
cat(sprintf("  %-28s  %-18s  %-18s\n",
            "Verum pre→post p",
            fmt_p(res1$t_verm$p.value), fmt_p(res2$t_verm$p.value)))

cat("\nDone.\n")
