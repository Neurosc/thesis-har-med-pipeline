# exploratory/ie_gap_threeway.R
#
# Interoception vs Exteroception gap: does it change with drug × session?
#
# Approach: subset model_ready.csv to Interoception + Exteroception rows only,
# fit auc ~ group * session * self_layer + (1+session|subject) + (1|roi_uid),
# then use emmeans to extract:
#   (1) Intero-Extero gap at each group × session cell
#   (2) How the gap changes pre→post within each group (simple effects of session on gap)
#   (3) Whether the gap change differs between groups (three-way interaction = key test)
#
# self_layer levels in this subset: "Exteroception" (ref), "Interoception"
# group   : "placebo" (ref), "verum"
# session : "ses-01"  (ref = pre), "ses-02" (= post)
# → group:session:self_layerInteroception = change in IE gap (verum post-pre) − (placebo post-pre)
#
# Run from repo root:
#   Rscript 04_statistics/scripts/exploratory/ie_gap_threeway.R

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
OUT_CSV    <- file.path(TABLES_DIR, "ie_gap_threeway.csv")

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\nie_gap_threeway.R — config: ", TAG, "\n", SEP, "\n\n", sep = "")

if (!file.exists(DATA_CSV))
  stop("Input not found — run 01_build_dataframe.jl + 02_lmm.R first: ", DATA_CSV)
if (ATLAS_METHOD != "parcels")
  stop("Requires atlas_method = 'parcels'.")

EXCLUDED <- c("sub-06", "sub-08", "sub-12", "sub-26", "sub-36")

# ── Load & subset ───────────────────────────────────────────────────────────────
df_raw <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
df_raw <- df_raw[!df_raw$subject %in% EXCLUDED, ]

df_ie <- df_raw[df_raw$self_layer %in% c("Interoception", "Exteroception"), ]
df_ie$self_layer <- relevel(factor(df_ie$self_layer), ref = "Exteroception")
df_ie$group      <- relevel(factor(df_ie$group),      ref = "placebo")
df_ie$session    <- relevel(factor(df_ie$session),    ref = "ses-01")
df_ie$subject    <- factor(df_ie$subject)
df_ie$roi_uid    <- factor(droplevels(df_ie$roi_uid))

cat(sprintf("Subset: %d rows | %d subjects | %d ROIs (%d Intero, %d Extero)\n\n",
            nrow(df_ie),
            length(unique(df_ie$subject)),
            nlevels(df_ie$roi_uid),
            sum(df_ie$self_layer == "Interoception") / length(unique(df_ie$subject)) /
              length(unique(df_ie$session)),
            sum(df_ie$self_layer == "Exteroception") / length(unique(df_ie$subject)) /
              length(unique(df_ie$session))))

ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5),
                    check.conv.grad = .makeCC("warning", tol = 0.002))

# ── Fit model (full RE first, fall back if singular) ────────────────────────────
cat(SEP, "\nFitting: auc ~ group * session * self_layer + (1+session|subject) + (1|roi_uid)\n",
    SEP, "\n", sep = "")

m <- tryCatch(
  suppressWarnings(
    lmer(auc ~ group * session * self_layer + (1 + session | subject) + (1 | roi_uid),
         data = df_ie, REML = TRUE, control = ctrl)
  ),
  error = function(e) { cat(sprintf("Full RE failed: %s\n", e$message)); NULL }
)

if (is.null(m) || isSingular(m)) {
  cat("Singular or failed — falling back to (1|subject) + (1|roi_uid)\n")
  m <- suppressWarnings(
    lmer(auc ~ group * session * self_layer + (1 | subject) + (1 | roi_uid),
         data = df_ie, REML = TRUE, control = ctrl)
  )
}

conv <- m@optinfo$conv$lme4$messages
cat(if (!is.null(conv)) sprintf("Convergence: %s\n", paste(conv, collapse = "; "))
    else "Convergence: OK\n")
if (isSingular(m)) cat("WARNING: singular fit\n")

cat("\nFixed effects:\n")
print(coef(summary(m)))

# ── emmeans: gap at each group × session cell ───────────────────────────────────
cat("\n", SEP, "\nEMMEANS — Interoception − Exteroception gap\n", SEP, "\n", sep = "")

emm_options(lmerTest.limit = nrow(df_ie), pbkrtest.limit = nrow(df_ie))
emm <- emmeans(m, ~ self_layer | group + session)

# Gap (Intero - Extero) at each of the 4 cells
gap <- contrast(emm, list("Intero-Extero" = c(-1, 1)), adjust = "none")
gap_df <- as.data.frame(summary(gap, infer = TRUE))
cat("\nIntero−Extero gap per group × session:\n")
print(gap_df[, c("group", "session", "estimate", "SE", "df", "t.ratio", "p.value")])

# Change in gap pre→post within each group
gap_change <- contrast(gap, "consec", by = c("contrast", "group"), adjust = "none")
gap_change_df <- as.data.frame(summary(gap_change, infer = TRUE))
cat("\nChange in gap (post−pre) within each group:\n")
print(gap_change_df[, c("group", "estimate", "SE", "df", "t.ratio", "p.value")])

# Three-way: does the gap change differ between groups? (key test)
threeway <- contrast(gap_change, "consec", by = "contrast", adjust = "none")
threeway_df <- as.data.frame(summary(threeway, infer = TRUE))
cat("\nThree-way: (verum gap change) − (placebo gap change):\n")
print(threeway_df[, c("estimate", "SE", "df", "t.ratio", "p.value")])

# ── Tidy output table ────────────────────────────────────────────────────────────
out_rows <- list()

# Gap at each cell
for (i in seq_len(nrow(gap_df))) {
  r <- gap_df[i, ]
  out_rows[[length(out_rows) + 1]] <- data.frame(
    model = "ie_gap_threeway",
    term  = paste0("gap_", r$group, "_", r$session),
    estimate = r$estimate, SE = r$SE, df = r$df,
    t = r$t.ratio, p = r$p.value, stringsAsFactors = FALSE)
}

# Gap change within each group
for (i in seq_len(nrow(gap_change_df))) {
  r <- gap_change_df[i, ]
  out_rows[[length(out_rows) + 1]] <- data.frame(
    model = "ie_gap_threeway",
    term  = paste0("gap_change_postpre_", r$group),
    estimate = r$estimate, SE = r$SE, df = r$df,
    t = r$t.ratio, p = r$p.value, stringsAsFactors = FALSE)
}

# Three-way
out_rows[[length(out_rows) + 1]] <- data.frame(
  model = "ie_gap_threeway",
  term  = "threeway_verum_vs_placebo_gap_change",
  estimate = threeway_df$estimate[1], SE = threeway_df$SE[1],
  df = threeway_df$df[1], t = threeway_df$t.ratio[1],
  p = threeway_df$p.value[1], stringsAsFactors = FALSE)

out_tbl <- do.call(rbind, out_rows)
rownames(out_tbl) <- NULL
write.csv(out_tbl, OUT_CSV, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", OUT_CSV))

# ── Console summary ───────────────────────────────────────────────────────────────
cat("\n", SEP, "\nSUMMARY\n", SEP, "\n", sep = "")
cat("Intero−Extero gap at each cell (positive = Intero > Extero):\n")
for (i in seq_len(nrow(gap_df))) {
  r <- gap_df[i, ]
  cat(sprintf("  %-10s %-8s  gap = %7.4f  SE = %.4f  p = %.4g\n",
              r$group, r$session, r$estimate, r$SE, r$p.value))
}
cat("\nGap change post−pre within each group:\n")
for (i in seq_len(nrow(gap_change_df))) {
  r <- gap_change_df[i, ]
  cat(sprintf("  %-10s  Δgap = %7.4f  SE = %.4f  p = %.4g%s\n",
              r$group, r$estimate, r$SE, r$p.value,
              if (!is.na(r$p.value) && r$p.value < 0.05) "  *" else ""))
}
cat("\nKEY TEST — (verum gap change) − (placebo gap change):\n")
tw <- out_tbl[out_tbl$term == "threeway_verum_vs_placebo_gap_change", ]
cat(sprintf("  β = %.6f  SE = %.6f  df = %.1f  t = %.4f  p = %.4g  %s\n",
            tw$estimate, tw$SE, tw$df, tw$t, tw$p,
            if (!is.na(tw$p) && tw$p < 0.05) "SIGNIFICANT *" else "not significant"))
cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
