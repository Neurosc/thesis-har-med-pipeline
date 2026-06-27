# 02_lmm_per_parcel.R — Per-parcel Group x Session LMM on SampEn.
# Fits one independent model per Glasser parcel (no cross-parcel pooling):
#   sampen ~ group * session + (1 | subject)
# Extracts the group:session interaction term (= drug effect) for each parcel.
# Flags convergence failures; adds Benjamini-Hochberg FDR (p_fdr) across all parcels.
#
# Run from repo root:  Rscript 04_statistics/scripts/parcels_NoGSR/sample_entropy/02_lmm_per_parcel.R

for (pkg in c("lme4", "lmerTest", "dplyr")) {
  if (!requireNamespace(pkg, quietly = TRUE))
    install.packages(pkg, repos = "https://cloud.r-project.org", quiet = TRUE)
}
suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(dplyr)
})

# ── Paths ─────────────────────────────────────────────────────────────────────
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
TABLES_DIR <- file.path(REPO_ROOT, "04_statistics", "results", "parcels_NoGSR", "sampen_tables")

DATA_CSV        <- file.path(TABLES_DIR, "sampen_full_dataframe.csv")
TSNR_TSV        <- file.path(REPO_ROOT,  "excluded_rois_low_tsnr.tsv")
OUT_PER_PARCEL  <- file.path(TABLES_DIR, "perparcel_drug_effect.csv")

if (!file.exists(DATA_CSV))
  stop("Input not found — provide sampen_full_dataframe.csv (frozen artifact; see README): ", DATA_CSV)

SEP <- strrep("=", 70)
cat(SEP, "\n02_lmm_per_parcel.R — parcels_NoGSR\n", SEP, "\n\n", sep = "")

# ── Load and prepare ─────────────────────────────────────────────────────────
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n", nrow(df)))

excl <- read.delim(TSNR_TSV, stringsAsFactors = FALSE)$roi_id
n0   <- nrow(df)
df   <- df[!(df$roi_pos_id %in% excl), ]
cat(sprintf("Dropped %d rows from %d low-tSNR parcels (%d -> %d)\n",
            n0 - nrow(df), length(excl), n0, nrow(df)))

df$group   <- relevel(factor(df$drug_group), ref = "placebo")
df$session <- relevel(factor(df$session),    ref = "ses-01")
df$subject <- factor(df$subject)

cat(sprintf("Subjects: %d  |  Groups: %s  |  Sessions: %s\n",
            nlevels(df$subject),
            paste(levels(df$group),   collapse = "/"),
            paste(levels(df$session), collapse = "/")))

# ── lmer control (same as 03_LMM_sampen.R) ───────────────────────────────────
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5),
                    check.conv.grad = .makeCC("warning", tol = 0.002))

# ── Per-parcel fit helper ─────────────────────────────────────────────────────
# Returns a named list with status, message, term name, estimate, SE, p_value.
fit_parcel <- function(sub_df, ctrl) {
  warns <- character(0)

  m <- tryCatch(
    withCallingHandlers(
      lmer(sampen ~ group * session + (1 | subject),
           data = sub_df, REML = TRUE, control = ctrl),
      warning = function(w) {
        warns <<- c(warns, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    ),
    error = function(e) e
  )

  if (inherits(m, "error")) {
    return(list(status = "error", message = m$message,
                term = NA_character_, estimate = NA_real_, SE = NA_real_, p_value = NA_real_))
  }

  fe  <- coef(summary(m))
  # The Group:Session interaction term contains both predictor names.
  ix  <- which(grepl("group",   rownames(fe), ignore.case = TRUE) &
               grepl("session", rownames(fe), ignore.case = TRUE))

  if (length(ix) == 0) {
    return(list(status = "no_interaction_term", message = NA_character_,
                term = NA_character_, estimate = NA_real_, SE = NA_real_, p_value = NA_real_))
  }

  ix  <- ix[1]
  status <- if (isSingular(m))     "singular" else
            if (length(warns) > 0) "warning"  else "ok"

  list(
    status   = status,
    message  = if (length(warns) > 0) paste(warns, collapse = "; ") else NA_character_,
    term     = rownames(fe)[ix],
    estimate = unname(fe[ix, "Estimate"]),
    SE       = unname(fe[ix, "Std. Error"]),
    p_value  = unname(fe[ix, "Pr(>|t|)"])
  )
}

# ── Loop over parcels ─────────────────────────────────────────────────────────
parcels <- df %>%
  distinct(roi_pos_id, roi_name, self_layer) %>%
  arrange(self_layer, roi_pos_id)

n_parcels <- nrow(parcels)
cat(sprintf("\n%s\nFitting %d parcel models (sampen ~ group * session + (1 | subject))...\n%s\n",
            SEP, n_parcels, SEP))

results <- vector("list", n_parcels)

for (i in seq_len(n_parcels)) {
  pid    <- parcels$roi_pos_id[i]
  pname  <- parcels$roi_name[i]
  layer  <- parcels$self_layer[i]
  sub_df <- df[df$roi_pos_id == pid, ]
  n_obs  <- nrow(sub_df)

  r <- fit_parcel(sub_df, ctrl)

  results[[i]] <- data.frame(
    roi_pos_id = pid,
    roi_name   = pname,
    self_layer = layer,
    n_obs      = n_obs,
    term       = r$term,
    estimate   = r$estimate,
    SE         = r$SE,
    p_value    = r$p_value,
    status     = r$status,
    message    = r$message,
    stringsAsFactors = FALSE
  )

  if (r$status != "ok")
    cat(sprintf("  [%d/%d] %-25s  layer=%-14s  status=%s\n",
                i, n_parcels, pname, layer, r$status))
  else if (i %% 50 == 0)
    cat(sprintf("  [%d/%d] ...\n", i, n_parcels))
}

results_df <- do.call(rbind, results)

# ── Multiple-comparison correction (Benjamini-Hochberg FDR across all parcels) ──
results_df$p_fdr <- NA_real_
ok_p <- !is.na(results_df$p_value)
results_df$p_fdr[ok_p] <- p.adjust(results_df$p_value[ok_p], method = "BH")

# ── Summary ───────────────────────────────────────────────────────────────────
cat(sprintf("\n%s\nSUMMARY\n%s\n", SEP, SEP))
cat("Status counts:\n")
print(table(results_df$status, dnn = NULL))

failed <- results_df[!results_df$status %in% c("ok", "singular", "warning"), ]
if (nrow(failed) > 0) {
  cat(sprintf("\n%d parcel(s) returned error / no_interaction_term:\n", nrow(failed)))
  print(failed[, c("roi_pos_id", "roi_name", "self_layer", "status", "message")],
        row.names = FALSE)
}

cat(sprintf("\nRange of Group:Session estimates: [%.4f, %.4f]\n",
            min(results_df$estimate, na.rm = TRUE),
            max(results_df$estimate, na.rm = TRUE)))
cat(sprintf("Parcels with p < 0.05 (uncorrected): %d / %d\n",
            sum(results_df$p_value < 0.05, na.rm = TRUE),
            sum(!is.na(results_df$p_value))))
cat(sprintf("Parcels with q < 0.05 (BH-FDR):      %d / %d\n",
            sum(results_df$p_fdr < 0.05, na.rm = TRUE),
            sum(!is.na(results_df$p_fdr))))

# ── Write output ──────────────────────────────────────────────────────────────
write.csv(results_df, OUT_PER_PARCEL, row.names = FALSE)
cat(sprintf("\nSaved %d rows -> %s\n", nrow(results_df), OUT_PER_PARCEL))
cat(SEP, "\nDONE\n", SEP, "\n", sep = "")
