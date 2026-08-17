# 05_supplementary_tables.R — three supplementary tables (AUC), one per effect category,
# comparing the effect across all four pipeline × atlas configurations.
#
# Effect categories (one table each):
#   drug    — DiD (session:arm interaction)           <- _did.csv     (effect, dz, q)
#   time    — overall session main effect (post-pre)  <- _overall.csv (session_dz, session_q)
#   placebo — within-placebo post-pre simple effect   <- _simple.csv  (arm=placebo: dz, q)
#
# Each table: rows = 5 regions (ordered by mean |dz| across configs), columns = the 4 configs.
#   Each cell = dz, with a check mark when that effect survives BH-FDR (q<0.05) in that config.
#   Configs: Sphere·max (PRIMARY), Sphere·GLM, Parcel·max, Parcel·GLM.
#   (detrend is excluded from the stats stage; metric = AUC only.)
#
# Reads : 04_statistics/results/mixed_models/per_config/{atlas}_{pipeline}_{did,overall,simple}.csv
# Writes: 04_statistics/results/mixed_models/supplementary/supp_{drug,time,placebo}_effect.{csv,tex}
# Run   : Rscript 04_statistics/scripts/qin/05_supplementary_tables.R

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO <- local({
  m <- grep("^--file=", commandArgs(FALSE), value = TRUE)
  f <- if (length(m)) sub("^--file=", "", m[1]) else NULL
  if (is.null(f)) normalizePath(getwd())
  else normalizePath(file.path(dirname(normalizePath(f)), "..", "..", "..")) })
PC  <- file.path(REPO, "04_statistics", "results", "mixed_models", "per_config")
OUT <- file.path(REPO, "04_statistics", "results", "mixed_models", "supplementary")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)

# ── Configs (columns) and regions (rows) ───────────────────────────────────────
# `atlas` here is the per_config FILENAME token (matches the results tree and 03_mixed_models.R's
# ATLAS_TAG), not the qinspheres|qinparcels CLI argument.
CONFIGS <- list(
  list(atlas = "spheres",              pipe = "maximal", csv = "Sphere.max", tex = "Sphere $\\cdot$ max", primary = TRUE),
  list(atlas = "parcels_self_regions", pipe = "maximal", csv = "Parcel.max", tex = "Parcel $\\cdot$ max", primary = FALSE))
REGION_ORDER  <- c("intero", "extero", "cognition", "visual", "auditory")
REGION_LABELS <- c(intero = "Interoception", extero = "Exteroception", cognition = "Cognition",
                   visual = "Visual", auditory = "Auditory")

rd <- function(atlas, pipe, kind)                                  # per-config CSV or NULL
  { f <- file.path(PC, sprintf("%s_%s_%s.csv", atlas, pipe, kind))
    if (file.exists(f)) read.csv(f, stringsAsFactors = FALSE) else NULL }

# Return a per-region (dz, q) frame for one config and one effect category.
pull <- function(cfg, effect) {
  if (effect == "drug") {
    d <- rd(cfg$atlas, cfg$pipe, "did")
    if (is.null(d)) return(NULL)
    data.frame(region = d$region, dz = d$dz, q = d$q)
  } else if (effect == "time") {
    d <- rd(cfg$atlas, cfg$pipe, "overall")
    if (is.null(d)) return(NULL)
    data.frame(region = d$region, dz = d$session_dz, q = d$session_q)
  } else {                                                          # placebo
    d <- rd(cfg$atlas, cfg$pipe, "simple")
    if (is.null(d)) return(NULL)
    d <- d[d$arm == "placebo", ]
    data.frame(region = d$region, dz = d$dz, q = d$q)
  }
}

# ── Assemble dz and q matrices (region × config) for one effect ────────────────
# Returns list(dz, q) as regions-by-configs matrices, rows ordered by mean |dz| desc.
build_matrices <- function(effect) {
  ncfg <- length(CONFIGS)
  dz <- q <- matrix(NA_real_, length(REGION_ORDER), ncfg,
                    dimnames = list(REGION_ORDER, vapply(CONFIGS, function(c) c$csv, character(1))))
  for (i in seq_along(CONFIGS)) {
    p <- pull(CONFIGS[[i]], effect); if (is.null(p)) next
    idx <- match(REGION_ORDER, p$region); dz[, i] <- p$dz[idx]; q[, i] <- p$q[idx]
  }
  ord <- order(rowMeans(abs(dz), na.rm = TRUE), decreasing = TRUE)   # strongest effect first
  list(dz = dz[ord, , drop = FALSE], q = q[ord, , drop = FALSE])
}

# Cell text: dz to 2 dp, with a check mark appended when it survives FDR (q<0.05).
cell <- function(dz, q, check) {
  if (is.na(dz)) return("--")
  mark <- if (!is.na(q) && q < 0.05) paste0(" ", check) else ""
  sprintf("%.2f%s", dz, mark)
}

# ── CSV writer (UTF-8, check mark = ✓) ─────────────────────────────────────────
write_csv_table <- function(m, file) {
  regs <- REGION_LABELS[rownames(m$dz)]
  body <- t(vapply(seq_along(regs), function(r)
    vapply(seq_len(ncol(m$dz)), function(k) cell(m$dz[r, k], m$q[r, k], "✓"), character(1)),
    character(ncol(m$dz))))
  df <- data.frame(Region = unname(regs), body, check.names = FALSE, stringsAsFactors = FALSE)
  names(df)[-1] <- gsub("\\.", " · ", vapply(CONFIGS, function(c) c$csv, character(1)))
  con <- file(file, "w", encoding = "UTF-8"); on.exit(close(con))
  write.csv(df, con, row.names = FALSE)
  cat(sprintf("Wrote: %s\n", file)); df
}

# ── LaTeX writer (booktabs; check mark = \checkmark, needs amssymb) ─────────────
write_latex <- function(m, effect, caption, file) {
  tex_lbl <- vapply(CONFIGS, function(c) if (c$primary) paste0(c$tex, "$^{*}$") else c$tex, character(1))
  regs    <- REGION_LABELS[rownames(m$dz)]; ncfg <- ncol(m$dz)
  con <- file(file, "w"); on.exit(close(con))
  wl <- function(...) cat(..., "\n", sep = "", file = con)
  wl("% ", caption)
  wl("\\begin{table}[htbp]\\centering")
  wl("\\caption{", caption, "}")
  wl("\\label{tab:supp_", effect, "_auc}")
  wl("\\small\\setlength{\\tabcolsep}{6pt}")
  wl("\\begin{tabular}{l", strrep("c", ncfg), "}")
  wl("\\toprule")
  wl("Region & ", paste(tex_lbl, collapse = " & "), " \\\\")
  wl("\\midrule")
  for (r in seq_along(regs)) {
    cells <- vapply(seq_len(ncfg), function(k) cell(m$dz[r, k], m$q[r, k], "\\checkmark"), character(1))
    wl(regs[r], " & ", paste(cells, collapse = " & "), " \\\\")
  }
  wl("\\bottomrule")
  wl("\\end{tabular}")
  wl("\\begin{flushleft}\\footnotesize $^{*}$Primary configuration. Each cell is Cohen's $d_z$ ",
     "(within-subject standardized effect; for the drug effect, the between-arm standardized ",
     "difference in change); \\checkmark\\ = survives BH-FDR across the 5 regions ($q<0.05$). ",
     "Rows ordered by mean $|d_z|$. $n=35$; motor excluded.\\end{flushleft}")
  wl("\\end{table}")
  cat(sprintf("Wrote: %s\n", file))
}

# ── Build, write CSV + LaTeX for all three categories ──────────────────────────
SPECS <- list(
  drug    = "AUC drug effect (DiD, session $\\times$ arm) --- robustness across pipeline and ROI ($d_z$, \\checkmark\\ = survives FDR)",
  time    = "AUC time effect (post$-$pre) --- robustness across pipeline and ROI ($d_z$, \\checkmark\\ = survives FDR)",
  placebo = "AUC placebo effect (within-placebo post$-$pre) --- robustness across pipeline and ROI ($d_z$, \\checkmark\\ = survives FDR)")

for (effect in names(SPECS)) {
  m <- build_matrices(effect)
  df <- write_csv_table(m, file.path(OUT, sprintf("supp_%s_effect.csv", effect)))
  write_latex(m, effect, SPECS[[effect]], file.path(OUT, sprintf("supp_%s_effect.tex", effect)))
  cat(sprintf("\n==== %s effect (AUC) ====\n", toupper(effect)))
  print(df, row.names = FALSE)
}
cat("\ndone.\n")
