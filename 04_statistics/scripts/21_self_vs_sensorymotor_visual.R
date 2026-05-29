# 21_self_vs_sensorymotor_visual.R
# Step 14: sensory-motor nonself subset + self layers visual comparison
#
# NOTE: task requested filename 20_; script 20 already exists. Using 21_.
#
# Selects Visual1/Visual2/Somatomotor/Auditory ROIs from the nonself atlas
# (via CAB-NP network assignments) and combines them with the self atlas
# (Interoception/Exteroception/Cognition layers) to visualise drug × session
# effects across 4 region categories.
#
# Run from repo root:
#   Rscript 04_statistics/scripts/21_self_vs_sensorymotor_visual.R

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(plotly)
  library(htmlwidgets)
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

AUC_CSV    <- file.path(REPO_ROOT, "04_statistics", "results", "analysis_long_format_auc.csv")
LOOKUP_CSV <- file.path(REPO_ROOT, "04_statistics", "results", "glasser_g1_lookup.csv")
CABNET_TXT <- file.path(REPO_ROOT,
  "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey (1).txt")

OUT_COMBINED <- file.path(REPO_ROOT, "04_statistics", "results", "sensory_motor_self_combined.csv")
OUT_EFFECTS  <- file.path(REPO_ROOT, "04_statistics", "results", "sensory_motor_self_effects.csv")
OUT_FIG18    <- file.path(REPO_ROOT, "04_statistics", "figures",  "fig18_self_vs_sensorymotor_rainfall.html")

# ── Idempotent guard ───────────────────────────────────────────────────────────
if (all(file.exists(c(OUT_COMBINED, OUT_EFFECTS, OUT_FIG18)))) {
  cat("All outputs already exist — nothing to do. Delete to rerun.\n")
  quit(status = 0)
}

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n21_self_vs_sensorymotor_visual.R — Step 14\n", SEP, "\n\n", sep = "")

# ── Hardcoded self atlas layer mapping (37 ROIs, Qin et al. 2020) ─────────────
LAYER_MAP <- data.frame(
  roi_pos_id = 1:37,
  roi_name = c(
    "Interoception_R_Insula","Interoception_L_DACC","Interoception_R_Thalamus",
    "Interoception_R_Parahippocampal","Interoception_L_Parahippocampal",
    "Interoception_L_Insula_1","Interoception_L_Insula_2","Interoception_R_IPL",
    "Interoception_R_SFG","Interoception_L_STG","Interoception_L_Postcentral",
    "Exteroception_R_Fusiform","Exteroception_R_IFG","Exteroception_R_Premotor",
    "Exteroception_R_Insula","Exteroception_L_Fusiform","Exteroception_R_SPL",
    "Exteroception_R_Postcentral","Exteroception_R_IPL","Exteroception_L_Insula",
    "Exteroception_R_IOG","Exteroception_L_IPL","Exteroception_L_SPL",
    "Exteroception_R_Cingulate","Exteroception_L_mPFC",
    "Cognition_L_ACC","Cognition_L_PCC","Cognition_L_Insula",
    "Cognition_L_MTG","Cognition_L_Thalamus","Cognition_L_SFG_1",
    "Cognition_Cingulate","Cognition_L_ITG","Cognition_R_MTG",
    "Cognition_R_Insula","Cognition_L_SFG_2","Cognition_R_Premotor"
  ),
  layer = c(rep("Interoception", 11), rep("Exteroception", 14), rep("Cognition", 12)),
  stringsAsFactors = FALSE
)

# ─────────────────────────────────────────────────────────────────────────────
cat(SEP, "\nPART 1 — Select sensory-motor nonself ROIs\n", SEP, "\n", sep = "")

SM_NETWORKS <- c("Visual1", "Visual2", "Somatomotor", "Auditory")

# Read CAB-NP (tab-separated, Windows line endings)
cabnet <- read.delim(CABNET_TXT, sep = "\t", header = TRUE,
                     stringsAsFactors = FALSE, check.names = FALSE)
# Strip \r from all character columns (Windows CRLF safety)
cabnet[] <- lapply(cabnet, function(x) if (is.character(x)) trimws(x) else x)

# Cortical parcels only (KEYVALUE 1-360) belonging to sensory-motor networks
sm_cab <- cabnet[cabnet$KEYVALUE <= 360 & cabnet$NETWORK %in% SM_NETWORKS,
                 c("KEYVALUE", "HEMISPHERE", "NETWORK", "GLASSERLABELNAME")]
names(sm_cab) <- c("keyvalue", "hemisphere", "network", "roi_name")

cat(sprintf("Sensory-motor ROIs in CAB-NP (cortical only):\n"))
for (net in SM_NETWORKS) {
  cat(sprintf("  %-15s : %d\n", net, sum(sm_cab$network == net)))
}
cat(sprintf("  %-15s : %d\n", "TOTAL", nrow(sm_cab)))

# Load glasser_g1_lookup (already filtered to 258 tSNR-surviving nonself ROIs)
lookup <- read.csv(LOOKUP_CSV, stringsAsFactors = FALSE)
# nonself_pos_id == roi_pos_id in the AUC data (verified in script 15)
names(lookup)[names(lookup) == "nonself_pos_id"] <- "roi_pos_id"

# Join: CAB-NP sensory-motor → lookup
sm_joined <- merge(sm_cab, lookup[, c("roi_pos_id", "roi_name")],
                   by = "roi_name", all.x = TRUE)

n_total_before <- nrow(sm_cab)
n_matched      <- sum(!is.na(sm_joined$roi_pos_id))
n_excluded     <- n_total_before - n_matched

cat(sprintf("\nAfter tSNR-filter join:\n"))
for (net in SM_NETWORKS) {
  n_net <- sum(!is.na(sm_joined$roi_pos_id[sm_joined$network == net]))
  cat(sprintf("  %-15s : %d\n", net, n_net))
}
cat(sprintf("  %-15s : %d\n", "TOTAL matched", n_matched))
cat(sprintf("  Excluded by tSNR : %d\n", n_excluded))

if (n_excluded > 0) {
  excl <- sm_joined$roi_name[is.na(sm_joined$roi_pos_id)]
  cat(sprintf("  Excluded ROIs    : %s\n", paste(excl, collapse = ", ")))
}

sm_rois <- sm_joined[!is.na(sm_joined$roi_pos_id), ]

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 2 — Build combined dataset\n", SEP, "\n", sep = "")

df_all <- read.csv(AUC_CSV, stringsAsFactors = FALSE)
cat(sprintf("Full AUC data: %d rows\n", nrow(df_all)))

# Nonself: filter to sensory-motor ROIs
df_ns_sm <- df_all[df_all$atlas == "nonself" &
                     df_all$roi_pos_id %in% sm_rois$roi_pos_id, ]
df_ns_sm <- merge(df_ns_sm,
                  sm_rois[, c("roi_pos_id", "roi_name", "network")],
                  by = "roi_pos_id", all.x = TRUE)
df_ns_sm$region_category <- "Sensory-Motor"
df_ns_sm$atlas_source    <- "nonself"

# Self: all 37 ROIs with layer assignments
df_self <- df_all[df_all$atlas == "self", ]
df_self <- merge(df_self, LAYER_MAP[, c("roi_pos_id", "roi_name", "layer")],
                 by = "roi_pos_id", all.x = TRUE)
df_self$region_category <- df_self$layer
df_self$atlas_source    <- "self"
df_self$network         <- NA_character_

# Harmonise column names (nonself uses drug_group, consistent with self)
keep_cols <- c("subject", "session", "roi_pos_id", "roi_name",
               "auc", "drug_group", "region_category", "atlas_source")
df_combined <- rbind(df_ns_sm[, keep_cols], df_self[, keep_cols])
df_combined$roi_uid <- paste(df_combined$atlas_source, df_combined$roi_pos_id, sep = "_")

cat("\nRows per region_category:\n")
rc_counts <- table(df_combined$region_category)
for (rc in names(rc_counts)) cat(sprintf("  %-15s : %d\n", rc, rc_counts[rc]))
cat(sprintf("  %-15s : %d\n", "TOTAL", nrow(df_combined)))

expected_sm   <- n_matched * 35 * 2
expected_self <- 37 * 35 * 2
if (nrow(df_combined) != expected_sm + expected_self) {
  cat(sprintf("WARNING: expected %d rows, got %d\n",
              expected_sm + expected_self, nrow(df_combined)))
}

write.csv(df_combined, OUT_COMBINED, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", OUT_COMBINED))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 3 — Per-ROI drug × session effect\n", SEP, "\n", sep = "")

# Per-subject, per-ROI session change (post - pre)
df_change <- df_combined %>%
  mutate(ses_lbl = ifelse(session == "ses-01", "pre", "post")) %>%
  pivot_wider(id_cols   = c(roi_uid, roi_name, atlas_source,
                             region_category, drug_group, subject),
              names_from  = ses_lbl,
              values_from = auc) %>%
  mutate(ses_change = post - pre)

# Per-ROI mean change by drug group, then double difference
roi_effects <- df_change %>%
  group_by(roi_uid, roi_name, atlas_source, region_category, drug_group) %>%
  summarise(mean_change = mean(ses_change, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(id_cols     = c(roi_uid, roi_name, atlas_source, region_category),
              names_from  = drug_group,
              values_from = mean_change) %>%
  mutate(drug_ses_effect = verum - placebo)

cat("Per-category summary (drug × session effect):\n")
ci95 <- function(x) qt(0.975, df = max(length(x) - 1, 1)) * sd(x, na.rm = TRUE) /
                     sqrt(sum(!is.na(x)))

CAT_ORDER <- c("Sensory-Motor", "Interoception", "Exteroception", "Cognition")
for (cat in CAT_ORDER) {
  sub <- roi_effects$drug_ses_effect[roi_effects$region_category == cat]
  cat(sprintf("\n  %s (N=%d)\n", cat, length(sub[!is.na(sub)])))
  cat(sprintf("    mean=%.5f  SD=%.5f  min=%.5f  max=%.5f\n",
              mean(sub, na.rm = TRUE), sd(sub, na.rm = TRUE),
              min(sub, na.rm = TRUE),  max(sub, na.rm = TRUE)))

  top3 <- head(roi_effects[roi_effects$region_category == cat, ][
                 order(-roi_effects$drug_ses_effect[roi_effects$region_category == cat]),
               c("roi_name", "drug_ses_effect")], 3)
  bot3 <- head(roi_effects[roi_effects$region_category == cat, ][
                 order( roi_effects$drug_ses_effect[roi_effects$region_category == cat]),
               c("roi_name", "drug_ses_effect")], 3)
  cat("    Top 3 positive:\n")
  for (i in seq_len(nrow(top3)))
    cat(sprintf("      %s  (%.5f)\n", top3$roi_name[i], top3$drug_ses_effect[i]))
  cat("    Top 3 negative:\n")
  for (i in seq_len(nrow(bot3)))
    cat(sprintf("      %s  (%.5f)\n", bot3$roi_name[i], bot3$drug_ses_effect[i]))
}

write.csv(roi_effects, OUT_EFFECTS, row.names = FALSE)
cat(sprintf("\nSaved: %s\n", OUT_EFFECTS))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 4 — Visualization (fig18)\n", SEP, "\n", sep = "")

CAT_COLORS <- c(
  "Sensory-Motor" = "#888780",
  "Interoception" = "#1f77b4",
  "Exteroception" = "#ff7f0e",
  "Cognition"     = "#2ca02c"
)
CAT_X <- setNames(seq_along(CAT_ORDER), CAT_ORDER)  # numeric positions 1-4

cat_summary <- roi_effects %>%
  group_by(region_category) %>%
  summarise(
    n       = sum(!is.na(drug_ses_effect)),
    mean_fx = mean(drug_ses_effect, na.rm = TRUE),
    ci      = ci95(drug_ses_effect),
    .groups = "drop"
  ) %>%
  mutate(region_category = factor(region_category, levels = CAT_ORDER)) %>%
  arrange(region_category)

fig <- plot_ly()

# Zero reference line
fig <- fig %>%
  add_trace(x = c(0.5, 4.5), y = c(0, 0),
            type = "scatter", mode = "lines",
            line = list(color = "black", dash = "dash", width = 1),
            showlegend = FALSE, hoverinfo = "skip")

# Individual ROI dots (jittered)
set.seed(42)
for (cat in CAT_ORDER) {
  sub  <- roi_effects[roi_effects$region_category == cat & !is.na(roi_effects$drug_ses_effect), ]
  jit  <- runif(nrow(sub), -0.18, 0.18)
  xpos <- CAT_X[[cat]] + jit

  fig <- fig %>%
    add_trace(x = xpos, y = sub$drug_ses_effect,
              type = "scatter", mode = "markers",
              marker = list(size = 6, color = CAT_COLORS[[cat]], opacity = 0.6,
                            line = list(color = "white", width = 0.3)),
              text = sub$roi_name,
              hovertemplate = "%{text}<br>effect = %{y:.4f}<extra></extra>",
              name = cat, legendgroup = cat, showlegend = TRUE)
}

# Mean bar + 95% CI for each category
for (i in seq_len(nrow(cat_summary))) {
  r   <- cat_summary[i, ]
  cat_i <- as.character(r$region_category)
  xc    <- CAT_X[[cat_i]]
  col_i <- CAT_COLORS[[cat_i]]
  n_i   <- r$n
  lbl_i <- sprintf("%s (N=%d)", cat_i, n_i)

  # CI vertical bar
  fig <- fig %>%
    add_trace(x = c(xc, xc), y = c(r$mean_fx - r$ci, r$mean_fx + r$ci),
              type = "scatter", mode = "lines",
              line = list(color = col_i, width = 3),
              showlegend = FALSE, hoverinfo = "skip")

  # Mean horizontal tick (±0.10 wide)
  fig <- fig %>%
    add_trace(x = c(xc - 0.10, xc + 0.10), y = c(r$mean_fx, r$mean_fx),
              type = "scatter", mode = "lines",
              line = list(color = col_i, width = 3),
              showlegend = FALSE, hoverinfo = "skip")

  # CI caps
  for (y_cap in c(r$mean_fx - r$ci, r$mean_fx + r$ci)) {
    fig <- fig %>%
      add_trace(x = c(xc - 0.07, xc + 0.07), y = c(y_cap, y_cap),
                type = "scatter", mode = "lines",
                line = list(color = col_i, width = 2),
                showlegend = FALSE, hoverinfo = "skip")
  }
}

# Dashed trend line connecting category means (left to right)
trend_x <- vapply(CAT_ORDER, function(c) CAT_X[[c]], numeric(1))
trend_y <- vapply(CAT_ORDER, function(c) {
  cat_summary$mean_fx[as.character(cat_summary$region_category) == c]
}, numeric(1))

fig <- fig %>%
  add_trace(x = trend_x, y = trend_y,
            type = "scatter", mode = "lines",
            line = list(color = "grey30", dash = "dash", width = 1.5),
            showlegend = FALSE, hoverinfo = "skip")

# X-axis tick labels with N
tick_labels <- vapply(CAT_ORDER, function(c) {
  n_i <- cat_summary$n[as.character(cat_summary$region_category) == c]
  sprintf("%s<br>(N=%d)", c, n_i)
}, character(1))

fig <- fig %>%
  layout(
    title = list(
      text = "Drug × Session effect: sensory-motor nonself vs self layers",
      x = 0.5
    ),
    xaxis = list(
      tickvals   = 1:4,
      ticktext   = tick_labels,
      title      = "",
      range      = c(0.4, 4.6),
      showgrid   = FALSE
    ),
    yaxis = list(
      title    = "Drug × Session effect on AUC (s)",
      zeroline = FALSE
    ),
    legend = list(x = 1.01, y = 0.95),
    plot_bgcolor  = "white",
    paper_bgcolor = "white"
  )

saveWidget(fig, OUT_FIG18, selfcontained = FALSE,
           title = "Drug x Session: Sensory-Motor vs Self Layers")
cat(sprintf("Saved: %s\n", OUT_FIG18))

# ─────────────────────────────────────────────────────────────────────────────
cat("\n", SEP, "\nPART 5 — Console summary\n", SEP, "\n", sep = "")

cat("\nCategory means (left → right in fig18):\n")
for (cat in CAT_ORDER) {
  m <- cat_summary$mean_fx[as.character(cat_summary$region_category) == cat]
  cat(sprintf("  %-15s : %.5f\n", cat, m))
}

means_ordered <- vapply(CAT_ORDER, function(c)
  cat_summary$mean_fx[as.character(cat_summary$region_category) == c], numeric(1))

monotone_inc <- all(diff(means_ordered) > 0)
monotone_dec <- all(diff(means_ordered) < 0)
cat(sprintf(
  "\nMonotonic trend (Sensory-Motor → Interoception → Exteroception → Cognition)? %s\n",
  if (monotone_inc) "YES (increasing)" else if (monotone_dec) "YES (decreasing)" else "NO"
))

# One-line interpretation
range_fx <- range(means_ordered)
dominant_dir <- if (means_ordered[4] > means_ordered[1]) "larger" else "smaller"
cat(sprintf(
  "\nInterpretation: Drug × session effect ranges from %.4f to %.4f s across the\n4 region categories (sensory-motor nonself → self-cognitive hierarchy).\nEffects are %s at the Cognition end than at the Sensory-Motor end.\n",
  range_fx[1], range_fx[2], dominant_dir
))

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
