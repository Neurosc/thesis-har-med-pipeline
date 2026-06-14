# exploratory/ie_diff_prepost_3panel.R
#
# EXPLORATORY, ALL NON-SIGNIFICANT — presented honestly (actual p-values, no
# significance stars, no language implying an effect).
#
# 3-panel paired pre→post figure of the subject-level IE_diff results from
# 05_per_category_glasser.R, one value per subject per session:
#   IE_diff_1 = mean(Interoception) − mean(Exteroception)
#   IE_diff_2 = mean(Interoception) − mean(Sensory-Motor)            [SM == nonself]
#   IE_diff_3 = mean(Interoception) − mean(Exteroception + Sensory-Motor pooled)
#
# "Sensory-Motor" == the broad `nonself` self_layer (CAT_MAP: nonself ->
# "Sensory-Motor"). For IE_diff_3 the pooled "External" is the parcel-level mean of
# AUC over ALL Exteroception + nonself rows (NOT the average of the two layer means),
# exactly as 05's external_means computes it.
#
# 05 writes no output table (console-only), so the diffs are recomputed here with
# 05's identical filter/pivot/join/diff pipeline. The script self-checks against
# 05's published group×session means and p-values and aborts if they don't reproduce.
#
# Tests reused verbatim from 05_per_category_glasser.R run_tests() — NOT a new model:
#   placebo / verum pre→post : paired t.test(post, pre, paired = TRUE)
#   drug × session           : Welch t.test(verum Δ, placebo Δ, var.equal = FALSE)
#                              (Δ = per-subject post − pre)
#
# Subjects come from model_ready.csv, already filtered to the canonical 35 included
# subjects upstream (01_build_dataframe.jl) — no subject list hardcoded.
#
# Reuses the style/palette/annotation scheme of the 4-panel "Pre → Post AUC change
# by drug group" figure (make_paired_panel() in 06_figures.R).
#
# Run from repo root:
#   Rscript 04_statistics/scripts/exploratory/ie_diff_prepost_3panel.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(patchwork)
  library(RcppTOML)
})

# ── Paths ────────────────────────────────────────────────────────────────────
args     <- commandArgs(trailingOnly = FALSE)
file_arg <- args[grep("--file=", args)]
if (length(file_arg) > 0) {
  SCRIPT_DIR <- dirname(normalizePath(sub("--file=", "", file_arg)))
  REPO_ROOT  <- normalizePath(file.path(SCRIPT_DIR, "..", "..", ".."))
} else {
  REPO_ROOT <- normalizePath(".")
}

cfg              <- parseTOML(file.path(REPO_ROOT, "config.toml"))
ATLAS_METHOD     <- cfg$active$atlas_method
DENOISING_METHOD <- cfg$active$denoising_method
TAG              <- paste0(ATLAS_METHOD, "_", DENOISING_METHOD)

TABLES_DIR <- file.path(REPO_ROOT, "04_statistics", "results", TAG, "tables")
FIGS_DIR   <- file.path(REPO_ROOT, "04_statistics", "results", TAG, "figures")
dir.create(FIGS_DIR, showWarnings = FALSE, recursive = TRUE)

DATA_CSV <- file.path(TABLES_DIR, "model_ready.csv")
if (!file.exists(DATA_CSV))
  stop("Input not found — run 02_lmm.R first: ", DATA_CSV)

SEP <- paste(rep("=", 72), collapse = "")
cat(SEP, "\nie_diff_prepost_3panel.R — config: ", TAG, "\n", SEP, "\n\n", sep = "")

# ── Build IE_diff_{1,2,3} (identical to 05_per_category_glasser.R) ────────────
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n", nrow(df)))

layer_means <- df %>%
  filter(self_layer %in% c("Interoception", "Exteroception", "nonself")) %>%
  group_by(subject, session, group, self_layer) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = self_layer, values_from = mean_auc)

# Pooled External = parcel-level mean over ALL Exteroception + nonself rows
external_means <- df %>%
  filter(self_layer %in% c("Exteroception", "nonself")) %>%
  group_by(subject, session, group) %>%
  summarise(External = mean(auc, na.rm = TRUE), .groups = "drop")

diff_df <- layer_means %>%
  left_join(external_means, by = c("subject", "session", "group")) %>%
  mutate(IE_diff_1 = Interoception - Exteroception,
         IE_diff_2 = Interoception - nonself,
         IE_diff_3 = Interoception - External) %>%
  as.data.frame()

diff_df$group   <- factor(diff_df$group,   levels = c("placebo", "verum"))
diff_df$session <- factor(diff_df$session, levels = c("ses-01", "ses-02"))

cat(sprintf("IE_diff rows: %d (expect 70)   Subjects: %d\n\n",
            nrow(diff_df), length(unique(diff_df$subject))))

# ── Tests (verbatim from 05 run_tests) ───────────────────────────────────────
diff_tests <- function(diff_col) {
  pre  <- diff_df[diff_df$session == "ses-01", c("subject", "group", diff_col)]
  post <- diff_df[diff_df$session == "ses-02", c("subject", diff_col)]
  names(pre)[3] <- "val_pre"; names(post)[2] <- "val_post"
  both <- merge(pre, post, by = "subject")
  both$delta <- both$val_post - both$val_pre
  plac_delta <- both$delta[both$group == "placebo"]
  verm_delta <- both$delta[both$group == "verum"]
  bp <- both[both$group == "placebo", ]; bv <- both[both$group == "verum", ]
  list(p_plac  = t.test(bp$val_post, bp$val_pre, paired = TRUE)$p.value,    # placebo pre→post
       p_verm  = t.test(bv$val_post, bv$val_pre, paired = TRUE)$p.value,    # verum   pre→post
       p_inter = t.test(verm_delta, plac_delta, var.equal = FALSE)$p.value) # drug × session
}

cell_mean <- function(grp, ses, col)
  mean(diff_df[[col]][diff_df$group == grp & diff_df$session == ses], na.rm = TRUE)

# ── Variant spec + self-check against 05's published numbers ──────────────────
variants <- list(
  list(col = "IE_diff_1", title = "Intero − Extero",
       means = list(list("placebo","ses-01", 0.002), list("placebo","ses-02",-0.045),
                    list("verum",  "ses-01", 0.011), list("verum",  "ses-02", 0.029)),
       p = c(plac = 0.640, verum = 0.830, inter = 0.618)),
  list(col = "IE_diff_2", title = "Intero − Sensory-Motor",
       means = list(list("placebo","ses-01",-0.153), list("placebo","ses-02",-0.195),
                    list("verum",  "ses-01",-0.185), list("verum",  "ses-02",-0.121)),
       p = c(plac = 0.666, verum = 0.270, inter = 0.346)),
  list(col = "IE_diff_3", title = "Intero − (Extero+SM)",
       means = list(list("placebo","ses-01",-0.144), list("placebo","ses-02",-0.186),
                    list("verum",  "ses-01",-0.174), list("verum",  "ses-02",-0.112)),
       p = c(plac = 0.656, verum = 0.279, inter = 0.346)))

M_TOL <- 0.002; P_TOL <- 0.005
ptests <- list()
for (v in variants) {
  cat(SEP, "\n", v$title, "  (", v$col, ")\n", SEP, "\n", sep = "")
  for (m in v$means) {
    got <- cell_mean(m[[1]], m[[2]], v$col); exp <- m[[3]]
    cat(sprintf("  %-8s %-7s  mean = %+.3f  (05 ref %+.3f)\n", m[[1]], m[[2]], got, exp))
    if (abs(got - exp) > M_TOL)
      stop(sprintf("Mean mismatch %s %s/%s: got %.4f, expected %.3f (tol %.3f). 05 not reproduced — stopping.",
                   v$col, m[[1]], m[[2]], got, exp, M_TOL))
  }
  pt <- diff_tests(v$col)
  for (nm in c("plac", "verum", "inter")) {
    got <- switch(nm, plac = pt$p_plac, verum = pt$p_verm, inter = pt$p_inter)
    exp <- v$p[[nm]]
    cat(sprintf("  p_%-6s = %.3f  (05 ref %.3f)\n", nm, got, exp))
    if (abs(got - exp) > P_TOL)
      stop(sprintf("p mismatch %s %s: got %.4f, expected %.3f (tol %.3f). 05 not reproduced — stopping.",
                   v$col, nm, got, exp, P_TOL))
  }
  ptests[[v$col]] <- pt
  cat("\n")
}
cat("All 05 reference values reproduced within tolerance.\n\n")

# ── Shared y-axis geometry (global across all three diffs) ────────────────────
all_vals <- c(diff_df$IE_diff_1, diff_df$IE_diff_2, diff_df$IE_diff_3)
g_min    <- min(all_vals, na.rm = TRUE)
g_max    <- max(all_vals, na.rm = TRUE)
y_span   <- g_max - g_min
y_br1 <- g_max + y_span * 0.05; y_t1 <- y_br1 - y_span * 0.02
y_br2 <- y_br1 + y_span * 0.05; y_t2 <- y_br2 - y_span * 0.02
y_br3 <- max(y_br1, y_br2) + y_span * 0.08; y_t3 <- y_br3 - y_span * 0.02
ylim_bottom <- g_min - y_span * 0.05
ylim_top    <- y_br3 + y_span * 0.08

# ── Panel builder (style = make_paired_panel in 06_figures.R) ────────────────
XPOS <- c("Placebo Pre"  = 1.0, "Placebo Post" = 2.0,
          "Verum Pre"    = 3.2, "Verum Post"   = 4.2)
COND_LEVELS <- c("Placebo Pre", "Verum Pre", "Placebo Post", "Verum Post")
# Honest reporting: actual p-values only, never significance stars.
fmt_p <- function(p) {
  if (is.na(p)) return("n.s.")
  if (p < 0.001) return("p<0.001")
  sprintf("p=%.3f", p)
}

make_panel <- function(diff_col, panel_title, pt, show_y_title) {
  d <- diff_df
  d$val <- d[[diff_col]]
  d$condition <- factor(
    paste0(ifelse(d$group == "placebo", "Placebo", "Verum"), " ",
           ifelse(d$session == "ses-01", "Pre", "Post")),
    levels = COND_LEVELS)
  d$x_pos <- XPOS[as.character(d$condition)]

  mk_lines <- function(grp) {
    pre_l  <- d[d$group == grp & d$session == "ses-01", c("subject", "val", "x_pos")]
    post_l <- d[d$group == grp & d$session == "ses-02", c("subject", "val", "x_pos")]
    merge(pre_l, post_l, by = "subject", suffixes = c("_pre", "_post"))
  }
  plac_segs <- mk_lines("placebo"); verm_segs <- mk_lines("verum")

  p <- ggplot() +
    geom_segment(data = plac_segs,
                 aes(x = x_pos_pre, xend = x_pos_post, y = val_pre, yend = val_post),
                 color = "#CD5C5C", alpha = 0.30, linewidth = 0.5) +
    geom_segment(data = verm_segs,
                 aes(x = x_pos_pre, xend = x_pos_post, y = val_pre, yend = val_post),
                 color = "#4682B4", alpha = 0.30, linewidth = 0.5) +
    geom_point(data = d[d$group == "placebo", ], aes(x = x_pos, y = val),
               color = "#CD5C5C", alpha = 0.60, size = 2, shape = 16) +
    geom_point(data = d[d$group == "verum", ], aes(x = x_pos, y = val),
               color = "#4682B4", alpha = 0.60, size = 2, shape = 16) +
    geom_boxplot(data = d[d$group == "placebo", ],
                 aes(x = x_pos, y = val, group = condition),
                 fill = NA, color = "#CD5C5C", width = 0.22, outlier.shape = NA,
                 linewidth = 0.60) +
    geom_boxplot(data = d[d$group == "verum", ],
                 aes(x = x_pos, y = val, group = condition),
                 fill = NA, color = "#4682B4", width = 0.22, outlier.shape = NA,
                 linewidth = 0.60) +
    scale_x_continuous(breaks = unname(XPOS),
                       labels = c("Placebo\nPre", "Placebo\nPost",
                                  "Verum\nPre", "Verum\nPost"),
                       expand = expansion(add = c(0.5, 0.5))) +
    labs(title = panel_title,
         y = if (show_y_title) "AUC difference (s)" else NULL, x = NULL) +
    theme_minimal(base_size = 11, base_family = "serif") +
    theme(panel.background   = element_rect(fill = "white", color = NA),
          plot.background    = element_rect(fill = "white", color = NA),
          panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
          panel.grid.major.y = element_line(color = "#e8e8e8", linewidth = 0.4),
          plot.title   = element_text(face = "bold", hjust = 0.5, size = 12,
                                      family = "serif"),
          axis.text.x  = element_text(size = 9, lineheight = 0.85),
          axis.title.y = element_text(size = 10), legend.position = "none")

  # annotations at the SHARED global y positions
  p <- p +
    annotate("segment", x=1.0, xend=2.0, y=y_br1, yend=y_br1, color="#CD5C5C", linewidth=0.5) +
    annotate("segment", x=1.0, xend=1.0, y=y_br1, yend=y_t1,  color="#CD5C5C", linewidth=0.5) +
    annotate("segment", x=2.0, xend=2.0, y=y_br1, yend=y_t1,  color="#CD5C5C", linewidth=0.5) +
    annotate("text", x=1.5, y=y_br1+y_span*0.02, label=fmt_p(pt$p_plac),
             size=3, hjust=0.5, color="#CD5C5C") +
    annotate("segment", x=3.2, xend=4.2, y=y_br2, yend=y_br2, color="#4682B4", linewidth=0.5) +
    annotate("segment", x=3.2, xend=3.2, y=y_br2, yend=y_t2,  color="#4682B4", linewidth=0.5) +
    annotate("segment", x=4.2, xend=4.2, y=y_br2, yend=y_t2,  color="#4682B4", linewidth=0.5) +
    annotate("text", x=3.7, y=y_br2+y_span*0.02, label=fmt_p(pt$p_verm),
             size=3, hjust=0.5, color="#4682B4") +
    annotate("segment", x=2.0, xend=4.2, y=y_br3, yend=y_br3, color="black", linewidth=0.5) +
    annotate("segment", x=2.0, xend=2.0, y=y_br3, yend=y_t3, color="black", linewidth=0.5) +
    annotate("segment", x=4.2, xend=4.2, y=y_br3, yend=y_t3, color="black", linewidth=0.5) +
    annotate("text", x=3.1, y=y_br3+y_span*0.02, label=fmt_p(pt$p_inter),
             size=3, hjust=0.5, color="black") +
    coord_cartesian(ylim = c(ylim_bottom, ylim_top))   # identical limits → shared y-axis
  p
}

p1 <- make_panel("IE_diff_1", "Intero − Extero",         ptests[["IE_diff_1"]], show_y_title = TRUE)
p2 <- make_panel("IE_diff_2", "Intero − Sensory-Motor",  ptests[["IE_diff_2"]], show_y_title = FALSE)
p3 <- make_panel("IE_diff_3", "Intero − (Extero+SM)",    ptests[["IE_diff_3"]], show_y_title = FALSE)

fig <- (p1 | p2 | p3) +
  plot_annotation(
    title = "Internal–External AUC differences, pre→post by group (exploratory)",
    theme = theme(plot.title = element_text(face = "plain", hjust = 0.5,
                                            size = 13, family = "serif")))

OUT_FIG <- file.path(FIGS_DIR, "fig_ie_diff_prepost_3panel.png")
ggsave(OUT_FIG, fig, width = 15, height = 5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n", OUT_FIG))

cat("\n", SEP, "\nDONE (exploratory, all non-significant — actual p-values shown, no stars)\n",
    SEP, "\n", sep = "")
