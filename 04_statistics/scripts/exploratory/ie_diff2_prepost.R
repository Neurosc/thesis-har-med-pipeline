# exploratory/ie_diff2_prepost.R
#
# EXPLORATORY, NON-SIGNIFICANT result — presented honestly (actual p-values,
# no significance stars, no language implying an effect).
#
# Plots the subject-level IE_diff_2 result from 05_per_category_glasser.R:
#   IE_diff_2 = mean(AUC | Interoception) − mean(AUC | Sensory-Motor)
#   one value per subject per session.
# "Sensory-Motor" == the broad `nonself` self_layer (CAT_MAP: nonself ->
# "Sensory-Motor"); IE_diff_2 = Interoception − nonself, exactly as 05 defines it.
#
# 05 writes no output table (console-only), so IE_diff_2 is recomputed here with
# 05's identical filter/pivot/diff pipeline. The script self-checks against 05's
# published group×session means and p-values and aborts if they don't reproduce.
#
# Tests reused verbatim from 05_per_category_glasser.R run_tests() — NOT a new model:
#   placebo pre→post : paired t.test(post, pre, paired = TRUE)         -> p ≈ 0.666
#   verum   pre→post : paired t.test(post, pre, paired = TRUE)         -> p ≈ 0.270
#   drug × session   : Welch t.test(verum Δ, placebo Δ, var.equal = F) -> p ≈ 0.346
#                      (Δ = per-subject post − pre)
#
# Subjects come from model_ready.csv, which is already filtered to the canonical
# 35 included subjects upstream (01_build_dataframe.jl) — no subject list hardcoded.
#
# Reuses the style/palette/annotation scheme of the 4-panel "Pre → Post AUC change
# by drug group" figure (make_paired_panel() in 06_figures.R).
#
# Run from repo root:
#   Rscript 04_statistics/scripts/exploratory/ie_diff2_prepost.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
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
cat(SEP, "\nie_diff2_prepost.R — config: ", TAG, "\n", SEP, "\n\n", sep = "")

# ── Build IE_diff_2 (identical to 05_per_category_glasser.R) ──────────────────
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n", nrow(df)))

diff_df <- df %>%
  filter(self_layer %in% c("Interoception", "nonself")) %>%
  group_by(subject, group, session, self_layer) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = self_layer, values_from = mean_auc) %>%
  mutate(ie_diff2 = Interoception - nonself) %>%
  as.data.frame()

diff_df$group   <- factor(diff_df$group,   levels = c("placebo", "verum"))
diff_df$session <- factor(diff_df$session, levels = c("ses-01", "ses-02"))

cat(sprintf("IE_diff_2 rows: %d (expect 70)   Subjects: %d\n\n",
            nrow(diff_df), length(unique(diff_df$subject))))

# ── Tests (verbatim from 05 run_tests) ───────────────────────────────────────
pre  <- diff_df[diff_df$session == "ses-01", c("subject", "group", "ie_diff2")]
post <- diff_df[diff_df$session == "ses-02", c("subject", "ie_diff2")]
names(pre)[3] <- "val_pre"; names(post)[2] <- "val_post"
both <- merge(pre, post, by = "subject")
both$delta <- both$val_post - both$val_pre

plac_delta <- both$delta[both$group == "placebo"]
verm_delta <- both$delta[both$group == "verum"]
t_inter <- t.test(verm_delta, plac_delta, var.equal = FALSE)   # drug × session
bp <- both[both$group == "placebo", ]; bv <- both[both$group == "verum", ]
t_plac <- t.test(bp$val_post, bp$val_pre, paired = TRUE)        # placebo pre→post
t_verm <- t.test(bv$val_post, bv$val_pre, paired = TRUE)        # verum   pre→post

p_plac  <- t_plac$p.value
p_verm  <- t_verm$p.value
p_inter <- t_inter$p.value

# ── Self-check against 05's published numbers; abort if not reproduced ────────
cell_mean <- function(grp, ses)
  mean(diff_df$ie_diff2[diff_df$group == grp & diff_df$session == ses], na.rm = TRUE)

ref_means <- list(
  list("placebo", "ses-01", -0.153), list("placebo", "ses-02", -0.195),
  list("verum",   "ses-01", -0.185), list("verum",   "ses-02", -0.121))
M_TOL <- 0.002
cat("Group × session means (IE_diff_2 = Intero − Sensory-Motor):\n")
for (rm in ref_means) {
  got <- cell_mean(rm[[1]], rm[[2]]); exp <- rm[[3]]
  cat(sprintf("  %-8s %-7s  mean = %+.3f  (05 ref %+.3f)\n", rm[[1]], rm[[2]], got, exp))
  if (abs(got - exp) > M_TOL)
    stop(sprintf("Mean mismatch for %s/%s: got %.4f, expected %.3f (tol %.3f). 05 not reproduced — stopping.",
                 rm[[1]], rm[[2]], got, exp, M_TOL))
}

P_TOL <- 0.005
ref_p <- list(list("placebo pre→post", p_plac, 0.666),
              list("verum pre→post",   p_verm, 0.270),
              list("drug × session",   p_inter, 0.346))
cat("\nTests (reused from 05):\n")
for (rp in ref_p) {
  cat(sprintf("  %-18s p = %.3f  (05 ref %.3f)\n", rp[[1]], rp[[2]], rp[[3]]))
  if (abs(rp[[2]] - rp[[3]]) > P_TOL)
    stop(sprintf("p mismatch for %s: got %.4f, expected %.3f (tol %.3f). 05 not reproduced — stopping.",
                 rp[[1]], rp[[2]], rp[[3]], P_TOL))
}
cat("\nAll 05 reference values reproduced within tolerance.\n\n")

# ── Figure: single paired panel (style = make_paired_panel in 06_figures.R) ──
XPOS <- c("Placebo Pre"  = 1.0, "Placebo Post" = 2.0,
          "Verum Pre"    = 3.2, "Verum Post"   = 4.2)
COND_LEVELS <- c("Placebo Pre", "Verum Pre", "Placebo Post", "Verum Post")
# Honest reporting: actual p-values only, never significance stars.
fmt_p <- function(p) {
  if (is.na(p)) return("n.s.")
  if (p < 0.001) return("p<0.001")
  sprintf("p=%.3f", p)
}

d <- diff_df
d$condition <- factor(
  paste0(ifelse(d$group == "placebo", "Placebo", "Verum"), " ",
         ifelse(d$session == "ses-01", "Pre", "Post")),
  levels = COND_LEVELS)
d$x_pos <- XPOS[as.character(d$condition)]

make_lines <- function(grp) {
  pre_l  <- d[d$group == grp & d$session == "ses-01", c("subject", "ie_diff2", "x_pos")]
  post_l <- d[d$group == grp & d$session == "ses-02", c("subject", "ie_diff2", "x_pos")]
  merge(pre_l, post_l, by = "subject", suffixes = c("_pre", "_post"))
}
plac_segs <- make_lines("placebo"); verm_segs <- make_lines("verum")

y_max  <- max(d$ie_diff2, na.rm = TRUE)
y_span <- diff(range(d$ie_diff2, na.rm = TRUE))

p <- ggplot() +
  geom_segment(data = plac_segs,
               aes(x = x_pos_pre, xend = x_pos_post, y = ie_diff2_pre, yend = ie_diff2_post),
               color = "#CD5C5C", alpha = 0.30, linewidth = 0.5) +
  geom_segment(data = verm_segs,
               aes(x = x_pos_pre, xend = x_pos_post, y = ie_diff2_pre, yend = ie_diff2_post),
               color = "#4682B4", alpha = 0.30, linewidth = 0.5) +
  geom_point(data = d[d$group == "placebo", ], aes(x = x_pos, y = ie_diff2),
             color = "#CD5C5C", alpha = 0.60, size = 2, shape = 16) +
  geom_point(data = d[d$group == "verum", ], aes(x = x_pos, y = ie_diff2),
             color = "#4682B4", alpha = 0.60, size = 2, shape = 16) +
  geom_boxplot(data = d[d$group == "placebo", ],
               aes(x = x_pos, y = ie_diff2, group = condition),
               fill = NA, color = "#CD5C5C", width = 0.22, outlier.shape = NA,
               linewidth = 0.60) +
  geom_boxplot(data = d[d$group == "verum", ],
               aes(x = x_pos, y = ie_diff2, group = condition),
               fill = NA, color = "#4682B4", width = 0.22, outlier.shape = NA,
               linewidth = 0.60) +
  scale_x_continuous(breaks = unname(XPOS),
                     labels = c("Placebo\nPre", "Placebo\nPost",
                                "Verum\nPre", "Verum\nPost"),
                     expand = expansion(add = c(0.5, 0.5))) +
  labs(title = "Intero − Sensory-Motor gap, pre→post by group (exploratory)",
       y = "Interoception − Sensory-Motor AUC (s)", x = NULL) +
  theme_minimal(base_size = 11, base_family = "serif") +
  theme(panel.background   = element_rect(fill = "white", color = NA),
        plot.background    = element_rect(fill = "white", color = NA),
        panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(color = "#e8e8e8", linewidth = 0.4),
        plot.title   = element_text(face = "bold", hjust = 0.5, size = 11,
                                    family = "serif"),
        axis.text.x  = element_text(size = 9, lineheight = 0.85),
        axis.title.y = element_text(size = 10), legend.position = "none")

# within-group brackets (color-matched) + between-group drug×session bracket (black)
y_br1 <- y_max + y_span * 0.05; y_t1 <- y_br1 - y_span * 0.02
y_br2 <- y_br1 + y_span * 0.05; y_t2 <- y_br2 - y_span * 0.02
p <- p +
  annotate("segment", x=1.0, xend=2.0, y=y_br1, yend=y_br1, color="#CD5C5C", linewidth=0.5) +
  annotate("segment", x=1.0, xend=1.0, y=y_br1, yend=y_t1,  color="#CD5C5C", linewidth=0.5) +
  annotate("segment", x=2.0, xend=2.0, y=y_br1, yend=y_t1,  color="#CD5C5C", linewidth=0.5) +
  annotate("text", x=1.5, y=y_br1+y_span*0.02, label=fmt_p(p_plac),
           size=3, hjust=0.5, color="#CD5C5C") +
  annotate("segment", x=3.2, xend=4.2, y=y_br2, yend=y_br2, color="#4682B4", linewidth=0.5) +
  annotate("segment", x=3.2, xend=3.2, y=y_br2, yend=y_t2,  color="#4682B4", linewidth=0.5) +
  annotate("segment", x=4.2, xend=4.2, y=y_br2, yend=y_t2,  color="#4682B4", linewidth=0.5) +
  annotate("text", x=3.7, y=y_br2+y_span*0.02, label=fmt_p(p_verm),
           size=3, hjust=0.5, color="#4682B4")
y_br3 <- max(y_br1, y_br2) + y_span * 0.08; y_t3 <- y_br3 - y_span * 0.02
p <- p +
  annotate("segment", x=2.0, xend=4.2, y=y_br3, yend=y_br3, color="black", linewidth=0.5) +
  annotate("segment", x=2.0, xend=2.0, y=y_br3, yend=y_t3, color="black", linewidth=0.5) +
  annotate("segment", x=4.2, xend=4.2, y=y_br3, yend=y_t3, color="black", linewidth=0.5) +
  annotate("text", x=3.1, y=y_br3+y_span*0.02, label=fmt_p(p_inter),
           size=3, hjust=0.5, color="black") +
  coord_cartesian(ylim = c(NA, y_br3 + y_span * 0.08))

# Per-panel footprint of the 4-panel grid is 5×5 in; match for this single panel.
OUT_FIG <- file.path(FIGS_DIR, "fig_ie_diff2_prepost.png")
ggsave(OUT_FIG, p, width = 5, height = 5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n", OUT_FIG))

cat("\n", SEP, "\nDONE (exploratory, non-significant — actual p-values shown, no stars)\n",
    SEP, "\n", sep = "")
