# exploratory/sm_intero_convergence.R
#
# Within-subject convergence of the Sensory-Motor vs Interoception AUC gap, pre→post.
#
# Question: does the gap between sensory-motor and interoceptive timescale shrink
# from pre to post, more under verum than placebo? (CONVERGENCE = gap decreases.)
#
# "Sensory-Motor" == the broad `nonself` self_layer (267 Glasser parcels), the SAME
# definition used by the 4-panel paired figure (CAT_MAP: nonself -> "Sensory-Motor"
# in 06_figures.R) and by IE_diff_2 in 05_per_category_glasser.R. This is the broad
# nonself, NOT a CAB-NP ~99 sensory-network subset.
#
# Per subject × session:
#   intero_mean = mean(AUC | self_layer == "Interoception")   (14 parcels)
#   sm_mean     = mean(AUC | self_layer == "nonself")          (267 parcels)
#   gap         = sm_mean - intero_mean   (positive: SM above intero)
# -> one row per subject per session (35 × 2 = 70).
#
# Subject-level test (the proper, non-pseudoreplicated one):
#   lmer(gap ~ group * session + (1 | subject))
#   group×session interaction = the convergence drug effect (verum shrinks gap more).
#   With 70 subject×session rows the interaction df is ≈ 33 (subject-level). If it
#   comes out in the thousands the model is pseudoreplicated — the script aborts.
#
# Reuses the style/palette/annotation scheme of the 4-panel "Pre → Post AUC change
# by drug group" figure (make_paired_panel() in 06_figures.R).
#
# Run from repo root:
#   Rscript 04_statistics/scripts/exploratory/sm_intero_convergence.R

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(lme4)
  library(lmerTest)
  library(emmeans)
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
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE)
dir.create(FIGS_DIR,   showWarnings = FALSE, recursive = TRUE)

DATA_CSV <- file.path(TABLES_DIR, "model_ready.csv")
if (!file.exists(DATA_CSV))
  stop("Input not found — run 02_lmm.R first: ", DATA_CSV)

SEP <- paste(rep("=", 72), collapse = "")
cat(SEP, "\nsm_intero_convergence.R — config: ", TAG, "\n", SEP, "\n\n", sep = "")

# ── Load & build per-subject × session gap ───────────────────────────────────
df <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n", nrow(df)))

gap_df <- df %>%
  filter(self_layer %in% c("Interoception", "nonself")) %>%
  group_by(subject, group, session, self_layer) %>%
  summarise(mean_auc = mean(auc, na.rm = TRUE), .groups = "drop") %>%
  pivot_wider(names_from = self_layer, values_from = mean_auc) %>%
  rename(intero_mean = Interoception, sm_mean = nonself) %>%
  mutate(gap = sm_mean - intero_mean)

gap_df$group   <- factor(gap_df$group,   levels = c("placebo", "verum"))
gap_df$session <- factor(gap_df$session, levels = c("ses-01", "ses-02"))
gap_df$subject <- factor(gap_df$subject)

cat(sprintf("Gap rows: %d (expect 70)   Subjects: %d\n\n",
            nrow(gap_df), nlevels(gap_df$subject)))

# ── Subject-level model ──────────────────────────────────────────────────────
ctrl <- lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))

fit_gap <- function(d, label) {
  m <- lmer(gap ~ group * session + (1 | subject), data = d, REML = TRUE,
            control = ctrl)
  cat(sprintf("[%s] fitted on %d rows, %d subjects\n",
              label, nrow(d), nlevels(droplevels(d$subject))))
  m
}

m_full <- fit_gap(gap_df, "full")

# df sanity check: subject-level interaction df should be ≈ 33, NOT thousands.
fe_full   <- coef(summary(m_full))
INT_TERM  <- "groupverum:sessionses-02"
int_df    <- fe_full[INT_TERM, "df"]
cat(sprintf("Interaction df (Satterthwaite) = %.2f  (expected ≈ 33)\n", int_df))
if (int_df > 100)
  stop(sprintf(paste0("Interaction df = %.1f — far above the ~33 expected for a ",
                      "subject-level model. The data look pseudoreplicated ",
                      "(ROI-level rows leaking in). Aborting."), int_df))

# Within-group pre→post simple effects (post − pre, per group) via emmeans.
# revpairwise gives "ses-02 - ses-01" = post − pre.
emm_ss  <- emmeans(m_full, ~ session | group)
ss_df   <- as.data.frame(summary(contrast(emm_ss, method = "revpairwise"),
                                 infer = TRUE))
p_plac  <- ss_df$p.value[ss_df$group == "placebo"][1]
p_verm  <- ss_df$p.value[ss_df$group == "verum"][1]
p_inter <- fe_full[INT_TERM, "Pr(>|t|)"]

cat("\nWithin-group pre→post simple effects (post − pre):\n")
print(ss_df[, intersect(c("contrast", "group", "estimate", "SE", "df",
                          "t.ratio", "p.value"), names(ss_df))], row.names = FALSE)
cat(sprintf("\nGroup × session interaction (convergence drug effect):\n"))
cat(sprintf("  estimate = %.4f  SE = %.4f  df = %.2f  t = %.3f  p = %.4g\n\n",
            fe_full[INT_TERM, "Estimate"], fe_full[INT_TERM, "Std. Error"],
            int_df, fe_full[INT_TERM, "t value"], p_inter))

# ── Sensitivity: residual trimming (|resid| > 2.5 SD), matching 02_lmm.R ──────
resids    <- resid(m_full)
resid_sd  <- sd(resids)
resid_thr <- 2.5 * resid_sd
flagged   <- abs(resids) > resid_thr
cat(sprintf("Sensitivity: residual SD = %.4f, threshold = ±%.4f (2.5×SD); %d/%d rows flagged\n",
            resid_sd, resid_thr, sum(flagged), length(resids)))

gap_trim <- gap_df[!flagged, ]
m_trim   <- fit_gap(gap_trim, "trimmed")
fe_trim  <- coef(summary(m_trim))
p_inter_trim <- fe_trim[INT_TERM, "Pr(>|t|)"]
cat(sprintf("Trimmed interaction: estimate = %.4f  p = %.4g  (full p = %.4g)%s\n\n",
            fe_trim[INT_TERM, "Estimate"], p_inter_trim, p_inter,
            if ((p_inter < 0.05) != (p_inter_trim < 0.05))
              "  [significance CHANGES vs full]" else "  [significance unchanged vs full]"))

# ── Models table (full + trimmed fixed effects) ──────────────────────────────
fe_to_df <- function(m, model_label) {
  fe <- as.data.frame(coef(summary(m)))
  data.frame(model    = model_label,
             term     = rownames(fe),
             estimate = fe[, "Estimate"],
             SE       = fe[, "Std. Error"],
             df       = fe[, "df"],
             t        = fe[, "t value"],
             p        = fe[, "Pr(>|t|)"],
             row.names = NULL, stringsAsFactors = FALSE)
}
models_tbl <- rbind(fe_to_df(m_full, "full"),
                    fe_to_df(m_trim, "trimmed"))
OUT_CSV <- file.path(TABLES_DIR, "sm_intero_convergence_models.csv")
write.csv(models_tbl, OUT_CSV, row.names = FALSE)
cat(sprintf("Table saved: %s\n", OUT_CSV))

# ── Figure: single paired panel (style = make_paired_panel in 06_figures.R) ──
XPOS <- c("Placebo Pre"  = 1.0, "Placebo Post" = 2.0,
          "Verum Pre"    = 3.2, "Verum Post"   = 4.2)
COND_LEVELS <- c("Placebo Pre", "Verum Pre", "Placebo Post", "Verum Post")
fmt_p <- function(p) {
  if (is.na(p)) return("n.s.")
  if (p < 0.001) return("p<0.001")
  sprintf("p=%.3f", p)
}

d <- gap_df
d$condition <- factor(
  paste0(ifelse(d$group == "placebo", "Placebo", "Verum"), " ",
         ifelse(d$session == "ses-01", "Pre", "Post")),
  levels = COND_LEVELS)
d$x_pos <- XPOS[as.character(d$condition)]

make_lines <- function(grp) {
  pre  <- d[d$group == grp & d$session == "ses-01", c("subject", "gap", "x_pos")]
  post <- d[d$group == grp & d$session == "ses-02", c("subject", "gap", "x_pos")]
  merge(pre, post, by = "subject", suffixes = c("_pre", "_post"))
}
plac_segs <- make_lines("placebo"); verm_segs <- make_lines("verum")

y_max  <- max(d$gap, na.rm = TRUE)
y_span <- diff(range(d$gap, na.rm = TRUE))

p <- ggplot() +
  geom_segment(data = plac_segs,
               aes(x = x_pos_pre, xend = x_pos_post, y = gap_pre, yend = gap_post),
               color = "#CD5C5C", alpha = 0.30, linewidth = 0.5) +
  geom_segment(data = verm_segs,
               aes(x = x_pos_pre, xend = x_pos_post, y = gap_pre, yend = gap_post),
               color = "#4682B4", alpha = 0.30, linewidth = 0.5) +
  geom_point(data = d[d$group == "placebo", ], aes(x = x_pos, y = gap),
             color = "#CD5C5C", alpha = 0.60, size = 2, shape = 16) +
  geom_point(data = d[d$group == "verum", ], aes(x = x_pos, y = gap),
             color = "#4682B4", alpha = 0.60, size = 2, shape = 16) +
  geom_boxplot(data = d[d$group == "placebo", ],
               aes(x = x_pos, y = gap, group = condition),
               fill = NA, color = "#CD5C5C", width = 0.22, outlier.shape = NA,
               linewidth = 0.60) +
  geom_boxplot(data = d[d$group == "verum", ],
               aes(x = x_pos, y = gap, group = condition),
               fill = NA, color = "#4682B4", width = 0.22, outlier.shape = NA,
               linewidth = 0.60) +
  scale_x_continuous(breaks = unname(XPOS),
                     labels = c("Placebo\nPre", "Placebo\nPost",
                                "Verum\nPre", "Verum\nPost"),
                     expand = expansion(add = c(0.5, 0.5))) +
  labs(title = "Within-subject SM–Intero gap, pre→post",
       y = "Sensory-Motor − Interoceptive AUC gap (s)", x = NULL) +
  theme_minimal(base_size = 11, base_family = "serif") +
  theme(panel.background   = element_rect(fill = "white", color = NA),
        plot.background    = element_rect(fill = "white", color = NA),
        panel.grid.major.x = element_blank(), panel.grid.minor = element_blank(),
        panel.grid.major.y = element_line(color = "#e8e8e8", linewidth = 0.4),
        plot.title   = element_text(face = "bold", hjust = 0.5, size = 12,
                                    family = "serif"),
        axis.text.x  = element_text(size = 9, lineheight = 0.85),
        axis.title.y = element_text(size = 10), legend.position = "none")

# within-group brackets (color-matched) + between-group interaction bracket (black)
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

# Per-panel footprint of the 4-panel grid is 5×5 in; match that for this single panel.
OUT_FIG <- file.path(FIGS_DIR, "fig_sm_intero_convergence.png")
ggsave(OUT_FIG, p, width = 5, height = 5, dpi = 300, bg = "white")
cat(sprintf("Figure saved: %s\n", OUT_FIG))

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
