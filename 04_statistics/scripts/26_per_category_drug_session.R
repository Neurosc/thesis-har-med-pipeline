# 26_per_category_drug_session.R
# Step 17: per-category drug × session interaction test + raincloud plots
#
# NOTE: task requested filename 24_; 24_glasser_self_nonself_lmm.R already
#   exists. Using 26_.
#
# Input: glasser_self_nonself_model_ready.csv
#   self_layer values: "nonself" (316 parcels), "Interoception" (10),
#                      "Exteroception" (14), "Cognition" (11)
#   "nonself" is displayed as "Sensory-Motor" throughout.
#
# Run from repo root:
#   Rscript 04_statistics/scripts/26_per_category_drug_session.R

suppressPackageStartupMessages({
  library(lme4); library(lmerTest)
  library(dplyr); library(tidyr)
  library(plotly); library(htmlwidgets)
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

DATA_CSV   <- file.path(REPO_ROOT, "04_statistics", "results",
                         "glasser_self_nonself_model_ready.csv")
OUT_INT    <- file.path(REPO_ROOT, "04_statistics", "results",
                         "per_category_drug_session.csv")
OUT_SFX    <- file.path(REPO_ROOT, "04_statistics", "results",
                         "per_category_simple_effects.csv")
OUT_FIG25  <- file.path(REPO_ROOT, "04_statistics", "figures",
                         "fig25_per_category_drug_session.html")
OUT_FIG26  <- file.path(REPO_ROOT, "04_statistics", "figures",
                         "fig26_drug_session_per_category_bar.html")

all_out <- c(OUT_INT, OUT_SFX, OUT_FIG25, OUT_FIG26)
if (all(file.exists(all_out))) {
  cat("All outputs exist — nothing to do. Delete to rerun.\n"); quit(status = 0)
}

SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n26_per_category_drug_session.R — Step 17\n", SEP, "\n\n", sep = "")

# ── Load data ──────────────────────────────────────────────────────────────────
df_raw <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n\n", nrow(df_raw)))

# Rename "nonself" → "Sensory-Motor" for display; keep original for model
df_raw$category <- ifelse(df_raw$self_layer == "nonself",
                          "Sensory-Motor", df_raw$self_layer)
df_raw$group   <- relevel(factor(df_raw$group),   ref = "placebo")
df_raw$session <- relevel(factor(df_raw$session), ref = "ses-01")
df_raw$subject <- factor(df_raw$subject)
df_raw$roi_uid <- factor(df_raw$roi_uid)

CAT_ORDER  <- c("Sensory-Motor","Interoception","Exteroception","Cognition")
CAT_COLORS <- c("Sensory-Motor"="#888780", Interoception="#1f77b4",
                Exteroception="#ff7f0e", Cognition="#2ca02c")
GRP_COLORS <- c(placebo="#3D5A6C", verum="#8B7355")

ctrl <- lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5),
                    check.conv.grad=.makeCC("warning",tol=0.002))

# ── Part 1: Per-category LMMs ─────────────────────────────────────────────────
cat(SEP, "\nPART 1 — Per-category drug × session tests\n", SEP, "\n", sep = "")

int_rows <- list()
sfx_rows <- list()

for (cat in CAT_ORDER) {
  sub <- df_raw[df_raw$category == cat, ]
  sub$roi_uid <- droplevels(sub$roi_uid)
  n_roi <- nlevels(sub$roi_uid)

  m <- tryCatch(
    suppressWarnings(
      lmer(auc ~ group * session + (1|subject) + (1|roi_uid),
           data=sub, REML=TRUE, control=ctrl)
    ),
    error = function(e) { cat(sprintf("  [%s] ERROR: %s\n", cat, e$message)); NULL }
  )
  if (is.null(m)) next

  fe  <- coef(summary(m))
  int_term <- "groupverum:sessionses-02"
  ses_term <- "sessionses-02"

  b_int <- fe[int_term, "Estimate"]
  se_int<- fe[int_term, "Std. Error"]
  t_int <- fe[int_term, "t value"]
  p_int <- fe[int_term, "Pr(>|t|)"]

  int_rows[[cat]] <- data.frame(
    category=cat, n_roi=n_roi,
    drug_session_b=b_int, drug_session_SE=se_int,
    drug_session_t=t_int, drug_session_p=p_int,
    stringsAsFactors=FALSE)

  # Simple effects: placebo pre→post = sessionses-02
  #                 verum   pre→post = sessionses-02 + groupverum:sessionses-02
  b_pl <- fe[ses_term, "Estimate"]
  se_pl<- fe[ses_term, "Std. Error"]
  t_pl <- fe[ses_term, "t value"]
  p_pl <- fe[ses_term, "Pr(>|t|)"]
  b_ve <- b_pl + b_int
  # SE for verum: sqrt(var_ses + var_int + 2*cov(ses,int))
  vcv  <- as.matrix(vcov(m))
  se_ve <- sqrt(vcv[ses_term,ses_term] + vcv[int_term,int_term] +
                2 * vcv[ses_term, int_term])
  t_ve <- b_ve / se_ve
  df_ve <- fe[int_term, "df"]
  p_ve <- 2 * pt(-abs(t_ve), df=df_ve)

  sfx_rows[[cat]] <- rbind(
    data.frame(category=cat, group="placebo",
               preto_post_b=b_pl, preto_post_SE=se_pl,
               preto_post_t=t_pl, preto_post_p=p_pl, stringsAsFactors=FALSE),
    data.frame(category=cat, group="verum",
               preto_post_b=b_ve, preto_post_SE=se_ve,
               preto_post_t=t_ve, preto_post_p=p_ve, stringsAsFactors=FALSE))
}

int_tbl <- do.call(rbind, int_rows); rownames(int_tbl) <- NULL
sfx_tbl <- do.call(rbind, sfx_rows); rownames(sfx_tbl) <- NULL

cat("\nDrug × Session interaction per category:\n")
cat(sprintf("  %-16s  %5s  %10s  %8s  %6s  %8s\n",
            "Category","N_ROI","β","SE","t","p"))
cat(strrep("-",60),"\n")
for (i in seq_len(nrow(int_tbl))) {
  r <- int_tbl[i,]
  cat(sprintf("  %-16s  %5d  %10.6f  %8.6f  %6.3f  %8.4g%s\n",
              r$category, r$n_roi,
              r$drug_session_b, r$drug_session_SE,
              r$drug_session_t, r$drug_session_p,
              if(!is.na(r$drug_session_p) && r$drug_session_p<0.05) " *" else ""))
}

cat("\nSimple effects (pre → post per group):\n")
cat(sprintf("  %-16s  %-8s  %10s  %8s  %6s  %8s\n",
            "Category","Group","β","SE","t","p"))
cat(strrep("-",60),"\n")
for (i in seq_len(nrow(sfx_tbl))) {
  r <- sfx_tbl[i,]
  cat(sprintf("  %-16s  %-8s  %10.6f  %8.6f  %6.3f  %8.4g%s\n",
              r$category, r$group,
              r$preto_post_b, r$preto_post_SE,
              r$preto_post_t, r$preto_post_p,
              if(!is.na(r$preto_post_p) && r$preto_post_p<0.05) " *" else ""))
}

sig_cats <- int_tbl$category[!is.na(int_tbl$drug_session_p) &
                               int_tbl$drug_session_p < 0.05]
ns_cats  <- setdiff(CAT_ORDER, sig_cats)
cat(sprintf("\nDrug × session interaction significant in   : %s\n",
            if(length(sig_cats)>0) paste(sig_cats,collapse=", ") else "none"))
cat(sprintf("Drug × session interaction NOT significant in: %s\n",
            if(length(ns_cats)>0) paste(ns_cats,collapse=", ") else "none"))

write.csv(int_tbl, OUT_INT, row.names=FALSE); cat(sprintf("\nSaved: %s\n", OUT_INT))
write.csv(sfx_tbl, OUT_SFX, row.names=FALSE); cat(sprintf("Saved: %s\n", OUT_SFX))

# ── Part 2: Raincloud / spaghetti figure ──────────────────────────────────────
cat("\n", SEP, "\nPART 2 — Raincloud plots (fig25)\n", SEP, "\n", sep = "")

# Subject means per (category, group, session)
subj_means <- df_raw %>%
  group_by(category, subject, group, session) %>%
  summarise(mean_auc = mean(auc, na.rm=TRUE), .groups="drop")

# x-positions: placebo_pre=1, placebo_post=2, verum_pre=3, verum_post=4
COND_X <- list(list(g="placebo",s="ses-01",x=1,lbl="Placebo\nPre"),
               list(g="placebo",s="ses-02",x=2,lbl="Placebo\nPost"),
               list(g="verum",  s="ses-01",x=3,lbl="Verum\nPre"),
               list(g="verum",  s="ses-02",x=4,lbl="Verum\nPost"))

make_raincloud <- function(cat_label, show_leg) {
  sub  <- subj_means[subj_means$category == cat_label, ]
  col  <- CAT_COLORS[[cat_label]]
  irow <- int_tbl[int_tbl$category == cat_label, ]
  b_v  <- if (nrow(irow)>0) irow$drug_session_b[1] else NA
  p_v  <- if (nrow(irow)>0) irow$drug_session_p[1] else NA
  annot_txt <- if (!is.na(b_v) && !is.na(p_v))
    sprintf("Drug×Session: β=%.4f, p=%.4g%s", b_v, p_v,
            if(p_v<0.05)" *" else "") else ""

  p <- plot_ly()

  # Spaghetti lines within placebo (x 1→2) and verum (x 3→4)
  for (grp in c("placebo","verum")) {
    xs <- if(grp=="placebo") c(1,2) else c(3,4)
    subj_list <- unique(sub$subject[sub$group==grp])
    gcol <- GRP_COLORS[[grp]]
    for (subj in subj_list) {
      d <- sub[sub$group==grp & sub$subject==subj, ]
      d <- d[order(d$session), ]
      if (nrow(d) == 2)
        p <- p %>%
          add_trace(x=xs, y=d$mean_auc, type="scatter", mode="lines",
                    line=list(color=gcol, width=0.6, opacity=0.3),
                    showlegend=FALSE, hoverinfo="skip")
    }
  }

  # Box plots
  for (cond in COND_X) {
    d    <- sub[sub$group==cond$g & sub$session==cond$s, ]
    gcol <- GRP_COLORS[[cond$g]]
    lbl  <- if(cond$g=="placebo")"Placebo" else "Verum"
    p <- p %>%
      add_trace(x=rep(cond$x, nrow(d)), y=d$mean_auc,
                type="box", boxpoints="all",
                jitter=0.4, pointpos=0,
                marker=list(size=5, color=gcol, opacity=0.55,
                            line=list(color="white",width=0.3)),
                line=list(color=gcol), fillcolor=paste0(gcol,"33"),
                name=lbl, legendgroup=lbl,
                showlegend=(cond$s=="ses-01" && show_leg))
  }

  # Asterisk bracket if significant
  annots <- list(list(x=0.5, y=1.05, xref="paper", yref="paper",
                      text=annot_txt, showarrow=FALSE,
                      font=list(size=10, color="grey30")))
  if (!is.na(p_v) && p_v < 0.05)
    annots <- c(annots, list(list(
      x=2.5, y=max(sub$mean_auc, na.rm=TRUE)*1.02,
      xref="x", yref="y",
      text="*", showarrow=FALSE, font=list(size=18, color="black"))))

  p %>% layout(
    title     = list(text=cat_label, font=list(size=13)),
    xaxis     = list(tickvals=1:4, ticktext=sapply(COND_X,`[[`,"lbl"),
                     showgrid=FALSE, title=""),
    yaxis     = list(title="Mean AUC (s)", showgrid=TRUE,
                     gridcolor="#eeeeee"),
    boxmode   = "overlay",
    plot_bgcolor ="white", paper_bgcolor="white",
    annotations = annots,
    showlegend  = show_leg
  )
}

panels <- lapply(seq_along(CAT_ORDER), function(i)
  make_raincloud(CAT_ORDER[i], show_leg=(i==1)))

fig25 <- subplot(panels[[1]], panels[[2]], panels[[3]], panels[[4]],
                 nrows=2, titleX=TRUE, titleY=TRUE,
                 shareY=FALSE, margin=0.07) %>%
  layout(title=list(
    text="Drug × Session effect per region category — Glasser parcels",
    x=0.5),
    legend=list(x=1.01, y=0.95))

saveWidget(fig25, OUT_FIG25, selfcontained=FALSE,
           title="Per-category Drug x Session")
cat(sprintf("Saved: %s\n", OUT_FIG25))

# ── Part 3: Bar chart of interaction effects ──────────────────────────────────
cat("\n", SEP, "\nPART 3 — Interaction bar chart (fig26)\n", SEP, "\n", sep = "")

# Sort by β (data-driven)
int_sorted <- int_tbl[order(int_tbl$drug_session_b), ]
ci95_half  <- 1.96 * int_sorted$drug_session_SE

fig26 <- plot_ly() %>%
  add_trace(x=c(-0.5, nrow(int_sorted)-0.5), y=c(0,0),
            type="scatter", mode="lines",
            line=list(color="black",dash="dash",width=1),
            showlegend=FALSE, hoverinfo="skip")

for (i in seq_len(nrow(int_sorted))) {
  r    <- int_sorted[i, ]
  col  <- CAT_COLORS[[r$category]]
  sig  <- !is.na(r$drug_session_p) && r$drug_session_p < 0.05
  fig26 <- fig26 %>%
    add_trace(x=i-1, y=r$drug_session_b,
              type="bar",
              marker=list(color=col, opacity=0.8,
                          line=list(color="white",width=0.5)),
              error_y=list(type="data", array=ci95_half[i],
                           color="black", thickness=1.5, width=5),
              name=r$category, showlegend=TRUE,
              hovertemplate=sprintf(
                "%s<br>β=%.4f<br>SE=%.4f<br>p=%.4g<extra></extra>",
                r$category, r$drug_session_b,
                r$drug_session_SE, r$drug_session_p)) %>%
    { if (sig) add_annotations(.,
        x=i-1, y=r$drug_session_b + ci95_half[i] + 0.003,
        text="*", showarrow=FALSE,
        font=list(size=16, color="black")) else . }
}

fig26 <- fig26 %>%
  layout(
    title = list(text="Drug × Session interaction per region category (sorted by β)",
                 x=0.5),
    xaxis = list(tickvals=seq(0, nrow(int_sorted)-1),
                 ticktext=int_sorted$category,
                 title="", showgrid=FALSE),
    yaxis = list(title="Drug × Session interaction β (AUC, s)",
                 zeroline=FALSE, showgrid=TRUE, gridcolor="#eeeeee"),
    plot_bgcolor ="white", paper_bgcolor="white",
    bargap=0.3
  )

saveWidget(fig26, OUT_FIG26, selfcontained=FALSE,
           title="Drug x Session per Category Bar")
cat(sprintf("Saved: %s\n", OUT_FIG26))

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
