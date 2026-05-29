# 26_per_category_drug_session_with_contrasts.R
# Step 17: per-category drug × session tests + Exteroception vs nonself
#          formal contrast + Keskin-style raincloud plots
#
# NOTE: task requested filename 24_per_category_drug_session_with_contrasts.R
#   but 24_glasser_self_nonself_lmm.R already exists. Using 26_.
#
# Part 1: Per-category LMMs (auc ~ group * session + RE)
# Part A: Exteroception vs nonself formal contrast using the full Step 16 model
# Part 2: Raincloud / spaghetti figure (fig25)
# Part 3: Bar chart of interaction effects with Ext vs nonself bracket (fig26)
#
# Run from repo root:
#   Rscript 04_statistics/scripts/26_per_category_drug_session_with_contrasts.R

if (!requireNamespace("emmeans", quietly = TRUE))
  install.packages("emmeans", repos = "https://cloud.r-project.org", quiet = TRUE)

suppressPackageStartupMessages({
  library(lme4); library(lmerTest); library(emmeans)
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

DATA_CSV  <- file.path(REPO_ROOT, "04_statistics", "results",
                        "glasser_self_nonself_model_ready.csv")
OUT_INT   <- file.path(REPO_ROOT, "04_statistics", "results",
                        "per_category_drug_session.csv")
OUT_SFX   <- file.path(REPO_ROOT, "04_statistics", "results",
                        "per_category_simple_effects.csv")
OUT_FIG25 <- file.path(REPO_ROOT, "04_statistics", "figures",
                        "fig25_per_category_drug_session.html")
OUT_FIG26 <- file.path(REPO_ROOT, "04_statistics", "figures",
                        "fig26_drug_session_per_category_bar.html")

# Idempotent: delete outputs to re-run (script always overwrites)
SEP <- paste(rep("=", 70), collapse = "")
cat(SEP, "\n26_per_category_drug_session_with_contrasts.R — Step 17\n",
    SEP, "\n\n", sep = "")

# ── Load data ──────────────────────────────────────────────────────────────────
df_raw <- read.csv(DATA_CSV, stringsAsFactors = FALSE)
cat(sprintf("Loaded: %d rows\n\n", nrow(df_raw)))

# "nonself" displayed as "Sensory-Motor" in figures/tables
df_raw$category   <- ifelse(df_raw$self_layer == "nonself",
                             "Sensory-Motor", df_raw$self_layer)
df_raw$self_layer <- relevel(factor(df_raw$self_layer), ref = "nonself")
df_raw$group      <- relevel(factor(df_raw$group),      ref = "placebo")
df_raw$session    <- relevel(factor(df_raw$session),    ref = "ses-01")
df_raw$subject    <- factor(df_raw$subject)
df_raw$roi_uid    <- factor(df_raw$roi_uid)

CAT_ORDER  <- c("Sensory-Motor","Interoception","Exteroception","Cognition")
CAT_COLORS <- c("Sensory-Motor"="#888780", Interoception="#1f77b4",
                Exteroception="#ff7f0e", Cognition="#2ca02c")
GRP_COLORS <- c(placebo="#3D5A6C", verum="#8B7355")

ctrl <- lmerControl(optimizer="bobyqa", optCtrl=list(maxfun=2e5),
                    check.conv.grad=.makeCC("warning", tol=0.002))

# ── Part 1: Per-category LMMs ─────────────────────────────────────────────────
cat(SEP, "\nPART 1 — Per-category drug × session tests\n", SEP, "\n", sep = "")

int_rows <- list()
sfx_rows <- list()

for (cat in CAT_ORDER) {
  sub <- df_raw[df_raw$category == cat, ]
  sub$roi_uid <- droplevels(sub$roi_uid)
  n_roi <- nlevels(sub$roi_uid)

  m_cat <- tryCatch(
    suppressWarnings(
      lmer(auc ~ group * session + (1|subject) + (1|roi_uid),
           data=sub, REML=TRUE, control=ctrl)
    ),
    error = function(e) { cat(sprintf("  [%s] ERROR: %s\n", cat, e$message)); NULL }
  )
  if (is.null(m_cat)) next

  fe   <- coef(summary(m_cat))
  b_int<- fe["groupverum:sessionses-02","Estimate"]
  se_i <- fe["groupverum:sessionses-02","Std. Error"]
  t_i  <- fe["groupverum:sessionses-02","t value"]
  p_i  <- fe["groupverum:sessionses-02","Pr(>|t|)"]

  int_rows[[cat]] <- data.frame(category=cat, n_roi=n_roi,
    drug_session_b=b_int, drug_session_SE=se_i,
    drug_session_t=t_i, drug_session_p=p_i, stringsAsFactors=FALSE)

  # Simple effects
  b_pl <- fe["sessionses-02","Estimate"]
  se_pl<- fe["sessionses-02","Std. Error"]
  t_pl <- fe["sessionses-02","t value"]
  p_pl <- fe["sessionses-02","Pr(>|t|)"]
  b_ve <- b_pl + b_int
  vcv  <- as.matrix(vcov(m_cat))
  se_ve <- sqrt(vcv["sessionses-02","sessionses-02"] +
                vcv["groupverum:sessionses-02","groupverum:sessionses-02"] +
                2 * vcv["sessionses-02","groupverum:sessionses-02"])
  df_ve <- fe["groupverum:sessionses-02","df"]
  t_ve  <- b_ve / se_ve
  p_ve  <- 2 * pt(-abs(t_ve), df=df_ve)

  sfx_rows[[cat]] <- rbind(
    data.frame(category=cat, group="placebo",
               preto_post_b=b_pl, SE=se_pl, t=t_pl, p=p_pl, stringsAsFactors=FALSE),
    data.frame(category=cat, group="verum",
               preto_post_b=b_ve, SE=se_ve, t=t_ve, p=p_ve, stringsAsFactors=FALSE))
}

int_tbl <- do.call(rbind, int_rows); rownames(int_tbl) <- NULL
sfx_tbl <- do.call(rbind, sfx_rows); rownames(sfx_tbl) <- NULL

cat("\nDrug × Session interaction per category:\n")
cat(sprintf("  %-16s %5s %10s %8s %6s %8s\n","Category","N_ROI","β","SE","t","p"))
cat(strrep("-",58),"\n")
for (i in seq_len(nrow(int_tbl))) {
  r <- int_tbl[i,]
  cat(sprintf("  %-16s %5d %10.6f %8.6f %6.3f %8.4g%s\n",
              r$category,r$n_roi,r$drug_session_b,r$drug_session_SE,
              r$drug_session_t,r$drug_session_p,
              if(!is.na(r$drug_session_p)&&r$drug_session_p<0.05)" *" else ""))
}

cat("\nSimple effects:\n")
cat(sprintf("  %-16s %-8s %10s %8s %6s %8s\n","Category","Group","β","SE","t","p"))
cat(strrep("-",58),"\n")
for (i in seq_len(nrow(sfx_tbl))) {
  r <- sfx_tbl[i,]
  cat(sprintf("  %-16s %-8s %10.6f %8.6f %6.3f %8.4g%s\n",
              r$category,r$group,r$preto_post_b,r$SE,r$t,r$p,
              if(!is.na(r$p)&&r$p<0.05)" *" else ""))
}

sig_cats <- int_tbl$category[!is.na(int_tbl$drug_session_p) & int_tbl$drug_session_p<0.05]
cat(sprintf("\nSignificant: %s\n", if(length(sig_cats)>0) paste(sig_cats,collapse=", ") else "none"))
cat(sprintf("Not significant: %s\n", paste(setdiff(CAT_ORDER,sig_cats),collapse=", ")))

write.csv(int_tbl, OUT_INT, row.names=FALSE); cat(sprintf("Saved: %s\n", OUT_INT))
write.csv(sfx_tbl, OUT_SFX, row.names=FALSE); cat(sprintf("Saved: %s\n", OUT_SFX))

# ── Part A: Exteroception vs nonself formal contrast (full Step 16 model) ─────
cat("\n", SEP, "\nPART A — Exteroception vs Sensory-Motor (nonself) formal contrast\n",
    SEP, "\n", sep = "")
cat("Fitting full Step 16 model: auc ~ group * session * self_layer + RE...\n")

m_full <- tryCatch(
  suppressWarnings(
    lmer(auc ~ group * session * self_layer +
           (1 + session | subject) + (1 | roi_uid),
         data=df_raw, REML=TRUE, control=ctrl)
  ),
  error = function(e) {
    cat(sprintf("  Full RE failed: %s\nFalling back to (1|subject)...\n", e$message))
    suppressWarnings(
      lmer(auc ~ group * session * self_layer + (1|subject) + (1|roi_uid),
           data=df_raw, REML=TRUE, control=ctrl)
    )
  }
)

if (!is.null(m_full) && isSingular(m_full)) {
  cat("  Singular — refitting with (1|subject)+(1|roi_uid)...\n")
  m_full <- suppressWarnings(
    lmer(auc ~ group * session * self_layer + (1|subject) + (1|roi_uid),
         data=df_raw, REML=TRUE, control=ctrl))
}

conv <- m_full@optinfo$conv$lme4$messages
cat(if(!is.null(conv)) sprintf("  Convergence: %s\n",paste(conv,collapse="; "))
    else "  Convergence: OK\n")

# EMMs with Satterthwaite df
emm_options(lmerTest.limit=nrow(df_raw), pbkrtest.limit=nrow(df_raw))
emm_by <- emmeans(m_full, ~ self_layer | group + session)

# Layer order: nonself(1), Interoception(2), Exteroception(3), Cognition(4)
gaps_ext <- contrast(emm_by,
  list("Exteroception - nonself" = c(-1, 0, 1, 0)), adjust="none")
gap_ses_ext <- contrast(gaps_ext, "consec", by=c("contrast","group"), adjust="none")
flat_ext    <- contrast(gap_ses_ext, "consec", by="contrast", adjust="none")
flat_ext_df <- as.data.frame(summary(flat_ext, infer=TRUE))
if (!"lower.CL" %in% names(flat_ext_df)) {
  flat_ext_df$lower.CL <- flat_ext_df$estimate - 1.96 * flat_ext_df$SE
  flat_ext_df$upper.CL <- flat_ext_df$estimate + 1.96 * flat_ext_df$SE
}

ext_b  <- flat_ext_df$estimate[1]
ext_se <- flat_ext_df$SE[1]
ext_df <- flat_ext_df$df[1]
ext_t  <- flat_ext_df$t.ratio[1]
ext_p  <- flat_ext_df$p.value[1]
ext_lo <- flat_ext_df$lower.CL[1]
ext_hi <- flat_ext_df$upper.CL[1]

cat(sprintf("\n%s\n", strrep("─",70)))
cat(sprintf("EXTEROCEPTION vs NONSELF CONTRAST\n"))
cat(sprintf("Contrast: (Exteroception drug×session) − (Sensory-Motor drug×session)\n"))
cat(sprintf("Estimate: %.6f\n", ext_b))
cat(sprintf("SE      : %.6f\n", ext_se))
cat(sprintf("t(%6.1f): %.4f\n", ext_df, ext_t))
cat(sprintf("p       : %.4g\n", ext_p))
cat(sprintf("95%% CI  : [%.6f, %.6f]\n", ext_lo, ext_hi))
sig_str <- if(!is.na(ext_p) && ext_p < 0.05) "significantly" else "NOT significantly"
cat(sprintf(
  "\nInterpretation: The drug × session effect in Exteroception is %s\ndifferent from Sensory-Motor (nonself) regions (β = %.5f, p = %.4g).\n",
  sig_str, ext_b, ext_p))
cat(sprintf("%s\n", strrep("─",70)))

# ── Part 2: Raincloud / spaghetti figure (fig25) ──────────────────────────────
cat("\n", SEP, "\nPART 2 — Raincloud plots (fig25)\n", SEP, "\n", sep = "")

subj_means <- df_raw %>%
  group_by(category, subject, group, session) %>%
  summarise(mean_auc=mean(auc, na.rm=TRUE), .groups="drop")

COND_X <- list(list(g="placebo",s="ses-01",x=1,lbl="Placebo\nPre"),
               list(g="placebo",s="ses-02",x=2,lbl="Placebo\nPost"),
               list(g="verum",  s="ses-01",x=3,lbl="Verum\nPre"),
               list(g="verum",  s="ses-02",x=4,lbl="Verum\nPost"))

make_raincloud <- function(cat_label, show_leg) {
  sub  <- subj_means[subj_means$category == cat_label, ]
  gcol <- CAT_COLORS[[cat_label]]
  irow <- int_tbl[int_tbl$category == cat_label, ]
  b_v  <- if(nrow(irow)>0) irow$drug_session_b[1] else NA
  p_v  <- if(nrow(irow)>0) irow$drug_session_p[1] else NA
  atxt <- if(!is.na(b_v)&&!is.na(p_v))
    sprintf("Drug×Session: β=%.4f, p=%.4g%s", b_v, p_v,
            if(!is.na(p_v)&&p_v<0.05)" *" else "") else ""

  p <- plot_ly()
  for (grp in c("placebo","verum")) {
    xs   <- if(grp=="placebo") c(1,2) else c(3,4)
    gcol2 <- GRP_COLORS[[grp]]
    for (subj in unique(sub$subject[sub$group==grp])) {
      d <- sub[sub$group==grp & sub$subject==subj, ]
      d <- d[order(d$session), ]
      if (nrow(d)==2)
        p <- p %>% add_trace(x=xs, y=d$mean_auc, type="scatter", mode="lines",
                             line=list(color=gcol2,width=0.6,opacity=0.3),
                             showlegend=FALSE, hoverinfo="skip")
    }
  }
  for (cond in COND_X) {
    d     <- sub[sub$group==cond$g & sub$session==cond$s, ]
    gcol2 <- GRP_COLORS[[cond$g]]
    lbl   <- if(cond$g=="placebo") "Placebo" else "Verum"
    p <- p %>% add_trace(x=rep(cond$x,nrow(d)), y=d$mean_auc,
                         type="box", boxpoints="all",
                         jitter=0.4, pointpos=0,
                         marker=list(size=5,color=gcol2,opacity=0.55,
                                     line=list(color="white",width=0.3)),
                         line=list(color=gcol2),
                         fillcolor=paste0(gcol2,"33"),
                         name=lbl, legendgroup=lbl,
                         showlegend=(cond$s=="ses-01"&&show_leg))
  }
  annots <- list(list(x=0.5,y=1.05,xref="paper",yref="paper",
                      text=atxt,showarrow=FALSE,font=list(size=10,color="grey30")))
  if (!is.na(p_v) && p_v < 0.05)
    annots <- c(annots, list(list(x=2.5, y=max(sub$mean_auc,na.rm=TRUE)*1.02,
                                  xref="x",yref="y",text="*",showarrow=FALSE,
                                  font=list(size=18,color="black"))))
  p %>% layout(title=list(text=cat_label,font=list(size=13)),
               xaxis=list(tickvals=1:4,ticktext=sapply(COND_X,`[[`,"lbl"),
                          showgrid=FALSE,title=""),
               yaxis=list(title="Mean AUC (s)",showgrid=TRUE,gridcolor="#eeeeee"),
               boxmode="overlay",plot_bgcolor="white",paper_bgcolor="white",
               annotations=annots, showlegend=show_leg)
}

panels <- lapply(seq_along(CAT_ORDER), function(i)
  make_raincloud(CAT_ORDER[i], show_leg=(i==1)))
fig25 <- subplot(panels[[1]],panels[[2]],panels[[3]],panels[[4]],
                 nrows=2,titleX=TRUE,titleY=TRUE,shareY=FALSE,margin=0.07) %>%
  layout(title=list(text="Drug × Session effect per region category (Glasser)", x=0.5),
         legend=list(x=1.01,y=0.95))
saveWidget(fig25, OUT_FIG25, selfcontained=FALSE, title="Per-category Drug x Session")
cat(sprintf("Saved: %s\n", OUT_FIG25))

# ── Part 3: Bar chart with Ext vs nonself bracket (fig26) ─────────────────────
cat("\n", SEP, "\nPART 3 — Interaction bar chart with contrast bracket (fig26)\n",
    SEP, "\n", sep = "")

int_sorted <- int_tbl[order(int_tbl$drug_session_b), ]
ci_half    <- 1.96 * int_sorted$drug_session_SE
x_pos      <- seq(0, nrow(int_sorted)-1)
names(x_pos) <- int_sorted$category

fig26 <- plot_ly() %>%
  add_trace(x=c(-0.5, nrow(int_sorted)-0.5), y=c(0,0),
            type="scatter",mode="lines",
            line=list(color="black",dash="dash",width=1),
            showlegend=FALSE,hoverinfo="skip")

for (i in seq_len(nrow(int_sorted))) {
  r   <- int_sorted[i,]
  col <- CAT_COLORS[[r$category]]
  sig <- !is.na(r$drug_session_p) && r$drug_session_p < 0.05
  fig26 <- fig26 %>%
    add_trace(x=x_pos[[r$category]], y=r$drug_session_b,
              type="bar",
              marker=list(color=col,opacity=0.8,line=list(color="white",width=0.5)),
              error_y=list(type="data",array=ci_half[i],
                           color="black",thickness=1.5,width=5),
              name=r$category, showlegend=TRUE,
              hovertemplate=sprintf(
                "%s<br>β=%.4f<br>SE=%.4f<br>p=%.4g<extra></extra>",
                r$category,r$drug_session_b,r$drug_session_SE,r$drug_session_p)) %>%
    { if (sig) add_annotations(., x=x_pos[[r$category]],
                               y=r$drug_session_b + ci_half[i] + 0.003,
                               text="*", showarrow=FALSE,
                               font=list(size=16,color="black")) else . }
}

# Bracket: Exteroception vs Sensory-Motor
x_ext <- x_pos[["Exteroception"]]
x_sm  <- x_pos[["Sensory-Motor"]]
y_br  <- max(int_sorted$drug_session_b + ci_half, na.rm=TRUE) + 0.010
y_tick <- y_br - 0.003

fig26 <- fig26 %>%
  add_trace(x=c(x_sm, x_sm, x_ext, x_ext),
            y=c(y_tick, y_br, y_br, y_tick),
            type="scatter", mode="lines",
            line=list(color="black", width=1.2),
            showlegend=FALSE, hoverinfo="skip") %>%
  add_annotations(x=(x_sm+x_ext)/2, y=y_br + 0.002,
                  text=sprintf("p=%.4g", ext_p),
                  showarrow=FALSE,
                  font=list(size=11, color="black")) %>%
  layout(
    title=list(text="Drug × Session interaction per region category (sorted by β)",
               x=0.5),
    xaxis=list(tickvals=unname(x_pos),
               ticktext=names(x_pos),
               title="", showgrid=FALSE),
    yaxis=list(title="Drug × Session interaction β (AUC, s)",
               zeroline=FALSE, showgrid=TRUE, gridcolor="#eeeeee"),
    plot_bgcolor="white", paper_bgcolor="white", bargap=0.3
  )
saveWidget(fig26, OUT_FIG26, selfcontained=FALSE, title="Drug x Session per Category Bar")
cat(sprintf("Saved: %s\n", OUT_FIG26))

cat("\n", SEP, "\nDONE\n", SEP, "\n", sep = "")
