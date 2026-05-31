# 12_spaghetti_pre_post.jl — Step 9b: per-subject pre→post spaghetti plot
#
# Visualises each subject's median AUC change from ses-01 to ses-02.
# Three side-by-side panels: self | nonself | self−nonself diff.
# Individual lines (opacity 0.45) colored by drug group; group-mean overlay
# with 95% CI error bars on top.
#
# Run from repo root:
#   julia 04_statistics/scripts/12_spaghetti_pre_post.jl

using CSV, DataFrames, PlotlyJS, Statistics, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const AGG_CSV   = joinpath(REPO_ROOT, "04_statistics", "results",
                           "aggregated_subject_session.csv")
const ANOVA_CSV = joinpath(REPO_ROOT, "04_statistics", "results",
                           "anova_results_median.csv")
const FIG_DIR   = joinpath(REPO_ROOT, "04_statistics", "figures")
const OUT_FIG   = joinpath(FIG_DIR, "fig10_spaghetti_pre_post.html")

# ── Colors ────────────────────────────────────────────────────────────────────
const COLOR_PLACEBO = "#3D5A6C"
const COLOR_VERUM   = "#8B7355"
const COLOR_MAP     = Dict("placebo" => COLOR_PLACEBO, "verum" => COLOR_VERUM)

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(OUT_FIG)
    println("[skip] output already exists at $OUT_FIG")
    exit()
end

for f in [AGG_CSV, ANOVA_CSV]
    isfile(f) || error("Missing required input: $f")
end
mkpath(FIG_DIR)

# ── Load and pivot data ───────────────────────────────────────────────────────
println("Loading data …")
agg   = CSV.read(AGG_CSV,   DataFrame)
anova = CSV.read(ANOVA_CSV, DataFrame)
@printf("  aggregated_subject_session: %d rows × %d cols\n", nrow(agg), ncol(agg))
@printf("  anova_results_median:       %d rows × %d cols\n", nrow(anova), ncol(anova))

# One row per subject with both sessions side by side
ses01 = filter(r -> r.session == "ses-01", agg)
ses02 = filter(r -> r.session == "ses-02", agg)

wide = innerjoin(
    select(ses01, :subject, :drug_group,
           :self_median    => :self_ses01,
           :nonself_median => :nonself_ses01,
           :diff_median    => :diff_ses01),
    select(ses02, :subject,
           :self_median    => :self_ses02,
           :nonself_median => :nonself_ses02,
           :diff_median    => :diff_ses02),
    on = :subject,
)
sort!(wide, :subject)
N = nrow(wide)
@printf("  Pivoted to %d subjects × 2 sessions\n", N)

# ── ANOVA stats lookup ────────────────────────────────────────────────────────
function anova_row(outcome_name)
    rows = filter(r -> r.outcome == outcome_name && r.effect == "drug_x_session", anova)
    nrow(rows) == 1 || error("Expected 1 row for $outcome_name / drug_x_session")
    first(rows)
end

stats_self    = anova_row("self_median")
stats_nonself = anova_row("nonself_median")
stats_diff    = anova_row("diff_median")

# ── Helpers ───────────────────────────────────────────────────────────────────
short(s)  = s[5:end]                               # "sub-01" → "01"
ci95(v)   = 1.96 * std(v) / sqrt(length(v))        # 95% CI half-width

function stats_text(s)
    @sprintf("Drug×Sess: F=%.2f, p=%.3f, η²=%.3f", s.F, s.p, s.eta2_partial)
end

# ── Panel definitions ─────────────────────────────────────────────────────────
PANELS = [
    (col=1,
     s01=:self_ses01,    s02=:self_ses02,
     title="Self atlas — per-subject change",
     stats=stats_self),
    (col=2,
     s01=:nonself_ses01, s02=:nonself_ses02,
     title="Nonself atlas — per-subject change",
     stats=stats_nonself),
    (col=3,
     s01=:diff_ses01,    s02=:diff_ses02,
     title="Self − Nonself — per-subject change",
     stats=stats_diff),
]

# ── Panel geometry in paper coords ────────────────────────────────────────────
const H_SPACING   = 0.05
const PANEL_W     = (1.0 - 2 * H_SPACING) / 3           # ≈ 0.300
const PANEL_LEFT  = [0.0, PANEL_W + H_SPACING, 2*(PANEL_W + H_SPACING)]
const PANEL_CX    = PANEL_LEFT .+ PANEL_W / 2            # centres ≈ 0.15, 0.50, 0.85
const STATS_XRGT  = PANEL_LEFT .+ PANEL_W .- 0.01        # right-edge anchors

# ══════════════════════════════════════════════════════════════════════════════
# Build figure
# ══════════════════════════════════════════════════════════════════════════════
println("Building figure …")

fig = make_subplots(
    rows               = 1,
    cols               = 3,
    horizontal_spacing = H_SPACING,
)

const SESSIONS = ["ses-01", "ses-02"]

for panel in PANELS
    c = panel.col

    # ── Individual subject lines ───────────────────────────────────────────
    for row in eachrow(wide)
        color = COLOR_MAP[row.drug_group]
        add_trace!(fig,
            scatter(
                x            = SESSIONS,
                y            = [row[panel.s01], row[panel.s02]],
                mode         = "lines+markers+text",
                opacity      = 0.45,
                line         = attr(color=color, width=1.5),
                marker       = attr(color=color, size=5, symbol="circle"),
                text         = ["", short(row.subject)],
                textposition = "middle right",
                textfont     = attr(size=8, color=color, family="Times New Roman"),
                name         = row.subject,
                legendgroup  = row.drug_group,
                showlegend   = false,
            ),
            row=1, col=c)
    end

    # ── Group mean lines with 95% CI error bars ────────────────────────────
    for (grp, label) in [("placebo", "Placebo"), ("verum", "Verum")]
        color  = COLOR_MAP[grp]
        sub    = filter(r -> r.drug_group == grp, wide)
        y1     = sub[:, panel.s01]
        y2     = sub[:, panel.s02]

        add_trace!(fig,
            scatter(
                x           = SESSIONS,
                y           = [mean(y1), mean(y2)],
                mode        = "lines+markers",
                line        = attr(color=color, width=4),
                marker      = attr(color=color, size=12, symbol="circle"),
                error_y     = attr(
                    type      = "data",
                    array     = [ci95(y1), ci95(y2)],
                    visible   = true,
                    color     = color,
                    thickness = 2,
                    width     = 6,
                ),
                name        = label,
                legendgroup = grp,
                showlegend  = (c == 1),
            ),
            row=1, col=c)
    end
end

# ── Annotations: panel titles + stats boxes ───────────────────────────────────
annotations = Any[]

for (i, panel) in enumerate(PANELS)
    push!(annotations, attr(
        text      = "<b>$(panel.title)</b>",
        xref      = "paper", yref      = "paper",
        x         = PANEL_CX[i],  y    = 1.02,
        xanchor   = "center",     yanchor = "bottom",
        showarrow = false,
        font      = attr(size=11, family="Times New Roman"),
    ))

    push!(annotations, attr(
        text        = stats_text(panel.stats),
        xref        = "paper", yref        = "paper",
        x           = STATS_XRGT[i], y     = 0.03,
        xanchor     = "right",  yanchor     = "bottom",
        showarrow   = false,
        font        = attr(size=9, family="Times New Roman"),
        bgcolor     = "rgba(255,255,255,0.85)",
        bordercolor = "#aaaaaa",
        borderpad   = 3,
    ))
end

relayout!(fig, annotations=annotations)

relayout!(fig,
    title         = attr(
        text = "Per-subject AUC pre→post by drug group · spaghetti plot",
        font = attr(size=14, family="Times New Roman"),
    ),
    height        = 580,
    width         = 1400,
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    font          = attr(family="Times New Roman", size=11),
    legend        = attr(x=1.01, y=0.95,
                         bgcolor="rgba(255,255,255,0.9)",
                         bordercolor="#cccccc", borderwidth=1),
    yaxis         = attr(title="Median AUC (s)"),
    yaxis2        = attr(title="Median AUC (s)"),
    yaxis3        = attr(title="Median AUC (s)"),
    margin        = attr(t=90, b=60, l=70, r=130),
)

PlotlyJS.savefig(fig, OUT_FIG)
println("  Figure written: $OUT_FIG")

# ══════════════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════════════
println()
println("═" ^ 72)
println("Console Summary")
println("═" ^ 72)

for (atlas_label, s01_col, s02_col) in [
        ("self",    :self_ses01,    :self_ses02),
        ("nonself", :nonself_ses01, :nonself_ses02),
    ]

    println()
    println("  ─── Atlas: $atlas_label ────────────────────────────────────────────")

    grp_dirs = Dict{String,Float64}()

    for grp in ["placebo", "verum"]
        sub     = filter(r -> r.drug_group == grp, wide)
        changes = sub[:, s02_col] .- sub[:, s01_col]
        n_pos   = count(>(0.0), changes)
        n_neg   = count(<(0.0), changes)
        n_zero  = count(==(0.0), changes)
        med_Δ   = median(changes)

        println()
        @printf("  %-10s  n=%d  pos=%d  neg=%d  zero=%d  median_Δ=%+.4f\n",
                grp, length(changes), n_pos, n_neg, n_zero, med_Δ)

        # Outlier responders: |Δ| > median(|Δ|) + 2 × SD(|Δ|) within the group
        abs_Δ   = abs.(changes)
        med_abs = median(abs_Δ)
        sd_abs  = std(abs_Δ)
        thresh  = med_abs + 2 * sd_abs

        outliers = [(sub[i, :subject], changes[i])
                    for i in 1:length(changes) if abs(changes[i]) > thresh]

        if isempty(outliers)
            @printf("  Outlier responders (|Δ| > %.4f): none\n", thresh)
        else
            @printf("  Outlier responders (|Δ| > med_abs+2SD = %.4f):\n", thresh)
            for (subj, ch) in outliers
                @printf("    %-10s  Δ=%+.4f  (%+.2f SD above med_abs)\n",
                        subj, ch, (abs(ch) - med_abs) / sd_abs)
            end
        end

        grp_dirs[grp] = mean(changes)
    end

    # Group-mean direction comparison
    p_dir    = grp_dirs["placebo"]
    v_dir    = grp_dirs["verum"]
    dir_desc = sign(p_dir) == sign(v_dir) ? "same direction (↑↑ or ↓↓)" :
                                             "OPPOSITE directions (↑↓)"
    println()
    @printf("  Group-mean Δ:  placebo=%+.4f  verum=%+.4f  → %s\n",
            p_dir, v_dir, dir_desc)
end

println()
println("Done.")
