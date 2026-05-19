# 10_baseline_drug_balance.jl — Step 7: drug-group baseline balance check
#
# Tests whether placebo and verum groups differ on AUC at ses-01 (pre-
# intervention baseline). Both Welch's t-test and Mann-Whitney U test are
# reported so the conclusion is robust to normality assumptions.
#
# Run from repo root:
#   julia 04_statistics/scripts/10_baseline_drug_balance.jl

using CSV, DataFrames, HypothesisTests, PlotlyJS, Statistics, StatsBase, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT  = normpath(joinpath(@__DIR__, "..", ".."))
const IN_CSV     = joinpath(REPO_ROOT, "04_statistics", "results",
                            "analysis_long_format_auc.csv")
const RES_DIR    = joinpath(REPO_ROOT, "04_statistics", "results")
const FIG_DIR    = joinpath(REPO_ROOT, "04_statistics", "figures")
const OUT_CSV    = joinpath(RES_DIR,  "baseline_summary.csv")
const OUT_FIG    = joinpath(FIG_DIR,  "fig09_baseline_drug_balance.html")

# ── Colors ────────────────────────────────────────────────────────────────────
const COLOR_PLACEBO_SOLID = "#3D5A6C"
const COLOR_VERUM_SOLID   = "#8B7355"
const COLOR_PLACEBO_FILL  = "rgba(61,90,108,0.50)"
const COLOR_VERUM_FILL    = "rgba(139,115,85,0.50)"

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(OUT_CSV) && isfile(OUT_FIG)
    println("[skip] all outputs already exist; delete to rerun")
    exit()
end

isfile(IN_CSV) || error("Missing required input: $IN_CSV")
mkpath(RES_DIR)
mkpath(FIG_DIR)

# ── Load and subset to ses-01 ─────────────────────────────────────────────────
println("Loading $IN_CSV …")
df     = CSV.read(IN_CSV, DataFrame)
ses01  = filter(r -> r.session == "ses-01", df)
@printf("  Total rows: %d  →  ses-01 subset: %d\n", nrow(df), nrow(ses01))

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Per-subject baseline summary
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 1: Per-subject baseline medians ═══\n")

self_med    = combine(groupby(filter(r -> r.atlas == "self",    ses01), :subject),
                      :auc => median => :self_baseline_median,
                      :drug_group => first => :drug_group)
nonself_med = combine(groupby(filter(r -> r.atlas == "nonself", ses01), :subject),
                      :auc => median => :nonself_baseline_median)

baseline = leftjoin(self_med, nonself_med; on=:subject)
sort!(baseline, :subject)

@printf("  %-10s  %-10s  %20s  %23s\n",
        "subject", "drug_group", "self_baseline_median", "nonself_baseline_median")
println("  " * "─" ^ 70)
for r in eachrow(baseline)
    @printf("  %-10s  %-10s  %20.4f  %23.4f\n",
            r.subject, r.drug_group, r.self_baseline_median, r.nonself_baseline_median)
end

CSV.write(OUT_CSV, select(baseline, :subject, :drug_group,
                          :self_baseline_median, :nonself_baseline_median))
println("\n  Saved: $OUT_CSV  ($(nrow(baseline)) rows)")

# ── Helper functions ──────────────────────────────────────────────────────────

function cohens_d(x, y)
    nx, ny = length(x), length(y)
    s_pool = sqrt(((nx - 1) * var(x) + (ny - 1) * var(y)) / (nx + ny - 2))
    s_pool ≈ 0 && return 0.0
    (mean(x) - mean(y)) / s_pool
end

function d_label(d)
    a = abs(d)
    a < 0.2 ? "negligible" :
    a < 0.5 ? "small"      :
    a < 0.8 ? "medium"     : "large"
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Group comparison (Welch's t-test + Mann-Whitney U)
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 2: Group comparison ═══")

# Store results for use in figure annotations
stats_results = Dict{String, NamedTuple}()

for (atlas_label, med_col) in [("self", :self_baseline_median),
                                ("nonself", :nonself_baseline_median)]
    placebo_vals = Float64.(baseline[baseline.drug_group .== "placebo", med_col])
    verum_vals   = Float64.(baseline[baseline.drug_group .== "verum",   med_col])
    n_p, n_v     = length(placebo_vals), length(verum_vals)

    # Descriptive statistics
    iqr_p = quantile(placebo_vals, 0.75) - quantile(placebo_vals, 0.25)
    iqr_v = quantile(verum_vals,   0.75) - quantile(verum_vals,   0.25)

    println()
    println("  ─── Atlas: $atlas_label ───────────────────────────────────────────")
    @printf("  %-12s  %3s  %8s  %8s  %8s  %8s\n",
            "drug_group", "n", "mean", "median", "SD", "IQR")
    @printf("  %-12s  %3d  %8.4f  %8.4f  %8.4f  %8.4f\n",
            "placebo", n_p,
            mean(placebo_vals), median(placebo_vals), std(placebo_vals), iqr_p)
    @printf("  %-12s  %3d  %8.4f  %8.4f  %8.4f  %8.4f\n",
            "verum", n_v,
            mean(verum_vals), median(verum_vals), std(verum_vals), iqr_v)

    # Welch's t-test (unequal variance)
    t_res  = UnequalVarianceTTest(placebo_vals, verum_vals)
    t_stat = t_res.t
    t_df   = t_res.df
    t_p    = pvalue(t_res)

    # Mann-Whitney U test
    mw_res = MannWhitneyUTest(placebo_vals, verum_vals)
    mw_U   = mw_res.U
    mw_p   = pvalue(mw_res)

    # Cohen's d
    d      = cohens_d(placebo_vals, verum_vals)
    d_lab  = d_label(d)

    println()
    @printf("  Welch's t-test:   t = %6.3f,  df = %5.2f,  p = %.4f\n",
            t_stat, t_df, t_p)
    @printf("  Mann-Whitney U:   U = %6.1f,               p = %.4f\n",
            mw_U, mw_p)
    @printf("  Cohen's d:        d = %+.4f  (%s effect)\n", d, d_lab)

    balanced = t_p >= 0.05 && mw_p >= 0.05 ? "balanced" : "not balanced"
    primary_p = max(t_p, mw_p)
    println()
    @printf("  → Groups are %s at baseline for %s atlas\n",
            balanced, atlas_label)
    @printf("    (Welch p=%.4f, MW p=%.4f, Cohen's d=%.3f [%s])\n",
            t_p, mw_p, d, d_lab)

    stats_results[atlas_label] = (
        t_stat=t_stat, t_df=t_df, t_p=t_p,
        mw_U=mw_U, mw_p=mw_p,
        d=d, d_lab=d_lab, balanced=balanced,
    )
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Figure
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 3: Building figure ═══")

fig = make_subplots(
    rows               = 1,
    cols               = 2,
    horizontal_spacing = 0.10,
)

for (col_idx, (atlas_label, med_col, ptitle)) in enumerate([
        ("self",    :self_baseline_median,    "Self atlas — baseline (ses-01)"),
        ("nonself", :nonself_baseline_median,  "Nonself atlas — baseline (ses-01)"),
    ])

    placebo_vals = Float64.(baseline[baseline.drug_group .== "placebo", med_col])
    verum_vals   = Float64.(baseline[baseline.drug_group .== "verum",   med_col])

    # Violin + overlaid jittered points for Placebo
    add_trace!(fig,
        violin(
            y         = placebo_vals,
            x         = fill("Placebo", length(placebo_vals)),
            name      = "Placebo",
            fillcolor = COLOR_PLACEBO_FILL,
            line      = attr(color=COLOR_PLACEBO_SOLID, width=1.5),
            marker    = attr(color=COLOR_PLACEBO_SOLID, size=6, opacity=0.85),
            points    = "all",
            jitter    = 0.35,
            pointpos  = 0,
            box       = attr(visible=true, width=0.1,
                             line=attr(color=COLOR_PLACEBO_SOLID, width=1)),
            meanline  = attr(visible=false),
            showlegend = col_idx == 1,
            legendgroup = "placebo",
        ),
        row=1, col=col_idx)

    # Violin + overlaid jittered points for Verum
    add_trace!(fig,
        violin(
            y         = verum_vals,
            x         = fill("Verum", length(verum_vals)),
            name      = "Verum",
            fillcolor = COLOR_VERUM_FILL,
            line      = attr(color=COLOR_VERUM_SOLID, width=1.5),
            marker    = attr(color=COLOR_VERUM_SOLID, size=6, opacity=0.85),
            points    = "all",
            jitter    = 0.35,
            pointpos  = 0,
            box       = attr(visible=true, width=0.1,
                             line=attr(color=COLOR_VERUM_SOLID, width=1)),
            meanline  = attr(visible=false),
            showlegend = col_idx == 1,
            legendgroup = "verum",
        ),
        row=1, col=col_idx)
end

# ── Annotations: panel titles + stats ─────────────────────────────────────────
# horizontal_spacing=0.10, 2 cols → x-domains ≈ [0, 0.45] and [0.55, 1.0]
# Panel x-centres (paper coords): 0.225 and 0.775
const PANEL_X_CENTRES = [0.225, 0.775]
const PANEL_TITLES    = ["Self atlas — baseline (ses-01)",
                         "Nonself atlas — baseline (ses-01)"]
const STATS_X_RIGHT   = [0.42, 0.98]   # right-edge anchor for stats box

annotations = Any[]

for (col_idx, atlas_label) in enumerate(["self", "nonself"])
    # ── Panel title (replaces subplot_titles) ─────────────────────────────────
    push!(annotations, attr(
        text      = "<b>$(PANEL_TITLES[col_idx])</b>",
        xref      = "paper",
        yref      = "paper",
        x         = PANEL_X_CENTRES[col_idx],
        y         = 1.01,
        xanchor   = "center",
        yanchor   = "bottom",
        showarrow = false,
        font      = attr(size=12, family="Times New Roman"),
    ))

    # ── Stats box (bottom-right of each panel) ────────────────────────────────
    s    = stats_results[atlas_label]
    text = @sprintf("t=%.3f (df=%.1f), p=%.3f<br>U=%.0f, p=%.3f<br>d=%+.3f (%s)",
                    s.t_stat, s.t_df, s.t_p,
                    s.mw_U, s.mw_p,
                    s.d, s.d_lab)
    push!(annotations, attr(
        text        = text,
        xref        = "paper",
        yref        = "paper",
        x           = STATS_X_RIGHT[col_idx],
        y           = 0.04,
        xanchor     = "right",
        yanchor     = "bottom",
        showarrow   = false,
        font        = attr(size=10, family="Times New Roman"),
        bgcolor     = "rgba(255,255,255,0.85)",
        bordercolor = "#aaaaaa",
        borderpad   = 4,
    ))
end
relayout!(fig, annotations=annotations)

relayout!(fig,
    title         = attr(
        text = "Per-subject AUC at baseline (ses-01) by drug group",
        font = attr(size=14, family="Times New Roman"),
    ),
    violinmode    = "group",
    height        = 550,
    width         = 950,
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    font          = attr(family="Times New Roman", size=12),
    legend        = attr(x=1.01, y=0.95),
    yaxis         = attr(title="Median AUC (s)"),
    yaxis2        = attr(title="Median AUC (s)"),
    margin        = attr(t=90, b=60, l=70, r=130),
)

PlotlyJS.savefig(fig, OUT_FIG)
println("  Figure written: $OUT_FIG")
println("\nDone.")
