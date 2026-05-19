# 09_per_subject_auc_distribution.jl — Step 6: per-subject AUC distribution
#
# 2×2 panel grid of per-subject AUC boxplots:
#   rows = atlas (self / nonself)
#   cols = session (ses-01 / ses-02)
# Each panel: 35 boxes (one per included subject), colored by drug group.
#
# Run from repo root:
#   julia 04_statistics/scripts/09_per_subject_auc_distribution.jl

using CSV, DataFrames, PlotlyJS, Statistics, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const IN_CSV    = joinpath(REPO_ROOT, "04_statistics", "results",
                           "analysis_long_format_auc.csv")
const FIG_DIR   = joinpath(REPO_ROOT, "04_statistics", "figures")
const FIG_PATH  = joinpath(FIG_DIR, "fig08_per_subject_auc_distribution.html")

# ── Subjects in canonical order (35; excluded: sub-06, sub-08, sub-12, sub-26, sub-36) ──
const SUBJECTS = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
    "sub-07",
    "sub-09", "sub-10", "sub-11",
    "sub-13", "sub-14", "sub-15", "sub-16", "sub-17", "sub-18",
    "sub-19", "sub-20", "sub-21", "sub-22", "sub-23", "sub-24", "sub-25",
    "sub-27", "sub-28", "sub-29", "sub-30", "sub-31", "sub-32", "sub-33",
    "sub-34", "sub-35",
    "sub-37", "sub-38", "sub-39", "sub-40"
]

# ── Colors ────────────────────────────────────────────────────────────────────
# Placebo: slate  #3D5A6C = rgb(61, 90, 108)
# Verum:   taupe  #8B7355 = rgb(139, 115, 85)
const COLOR_PLACEBO_SOLID = "#3D5A6C"
const COLOR_VERUM_SOLID   = "#8B7355"
const COLOR_PLACEBO_FILL  = "rgba(61,90,108,0.55)"
const COLOR_VERUM_FILL    = "rgba(139,115,85,0.55)"

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(FIG_PATH)
    println("[skip] output already exists at $FIG_PATH")
    exit()
end

isfile(IN_CSV) || error("Missing required input: $IN_CSV")
mkpath(FIG_DIR)

# ── Load data ─────────────────────────────────────────────────────────────────
println("Loading $IN_CSV …")
df = CSV.read(IN_CSV, DataFrame)
@printf("  Loaded %d rows × %d columns\n", nrow(df), ncol(df))
nrow(df) == 20632 || @warn "Expected 20,632 rows; got $(nrow(df))"

drug_group_dict = Dict{String,String}(row.subject => row.drug_group
                                       for row in eachrow(df))

# Short label for x-axis: "sub-01" → "01"
short(s) = s[5:end]

# ── Panel definitions ─────────────────────────────────────────────────────────
const PANELS = [
    (atlas="self",    session="ses-01", row=1, col=1, title="Self — ses-01"),
    (atlas="self",    session="ses-02", row=1, col=2, title="Self — ses-02"),
    (atlas="nonself", session="ses-01", row=2, col=1, title="Nonself — ses-01"),
    (atlas="nonself", session="ses-02", row=2, col=2, title="Nonself — ses-02"),
]

# ══════════════════════════════════════════════════════════════════════════════
# Build figure
# ══════════════════════════════════════════════════════════════════════════════
println("Building figure …")

fig = make_subplots(
    rows               = 2,
    cols               = 2,
    subplot_titles     = permutedims(reshape([p.title for p in PANELS], 2, 2)),
    horizontal_spacing = 0.04,
    vertical_spacing   = 0.14,
)

for panel in PANELS
    panel_df = filter(r -> r.atlas == panel.atlas && r.session == panel.session, df)

    for subject in SUBJECTS
        subj_auc = filter(r -> r.subject == subject, panel_df).auc
        isempty(subj_auc) && continue

        dg         = get(drug_group_dict, subject, "unknown")
        fill_color = dg == "placebo" ? COLOR_PLACEBO_FILL  : COLOR_VERUM_FILL
        line_color = dg == "placebo" ? COLOR_PLACEBO_SOLID : COLOR_VERUM_SOLID

        add_trace!(fig,
            box(
                y           = subj_auc,
                name        = short(subject),
                fillcolor   = fill_color,
                line        = attr(color=line_color, width=1),
                marker      = attr(color=line_color, size=3),
                boxpoints   = "outliers",
                legendgroup = dg,
                showlegend  = false,
            ),
            row=panel.row, col=panel.col)
    end
end

# ── Legend entries (visible="legendonly" so they appear only in the legend) ──
for (dg, label, fill, solid) in [
    ("placebo", "Placebo", COLOR_PLACEBO_FILL, COLOR_PLACEBO_SOLID),
    ("verum",   "Verum",   COLOR_VERUM_FILL,   COLOR_VERUM_SOLID),
]
    add_trace!(fig,
        box(
            y           = [0.0],
            name        = label,
            fillcolor   = fill,
            line        = attr(color=solid, width=1),
            marker      = attr(color=solid, size=1),
            legendgroup = dg,
            showlegend  = true,
            visible     = "legendonly",
        ),
        row=1, col=1)
end

# ── Shared y-axis: link yaxis2, yaxis3, yaxis4 to yaxis1 ─────────────────────
relayout!(fig, Dict(
    Symbol("yaxis2.matches") => "y",
    Symbol("yaxis3.matches") => "y",
    Symbol("yaxis4.matches") => "y",
))

# ── Rotate x-axis tick labels in all four panels ──────────────────────────────
relayout!(fig, Dict(
    Symbol("xaxis.tickangle")      => -90,
    Symbol("xaxis2.tickangle")     => -90,
    Symbol("xaxis3.tickangle")     => -90,
    Symbol("xaxis4.tickangle")     => -90,
    Symbol("xaxis.tickfont.size")  => 9,
    Symbol("xaxis2.tickfont.size") => 9,
    Symbol("xaxis3.tickfont.size") => 9,
    Symbol("xaxis4.tickfont.size") => 9,
))

relayout!(fig,
    title         = attr(
        text = "Per-subject AUC distribution by atlas × session",
        font = attr(size=15, family="Times New Roman"),
    ),
    height        = 900,
    width         = 1700,
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    font          = attr(family="Times New Roman", size=11),
    legend        = attr(x=1.01, y=0.95, bgcolor="rgba(255,255,255,0.9)",
                         bordercolor="#cccccc", borderwidth=1),
    margin        = attr(t=80, b=60, l=60, r=120),
)

PlotlyJS.savefig(fig, FIG_PATH)
println("Figure written: $FIG_PATH")

# ══════════════════════════════════════════════════════════════════════════════
# Console summary
# ══════════════════════════════════════════════════════════════════════════════
println()
println("═" ^ 72)
println("Console Summary")
println("═" ^ 72)

for panel in PANELS
    panel_df = filter(r -> r.atlas == panel.atlas && r.session == panel.session, df)

    subj_stats = combine(
        groupby(panel_df, :subject),
        :auc => median                                      => :med,
        :auc => (v -> quantile(v, 0.75) - quantile(v, 0.25)) => :iqr,
    )
    sort!(subj_stats, :med; rev=true)

    n_subj      = nrow(subj_stats)
    med_iqr     = median(subj_stats.iqr)
    iqr_thresh  = 1.5 * med_iqr
    med_of_meds = median(subj_stats.med)
    range_min   = minimum(subj_stats.med)
    range_max   = maximum(subj_stats.med)

    println()
    println("Panel: $(panel.title)  ($n_subj subjects)")
    println("  ─────────────────────────────────────────────────")

    println("  Top 3 highest median AUC:")
    for r in eachrow(first(subj_stats, 3))
        @printf("    %-10s  median=%8.4f  IQR=%7.4f\n",
                r.subject, r.med, r.iqr)
    end

    println("  Bottom 3 lowest median AUC:")
    for r in eachrow(last(subj_stats, 3))
        @printf("    %-10s  median=%8.4f  IQR=%7.4f\n",
                r.subject, r.med, r.iqr)
    end

    wide = filter(r -> r.iqr > iqr_thresh, subj_stats)
    if nrow(wide) > 0
        @printf("  Subjects with IQR > 1.5× median IQR (threshold=%.4f):\n", iqr_thresh)
        for r in eachrow(wide)
            @printf("    %-10s  IQR=%7.4f  (%.2f× median IQR)\n",
                    r.subject, r.iqr, r.iqr / med_iqr)
        end
    else
        @printf("  No subjects with IQR > 1.5× median IQR (threshold=%.4f). OK\n",
                iqr_thresh)
    end

    println("  Summary statistics:")
    @printf("    Median of medians : %.4f\n", med_of_meds)
    @printf("    Range of medians  : %.4f – %.4f\n", range_min, range_max)
    @printf("    Median IQR        : %.4f\n", med_iqr)
end

println()
println("Done.")
