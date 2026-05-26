# 14_q1_ranked_rainfall.jl — Step 11: Q1 per-ROI drug × session ranked rainfall plot
#
# Each ROI's drug × session interaction value is plotted as a jittered dot grouped
# by subgroup (layer for self atlas, CAB-NP network for nonself atlas), with mean
# bars and 95% CI brackets overlaid.
#
# Nonself network columns are sorted by mean effect (most negative → most positive)
# to reveal whether sensory vs association networks cluster differentially.
#
# Effect computation replicates Step 10 (13_q1_drug_session_histogram.jl).
#
# PATH NOTES:
#   CAB-NP file: at repo root (HCP_MMP1_atlas/ exists but lacks this file).
#   Self coords: from _old/Thesis/01_atlases/ (02_timeseries_extraction/ version absent locally).
#
# Run from repo root:
#   julia 04_statistics/scripts/14_q1_ranked_rainfall.jl

using CSV, DataFrames, PlotlyJS, Statistics, Printf, Random

# ── Paths ──────────────────────────────────────────────────────────────────────
const REPO_ROOT      = normpath(joinpath(@__DIR__, "..", ".."))
const AUC_CSV        = joinpath(REPO_ROOT, "04_statistics", "results",
                                "analysis_long_format_auc.csv")
const SELF_COORDS    = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                                "self_coordinates.txt")
const NONSELF_COORDS = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                                "glasser_coordinates_nonself_clean_1mm.txt")
const CABNP_KEY      = joinpath(REPO_ROOT,
    "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey (1).txt")
const FIG_DIR        = joinpath(REPO_ROOT, "04_statistics", "figures")
const OUT_FIG        = joinpath(FIG_DIR, "fig12_q1_ranked_rainfall.html")

# ── Idempotency ────────────────────────────────────────────────────────────────
if isfile(OUT_FIG)
    println("[skip] output already exists at $OUT_FIG")
    exit()
end

for f in [AUC_CSV, SELF_COORDS, NONSELF_COORDS, CABNP_KEY]
    isfile(f) || error("Missing required input: $f")
end
mkpath(FIG_DIR)

# ── Color palettes ─────────────────────────────────────────────────────────────
const LAYER_COLORS = Dict(
    "Interoception" => "#1f77b4",
    "Exteroception" => "#ff7f0e",
    "Cognition"     => "#2ca02c",
)

const NETWORK_COLORS = Dict(
    "Visual1"              => "#0000FF",
    "Visual2"              => "#6400FF",
    "Somatomotor"          => "#00FFFF",
    "Cingulo-Opercular"    => "#990099",
    "Dorsal-Attention"     => "#00FF00",
    "Language"             => "#009B9B",
    "Frontoparietal"       => "#FFFF00",
    "Auditory"             => "#FA3EFB",
    "Default"              => "#FF0000",
    "Posterior-Multimodal" => "#B15928",
    "Ventral-Multimodal"   => "#FF9900",
    "Orbito-Affective"     => "#417C00",
)

const LAYER_ORDER = ["Interoception", "Exteroception", "Cognition"]

# ── Geometry constants ─────────────────────────────────────────────────────────
const SELF_JITTER_HALF    = 0.12    # half-width of uniform jitter for self dots
const NONSELF_JITTER_HALF = 0.15    # wider jitter for denser nonself clusters
const SELF_BAR_HALF       = SELF_JITTER_HALF    * 1.30
const NONSELF_BAR_HALF    = NONSELF_JITTER_HALF * 1.30

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Load atlas metadata (replicates script 13)
# ══════════════════════════════════════════════════════════════════════════════
println("Loading atlas metadata …")

self_coords    = CSV.read(SELF_COORDS,    DataFrame; delim='\t')
nonself_coords = CSV.read(NONSELF_COORDS, DataFrame; delim='\t')
nonself_coords[!, :nonself_pos_id] = 1:nrow(nonself_coords)

cabnp          = CSV.read(CABNP_KEY, DataFrame; delim='\t')
cabnp_cortical = filter(r -> r.KEYVALUE <= 360, cabnp)

@printf("  Self: %d ROIs | Nonself clean: %d rows | CAB-NP cortical: %d\n",
        nrow(self_coords), nrow(nonself_coords), nrow(cabnp_cortical))

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Compute drug × session effect per ROI (replicates script 13)
# ══════════════════════════════════════════════════════════════════════════════
println("Loading AUC data and computing drug × session effects …")

auc_df = CSV.read(AUC_CSV, DataFrame)
@printf("  Loaded %d rows × %d columns\n", nrow(auc_df), ncol(auc_df))

cell_means = combine(
    groupby(auc_df, [:atlas, :roi_pos_id, :drug_group, :session]),
    :auc => mean => :mean_auc,
)

ses01 = filter(r -> r.session == "ses-01", cell_means)
ses02 = filter(r -> r.session == "ses-02", cell_means)

wide = innerjoin(
    select(ses01, :atlas, :roi_pos_id, :drug_group, :mean_auc => :auc_pre),
    select(ses02, :atlas, :roi_pos_id, :drug_group, :mean_auc => :auc_post),
    on = [:atlas, :roi_pos_id, :drug_group],
)
wide[!, :delta] = wide.auc_post .- wide.auc_pre

verum_df   = select(filter(r -> r.drug_group == "verum",   wide),
                    :atlas, :roi_pos_id, :delta => :delta_verum)
placebo_df = select(filter(r -> r.drug_group == "placebo", wide),
                    :atlas, :roi_pos_id, :delta => :delta_placebo)

roi_effects = innerjoin(verum_df, placebo_df, on = [:atlas, :roi_pos_id])
roi_effects[!, :effect] = roi_effects.delta_verum .- roi_effects.delta_placebo

self_effects    = filter(r -> r.atlas == "self",    roi_effects)
nonself_effects = filter(r -> r.atlas == "nonself", roi_effects)

# ── Attach layer / network labels ───────────────────────────────────────────────
self_info  = select(self_coords, :ROI_Number => :roi_pos_id, :ROI_Name, :Layer)
self_df    = innerjoin(self_effects, self_info, on = :roi_pos_id)

nonself_info = select(nonself_coords, :nonself_pos_id => :roi_pos_id, :ROI_Name)
nonself_df   = innerjoin(nonself_effects, nonself_info, on = :roi_pos_id)
cabnp_info   = select(cabnp_cortical, :GLASSERLABELNAME => :ROI_Name, :NETWORK)
nonself_df   = innerjoin(nonself_df, cabnp_info, on = :ROI_Name)

@printf("  Self: %d ROIs with layer | Nonself: %d ROIs with network\n",
        nrow(self_df), nrow(nonself_df))
nrow(self_df)    == 37  || @warn "Expected 37 self ROIs, got $(nrow(self_df))"
nrow(nonself_df) == 258 || @warn "Expected 258 nonself ROIs, got $(nrow(nonself_df))"

# ── Nonself network order: sorted by mean effect ────────────────────────────────
network_means = combine(groupby(nonself_df, :NETWORK), :effect => mean => :net_mean)
sort!(network_means, :net_mean)
NETWORK_ORDER = String.(network_means.NETWORK)   # most negative → most positive

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Console summary
# ══════════════════════════════════════════════════════════════════════════════

function print_atlas_summary(label, df, cluster_col, cluster_order)
    println()
    println("═" ^ 72)
    println("Atlas: $label")
    println("═" ^ 72)
    println()
    println("  Subgroups ranked by mean effect (most negative → most positive):")
    @printf("  %-28s  %4s  %+9s  %8s  %+9s  %+9s\n",
            cluster_col, "N", "Mean", "SE", "CI_lo", "CI_hi")
    println("  " * "─" ^ 70)

    non_empty = String[]
    sub_means = Float64[]
    sub_sds   = Float64[]

    for cl in cluster_order
        sub = filter(r -> getproperty(r, Symbol(cluster_col)) == cl, df)
        n   = nrow(sub)
        n == 0 && continue
        e   = sub.effect
        m   = mean(e)
        se  = n >= 2 ? std(e) / sqrt(n) : NaN
        ci_lo = isnan(se) ? NaN : m - 1.96 * se
        ci_hi = isnan(se) ? NaN : m + 1.96 * se

        push!(non_empty, string(cl))
        push!(sub_means, m)
        push!(sub_sds,   n >= 2 ? std(e) : 0.0)

        @printf("  %-28s  %4d  %+9.4f  %8.4f  %+9.4f  %+9.4f\n",
                cl, n, m,
                isnan(se)    ? 0.0 : se,
                isnan(ci_lo) ? m   : ci_lo,
                isnan(ci_hi) ? m   : ci_hi)
    end

    # Top 3 most positive and most negative overall
    println()
    sorted_pos = sort(df, :effect; rev=true)
    println("  Top 3 most positive ROIs:")
    for i in 1:min(3, nrow(sorted_pos))
        r = sorted_pos[i, :]
        @printf("    %d.  %-38s  %+.4f\n", i, r.ROI_Name, r.effect)
    end

    sorted_neg = sort(df, :effect)
    println("  Top 3 most negative ROIs:")
    for i in 1:min(3, nrow(sorted_neg))
        r = sorted_neg[i, :]
        @printf("    %d.  %-38s  %+.4f\n", i, r.ROI_Name, r.effect)
    end

    # One-line interpretation
    if length(non_empty) >= 2
        between_spread = maximum(sub_means) - minimum(sub_means)
        within_avg_sd  = mean(sub_sds)
        ratio          = within_avg_sd / max(between_spread, 1e-9)

        spread_desc  = ratio > 1.5  ? "larger than" :
                       ratio < 0.67 ? "smaller than" : "similar to"
        most_neg_cl  = non_empty[argmin(sub_means)]
        most_pos_cl  = non_empty[argmax(sub_means)]

        println()
        @printf("  Interpretation: %s subgroups range from %s (mean=%+.4f) to %s " *
                "(mean=%+.4f); within-subgroup spread is %s between-subgroup differences.\n",
                label,
                most_neg_cl, minimum(sub_means),
                most_pos_cl, maximum(sub_means),
                spread_desc)
    end
end

print_atlas_summary("Self",    self_df,    "Layer",   LAYER_ORDER)
print_atlas_summary("Nonself", nonself_df, "NETWORK", NETWORK_ORDER)

# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Build figure
# ══════════════════════════════════════════════════════════════════════════════
println()
println("Building figure …")

rng = MersenneTwister(42)   # fixed seed for reproducible jitter

# Shared y-axis range (symmetric around 0, 12% padding)
all_effects = vcat(self_df.effect, nonself_df.effect)
y_abs_max   = maximum(abs.(all_effects)) * 1.12
y_range     = [-y_abs_max, y_abs_max]

# ── Helper: add dots + mean bar + CI bracket for one cluster ───────────────────
# Returns Vector{Tuple{String,Float64,Float64}} of (roi_name, x_pos, y_val)
function add_cluster!(fig, cluster_name, effects, roi_names, x_center::Float64,
                      color, panel_col::Int, rng, jitter_half::Float64,
                      bar_half::Float64)
    N  = length(effects)
    jx = x_center .+ (rand(rng, N) .- 0.5) .* (2.0 * jitter_half)

    # Jittered dots
    add_trace!(fig,
        scatter(
            x             = jx,
            y             = effects,
            mode          = "markers",
            marker        = attr(color=color, size=6, opacity=0.7,
                                 line=attr(width=0.5, color="rgba(0,0,0,0.25)")),
            name          = cluster_name,
            customdata    = collect(roi_names),
            hovertemplate = "%{customdata}<br>Effect: %{y:.4f}<extra></extra>",
            legendgroup   = cluster_name,
        ),
        row=1, col=panel_col)

    # Mean bar (horizontal)
    m = mean(effects)
    add_trace!(fig,
        scatter(
            x           = [x_center - bar_half, x_center + bar_half],
            y           = [m, m],
            mode        = "lines",
            line        = attr(color="#2C2C2A", width=3),
            showlegend  = false,
            hoverinfo   = "skip",
            legendgroup = cluster_name,
        ),
        row=1, col=panel_col)

    # 95% CI bracket (vertical); skip if N < 3
    if N >= 3
        ci_half = 1.96 * std(effects) / sqrt(N)
        add_trace!(fig,
            scatter(
                x           = [x_center, x_center],
                y           = [m - ci_half, m + ci_half],
                mode        = "lines",
                line        = attr(color="#2C2C2A", width=1.5),
                showlegend  = false,
                hoverinfo   = "skip",
                legendgroup = cluster_name,
            ),
            row=1, col=panel_col)
    end

    return collect(zip(collect(roi_names), jx, effects))
end

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08)

# ── Left panel: self atlas, layers in fixed order ──────────────────────────────
self_dot_positions = Tuple{String, Float64, Float64}[]
self_tick_vals     = Float64[]
self_tick_texts    = String[]

for (i, layer) in enumerate(LAYER_ORDER)
    sub   = filter(r -> r.Layer == layer, self_df)
    n     = nrow(sub)
    color = LAYER_COLORS[layer]
    dots  = add_cluster!(fig, "$layer (N=$n)", sub.effect, sub.ROI_Name,
                         Float64(i), color, 1, rng,
                         SELF_JITTER_HALF, SELF_BAR_HALF)
    append!(self_dot_positions, dots)
    push!(self_tick_vals,  Float64(i))
    push!(self_tick_texts, "$layer (N=$n)")
end

# ── Right panel: nonself atlas, networks sorted by mean effect ──────────────────
nonself_dot_positions = Tuple{String, Float64, Float64}[]
net_tick_vals         = Float64[]
net_tick_texts        = String[]

for (i, net) in enumerate(NETWORK_ORDER)
    sub   = filter(r -> r.NETWORK == net, nonself_df)
    n     = nrow(sub)
    n == 0 && continue
    color = get(NETWORK_COLORS, net, "#888888")
    dots  = add_cluster!(fig, "$net (N=$n)", sub.effect, sub.ROI_Name,
                         Float64(i), color, 2, rng,
                         NONSELF_JITTER_HALF, NONSELF_BAR_HALF)
    append!(nonself_dot_positions, dots)
    push!(net_tick_vals,  Float64(i))
    push!(net_tick_texts, "$net (N=$n)")
end

# ── Extreme-ROI annotations (top 3 most positive + top 3 most negative) ────────
function extreme_annotations(dot_positions, xref_str, yref_str; n_extreme=3)
    sorted = sort(collect(dot_positions), by=t -> t[3])
    n      = length(sorted)
    ann    = Any[]

    for (name, x, y) in sorted[1:min(n_extreme, n)]
        push!(ann, attr(
            x=x, y=y, text=name, showarrow=true,
            arrowhead=2, arrowsize=0.7, arrowwidth=1.0,
            ax=30, ay=20,
            xref=xref_str, yref=yref_str,
            font=attr(size=7, family="Times New Roman"),
            bgcolor="rgba(255,255,255,0.75)",
            borderpad=2,
        ))
    end

    for (name, x, y) in sorted[max(1, n - n_extreme + 1):n]
        push!(ann, attr(
            x=x, y=y, text=name, showarrow=true,
            arrowhead=2, arrowsize=0.7, arrowwidth=1.0,
            ax=30, ay=-20,
            xref=xref_str, yref=yref_str,
            font=attr(size=7, family="Times New Roman"),
            bgcolor="rgba(255,255,255,0.75)",
            borderpad=2,
        ))
    end
    return ann
end

self_ann    = extreme_annotations(self_dot_positions,    "x",  "y")
nonself_ann = extreme_annotations(nonself_dot_positions, "x2", "y2")

# Panel title annotations (horizontal_spacing=0.08 → panel_width=0.46;
# left centre x=0.23, right centre x=0.77)
panel_title_ann = Any[
    attr(
        text    = "<b>Self atlas — drug × session effect by layer</b>",
        xref    = "paper", yref    = "paper",
        x       = 0.23,    y       = 1.04,
        xanchor = "center", yanchor = "bottom",
        showarrow = false,
        font    = attr(size=12, family="Times New Roman"),
    ),
    attr(
        text    = "<b>Nonself atlas — drug × session effect by CAB-NP network (sorted by mean)</b>",
        xref    = "paper", yref    = "paper",
        x       = 0.77,    y       = 1.04,
        xanchor = "center", yanchor = "bottom",
        showarrow = false,
        font    = attr(size=12, family="Times New Roman"),
    ),
]

all_annotations = vcat(panel_title_ann, self_ann, nonself_ann)

# ── y = 0 reference lines ───────────────────────────────────────────────────────
n_nets = length(net_tick_vals)
shapes = [
    attr(type="line",
         x0=0.5, x1=3.5,                     y0=0.0, y1=0.0,
         xref="x",  yref="y",
         line=attr(color="#888888", width=1.0, dash="dash")),
    attr(type="line",
         x0=0.5, x1=Float64(n_nets) + 0.5,   y0=0.0, y1=0.0,
         xref="x2", yref="y2",
         line=attr(color="#888888", width=1.0, dash="dash")),
]

relayout!(fig,
    title         = attr(
        text = "Question 1: Per-ROI drug × session effect, ranked by subgroup",
        font = attr(size=14, family="Times New Roman"),
        x    = 0.5,
    ),
    height        = 660,
    width         = 1600,
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    font          = attr(family="Times New Roman", size=10),
    annotations   = all_annotations,
    shapes        = shapes,
    xaxis  = attr(
        tickvals  = self_tick_vals,
        ticktext  = self_tick_texts,
        tickangle = 0,
        range     = [0.5, 3.5],
        showgrid  = false,
        zeroline  = false,
    ),
    xaxis2 = attr(
        tickvals  = net_tick_vals,
        ticktext  = net_tick_texts,
        tickangle = -45,
        range     = [0.5, Float64(n_nets) + 0.5],
        showgrid  = false,
        zeroline  = false,
    ),
    yaxis  = attr(
        title     = "Drug × Session effect (AUC, s)",
        range     = y_range,
        zeroline  = false,
        gridcolor = "rgba(200,200,200,0.5)",
        gridwidth = 1,
    ),
    yaxis2 = attr(
        title     = "Drug × Session effect (AUC, s)",
        range     = y_range,
        zeroline  = false,
        gridcolor = "rgba(200,200,200,0.5)",
        gridwidth = 1,
    ),
    legend = attr(
        x           = 1.01,
        y           = 0.99,
        xanchor     = "left",
        yanchor     = "top",
        bgcolor     = "rgba(255,255,255,0.9)",
        bordercolor = "#cccccc",
        borderwidth = 1,
        font        = attr(size=8),
        tracegroupgap = 4,
    ),
    margin = attr(t=100, b=130, l=70, r=210),
)

PlotlyJS.savefig(fig, OUT_FIG)
println("  Figure written: $OUT_FIG")
println()
println("Done.")
