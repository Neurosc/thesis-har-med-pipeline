# 15_q1_pre_post_change_by_group.jl — Step 12: Q1 pre→post change split by drug group
#
# Decomposes the drug×session interaction from Step 11 into two components:
#   placebo_change[ROI] = mean(placebo_post − placebo_pre)
#   verum_change[ROI]   = mean(verum_post   − verum_pre)
#
# Four-panel 2×2 figure (same ranked rainfall layout as fig12):
#   Top-left:    Self atlas — Placebo change
#   Top-right:   Self atlas — Verum change
#   Bottom-left: Nonself atlas — Placebo change
#   Bottom-right: Nonself atlas — Verum change
#
# Network/layer column ordering is identical across all four panels
# (sorted by drug×session interaction, replicating Step 11's ordering).
#
# PATH NOTES:
#   CAB-NP file: repo root (HCP_MMP1_atlas/ lacks this file).
#   Self coords: _old/Thesis/01_atlases/ (02_timeseries_extraction/ absent locally).
#
# Run from repo root:
#   julia 04_statistics/scripts/15_q1_pre_post_change_by_group.jl

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
const RES_DIR        = joinpath(REPO_ROOT, "04_statistics", "results")
const OUT_FIG        = joinpath(FIG_DIR, "fig13_q1_pre_post_change_by_group.html")
const OUT_CSV        = joinpath(RES_DIR, "pre_post_change_by_group.csv")

# ── Idempotency ────────────────────────────────────────────────────────────────
if isfile(OUT_FIG) && isfile(OUT_CSV)
    println("[skip] both outputs already exist; delete to rerun")
    exit()
end

for f in [AUC_CSV, SELF_COORDS, NONSELF_COORDS, CABNP_KEY]
    isfile(f) || error("Missing required input: $f")
end
mkpath(FIG_DIR)
mkpath(RES_DIR)

# ── Constants ──────────────────────────────────────────────────────────────────
const PLACEBO_COLOR = "#3D5A6C"
const VERUM_COLOR   = "#8B7355"
const LAYER_ORDER   = ["Interoception", "Exteroception", "Cognition"]

const SELF_JITTER_HALF    = 0.12
const NONSELF_JITTER_HALF = 0.15
const SELF_BAR_HALF       = SELF_JITTER_HALF    * 1.30
const NONSELF_BAR_HALF    = NONSELF_JITTER_HALF * 1.30

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Load atlas metadata
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
# Part 2 — Load AUC data, compute per-ROI group changes
# ══════════════════════════════════════════════════════════════════════════════
println("Loading AUC data …")
auc_df = CSV.read(AUC_CSV, DataFrame)
@printf("  Loaded %d rows × %d columns\n", nrow(auc_df), ncol(auc_df))

# Subject counts per group
subj_grp  = unique(select(auc_df, :subject, :drug_group))
n_placebo = count(r -> r.drug_group == "placebo", eachrow(subj_grp))
n_verum   = count(r -> r.drug_group == "verum",   eachrow(subj_grp))
@printf("  Subjects: %d placebo, %d verum\n", n_placebo, n_verum)

println("Computing pre→post changes per group …")

cell_means = combine(
    groupby(auc_df, [:atlas, :roi_pos_id, :drug_group, :session]),
    :auc => mean => :mean_auc,
)

ses01 = filter(r -> r.session == "ses-01", cell_means)
ses02 = filter(r -> r.session == "ses-02", cell_means)

deltas = innerjoin(
    select(ses01, :atlas, :roi_pos_id, :drug_group, :mean_auc => :auc_pre),
    select(ses02, :atlas, :roi_pos_id, :drug_group, :mean_auc => :auc_post),
    on = [:atlas, :roi_pos_id, :drug_group],
)
deltas[!, :delta] = deltas.auc_post .- deltas.auc_pre

placebo_delta = select(filter(r -> r.drug_group == "placebo", deltas),
                       :atlas, :roi_pos_id, :delta => :placebo_change)
verum_delta   = select(filter(r -> r.drug_group == "verum",   deltas),
                       :atlas, :roi_pos_id, :delta => :verum_change)

roi_data = innerjoin(placebo_delta, verum_delta, on = [:atlas, :roi_pos_id])
roi_data[!, :interaction] = roi_data.verum_change .- roi_data.placebo_change

# ── Attach atlas metadata ───────────────────────────────────────────────────────
self_info    = select(self_coords, :ROI_Number => :roi_pos_id, :ROI_Name, :Layer)
nonself_info = select(nonself_coords, :nonself_pos_id => :roi_pos_id, :ROI_Name)
cabnp_info   = select(cabnp_cortical, :GLASSERLABELNAME => :ROI_Name, :NETWORK)

self_base    = filter(r -> r.atlas == "self",    roi_data)
nonself_base = filter(r -> r.atlas == "nonself", roi_data)

self_full   = innerjoin(self_base,    self_info,    on = :roi_pos_id)
nonself_tmp = innerjoin(nonself_base, nonself_info, on = :roi_pos_id)
nonself_full = innerjoin(nonself_tmp, cabnp_info,   on = :ROI_Name)

# Unified subgroup column
self_full[!,    :subgroup] = String.(self_full.Layer)
nonself_full[!, :subgroup] = String.(nonself_full.NETWORK)

@printf("  Self: %d ROIs | Nonself: %d ROIs\n", nrow(self_full), nrow(nonself_full))
nrow(self_full)    == 37  || @warn "Expected 37 self ROIs, got $(nrow(self_full))"
nrow(nonself_full) == 258 || @warn "Expected 258 nonself ROIs, got $(nrow(nonself_full))"

# ── Network ordering: identical to Step 11 (sort by mean interaction) ───────────
net_ix    = combine(groupby(nonself_full, :subgroup), :interaction => mean => :mean_ix)
sort!(net_ix, :mean_ix)
sorted_networks = String.(net_ix.subgroup)   # most negative → most positive

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Console output
# ══════════════════════════════════════════════════════════════════════════════

function print_group_stats(label, df, change_col::Symbol, cluster_order)
    println()
    println("═" ^ 72)
    println(label)
    println("═" ^ 72)

    changes = df[!, change_col]
    @printf("  Mean change (all ROIs): %+.4f\n", mean(changes))
    @printf("  N ROIs positive: %d  |  N ROIs negative: %d\n",
            count(>(0), changes), count(<(0), changes))

    println()
    println("  Per-subgroup breakdown:")
    @printf("  %-28s  %4s  %+9s  %8s  %+9s  %+9s\n",
            "Subgroup", "N", "Mean", "SE", "CI_lo", "CI_hi")
    println("  " * "─"^68)

    for cl in cluster_order
        sub = filter(r -> r.subgroup == cl, df)
        n   = nrow(sub)
        n == 0 && continue
        e   = sub[!, change_col]
        m   = mean(e)
        se  = n >= 2 ? std(e) / sqrt(n) : NaN
        ci_lo = isnan(se) ? NaN : m - 1.96 * se
        ci_hi = isnan(se) ? NaN : m + 1.96 * se
        @printf("  %-28s  %4d  %+9.4f  %8.4f  %+9.4f  %+9.4f\n",
                cl, n, m,
                isnan(se)    ? 0.0 : se,
                isnan(ci_lo) ? m   : ci_lo,
                isnan(ci_hi) ? m   : ci_hi)
    end

    sorted_pos = sort(df, change_col; rev=true)
    println()
    println("  Top 3 ROIs — largest positive change:")
    for i in 1:min(3, nrow(sorted_pos))
        r = sorted_pos[i, :]
        @printf("    %d. %-38s  %+.4f\n", i, r.ROI_Name, getproperty(r, change_col))
    end

    sorted_neg = sort(df, change_col)
    println("  Top 3 ROIs — largest negative change:")
    for i in 1:min(3, nrow(sorted_neg))
        r = sorted_neg[i, :]
        @printf("    %d. %-38s  %+.4f\n", i, r.ROI_Name, getproperty(r, change_col))
    end
end

function print_comparison_table(atlas_label, df, cluster_order)
    println()
    println("═" ^ 72)
    println("$atlas_label — Placebo vs Verum per subgroup")
    println("═" ^ 72)
    @printf("  %-28s  %+12s  %+12s  %+12s  %s\n",
            "Subgroup", "Placebo Δ", "Verum Δ", "Interaction", "Pattern")
    println("  " * "─"^76)

    for cl in cluster_order
        sub = filter(r -> r.subgroup == cl, df)
        nrow(sub) == 0 && continue
        p  = mean(sub.placebo_change)
        v  = mean(sub.verum_change)
        ix = mean(sub.interaction)
        pattern = if v > 0 && p > 0
            v > p ? "↑↑ verum>placebo" : "↑↑ placebo≥verum"
        elseif v > 0 && p <= 0
            "↑ verum, ↓ placebo"
        elseif v <= 0 && p > 0
            "↓ verum, ↑ placebo"
        else
            abs(v) > abs(p) ? "↓↓ verum>placebo" : "↓↓ placebo≥verum"
        end
        @printf("  %-28s  %+12.4f  %+12.4f  %+12.4f  %s\n", cl, p, v, ix, pattern)
    end
end

print_group_stats("Self atlas — Placebo change",    self_full,    :placebo_change, LAYER_ORDER)
print_group_stats("Self atlas — Verum change",      self_full,    :verum_change,   LAYER_ORDER)
print_group_stats("Nonself atlas — Placebo change", nonself_full, :placebo_change, sorted_networks)
print_group_stats("Nonself atlas — Verum change",   nonself_full, :verum_change,   sorted_networks)

print_comparison_table("Self atlas",    self_full,    LAYER_ORDER)
print_comparison_table("Nonself atlas", nonself_full, sorted_networks)

# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Save CSV
# ══════════════════════════════════════════════════════════════════════════════
println("\nSaving CSV …")

csv_self    = select(self_full,    :atlas, :roi_pos_id,
                     :ROI_Name => :roi_name, :subgroup,
                     :placebo_change, :verum_change, :interaction)
csv_nonself = select(nonself_full, :atlas, :roi_pos_id,
                     :ROI_Name => :roi_name, :subgroup,
                     :placebo_change, :verum_change, :interaction)

roi_csv = vcat(csv_self, csv_nonself)
sort!(roi_csv, [:atlas, :subgroup, :interaction]; rev=[false, false, true])

CSV.write(OUT_CSV, roi_csv)
@printf("  Saved %d rows → %s\n", nrow(roi_csv), OUT_CSV)

# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — Build figure
# ══════════════════════════════════════════════════════════════════════════════
println("\nBuilding figure …")

rng = MersenneTwister(42)

# Shared y-axis range across all four panels
all_changes = vcat(self_full.placebo_change, self_full.verum_change,
                   nonself_full.placebo_change, nonself_full.verum_change)
y_abs_max = maximum(abs.(all_changes)) * 1.12
y_range   = [-y_abs_max, y_abs_max]

# Tick labels — same for both panels in each row
self_tick_vals  = Float64.(1:length(LAYER_ORDER))
self_tick_texts = [begin
    n = count(r -> r.subgroup == l, eachrow(self_full))
    "$l (N=$n)"
end for l in LAYER_ORDER]

n_nets         = length(sorted_networks)
net_tick_vals  = Float64.(1:n_nets)
net_tick_texts = [begin
    n = count(r -> r.subgroup == net, eachrow(nonself_full))
    "$net (N=$n)"
end for net in sorted_networks]

# ── Cluster helper: dots + mean bar + CI bracket ───────────────────────────────
function add_cluster!(fig, hover_name, changes, x_center::Float64,
                      color, panel_row::Int, panel_col::Int, rng,
                      jitter_half::Float64, bar_half::Float64)
    N  = length(changes)
    jx = x_center .+ (rand(rng, N) .- 0.5) .* (2.0 * jitter_half)

    add_trace!(fig,
        scatter(
            x             = jx,
            y             = changes,
            mode          = "markers",
            marker        = attr(color=color, size=6, opacity=0.7,
                                 line=attr(width=0.5, color="rgba(0,0,0,0.2)")),
            name          = hover_name,
            showlegend    = false,
            hovertemplate = "$hover_name<br>Change: %{y:.4f}<extra></extra>",
        ),
        row=panel_row, col=panel_col)

    m = mean(changes)
    add_trace!(fig,
        scatter(
            x          = [x_center - bar_half, x_center + bar_half],
            y          = [m, m],
            mode       = "lines",
            line       = attr(color="#2C2C2A", width=3),
            showlegend = false,
            hoverinfo  = "skip",
        ),
        row=panel_row, col=panel_col)

    if N >= 3
        ci = 1.96 * std(changes) / sqrt(N)
        add_trace!(fig,
            scatter(
                x          = [x_center, x_center],
                y          = [m - ci, m + ci],
                mode       = "lines",
                line       = attr(color="#2C2C2A", width=1.5),
                showlegend = false,
                hoverinfo  = "skip",
            ),
            row=panel_row, col=panel_col)
    end
end

fig = make_subplots(
    rows               = 2,
    cols               = 2,
    horizontal_spacing = 0.10,
    vertical_spacing   = 0.25,
)

# ── Panel 1 (row=1, col=1): Self — Placebo ──────────────────────────────────────
for (i, layer) in enumerate(LAYER_ORDER)
    sub = filter(r -> r.subgroup == layer, self_full)
    add_cluster!(fig, layer, sub.placebo_change, Float64(i),
                 PLACEBO_COLOR, 1, 1, rng, SELF_JITTER_HALF, SELF_BAR_HALF)
end

# ── Panel 2 (row=1, col=2): Self — Verum ───────────────────────────────────────
for (i, layer) in enumerate(LAYER_ORDER)
    sub = filter(r -> r.subgroup == layer, self_full)
    add_cluster!(fig, layer, sub.verum_change, Float64(i),
                 VERUM_COLOR, 1, 2, rng, SELF_JITTER_HALF, SELF_BAR_HALF)
end

# ── Panel 3 (row=2, col=1): Nonself — Placebo ──────────────────────────────────
for (i, net) in enumerate(sorted_networks)
    sub = filter(r -> r.subgroup == net, nonself_full)
    nrow(sub) == 0 && continue
    add_cluster!(fig, net, sub.placebo_change, Float64(i),
                 PLACEBO_COLOR, 2, 1, rng, NONSELF_JITTER_HALF, NONSELF_BAR_HALF)
end

# ── Panel 4 (row=2, col=2): Nonself — Verum ────────────────────────────────────
for (i, net) in enumerate(sorted_networks)
    sub = filter(r -> r.subgroup == net, nonself_full)
    nrow(sub) == 0 && continue
    add_cluster!(fig, net, sub.verum_change, Float64(i),
                 VERUM_COLOR, 2, 2, rng, NONSELF_JITTER_HALF, NONSELF_BAR_HALF)
end

# ── Legend: empty traces to register group colors ──────────────────────────────
add_trace!(fig,
    scatter(x=Float64[], y=Float64[], mode="markers",
            marker=attr(color=PLACEBO_COLOR, size=7, opacity=0.7),
            name="Placebo (n=$n_placebo)", showlegend=true),
    row=1, col=1)
add_trace!(fig,
    scatter(x=Float64[], y=Float64[], mode="markers",
            marker=attr(color=VERUM_COLOR, size=7, opacity=0.7),
            name="Verum (n=$n_verum)", showlegend=true),
    row=1, col=1)

# ── y = 0 reference lines ───────────────────────────────────────────────────────
shapes = [
    attr(type="line", x0=0.5, x1=3.5,                     y0=0.0, y1=0.0,
         xref="x",  yref="y",  line=attr(color="#888888", width=1.0, dash="dash")),
    attr(type="line", x0=0.5, x1=3.5,                     y0=0.0, y1=0.0,
         xref="x2", yref="y2", line=attr(color="#888888", width=1.0, dash="dash")),
    attr(type="line", x0=0.5, x1=Float64(n_nets) + 0.5,   y0=0.0, y1=0.0,
         xref="x3", yref="y3", line=attr(color="#888888", width=1.0, dash="dash")),
    attr(type="line", x0=0.5, x1=Float64(n_nets) + 0.5,   y0=0.0, y1=0.0,
         xref="x4", yref="y4", line=attr(color="#888888", width=1.0, dash="dash")),
]

# ── Panel title annotations ─────────────────────────────────────────────────────
# horizontal_spacing=0.10 → panel_width=0.45; col centres: 0.225, 0.775
# vertical_spacing=0.25   → row tops: row1=1.00, row2=0.375; title y: 1.03, 0.41
panel_ann = Any[
    attr(text="<b>Self atlas — Placebo change (n=$n_placebo)</b>",
         xref="paper", yref="paper", x=0.225, y=1.03,
         xanchor="center", yanchor="bottom", showarrow=false,
         font=attr(size=11, family="Times New Roman")),
    attr(text="<b>Self atlas — Verum change (n=$n_verum)</b>",
         xref="paper", yref="paper", x=0.775, y=1.03,
         xanchor="center", yanchor="bottom", showarrow=false,
         font=attr(size=11, family="Times New Roman")),
    attr(text="<b>Nonself atlas — Placebo change (n=$n_placebo)</b>",
         xref="paper", yref="paper", x=0.225, y=0.41,
         xanchor="center", yanchor="bottom", showarrow=false,
         font=attr(size=11, family="Times New Roman")),
    attr(text="<b>Nonself atlas — Verum change (n=$n_verum)</b>",
         xref="paper", yref="paper", x=0.775, y=0.41,
         xanchor="center", yanchor="bottom", showarrow=false,
         font=attr(size=11, family="Times New Roman")),
]

relayout!(fig,
    title  = attr(
        text = "Pre→post AUC change per ROI, split by drug group",
        font = attr(size=14, family="Times New Roman"),
        x    = 0.5,
    ),
    height        = 880,
    width         = 1400,
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    font          = attr(family="Times New Roman", size=10),
    annotations   = panel_ann,
    shapes        = shapes,
    # Top row (self, 3 layers)
    xaxis  = attr(tickvals=self_tick_vals, ticktext=self_tick_texts,
                  tickangle=0, range=[0.5, 3.5], showgrid=false, zeroline=false),
    xaxis2 = attr(tickvals=self_tick_vals, ticktext=self_tick_texts,
                  tickangle=0, range=[0.5, 3.5], showgrid=false, zeroline=false),
    # Bottom row (nonself, n_nets networks)
    xaxis3 = attr(tickvals=net_tick_vals, ticktext=net_tick_texts,
                  tickangle=-45, range=[0.5, Float64(n_nets) + 0.5],
                  showgrid=false, zeroline=false),
    xaxis4 = attr(tickvals=net_tick_vals, ticktext=net_tick_texts,
                  tickangle=-45, range=[0.5, Float64(n_nets) + 0.5],
                  showgrid=false, zeroline=false),
    # Y-axes: identical range; left panels get axis label
    yaxis  = attr(title="Pre→post change (AUC, s)", range=y_range, zeroline=false,
                  gridcolor="rgba(200,200,200,0.5)", gridwidth=1),
    yaxis2 = attr(range=y_range, zeroline=false,
                  gridcolor="rgba(200,200,200,0.5)", gridwidth=1),
    yaxis3 = attr(title="Pre→post change (AUC, s)", range=y_range, zeroline=false,
                  gridcolor="rgba(200,200,200,0.5)", gridwidth=1),
    yaxis4 = attr(range=y_range, zeroline=false,
                  gridcolor="rgba(200,200,200,0.5)", gridwidth=1),
    legend = attr(
        x=1.01, y=0.99, xanchor="left", yanchor="top",
        bgcolor="rgba(255,255,255,0.9)", bordercolor="#cccccc", borderwidth=1,
        font=attr(size=9),
    ),
    margin = attr(t=100, b=150, l=70, r=160),
)

PlotlyJS.savefig(fig, OUT_FIG)
println("  Figure written: $OUT_FIG")
println()
println("Done.")
