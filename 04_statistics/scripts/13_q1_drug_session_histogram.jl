# 13_q1_drug_session_histogram.jl — Step 10: Q1 drug × session interaction per ROI
#
# For each ROI (self and nonself) compute the drug × session interaction value:
#   effect[ROI] = (mean verum post − mean verum pre) − (mean placebo post − mean placebo pre)
# No statistical test — visual inspection only.
#
# Left panel:  self atlas (37 ROIs) stacked histogram, colored by layer
#              (Interoception / Exteroception / Cognition)
# Right panel: nonself atlas (258 ROIs) stacked histogram, colored by CAB-NP network
#
# Join path for nonself:
#   roi_pos_id → row position in glasser_coordinates_nonself_clean_1mm.txt → ROI_Name
#   ROI_Name = GLASSERLABELNAME in CAB-NP label key → NETWORK
#
# Run from repo root:
#   julia 04_statistics/scripts/13_q1_drug_session_histogram.jl

using CSV, DataFrames, PlotlyJS, Statistics, Printf

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
const OUT_FIG        = joinpath(FIG_DIR, "fig11_q1_drug_session_histogram.html")

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

const LAYER_ORDER   = ["Interoception", "Exteroception", "Cognition"]
const NETWORK_ORDER = ["Visual1", "Visual2", "Somatomotor", "Cingulo-Opercular",
                       "Dorsal-Attention", "Language", "Frontoparietal", "Auditory",
                       "Default", "Posterior-Multimodal", "Ventral-Multimodal",
                       "Orbito-Affective"]

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Load and join atlas metadata
# ══════════════════════════════════════════════════════════════════════════════
println("Loading atlas metadata …")

self_coords = CSV.read(SELF_COORDS, DataFrame; delim='\t')
@printf("  Self coords:    %d ROIs\n", nrow(self_coords))

nonself_coords = CSV.read(NONSELF_COORDS, DataFrame; delim='\t')
nonself_coords[!, :nonself_pos_id] = 1:nrow(nonself_coords)   # row order = position
@printf("  Nonself coords: %d ROIs (Glasser clean 1mm)\n", nrow(nonself_coords))

cabnp = CSV.read(CABNP_KEY, DataFrame; delim='\t')
cabnp_cortical = filter(r -> r.KEYVALUE <= 360, cabnp)         # drop subcortical
@printf("  CAB-NP cortical: %d rows (KEYVALUE 1-360)\n", nrow(cabnp_cortical))

# Verify uniqueness of GLASSERLABELNAME in CAB-NP cortical
label_counts = combine(groupby(cabnp_cortical, :GLASSERLABELNAME), nrow => :n)
dups = filter(r -> r.n > 1, label_counts)
nrow(dups) > 0 && @warn "Duplicate GLASSERLABELNAME in CAB-NP: $(dups.GLASSERLABELNAME)"

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Compute drug × session interaction per ROI
# ══════════════════════════════════════════════════════════════════════════════
println("Loading AUC data …")
auc_df = CSV.read(AUC_CSV, DataFrame)
@printf("  Loaded %d rows × %d columns\n", nrow(auc_df), ncol(auc_df))

println("Computing drug × session interaction per ROI …")

# Mean AUC per (atlas, roi_pos_id, drug_group, session)
cell_means = combine(
    groupby(auc_df, [:atlas, :roi_pos_id, :drug_group, :session]),
    :auc => mean => :mean_auc,
)

# Pre/post split; join to get delta = post − pre per (atlas, roi_pos_id, drug_group)
ses01 = filter(r -> r.session == "ses-01", cell_means)
ses02 = filter(r -> r.session == "ses-02", cell_means)

wide = innerjoin(
    select(ses01, :atlas, :roi_pos_id, :drug_group, :mean_auc => :auc_pre),
    select(ses02, :atlas, :roi_pos_id, :drug_group, :mean_auc => :auc_post),
    on = [:atlas, :roi_pos_id, :drug_group],
)
wide[!, :delta] = wide.auc_post .- wide.auc_pre

# Pivot drug_group to wide: effect = delta_verum − delta_placebo
verum_df   = select(filter(r -> r.drug_group == "verum",   wide),
                    :atlas, :roi_pos_id, :delta => :delta_verum)
placebo_df = select(filter(r -> r.drug_group == "placebo", wide),
                    :atlas, :roi_pos_id, :delta => :delta_placebo)

roi_effects = innerjoin(verum_df, placebo_df, on = [:atlas, :roi_pos_id])
roi_effects[!, :effect] = roi_effects.delta_verum .- roi_effects.delta_placebo

self_effects    = filter(r -> r.atlas == "self",    roi_effects)
nonself_effects = filter(r -> r.atlas == "nonself", roi_effects)

@printf("  Self effects:    %d ROIs\n", nrow(self_effects))
@printf("  Nonself effects: %d ROIs\n", nrow(nonself_effects))

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Attach layer / network labels
# ══════════════════════════════════════════════════════════════════════════════

# Self: roi_pos_id = ROI_Number in self_coords (1-37)
self_info = select(self_coords, :ROI_Number => :roi_pos_id, :ROI_Name, :Layer)
self_df   = innerjoin(self_effects, self_info, on = :roi_pos_id)

# Nonself: roi_pos_id = row position in nonself_coords; ROI_Name → GLASSERLABELNAME → NETWORK
nonself_info = select(nonself_coords, :nonself_pos_id => :roi_pos_id, :ROI_Name)
nonself_df   = innerjoin(nonself_effects, nonself_info, on = :roi_pos_id)

cabnp_info   = select(cabnp_cortical, :GLASSERLABELNAME => :ROI_Name, :NETWORK)
nonself_df   = innerjoin(nonself_df, cabnp_info, on = :ROI_Name)

@printf("  Self with layer:    %d ROIs\n", nrow(self_df))
@printf("  Nonself with network: %d ROIs\n", nrow(nonself_df))

nrow(self_df)    == 37  || @warn "Expected 37 self ROIs, got $(nrow(self_df))"
nrow(nonself_df) == 258 || @warn "Expected 258 nonself ROIs, got $(nrow(nonself_df))"

# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Console summary
# ══════════════════════════════════════════════════════════════════════════════

function print_atlas_summary(label, df, cluster_col, cluster_order)
    println()
    println("═" ^ 72)
    println("Atlas: $label")
    println("═" ^ 72)

    effs = df.effect
    @printf("  Total N:    %d\n",    length(effs))
    @printf("  Mean:       %+.4f\n", mean(effs))
    @printf("  Median:     %+.4f\n", median(effs))
    @printf("  SD:         %.4f\n",  std(effs))
    @printf("  N positive: %d\n",    count(>(0), effs))
    @printf("  N negative: %d\n",    count(<(0), effs))
    @printf("  N zero:     %d\n",    count(==(0), effs))

    println()
    println("  Per-cluster breakdown:")
    @printf("  %-28s  %4s  %+9s  %+9s  %+9s\n",
            cluster_col, "N", "Mean", "Min", "Max")
    for cl in cluster_order
        sub = filter(r -> getproperty(r, Symbol(cluster_col)) == cl, df)
        n   = nrow(sub)
        n == 0 && continue
        e   = sub.effect
        @printf("  %-28s  %4d  %+9.4f  %+9.4f  %+9.4f\n",
                cl, n, mean(e), minimum(e), maximum(e))
    end

    println()
    println("  Top 5 ROIs by |effect|:")
    sorted = df[sortperm(abs.(df.effect), rev=true), :]
    for i in 1:min(5, nrow(sorted))
        row = sorted[i, :]
        @printf("  %d.  %-35s  effect = %+.4f\n",
                i, row.ROI_Name, row.effect)
    end
end

print_atlas_summary("Self",    self_df,    "Layer",   LAYER_ORDER)
print_atlas_summary("Nonself", nonself_df, "NETWORK", NETWORK_ORDER)

# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — Build figure
# ══════════════════════════════════════════════════════════════════════════════
println()
println("Building figure …")

# Shared x-axis range: symmetric around 0, 10% padding
all_effects = vcat(self_df.effect, nonself_df.effect)
x_abs_max   = maximum(abs.(all_effects)) * 1.10
x_range     = [-x_abs_max, x_abs_max]

# Bin sizes (same range, different resolution per panel)
self_bin_size    = 2 * x_abs_max / 17.0   # ~17 bins (within 15-20)
nonself_bin_size = 2 * x_abs_max / 27.0   # ~27 bins (within 25-30)

N_self    = nrow(self_df)
N_nonself = nrow(nonself_df)

# subplot_titles not used here — this version of PlotlyBase requires
# Vector{Union{Missing,String}}; panel titles are added via annotations instead.
fig = make_subplots(
    rows               = 1,
    cols               = 2,
    horizontal_spacing = 0.12,
)

# Left panel: self atlas layers
for layer in LAYER_ORDER
    sub   = filter(r -> r.Layer == layer, self_df)
    n     = nrow(sub)
    color = LAYER_COLORS[layer]
    add_trace!(fig,
        histogram(
            x      = sub.effect,
            name   = "$layer (N=$n)",
            marker = attr(color=color, opacity=0.85),
            xbins  = attr(start=-x_abs_max, stop=x_abs_max, size=self_bin_size),
        ),
        row=1, col=1)
end

# Right panel: nonself CAB-NP networks
for net in NETWORK_ORDER
    sub   = filter(r -> r.NETWORK == net, nonself_df)
    n     = nrow(sub)
    n == 0 && continue
    color = get(NETWORK_COLORS, net, "#888888")
    add_trace!(fig,
        histogram(
            x      = sub.effect,
            name   = "$net (N=$n)",
            marker = attr(color=color, opacity=0.85),
            xbins  = attr(start=-x_abs_max, stop=x_abs_max, size=nonself_bin_size),
        ),
        row=1, col=2)
end

# Vertical reference lines at x=0 in each panel
shapes = [
    attr(
        type  = "line", xref = "x",  yref = "paper",
        x0=0, x1=0, y0=0, y1=1,
        line  = attr(color="black", width=1.5, dash="dash"),
    ),
    attr(
        type  = "line", xref = "x2", yref = "paper",
        x0=0, x1=0, y0=0, y1=1,
        line  = attr(color="black", width=1.5, dash="dash"),
    ),
]

# Panel centres: horizontal_spacing=0.12, 2 cols → panel_width=0.44
# left centre = 0.22, right centre = 0.78
panel_annotations = [
    attr(
        text      = "<b>Self atlas — drug × session by layer (N=$N_self)</b>",
        xref      = "paper", yref = "paper",
        x         = 0.22,    y    = 1.04,
        xanchor   = "center", yanchor = "bottom",
        showarrow = false,
        font      = attr(size=12, family="Times New Roman"),
    ),
    attr(
        text      = "<b>Nonself atlas — drug × session by CAB-NP network (N=$N_nonself)</b>",
        xref      = "paper", yref = "paper",
        x         = 0.78,    y    = 1.04,
        xanchor   = "center", yanchor = "bottom",
        showarrow = false,
        font      = attr(size=12, family="Times New Roman"),
    ),
]

relayout!(fig,
    title         = attr(
        text = "Question 1: Drug × Session interaction value per ROI",
        font = attr(size=15, family="Times New Roman"),
        x    = 0.5,
    ),
    annotations   = panel_annotations,
    barmode       = "stack",
    height        = 560,
    width         = 1500,
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    font          = attr(family="Times New Roman", size=11),
    shapes        = shapes,
    xaxis         = attr(
        title = "Drug × session effect (AUC, s)",
        range = x_range,
    ),
    xaxis2        = attr(
        title = "Drug × session effect (AUC, s)",
        range = x_range,
    ),
    yaxis  = attr(title = "Count"),
    yaxis2 = attr(title = "Count"),
    legend = attr(
        x           = 1.01,
        y           = 0.99,
        xanchor     = "left",
        yanchor     = "top",
        bgcolor     = "rgba(255,255,255,0.9)",
        bordercolor = "#cccccc",
        borderwidth = 1,
        font        = attr(size=9),
    ),
    margin = attr(t=90, b=60, l=70, r=240),
)

PlotlyJS.savefig(fig, OUT_FIG)
println("  Figure written: $OUT_FIG")
println()
println("Done.")
