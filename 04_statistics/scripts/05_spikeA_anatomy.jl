# 05_spikeA_anatomy.jl — Step 5b-i continued: Glasser anatomy for Spike A ROIs
#
# roi_id in fig04_spikeA_roi_frequency.csv is a 1-based positional index into the
# nonself timeseries CSVs (columns after "Timepoint"), NOT the Glasser ROI number.
# The column header of each nonself timeseries CSV gives the actual Glasser ROI number
# for each position. This script builds that mapping and joins to the atlas coordinates.
#
# Run from repo root: julia 04_statistics/scripts/05_spikeA_anatomy.jl

using CSV, DataFrames, PlotlyJS, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT     = normpath(joinpath(@__DIR__, "..", ".."))
const FREQ_CSV      = joinpath(REPO_ROOT, "04_statistics", "results", "fig04_spikeA_roi_frequency.csv")
const ATLAS_TXT     = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                               "glasser_coordinates_nonself_clean_1mm.txt")
# Any nonself timeseries CSV — we only need the header to get position → Glasser ROI mapping
const TS_HEADER_CSV = joinpath(REPO_ROOT, "02_timeseries_extraction", "results",
                               "timeseries_nonself", "denoisedNoGSR",
                               "sub-01_ses-01_nonself_timeseries.csv")
const RES_DIR       = joinpath(REPO_ROOT, "04_statistics", "results")
const FIG_DIR       = joinpath(REPO_ROOT, "04_statistics", "figures")
const OUT_CSV       = joinpath(RES_DIR, "fig05_spikeA_anatomy.csv")
const OUT_FIG       = joinpath(FIG_DIR, "fig05_spikeA_anatomy.html")

# ── Colors (Palette A) ────────────────────────────────────────────────────────
const COLOR_LEFT  = "#3D5A6C"   # slate
const COLOR_RIGHT = "#8B7355"   # taupe

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(OUT_CSV) && isfile(OUT_FIG)
    println("[skip] all outputs already exist; delete to rerun")
    exit()
end

# ── Check required inputs ─────────────────────────────────────────────────────
for f in [FREQ_CSV, ATLAS_TXT, TS_HEADER_CSV]
    isfile(f) || error("Missing required input: $f")
end

mkpath(RES_DIR)
mkpath(FIG_DIR)

# ── Load timeseries header → position (1-based) → Glasser ROI number ──────────
ts_header = open(TS_HEADER_CSV) do io
    split(readline(io), ",")
end
# First column is "Timepoint"; remaining columns are Glasser ROI numbers as strings
roi_positions = parse.(Int, ts_header[2:end])   # position i → Glasser ROI roi_positions[i]

# ── Load atlas: Glasser ROI number → (name, x, y, z) ─────────────────────────
atlas_df = CSV.read(ATLAS_TXT, DataFrame; delim='\t')

expected_atlas_cols = ["ROI_Number", "ROI_Name", "X", "Y", "Z"]
actual_cols = names(atlas_df)
for col in expected_atlas_cols
    col in actual_cols || error("Atlas file missing expected column '$col'. Found: $(actual_cols)")
end

atlas_lookup = Dict(
    row.ROI_Number => (name=row.ROI_Name, x=row.X, y=row.Y, z=row.Z)
    for row in eachrow(atlas_df)
)

# ── Load Spike A ROI frequency ────────────────────────────────────────────────
freq_df = CSV.read(FREQ_CSV, DataFrame)

# ── Build merged table ────────────────────────────────────────────────────────
result_rows = []
for row in eachrow(freq_df)
    pos_id     = row.roi_id                        # positional index (1-based)
    glasser_id = roi_positions[pos_id]             # actual Glasser ROI number

    haskey(atlas_lookup, glasser_id) ||
        error("Glasser ROI $glasser_id (from position $pos_id) not found in atlas")

    info = atlas_lookup[glasser_id]

    # Parse hemisphere and region from name like "R_V1_ROI" or "L_PFm_ROI"
    parts      = split(info.name, "_")             # ["R", "V1", "ROI"]  or  ["L", "PFm", "ROI"]
    hemisphere = parts[1]                          # "L" or "R"
    region     = join(parts[2:end-1], "_")         # middle portion(s); drops leading hemi and trailing ROI

    push!(result_rows, (
        roi_id              = pos_id,
        glasser_roi         = glasser_id,
        count               = row.count,
        percent_of_spike_A  = row.percent_of_spike_A,
        cumulative_percent  = row.cumulative_percent,
        hemisphere          = hemisphere,
        region              = region,
        x_mni               = info.x,
        y_mni               = info.y,
        z_mni               = info.z,
    ))
end

result_df = DataFrame(result_rows)
sort!(result_df, :count, rev=true)

# ── Console summary ───────────────────────────────────────────────────────────
println()
println("=" ^ 78)
println("Spike A ROI Anatomy — all 46 nonself ROIs (sorted by count)")
println("=" ^ 78)
@printf("  %6s  %10s  %5s  %6s  %2s  %-14s  %7s  %7s  %7s\n",
        "pos_id", "Glasser_id", "count", "%SpikeA", "H", "region", "X(mm)", "Y(mm)", "Z(mm)")
println("  " * "-" ^ 76)
for r in eachrow(result_df)
    @printf("  %6d  %10d  %5d  %5.1f%%  %2s  %-14s  %7.1f  %7.1f  %7.1f\n",
            r.roi_id, r.glasser_roi, r.count, r.percent_of_spike_A,
            r.hemisphere, r.region, r.x_mni, r.y_mni, r.z_mni)
end

# ── Hemisphere breakdown ──────────────────────────────────────────────────────
n_left  = count(==("L"), result_df.hemisphere)
n_right = count(==("R"), result_df.hemisphere)
println()
println("Hemisphere breakdown:")
@printf("  Left  (L): %d ROIs\n", n_left)
@printf("  Right (R): %d ROIs\n", n_right)

# ── Region summary (summed across hemispheres) ────────────────────────────────
region_counts = combine(groupby(result_df, :region), :count => sum => :total_count)
sort!(region_counts, :total_count, rev=true)
println()
println("Distinct regions (count summed across hemispheres):")
for r in eachrow(region_counts)
    @printf("  %-14s  %d\n", r.region, r.total_count)
end

# ── Output CSV ────────────────────────────────────────────────────────────────
out_csv_df = select(result_df, :roi_id, :count, :percent_of_spike_A, :cumulative_percent,
                    :hemisphere, :region, :x_mni, :y_mni, :z_mni)
CSV.write(OUT_CSV, out_csv_df)
println("\nCSV written: $OUT_CSV  ($(nrow(out_csv_df)) rows)")

# ── Output figure: 3D scatter in MNI space ───────────────────────────────────
colors  = [h == "L" ? COLOR_LEFT : COLOR_RIGHT for h in result_df.hemisphere]
sizes   = result_df.count ./ maximum(result_df.count) .* 24 .+ 6   # 6–30 px range
hovertexts = [
    @sprintf("ROI pos %d (Glasser %d)<br>%s %s<br>count: %d",
             r.roi_id, r.glasser_roi, r.hemisphere, r.region, r.count)
    for r in eachrow(result_df)
]

trace = PlotlyJS.scatter3d(
    x    = result_df.x_mni,
    y    = result_df.y_mni,
    z    = result_df.z_mni,
    mode = "markers",
    marker = attr(
        size  = sizes,
        color = colors,
    ),
    text      = hovertexts,
    hoverinfo = "text",
)

layout = Layout(
    title = attr(text="Anatomical location of Spike A nonself ROIs (n=46)", font_size=16),
    scene = attr(
        xaxis = attr(title="X (mm, R→L)"),
        yaxis = attr(title="Y (mm, P→A)"),
        zaxis = attr(title="Z (mm, I→S)"),
    ),
    margin = attr(l=0, r=0, b=0, t=50),
)

PlotlyJS.savefig(PlotlyJS.plot(trace, layout), OUT_FIG)
println("Figure written: $OUT_FIG")
