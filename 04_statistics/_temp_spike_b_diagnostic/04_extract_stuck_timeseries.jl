# 04_extract_stuck_timeseries.jl — Step 5b-v
# Extract BOLD time series for all 279 ROIs whose τ is stuck at the lag-2
# p0 value (5.193702147200268) in the post-tSNR-filtered analysis dataset.
#
# Outputs:
#   stuck_5194_timeseries_wide.csv  — 234 rows × 279 columns (one per stuck ROI)
#   stuck_5194_metadata.csv         — 279 rows, one per column, with case details
#
# Run from repo root:
#   julia 04_statistics/_temp_spike_b_diagnostic/04_extract_stuck_timeseries.jl

using CSV, DataFrames, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT  = normpath(joinpath(@__DIR__, "..", ".."))
const LONG_CSV   = joinpath(REPO_ROOT, "04_statistics", "results",
                             "analysis_long_format.csv")
const ATLAS_FILE = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                             "glasser_coordinates_nonself_clean_1mm.txt")
const SELF_LAB   = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                             "self_labels.txt")
const TS_BASE    = joinpath(REPO_ROOT, "02_timeseries_extraction", "results")
const OUT_DIR    = joinpath(REPO_ROOT, "04_statistics", "_temp_spike_b_diagnostic")
const OUT_WIDE   = joinpath(OUT_DIR, "stuck_5194_timeseries_wide.csv")
const OUT_META   = joinpath(OUT_DIR, "stuck_5194_metadata.csv")

# ── Constants ─────────────────────────────────────────────────────────────────
const STUCK_TAU     = 5.193702147200268
const TOL           = 1e-9
const DUMMY_VOLUMES = 6
const N_TIMEPOINTS  = 234   # 240 total − 6 discarded

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(OUT_WIDE) && isfile(OUT_META)
    println("[skip] both outputs already exist; delete to rerun")
    exit()
end

for f in [LONG_CSV, ATLAS_FILE, SELF_LAB]
    isfile(f) || error("Missing required input: $f")
end
mkpath(OUT_DIR)

# ── ROI name lookups ──────────────────────────────────────────────────────────
_atlas_df     = CSV.read(ATLAS_FILE, DataFrame; delim='\t')
nonself_names = Dict(pos => row.ROI_Name
                     for (pos, row) in enumerate(eachrow(_atlas_df)))

self_names = Dict{Int,String}()
for line in eachline(SELF_LAB)
    parts = split(strip(line))
    (isempty(parts) || length(parts) < 2) && continue
    self_names[parse(Int, parts[1])] = join(parts[2:end], " ")
end

get_roi_name(atlas, pos) = atlas == "self" ?
    get(self_names, pos, "ROI_$pos") : get(nonself_names, pos, "ROI_$pos")

# ── Load long_df and filter to stuck cases ────────────────────────────────────
long_df  = CSV.read(LONG_CSV, DataFrame)
stuck_df = filter(r -> abs(r.tau - STUCK_TAU) < TOL, long_df)

n_stuck       = nrow(stuck_df)
n_stuck_self  = count(==("self"),    stuck_df.atlas)
n_stuck_ns    = count(==("nonself"), stuck_df.atlas)

println("Stuck τ ≈ 5.194 filter:")
@printf("  Self:    %d  [expected  33]  %s\n", n_stuck_self, n_stuck_self ==  33 ? "OK" : "*** MISMATCH")
@printf("  Nonself: %d  [expected 246]  %s\n", n_stuck_ns,  n_stuck_ns  == 246 ? "OK" : "*** MISMATCH")
@printf("  Total:   %d  [expected 279]  %s\n", n_stuck,     n_stuck     == 279 ? "OK" : "*** MISMATCH")

(n_stuck == 279 && n_stuck_self == 33 && n_stuck_ns == 246) ||
    error("Stuck count mismatch — STOP; do not extract time series")

# ── Extract time series, grouped by (subject, session, atlas) ─────────────────
# Loading each CSV once for multiple ROIs from the same file.
col_names = String[]
col_data  = Vector{Float64}[]
meta_rows = []

verified_atlas = Set{String}()   # track which atlases have been verified

for gdf in groupby(stuck_df, [:subject, :session, :atlas])
    subject = gdf[1, :subject]
    session = gdf[1, :session]
    atlas   = gdf[1, :atlas]

    ts_path = joinpath(TS_BASE, "timeseries_$(atlas)", "denoisedNoGSR",
                       "$(subject)_$(session)_$(atlas)_timeseries.csv")
    isfile(ts_path) || error("Missing timeseries file: $ts_path — STOP")

    ts_df = CSV.read(ts_path, DataFrame)

    # One-time column-naming verification per atlas:
    # names(ts_df)[1]         = "Timepoint"
    # names(ts_df)[pos_id+1]  = Glasser ID string (nonself) or string(pos_id) (self)
    # Both match the roi_columns stored in the JLD2 files.
    if !(atlas in verified_atlas)
        col1 = names(ts_df)[1]
        col2 = names(ts_df)[2]
        col1 == "Timepoint" ||
            error("Expected first column 'Timepoint'; got '$col1' — STOP")
        tryparse(Int, col2) !== nothing ||
            error("Expected integer column names after Timepoint; got '$col2' — STOP")
        n_data_rows = nrow(ts_df)
        n_data_rows == 240 ||
            @warn "Expected 240 rows in $ts_path; got $n_data_rows"
        println("Column naming verified ($atlas, $subject $session): " *
                "names[1]=\"$col1\", names[2]=\"$col2\" [OK; " *
                "$(ncol(ts_df)-1) ROI columns, $n_data_rows timepoints]")
        push!(verified_atlas, atlas)
    end

    for row in eachrow(gdf)
        pos_id  = row.roi_pos_id
        col_key = "$(subject)__$(session)__$(atlas)__roi$(pos_id)"

        # pos_id + 1 because column 1 is "Timepoint"
        ts_col  = names(ts_df)[pos_id + 1]
        ts_vec  = Float64.(ts_df[(DUMMY_VOLUMES + 1):end, ts_col])

        length(ts_vec) == N_TIMEPOINTS ||
            error("Expected $N_TIMEPOINTS timepoints for $col_key; " *
                  "got $(length(ts_vec)) — STOP")

        push!(col_names, col_key)
        push!(col_data, ts_vec)
        push!(meta_rows, (
            column_name  = col_key,
            subject      = subject,
            session      = session,
            atlas        = atlas,
            roi_pos_id   = pos_id,
            roi_name     = get_roi_name(atlas, pos_id),
            tau_recorded = row.tau,
        ))
    end
end

@assert length(col_names) == 279  "Expected 279 columns; got $(length(col_names))"
@assert length(col_data)  == 279  "Expected 279 data vectors; got $(length(col_data))"

# ── Build wide DataFrame (234 rows × 279 columns) ────────────────────────────
wide_df = DataFrame(col_data, col_names)

println()
@printf("Wide DataFrame: %d rows × %d columns  [expected 234 × 279]\n",
        nrow(wide_df), ncol(wide_df))

# ── Write outputs ─────────────────────────────────────────────────────────────
meta_df = DataFrame(meta_rows)

CSV.write(OUT_WIDE, wide_df)
println("Wrote: $OUT_WIDE")
CSV.write(OUT_META, meta_df)
println("Wrote: $OUT_META")
println("Done.")
