# 01_compute_glasser_auc.jl — ACW-AUC for whole-cortex Glasser 360 parcels (G1-flattening).
#
# Reuses the EXACT acw() configuration from 03_intrinsic_neural_metrics/scripts/01_compute_acw.jl
# (TR 1.8 s, FS=1/TR, n_lags 100, 6 dummy volumes dropped, acwtypes [:auc,:acw50],
# skip_zero_lag false; AUC = acw_results[1]). Same ACF lag range / integration / band — nothing
# re-specified.
#
# Input : 02_timeseries_extraction/results/glasser360/{pipeline}/{sub}_{ses}_glasser360_timeseries.csv
#         (rows = Timepoint incl. 6 dummies, cols = parcel INDEX 1..360)
# Output: 03_intrinsic_neural_metrics/results/glasser360_auc/{pipeline}/glasser360_auc_{pipeline}.csv
#         columns: subject, session, parcel_id, auc   (parcel_id = CAB-NP INDEX 1..360)
#
# Run from repo root:  julia 03_intrinsic_neural_metrics/scripts/glasser_g1/01_compute_glasser_auc.jl

using IntrinsicTimescales, CSV, DataFrames

# ── Parameters (identical to 01_compute_acw.jl) ──────────────────────────────
const TR            = 1.8
const FS            = 1.0 / TR
const N_LAGS        = 100
const DUMMY_VOLUMES = 6
const ACW_TYPES     = [:auc, :acw50]   # AUC = acw_results[1]
const SKIP_ZERO_LAG = false

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const PIPELINES = ["detrend", "glm", "maximal"]
const TS_BASE   = joinpath(REPO_ROOT, "02_timeseries_extraction", "results", "glasser360")
const OUT_BASE  = joinpath(REPO_ROOT, "03_intrinsic_neural_metrics", "results", "glasser360_auc")

const FNAME_RE = r"^(sub-\d+)_(ses-\d+)_glasser360_timeseries\.csv$"

Row = NamedTuple{(:subject, :session, :parcel_id, :auc),
                 Tuple{String, String, String, Float64}}

for pipeline in PIPELINES
    out_dir = joinpath(OUT_BASE, pipeline); mkpath(out_dir)
    out_csv = joinpath(out_dir, "glasser360_auc_$(pipeline).csv")

    files = sort(filter(f -> occursin(FNAME_RE, f), readdir(joinpath(TS_BASE, pipeline))))
    println("\n=== $pipeline : $(length(files)) runs ===")

    rows = Row[]
    for f in files
        m   = match(FNAME_RE, f)
        sub, ses = m[1], m[2]
        df  = CSV.read(joinpath(TS_BASE, pipeline, f), DataFrame)
        ts  = Matrix(df[:, 2:end])'                  # parcel × time
        ts  = ts[:, (DUMMY_VOLUMES + 1):end]         # drop 6 dummies
        acw_obj = acw(ts, FS; dims = 2, acwtypes = ACW_TYPES,
                      n_lags = N_LAGS, skip_zero_lag = SKIP_ZERO_LAG)
        auc  = collect(acw_obj.acw_results[1])       # [1] = AUC
        pids = names(df)[2:end]
        @assert length(pids) == 360 "$sub $ses: $(length(pids)) parcels (expected 360)"
        for (pid, a) in zip(pids, auc)
            push!(rows, (subject = sub, session = ses,
                         parcel_id = String(pid), auc = Float64(a)))
        end
        println("  $sub $ses ... $(length(pids)) parcels")
    end

    CSV.write(out_csv, DataFrame(rows))
    println("  wrote $out_csv  ($(length(rows)) rows)")
end
println("\nDone.")
