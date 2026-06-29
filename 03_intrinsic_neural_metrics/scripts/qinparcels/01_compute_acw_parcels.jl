# 01_compute_acw_parcels.jl — ACW for the Glasser-PARCEL version of the 6 Qin regions.
# Identical acw() config to 03_intrinsic_neural_metrics/scripts/01_compute_acw.jl
# (TR 1.8, n_lags 100, 6 dummy dropped, acwtypes [:auc,:acw50], skip_zero_lag false).
#
# In : 02_timeseries_extraction/results/qinparcels/{pipeline}/{layer}/{sub}_{ses}_{layer}_timeseries.csv
# Out: 03_intrinsic_neural_metrics/results/acw_parcels/{pipeline}/{layer}/{sub}_{ses}.jld2
#      (vars: acw_results [1]=AUC [2]=ACW-50, parcel_ids). Subjects derived from files (n=35).
# Run from repo root: julia 03_intrinsic_neural_metrics/scripts/qinparcels/01_compute_acw_parcels.jl

using IntrinsicTimescales, CSV, DataFrames, JLD2

const TR = 1.8; const FS = 1.0 / TR; const N_LAGS = 100; const DUMMY = 6
const ACW_TYPES = [:auc, :acw50]; const SKIP_ZERO_LAG = false

const REPO     = normpath(joinpath(@__DIR__, "..", "..", ".."))
const PIPELINES = ["detrend", "glm", "maximal"]
const LAYERS    = ["intero", "extero", "mental", "visual", "motor", "auditory"]
const SESSIONS  = ["ses-01", "ses-02"]
const TS_BASE   = joinpath(REPO, "02_timeseries_extraction", "results", "qinparcels")
const OUT_BASE  = joinpath(REPO, "03_intrinsic_neural_metrics", "results", "acw_parcels")

# subjects present in the parcel timeseries (n=35)
fnre = r"^(sub-\d+)_ses-01_intero_timeseries\.csv$"
SUBS = sort(unique(String[m.captures[1] for f in readdir(joinpath(TS_BASE, "detrend", "intero"))
                          for m in (match(fnre, f),) if m !== nothing]))
println("Subjects: $(length(SUBS))")

done = skip = fail = 0
for pipeline in PIPELINES, layer in LAYERS, sub in SUBS, ses in SESSIONS
    global done, skip, fail
    csv     = joinpath(TS_BASE, pipeline, layer, "$(sub)_$(ses)_$(layer)_timeseries.csv")
    out_dir = joinpath(OUT_BASE, pipeline, layer)
    out     = joinpath(out_dir, "$(sub)_$(ses).jld2")
    if isfile(out); skip += 1; continue; end
    if !isfile(csv); fail += 1; continue; end
    try
        df = CSV.read(csv, DataFrame)
        ts = Matrix(df[:, 2:end])'
        ts = ts[:, (DUMMY + 1):end]
        acw_obj = acw(ts, FS; dims = 2, acwtypes = ACW_TYPES, n_lags = N_LAGS, skip_zero_lag = SKIP_ZERO_LAG)
        mkpath(out_dir)
        acw_results = acw_obj.acw_results
        parcel_ids  = names(df)[2:end]
        @save out acw_results parcel_ids
        done += 1
    catch e
        println("FAIL $pipeline $layer $sub $ses : $e"); fail += 1
    end
end
println("done=$done skip=$skip fail=$fail")
