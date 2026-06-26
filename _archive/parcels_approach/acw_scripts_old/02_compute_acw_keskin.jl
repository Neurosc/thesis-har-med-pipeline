# 02_compute_acw_keskin.jl
# Compute ACW (AUC) for Keskin self parcels from denoisedNoGSR timeseries CSVs.
# Analogous to 01_compute_acw.jl but for the Keskin atlas only.
#
# Input:  02_timeseries_extraction/results/timeseries_keskin/denoisedNoGSR/
#             sub-XX_ses-YY_keskin_timeseries.csv
# Output: 03_acw_analysis/results/acw/keskin/denoisedNoGSR/
#             sub-XX_ses-YY.jld2
#
# Idempotent: skips runs where the output JLD2 already exists.
#
# Run from repo root (after transferring timeseries CSVs from server):
#   julia 03_acw_analysis/scripts/02_compute_acw_keskin.jl

using IntrinsicTimescales, CSV, DataFrames, JLD2, Statistics

# ── Parameters (same as 01_compute_acw.jl) ──────────────────────────────────
const TR            = 1.8
const FS            = 1.0 / TR   # 0.556 Hz
const N_LAGS        = 100
const DUMMY_VOLUMES = 6
const ACW_TYPES     = [:auc, :tau]
const SKIP_ZERO_LAG = false
const ATLAS         = "keskin"
const VERSION       = "denoisedNoGSR"

const SUBJECTS = [
    "sub-01","sub-02","sub-03","sub-04","sub-05",
    "sub-07",
    "sub-09","sub-10","sub-11",
    "sub-13","sub-14","sub-15","sub-16","sub-17","sub-18",
    "sub-19","sub-20","sub-21","sub-22","sub-23","sub-24","sub-25",
    "sub-27","sub-28","sub-29","sub-30","sub-31","sub-32","sub-33",
    "sub-34","sub-35",
    "sub-37","sub-38","sub-39","sub-40"
]
const SESSIONS = ["ses-01","ses-02"]

const REPO_ROOT   = normpath(joinpath(@__DIR__, "..", ".."))
const TS_BASE     = joinpath(REPO_ROOT, "02_timeseries_extraction", "results",
                             "timeseries_self_glasser")
const OUTPUT_BASE = joinpath(REPO_ROOT, "03_acw_analysis", "results", "acw",
                             ATLAS, VERSION)

SEP = "=" ^ 70
println(SEP)
println("02_compute_acw_keskin.jl — ACW for Keskin self parcels")
println(SEP, "\n")

if !isdir(TS_BASE)
    error("Timeseries directory not found: $TS_BASE\n" *
          "Transfer Keskin timeseries CSVs from the server first.")
end

const TOTAL = length(SUBJECTS) * length(SESSIONS)
completed = 0
skipped   = 0
failed    = 0
run_idx   = 0

for subject in SUBJECTS, session in SESSIONS
    global run_idx, completed, skipped, failed
    run_idx += 1

    csv_path = joinpath(TS_BASE, "$(subject)_$(session)_keskin_timeseries.csv")
    out_dir  = OUTPUT_BASE
    out_path = joinpath(out_dir, "$(subject)_$(session).jld2")
    label    = "[$run_idx/$TOTAL] $subject $session"

    if isfile(out_path)
        println("$label ... SKIP (already exists)")
        skipped += 1
        continue
    end

    if !isfile(csv_path)
        println("$label ... FAIL (CSV not found: $csv_path)")
        failed += 1
        continue
    end

    t_start = time()
    try
        df  = CSV.read(csv_path, DataFrame)
        ts  = Matrix(df[:, 2:end])'           # ROI × time
        ts  = ts[:, (DUMMY_VOLUMES + 1):end]  # discard 6 dummy volumes
        n_rois = size(ts, 1)

        acw_obj = acw(ts, FS;
                      dims          = 2,
                      acwtypes      = ACW_TYPES,
                      n_lags        = N_LAGS,
                      skip_zero_lag = SKIP_ZERO_LAG)

        mkpath(out_dir)
        roi_columns = names(df)[2:end]  # Glasser ID strings
        acw_results = acw_obj.acw_results
        @save out_path acw_results subject session roi_columns

        elapsed = round(time() - t_start; digits = 2)
        println("$label ... DONE ($(elapsed)s, $n_rois ROIs)")
        completed += 1
    catch e
        println("$label ... FAIL ($e)")
        failed += 1
    end
end

println("\n─── ACW Keskin Summary ───")
println("Total:     $TOTAL")
println("Completed: $completed")
println("Skipped:   $skipped")
println("Failed:    $failed")
