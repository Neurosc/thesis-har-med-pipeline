# 01_compute_acw.jl — Compute ACW for the three-pipeline DMT-MED design.
#
# Loops the three denoising pipelines (detrend/glm/maximal) × six sphere layers,
# reading the qinspheres timeseries and writing one JLD2 per (pipeline, layer,
# subject, session). Sample = 39 (all minus sub-12), matching the extraction; the
# 35-subject inclusion filter is applied later at the statistics stage.
#
# Parameters: TR 1.8 s; dummy volumes = 6; n_lags = 100; acwtypes = [:auc, :tau]
#
# Input  : 02_timeseries_extraction/results/qinspheres/{pipeline}/{layer}/{sub}_{ses}_{layer}_timeseries.csv
# Output : 03_intrinsic_neural_metrics/results/acw/{pipeline}/{layer}/{sub}_{ses}.jld2
#          JLD2 vars: acw_results (ACWResults; [1]=AUC, [2]=tau), parcel_ids (Vector{String})
#
# Run from repo root:  julia 03_intrinsic_neural_metrics/scripts/01_compute_acw.jl
# Idempotent: skips runs whose output JLD2 already exists.

using IntrinsicTimescales, CSV, DataFrames, JLD2, Statistics

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# ── Parameters ───────────────────────────────────────────────────────────────
const TR            = 1.8
const FS            = 1.0 / TR
const N_LAGS        = 100
const DUMMY_VOLUMES = 6
const ACW_TYPES     = [:auc, :tau]
const SKIP_ZERO_LAG = false

# ── Design ───────────────────────────────────────────────────────────────────
const PIPELINES = ["detrend", "glm", "maximal"]
const LAYERS    = ["intero", "extero", "mental", "visual", "motor", "auditory"]

# 39 subjects (all minus sub-12) — matches utils/subject_filter.py:get_pipeline_subjects()
const SUBJECTS = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05", "sub-06",
    "sub-07", "sub-08", "sub-09", "sub-10", "sub-11",
    "sub-13", "sub-14", "sub-15", "sub-16", "sub-17", "sub-18",
    "sub-19", "sub-20", "sub-21", "sub-22", "sub-23", "sub-24", "sub-25",
    "sub-26", "sub-27", "sub-28", "sub-29", "sub-30", "sub-31", "sub-32", "sub-33",
    "sub-34", "sub-35", "sub-36",
    "sub-37", "sub-38", "sub-39", "sub-40",
]
const SESSIONS = ["ses-01", "ses-02"]

# ── Paths ────────────────────────────────────────────────────────────────────
const TS_BASE  = joinpath(REPO_ROOT, "02_timeseries_extraction", "results", "qinspheres")
const OUT_BASE = joinpath(REPO_ROOT, "03_intrinsic_neural_metrics", "results", "acw")

# ── Main loop ────────────────────────────────────────────────────────────────
TOTAL     = length(PIPELINES) * length(LAYERS) * length(SUBJECTS) * length(SESSIONS)
completed = 0
skipped   = 0
failed    = 0
run_idx   = 0

println("ACW: $(length(PIPELINES)) pipelines × $(length(LAYERS)) layers × " *
        "$(length(SUBJECTS)) subjects × $(length(SESSIONS)) sessions = $TOTAL runs")
println("Output base: $OUT_BASE\n")

for pipeline in PIPELINES, layer in LAYERS, subject in SUBJECTS, session in SESSIONS
    global run_idx, completed, skipped, failed
    run_idx += 1

    csv_path = joinpath(TS_BASE, pipeline, layer, "$(subject)_$(session)_$(layer)_timeseries.csv")
    out_dir  = joinpath(OUT_BASE, pipeline, layer)
    out_path = joinpath(out_dir, "$(subject)_$(session).jld2")
    label    = "[$run_idx/$TOTAL] $pipeline $layer $subject $session"

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
        df         = CSV.read(csv_path, DataFrame)
        ts         = Matrix(df[:, 2:end])'           # ROI × time
        ts         = ts[:, (DUMMY_VOLUMES + 1):end]  # discard dummies
        n_rois     = size(ts, 1)

        acw_obj    = acw(ts, FS;
                         dims          = 2,
                         acwtypes      = ACW_TYPES,
                         n_lags        = N_LAGS,
                         skip_zero_lag = SKIP_ZERO_LAG)

        mkpath(out_dir)
        parcel_ids  = names(df)[2:end]
        acw_results = acw_obj.acw_results
        @save out_path acw_results parcel_ids

        elapsed = round(time() - t_start; digits = 2)
        println("$label ... DONE ($(elapsed)s, $n_rois ROIs)")
        completed += 1
    catch e
        println("$label ... FAIL ($e)")
        failed += 1
    end
end

println("\n─── ACW summary ───")
println("Pipelines: $PIPELINES")
println("Total:     $TOTAL")
println("Completed: $completed")
println("Skipped (already existed): $skipped")
println("Failed:    $failed")
