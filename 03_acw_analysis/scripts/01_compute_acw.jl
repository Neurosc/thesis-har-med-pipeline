# 01_compute_acw.jl — Compute ACW for DMT-MED dataset
# Adapted from ACW_calculation.jl (task-med reference)
#
# Key changes vs reference:
#   TR 2.0 → 1.8 s; no session window (use full run after dummy removal);
#   no groups; two sessions (ses-01/02); two denoising versions; per-run JLD2 output
#
# Run from repo root:  julia 03_acw_analysis/scripts/01_compute_acw.jl
# Idempotent: skips runs where the output JLD2 already exists.

using IntrinsicTimescales, CSV, DataFrames, JLD2, Statistics

# ── Config ───────────────────────────────────────────────────────────────────
include(normpath(joinpath(@__DIR__, "..", "..", "utils", "config_loader.jl")))
let cfg = load_config()
    println("Config: atlas=$(cfg.atlas_method)  denoising=$(cfg.denoising_method)  tag=$(tag())")
end

# ── Parameters ───────────────────────────────────────────────────────────────
const TR            = 1.8
const FS            = 1.0 / TR       # 0.556 Hz
const N_LAGS        = 100
const DUMMY_VOLUMES = 6              # discard first 10.8 s (6 × 1.8 s)
const ACW_TYPES     = [:auc, :tau]
const SKIP_ZERO_LAG = false

# ── Included subjects (40 total; excluded: sub-06, sub-08, sub-12, sub-26, sub-36) ──
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

const SESSIONS = ["ses-01", "ses-02"]
const VERSIONS = ["raw", "denoisedNoGSR"]
const ATLASES  = ["self", "nonself"]

# ── Paths (REPO_ROOT provided by config_loader.jl) ───────────────────────────
const TS_BASE     = joinpath(REPO_ROOT, "02_timeseries_extraction", "results")
const OUTPUT_BASE = joinpath(REPO_ROOT, "03_acw_analysis", "results", tag(), "acw")

# ── Main loop ────────────────────────────────────────────────────────────────
const TOTAL = length(ATLASES) * length(VERSIONS) * length(SUBJECTS) * length(SESSIONS)
completed = 0
skipped   = 0
failed    = 0
run_idx   = 0

for atlas in ATLASES, version in VERSIONS, subject in SUBJECTS, session in SESSIONS
    global run_idx, completed, skipped, failed
    run_idx += 1

    csv_path = joinpath(TS_BASE, "timeseries_$(atlas)", version,
                        "$(subject)_$(session)_$(atlas)_timeseries.csv")
    out_dir  = joinpath(OUTPUT_BASE, atlas, version)
    out_path = joinpath(out_dir, "$(subject)_$(session).jld2")
    label    = "[$run_idx/$TOTAL] $subject $session $atlas $version"

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
        ts  = ts[:, (DUMMY_VOLUMES + 1):end]  # discard dummies → ROI × 234
        n_rois = size(ts, 1)

        acw_obj = acw(ts, FS;
                      dims          = 2,
                      acwtypes      = ACW_TYPES,
                      n_lags        = N_LAGS,
                      skip_zero_lag = SKIP_ZERO_LAG)

        mkpath(out_dir)
        roi_columns = names(df)[2:end]
        acw_results = acw_obj.acw_results
        @save out_path acw_results subject session atlas version roi_columns

        elapsed = round(time() - t_start; digits = 2)
        println("$label ... DONE ($(elapsed)s, $n_rois ROIs)")
        completed += 1
    catch e
        println("$label ... FAIL ($e)")
        failed += 1
    end
end

println("\n─── ACW Computation Summary ───")
println("Total runs:  $TOTAL")
println("Completed:   $completed")
println("Skipped:     $skipped")
println("Failed:      $failed")
