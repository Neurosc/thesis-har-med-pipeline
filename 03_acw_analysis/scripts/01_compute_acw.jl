# 01_compute_acw.jl — Compute ACW for DMT-MED dataset (multi-method)
# Reads config.toml → switches between sphere and parcel input directories automatically.
#
# Key parameters:
#   TR 1.8 s; dummy volumes = 6; n_lags = 100; acwtypes = [:auc, :tau]
#
# Run from repo root:  julia 03_acw_analysis/scripts/01_compute_acw.jl
# Idempotent: skips runs where the output JLD2 already exists.

using IntrinsicTimescales, CSV, DataFrames, JLD2, Statistics

include(joinpath(@__DIR__, "..", "..", "utils", "config_loader.jl"))

# ── Parameters ───────────────────────────────────────────────────────────────
const TR            = 1.8
const FS            = 1.0 / TR
const N_LAGS        = 100
const DUMMY_VOLUMES = 6
const ACW_TYPES     = [:auc, :tau]
const SKIP_ZERO_LAG = false

# ── Included subjects: 35 (excluded by FD>0.3mm censoring: sub-06/08/12/26/36) ──
# Matches utils/subject_filter.py:get_included_subjects() — the single source of
# truth for inclusion. The 4 censoring-excluded subjects (06/08/26/36) and sub-12
# are NOT computed; any stray outputs for them live under _archive/.
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

# ── Config ────────────────────────────────────────────────────────────────────
cfg = load_config()
tg  = tag()

println("Running ACW with: atlas=$(cfg.atlas_method), denoising=$(cfg.denoising_method) (tag=$tg)")

# Map config denoising_method ("NoGSR"/"GSR"/"raw") to the actual on-disk folder name.
# Extraction scripts write subdirs as "denoisedNoGSR", "denoisedGSR", or "raw".
# The output tag stays clean ("parcels_NoGSR"), only the input path uses the prefix.
denoising_folder = cfg.denoising_method == "raw" ? "raw" : "denoised" * cfg.denoising_method

# ── Input directory mapping ───────────────────────────────────────────────────
if cfg.atlas_method == "spheres"
    TS_BASE_M  = joinpath(REPO_ROOT, "02_timeseries_extraction", "results", "timeseries_spheres")
    atlas_dirs = [
        "self"    => joinpath(TS_BASE_M, "timeseries_self",    denoising_folder),
        "nonself" => joinpath(TS_BASE_M, "timeseries_nonself", denoising_folder),
    ]
    csv_suffix = "_timeseries.csv"
elseif cfg.atlas_method == "parcels"
    TS_BASE_M  = joinpath(REPO_ROOT, "02_timeseries_extraction", "results", "timeseries_parcels")
    atlas_dirs = [
        "self"          => joinpath(TS_BASE_M, "self",          denoising_folder),
        "nonself"       => joinpath(TS_BASE_M, "nonself",       denoising_folder),
        "interoceptive" => joinpath(TS_BASE_M, "interoceptive", denoising_folder),
        "exteroceptive" => joinpath(TS_BASE_M, "exteroceptive", denoising_folder),
        "mental"        => joinpath(TS_BASE_M, "mental",        denoising_folder),
    ]
    csv_suffix = "_parcel_timeseries.csv"
elseif cfg.atlas_method == "qinspheres"
    TS_BASE_M  = joinpath(REPO_ROOT, "02_timeseries_extraction", "results", "qinspheres")
    atlas_dirs = [
        "intero"   => joinpath(TS_BASE_M, "intero"),
        "extero"   => joinpath(TS_BASE_M, "extero"),
        "mental"   => joinpath(TS_BASE_M, "mental"),
        "auditory" => joinpath(TS_BASE_M, "auditory"),
        "motor"    => joinpath(TS_BASE_M, "motor"),
        "visual"   => joinpath(TS_BASE_M, "visual"),
    ]
    csv_suffix = "_timeseries.csv"
else
    error("Unknown atlas_method '$(cfg.atlas_method)' in config.toml. Expected 'spheres', 'parcels', or 'qinspheres'.")
end

# ── Output directory ──────────────────────────────────────────────────────────
const OUT_DIR = joinpath(REPO_ROOT, "03_acw_analysis", "results", tg)
println("Output dir: $OUT_DIR\n")

# ── Main loop ────────────────────────────────────────────────────────────────
TOTAL     = length(atlas_dirs) * length(SUBJECTS) * length(SESSIONS)
completed = 0
skipped   = 0
failed    = 0
run_idx   = 0

for (atlas_name, ts_dir) in atlas_dirs, subject in SUBJECTS, session in SESSIONS
    global run_idx, completed, skipped, failed
    run_idx += 1

    csv_path = joinpath(ts_dir, "$(subject)_$(session)_$(atlas_name)$(csv_suffix)")
    out_dir  = joinpath(OUT_DIR, atlas_name)
    out_path = joinpath(out_dir, "$(subject)_$(session).jld2")
    label    = "[$run_idx/$TOTAL] $subject $session $atlas_name"

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
        ts         = ts[:, (DUMMY_VOLUMES + 1):end]  # discard dummies → ROI × 234
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
println("Atlas:     $(cfg.atlas_method)")
println("Denoising: $(cfg.denoising_method)")
println("Total:     $TOTAL")
println("Completed: $completed")
println("Skipped (already existed): $skipped")
println("Failed:    $failed")
