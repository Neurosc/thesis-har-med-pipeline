# 01_build_dataframe.jl — Build analysis-ready DataFrames (parcels only)
#
# Config-aware: reads config.toml for atlas_method + denoising_method.
# Guards against sphere runs (use atlas_method = "parcels").
#
# Outputs to 04_statistics/results/{tag}/tables/:
#   keskin_auc_model_ready.csv   — self parcels per Keskin layer (no G1)
#   nonself_model_ready.csv      — nonself parcels (tSNR+NaN exclusions applied, no G1)
#   analysis_long_format_auc.csv — combined long format for QC scripts
#
# Run from repo root:
#   julia 04_statistics/scripts/01_build_dataframe.jl

using JLD2, DataFrames, CSV, Statistics, Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# ── Config ─────────────────────────────────────────────────────────────────────
include(joinpath(REPO_ROOT, "utils", "config_loader.jl"))
cfg = load_config()

if cfg.atlas_method == "spheres"
    error("Sphere pipeline not yet implemented in refactored statistics. " *
          "Set atlas_method = \"parcels\" in config.toml.")
end

const DENOISING = cfg.denoising_method
const TAG       = tag()
const ACW_BASE  = joinpath(REPO_ROOT, "03_acw_analysis", "results", "parcels_$(DENOISING)")
const EXCL_TSV  = joinpath(REPO_ROOT, "excluded_rois_low_tsnr.tsv")
const PARTS_TSV = joinpath(REPO_ROOT, "participants.tsv")
const OUT_DIR   = joinpath(REPO_ROOT, "04_statistics", "results", TAG, "tables")

mkpath(OUT_DIR)

println("Config: $TAG")
println("ACW base: $ACW_BASE")
println("Out dir:  $OUT_DIR\n")

for f in [EXCL_TSV, PARTS_TSV]
    isfile(f) || error("Missing required input: $f")
end

# ── Subjects (35; excluded: sub-06, sub-08, sub-12, sub-26, sub-36) ──────────
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
const AUC_IDX  = 1   # acw_results[1] = AUC; acw_results[2] = τ

const SELF_DIRS = [
    ("interoceptive", "Interoception"),
    ("exteroceptive", "Exteroception"),
    ("mental",        "Cognition"),
]

# ── Drug-group map ────────────────────────────────────────────────────────────
part_df        = CSV.read(PARTS_TSV, DataFrame; delim='\t')
drug_group_map = Dict{String,String}(row.participant_id => row.condition
                                     for row in eachrow(part_df))
println("Drug-group metadata: $(length(drug_group_map)) participants\n")

# ── tSNR exclusion (nonself only) ─────────────────────────────────────────────
println("═══ tSNR exclusion ═══")
excl_df          = CSV.read(EXCL_TSV, DataFrame; delim='\t')
excl_glasser_set = Set{Int}(excl_df.roi_id)
n_tsnr = length(excl_glasser_set)
@printf("  %d nonself Glasser parcels excluded by tSNR < 30\n\n", n_tsnr)

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Self parcels (per-category JLD2)
# ══════════════════════════════════════════════════════════════════════════════
println("═══ Part 1: Self parcels (per-category) ═══\n")

self_rows = NamedTuple{(:subject, :session, :atlas, :glasser_id, :auc, :layer, :drug_group),
                        Tuple{String,String,String,Int,Float64,String,String}}[]

file_idx = 0
TOTAL_SELF = length(SUBJECTS) * length(SESSIONS) * length(SELF_DIRS)

for (dir_name, layer_label) in SELF_DIRS, subject in SUBJECTS, session in SESSIONS
    global file_idx
    file_idx += 1
    jld_path = joinpath(ACW_BASE, dir_name, "$(subject)_$(session).jld2")
    if !isfile(jld_path)
        println("[$file_idx/$TOTAL_SELF] MISSING: $jld_path")
        continue
    end
    data        = load(jld_path)
    acw_results = data["acw_results"]
    parcel_ids  = data["parcel_ids"]
    auc_vals    = collect(acw_results[AUC_IDX])
    dg          = get(drug_group_map, subject, "unknown")
    for (pid_str, auc) in zip(parcel_ids, auc_vals)
        glasser_id = parse(Int, pid_str)
        push!(self_rows, (subject=subject, session=session, atlas="self",
                          glasser_id=glasser_id, auc=Float64(auc),
                          layer=layer_label, drug_group=dg))
    end
end

self_df = DataFrame(self_rows)

@printf("Self rows loaded: %d  (%.0f per run × %d runs)\n",
        nrow(self_df),
        nrow(self_df) / (length(SUBJECTS) * length(SESSIONS)),
        length(SUBJECTS) * length(SESSIONS))

n_nan_self = count(isnan, self_df.auc)
@printf("NaN self AUC: %d\n\n", n_nan_self)

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Nonself parcels (tSNR exclusion + NaN detection)
# ══════════════════════════════════════════════════════════════════════════════
println("═══ Part 2: Nonself parcels ═══\n")

ns_all = NamedTuple{(:subject, :session, :atlas, :glasser_id, :auc, :drug_group),
                     Tuple{String,String,String,Int,Float64,String}}[]

TOTAL_NS = length(SUBJECTS) * length(SESSIONS)
file_idx = 0
n_raw = 0; n_tsnr_dropped = 0

for subject in SUBJECTS, session in SESSIONS
    global file_idx, n_raw, n_tsnr_dropped
    file_idx += 1
    jld_path = joinpath(ACW_BASE, "nonself", "$(subject)_$(session).jld2")
    if !isfile(jld_path)
        println("[$file_idx/$TOTAL_NS] MISSING: $jld_path")
        continue
    end
    data        = load(jld_path)
    acw_results = data["acw_results"]
    parcel_ids  = data["parcel_ids"]
    auc_vals    = collect(acw_results[AUC_IDX])
    dg          = get(drug_group_map, subject, "unknown")
    for (pid_str, auc) in zip(parcel_ids, auc_vals)
        n_raw += 1
        glasser_id = parse(Int, pid_str)
        if glasser_id in excl_glasser_set
            n_tsnr_dropped += 1
            continue
        end
        push!(ns_all, (subject=subject, session=session, atlas="nonself",
                       glasser_id=glasser_id, auc=Float64(auc), drug_group=dg))
    end
end

ns_df_raw = DataFrame(ns_all)
@printf("Nonself raw obs:            %d\n", n_raw)
@printf("Dropped by tSNR exclusion:  %d\n", n_tsnr_dropped)
@printf("Remaining:                  %d\n\n", nrow(ns_df_raw))

# ── NaN detection ────────────────────────────────────────────────────────────
println("═══ NaN detection (nonself, post-tSNR) ═══\n")

nan_mask = isnan.(ns_df_raw.auc)
n_nan    = sum(nan_mask)
@printf("  NaN AUC observations: %d\n", n_nan)

if n_nan > 0
    nan_rows      = ns_df_raw[nan_mask, :]
    parcel_counts = combine(groupby(nan_rows, :glasser_id), nrow => :n_obs)
    sort!(parcel_counts, :n_obs, rev=true)
    n_distinct    = nrow(parcel_counts)
    max_recur     = maximum(parcel_counts.n_obs)
    n_runs        = length(SUBJECTS) * length(SESSIONS)

    println("  NaN cases by parcel:")
    @printf("  %10s  %s\n", "glasser_id", "n_obs (of $(n_runs))")
    for row in eachrow(parcel_counts)
        @printf("  %10d  %d\n", row.glasser_id, row.n_obs)
    end

    parcel_level = n_distinct <= 5 && (max_recur / n_runs > 0.5)
    if parcel_level
        println("\n  → PARCEL-LEVEL NaN exclusion ($n_distinct parcels dropped for all subjects)")
        nan_glasser_set = Set{Int}(parcel_counts.glasser_id)
        ns_df_raw = ns_df_raw[.!isnan.(ns_df_raw.auc) .&
                              .![g in nan_glasser_set for g in ns_df_raw.glasser_id], :]
    else
        println("\n  → OBS-LEVEL NaN exclusion ($n_nan rows dropped)")
        ns_df_raw = ns_df_raw[.!nan_mask, :]
    end
    @printf("  Nonself rows after NaN exclusion: %d\n\n", nrow(ns_df_raw))
else
    println("  No NaN AUC values — no additional exclusions needed.\n")
end

ns_df = ns_df_raw

# ── Post-exclusion NaN/Inf safety check ────────────────────────────────────────
n_nan_final = count(isnan, ns_df.auc)
n_inf_final = count(isinf, ns_df.auc)
@printf("Post-exclusion NaN: %d [%s]  Inf: %d [%s]\n\n",
        n_nan_final, n_nan_final == 0 ? "OK" : "*** CHECK",
        n_inf_final, n_inf_final == 0 ? "OK" : "*** CHECK")

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Drug-group join check
# ══════════════════════════════════════════════════════════════════════════════
println("═══ Part 3: Drug-group join check ═══\n")

for (label, df_chk) in [("self", self_df), ("nonself", ns_df)]
    n_unk = count(==("unknown"), df_chk.drug_group)
    if n_unk > 0
        @printf("  WARNING [%s]: %d rows with drug_group='unknown'\n", label, n_unk)
    else
        @printf("  [%s] All subjects have drug_group. OK\n", label)
    end
end
println()

# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Assemble and save outputs
# ══════════════════════════════════════════════════════════════════════════════
println("═══ Part 4: Save outputs ═══\n")

# --- keskin_auc_model_ready.csv ---
keskin_out = select(self_df, :subject, :session, :drug_group => :group,
                    :glasser_id, :auc, :layer)

out_keskin = joinpath(OUT_DIR, "keskin_auc_model_ready.csv")
CSV.write(out_keskin, keskin_out)
@printf("Saved: %s  (%d rows)\n", out_keskin, nrow(keskin_out))

# --- nonself_model_ready.csv ---
nonself_out = select(ns_df, :subject, :session, :drug_group => :group,
                     :glasser_id, :auc)

out_nonself = joinpath(OUT_DIR, "nonself_model_ready.csv")
CSV.write(out_nonself, nonself_out)
@printf("Saved: %s  (%d rows)\n", out_nonself, nrow(nonself_out))

# --- analysis_long_format_auc.csv ---
# Combine self + nonself for QC plots; self has no per-layer split here (atlas = "self")
self_long = select(self_df, :subject, :session, :atlas, :glasser_id, :auc,
                   :drug_group)
ns_long   = select(ns_df,   :subject, :session, :atlas, :glasser_id, :auc,
                   :drug_group)
long_df   = vcat(self_long, ns_long)

out_long = joinpath(OUT_DIR, "analysis_long_format_auc.csv")
CSV.write(out_long, long_df)
@printf("Saved: %s  (%d rows)\n", out_long, nrow(long_df))

# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — Summary statistics
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 5: Summary ═══\n")

n_self    = count(==("self"),    long_df.atlas)
n_nonself = count(==("nonself"), long_df.atlas)
@printf("  Self rows:    %d  (%.0f per run)\n",
        n_self,    n_self    / (length(SUBJECTS) * length(SESSIONS)))
@printf("  Nonself rows: %d  (%.0f per run)\n",
        n_nonself, n_nonself / (length(SUBJECTS) * length(SESSIONS)))
@printf("  Total:        %d\n\n", nrow(long_df))

for atlas_label in ["self", "nonself"]
    sub_df   = long_df[long_df.atlas .== atlas_label, :]
    finite_v = filter(isfinite, sub_df.auc)
    @printf("  %-8s  min=%.4f  median=%.4f  max=%.4f\n",
            atlas_label,
            isempty(finite_v) ? NaN : minimum(finite_v),
            isempty(finite_v) ? NaN : median(finite_v),
            isempty(finite_v) ? NaN : maximum(finite_v))
end

for layer_lbl in ["Interoception", "Exteroception", "Cognition"]
    sub_lk = keskin_out[keskin_out.layer .== layer_lbl, :]
    sub01  = sub_lk[sub_lk.subject .== "sub-01" .&& sub_lk.session .== "ses-01", :]
    @printf("  Keskin %s: %d parcels/run\n", layer_lbl, nrow(sub01))
end

println("\nDone.")
