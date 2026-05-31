# 08_build_analysis_dataframe_auc.jl — Step 9: AUC long-format DataFrame
#
# Builds the analysis-ready long-format DataFrame with AUC as the outcome.
# Two nonself exclusion rules applied:
#   1. tSNR < 30 parcel-level exclusion (58 parcels, all subjects)
#   2. AUC = NaN exclusion (18 obs outside tSNR set; granularity decided in Part 1)
# Self atlas: no exclusions.
#
# Run from repo root:  julia 04_statistics/scripts/08_build_analysis_dataframe_auc.jl

using JLD2, DataFrames, CSV, Statistics, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT    = normpath(joinpath(@__DIR__, "..", ".."))
const ATLAS_FILE   = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                              "glasser_coordinates_nonself_clean_1mm.txt")
const EXCL_TSV     = joinpath(REPO_ROOT, "excluded_rois_low_tsnr.tsv")
const NAN_CSV      = joinpath(REPO_ROOT, "99_QC", "troubleshooting", "results",
                              "auc_nan_not_in_tsnr_exclusion.csv")
const PARTICIPANTS = joinpath(REPO_ROOT, "participants.tsv")
const ACW_BASE     = joinpath(REPO_ROOT, "03_acw_analysis", "results", "acw")
const VERSION      = "denoisedNoGSR"
const RES_DIR      = joinpath(REPO_ROOT, "04_statistics", "results")
const OUT_CSV      = joinpath(RES_DIR, "analysis_long_format_auc.csv")

# ── Subjects (35; excluded: sub-06, sub-08, sub-12, sub-26, sub-36) ───────────
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
const SESSIONS    = ["ses-01", "ses-02"]
const ATLASES     = ["self", "nonself"]
const AUC_IDX     = 1   # acw_results[1] = AUC; acw_results[2] = τ
const TOTAL_FILES = length(SUBJECTS) * length(SESSIONS) * length(ATLASES)  # 140

# ── Idempotency ────────────────────────────────────────────────────────────────
if isfile(OUT_CSV)
    println("[skip] output already exists at $OUT_CSV")
    exit()
end

# ── Check required inputs ─────────────────────────────────────────────────────
for f in [ATLAS_FILE, EXCL_TSV, NAN_CSV, PARTICIPANTS]
    isfile(f) || error("Missing required input: $f")
end

mkpath(RES_DIR)

# ── Build Glasser→position mapping ────────────────────────────────────────────
# Row order in the clean atlas file defines nonself position numbering:
#   row 1 → position 1, row 2 → position 2, …
# roi_columns in JLD2 holds Glasser IDs (strings) for nonself, so we need
# this mapping to convert Glasser IDs to nonself position IDs.
atlas_df       = CSV.read(ATLAS_FILE, DataFrame; delim='\t')
glasser_to_pos = Dict{Int,Int}(row.ROI_Number => pos
                                for (pos, row) in enumerate(eachrow(atlas_df)))
n_atlas = length(glasser_to_pos)
n_atlas == 316 || error("Atlas mapping has $n_atlas entries; expected 316 — STOP")

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Load exclusion lists and decide AUC exclusion granularity
# ══════════════════════════════════════════════════════════════════════════════
println("═══ Part 1: Load exclusion lists ═══\n")

# tSNR exclusion (parcel-level, all subjects)
excl_df      = CSV.read(EXCL_TSV, DataFrame; delim='\t')
excl_pos_set = Set{Int}(glasser_to_pos[id] for id in excl_df.roi_id)
n_tsnr_excl  = length(excl_pos_set)
@printf("  tSNR exclusion: %d parcels [expected 58: %s]\n",
        n_tsnr_excl, n_tsnr_excl == 58 ? "OK" : "MISMATCH")
n_tsnr_excl == 58 || error("tSNR exclusion has $n_tsnr_excl entries; expected 58 — STOP")

println("  tSNR-excluded nonself_pos_ids:")
for p in sort(collect(excl_pos_set))
    print("  $p")
end
println()

# AUC NaN exclusion (outside tSNR set)
nan_df   = CSV.read(NAN_CSV, DataFrame)
n_nan_ex = nrow(nan_df)
@printf("\n  AUC NaN cases (outside tSNR excl): %d [expected 18: %s]\n",
        n_nan_ex, n_nan_ex == 18 ? "OK" : "MISMATCH")
n_nan_ex == 18 || error("AUC NaN list has $n_nan_ex rows; expected 18 — STOP")

println("  AUC NaN cases:")
@printf("  %-10s %-8s  %10s  %s\n", "subject", "session", "roi_pos_id", "roi_name")
println("  " * "─" ^ 52)
for row in eachrow(nan_df)
    @printf("  %-10s %-8s  %10d  %s\n",
            row.subject, row.session, row.roi_pos_id, row.roi_name)
end

# Recurrence analysis: how many distinct parcels, and how often does each recur?
parcel_counts          = combine(groupby(nan_df, [:roi_pos_id, :roi_name]), nrow => :n_obs)
sort!(parcel_counts, :n_obs, rev=true)
n_distinct_nan_parcels = nrow(parcel_counts)
max_recurrence         = maximum(parcel_counts.n_obs)

println("\n  Recurrence by parcel:")
@printf("  %10s  %-20s  %s\n", "roi_pos_id", "roi_name", "n_obs (of 70 possible)")
println("  " * "─" ^ 52)
for row in eachrow(parcel_counts)
    @printf("  %10d  %-20s  %d\n", row.roi_pos_id, row.roi_name, row.n_obs)
end
@printf("\n  Distinct NaN parcels: %d\n", n_distinct_nan_parcels)
@printf("  Max recurrence:       %d / 70 (%.1f%%)\n",
        max_recurrence, 100.0 * max_recurrence / 70)

# Decision rule:
#   Parcel-level if parcels are few (≤5) AND the most-affected parcel is NaN
#   in >50% of its possible observations (implying a structural signal issue).
#   Otherwise: (subject, session, parcel)-level (drop only those 18 rows).
parcel_level_auc_excl = n_distinct_nan_parcels <= 5 &&
                         (max_recurrence / 70.0 > 0.5)

if parcel_level_auc_excl
    println("\n  → Applying PARCEL-LEVEL AUC exclusion ",
            "($(n_distinct_nan_parcels) parcels dropped for all subjects).")
    auc_excl_parcels = Set{Int}(parcel_counts.roi_pos_id)
    nan_obs_set      = Set{Tuple{String,String,Int}}()
else
    println("\n  → Applying (SUBJECT, SESSION, PARCEL)-LEVEL AUC exclusion ",
            "(18 specific observations dropped).")
    auc_excl_parcels = Set{Int}()
    nan_obs_set      = Set{Tuple{String,String,Int}}(
        (row.subject, row.session, row.roi_pos_id) for row in eachrow(nan_df))
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Build long-format AUC DataFrame
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 2: Building long-format AUC DataFrame ═══\n")

isfile(PARTICIPANTS) || error("Drug-group metadata not found at $PARTICIPANTS — STOP")
part_df        = CSV.read(PARTICIPANTS, DataFrame; delim='\t')
drug_group_map = Dict{String,String}(row.participant_id => row.condition
                                     for row in eachrow(part_df))
println("Drug-group metadata loaded: $(length(drug_group_map)) participants")

rows = NamedTuple{(:subject, :session, :atlas, :roi_pos_id, :auc, :drug_group),
                  Tuple{String,String,String,Int,Float64,String}}[]

file_idx     = 0
n_unfiltered = 0

for subject in SUBJECTS, session in SESSIONS, atlas in ATLASES
    global file_idx, n_unfiltered
    file_idx += 1

    jld_path = joinpath(ACW_BASE, atlas, VERSION, "$(subject)_$(session).jld2")
    if !isfile(jld_path)
        println("[$file_idx/$TOTAL_FILES] MISSING: $jld_path")
        continue
    end

    data        = load(jld_path)
    acw_results = data["acw_results"]
    roi_columns = data["roi_columns"]  # nonself: Glasser ID strings; self: position ID strings
    auc_vals    = collect(acw_results[AUC_IDX])

    dg = get(drug_group_map, subject, "unknown")

    n_kept = 0
    for (col_name, auc) in zip(roi_columns, auc_vals)
        n_unfiltered += 1
        if atlas == "nonself"
            # Convert Glasser ID → nonself position
            glasser_id = parse(Int, col_name)
            pos_id     = glasser_to_pos[glasser_id]
            # tSNR exclusion (parcel-level)
            pos_id in excl_pos_set && continue
            # AUC NaN exclusion (parcel- or obs-level depending on Part 1 decision)
            pos_id in auc_excl_parcels && continue
            (subject, session, pos_id) in nan_obs_set && continue
        else
            pos_id = parse(Int, col_name)
        end
        push!(rows, (subject=subject, session=session, atlas=atlas,
                     roi_pos_id=pos_id, auc=Float64(auc), drug_group=dg))
        n_kept += 1
    end

    println("[$file_idx/$TOTAL_FILES] $subject $session $atlas → $n_kept ROIs kept")
end

long_df = DataFrame(rows)

@printf("\nUnfiltered observations loaded: %d [expected 24710: %s]\n",
        n_unfiltered, n_unfiltered == 24710 ? "OK" : "MISMATCH")

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Safety net: verify no NaN AUC remains
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 3: Post-exclusion NaN check ═══\n")

n_nan_remaining = count(isnan, long_df.auc)
n_inf_remaining = count(isinf, long_df.auc)
@printf("  NaN AUC remaining: %d [expected 0: %s]\n",
        n_nan_remaining, n_nan_remaining == 0 ? "OK" : "*** UNEXPECTED — investigate")
@printf("  Inf AUC remaining: %d [expected 0: %s]\n",
        n_inf_remaining, n_inf_remaining == 0 ? "OK" : "*** UNEXPECTED — investigate")

# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Verify drug-group join completeness
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 4: Drug-group join check ═══\n")

n_unknown = count(==("unknown"), long_df.drug_group)
if n_unknown > 0
    missing_subjs = unique(long_df.subject[long_df.drug_group .== "unknown"])
    @printf("  WARNING: %d rows have drug_group='unknown' — subjects: %s\n",
            n_unknown, join(missing_subjs, ", "))
else
    println("  All subjects have a drug_group assignment. OK")
end

drug_counts = combine(groupby(long_df, :drug_group), nrow => :n_rows)
println("\n  Drug-group row breakdown:")
for row in eachrow(drug_counts)
    @printf("    %-10s  %d rows\n", row.drug_group, row.n_rows)
end

# Session balance within each drug group (subject count, not rows)
println("\n  Subjects per drug_group × session (should be balanced):")
subj_sess = unique(select(long_df, :subject, :session, :drug_group))
bal = combine(groupby(subj_sess, [:drug_group, :session]), nrow => :n_subjects)
sort!(bal, [:drug_group, :session])
for row in eachrow(bal)
    @printf("    %-10s  %-8s  %d subjects\n",
            row.drug_group, row.session, row.n_subjects)
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — Sanity checks and save
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 5: Sanity checks ═══\n")

n_total   = nrow(long_df)
n_self    = count(==("self"),    long_df.atlas)
n_nonself = count(==("nonself"), long_df.atlas)

# Expected nonself depends on exclusion granularity chosen in Part 1
exp_nonself = if parcel_level_auc_excl
    35 * 2 * (316 - 58 - n_distinct_nan_parcels)
else
    35 * 2 * (316 - 58) - n_nan_ex   # 18060 - 18 = 18042
end
exp_self    = 35 * 2 * 37             # 2590
exp_total   = exp_self + exp_nonself

@printf("  %-18s  %8s  %8s\n", "", "actual", "expected")
println("  " * "─" ^ 38)
@printf("  %-18s  %8d  %8d  %s\n", "self",    n_self,    exp_self,
        n_self == exp_self ? "OK" : "MISMATCH")
@printf("  %-18s  %8d  %8d  %s\n", "nonself", n_nonself, exp_nonself,
        n_nonself == exp_nonself ? "OK" : "MISMATCH")
@printf("  %-18s  %8d  %8d  %s\n", "total",   n_total,   exp_total,
        n_total == exp_total ? "OK" : "MISMATCH")

println()
for sess in ["ses-01", "ses-02"]
    n = count(==(sess), long_df.session)
    @printf("  %s: %d rows\n", sess, n)
end

# AUC summary statistics per atlas (finite values only)
println("\n  AUC summary (finite values) per atlas:")
@printf("  %-10s  %10s  %10s  %10s  %8s  %8s\n",
        "atlas", "min", "median", "max", "n_NaN", "n_Inf")
println("  " * "─" ^ 62)
for at in ["self", "nonself"]
    sub = long_df[long_df.atlas .== at, :]
    auc_vals = sub.auc
    finite   = filter(isfinite, auc_vals)
    @printf("  %-10s  %10.4f  %10.4f  %10.4f  %8d  %8d\n",
            at,
            isempty(finite) ? NaN : minimum(finite),
            isempty(finite) ? NaN : median(finite),
            isempty(finite) ? NaN : maximum(finite),
            count(isnan, auc_vals),
            count(isinf, auc_vals))
end

# ── Save ───────────────────────────────────────────────────────────────────────
CSV.write(OUT_CSV, long_df)
println("\nWrote: $OUT_CSV  ($(nrow(long_df)) rows × $(ncol(long_df)) columns)")
println("\nDone.")
