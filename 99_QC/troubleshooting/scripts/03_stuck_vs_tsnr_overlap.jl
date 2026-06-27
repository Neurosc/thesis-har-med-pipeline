# 03_stuck_vs_tsnr_overlap.jl — Step 5b-iv: Cross-reference p0-stuck τ values
# with tSNR exclusion
#
# Identifies all τ observations that equal a known fit-initialization value
# (p0) to within 1e-9, then determines how many are already removed by the
# tSNR < 30 exclusion applied in Step 7 versus how many remain in the main
# analysis dataset.
#
# Run from repo root:
#   julia 99_QC/troubleshooting/scripts/03_stuck_vs_tsnr_overlap.jl

using CSV, DataFrames, JLD2, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT  = normpath(joinpath(@__DIR__, "..", "..", ".."))
const LONG_CSV   = joinpath(REPO_ROOT, "04_statistics", "results",
                             "analysis_long_format.csv")
const EXCL_CSV   = joinpath(REPO_ROOT, "04_statistics", "results",
                             "excluded_rois_translated.csv")
const ATLAS_FILE = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                             "glasser_coordinates_nonself_clean_1mm.txt")
const SELF_LAB   = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                             "self_labels.txt")
const ACW_BASE   = joinpath(REPO_ROOT, "03_intrinsic_neural_metrics", "results", "acw")
const OUT_DIR    = joinpath(REPO_ROOT, "99_QC", "troubleshooting", "results")
const OUT_CSV    = joinpath(OUT_DIR, "stuck_remaining_after_tsnr.csv")

# ── Constants (must match 01_compute_acw.jl and Step 7) ──────────────────────
const VERSION = "denoisedNoGSR"
const TAU_IDX = 2
const TOL     = 1e-9

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

# Confirmed stuck p0 values (empirically verified in Step 5b-iii)
const P0_VALUES = [1.0, 5.193702147200269, 7.790553220800402]
const P0_LABELS = [
    "NaN fallback (1.0)",
    "lag-2 p0 (ACW-50 ≈ 3.6 s)",
    "lag-3 p0 (ACW-50 ≈ 5.4 s)",
]

is_stuck(tau)      = any(abs(tau - p) < TOL for p in P0_VALUES)
which_p0_idx(tau)  = findfirst(p -> abs(tau - p) < TOL, P0_VALUES)

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(OUT_CSV)
    println("[skip] output already exists; delete to rerun")
    exit()
end

for f in [LONG_CSV, EXCL_CSV, ATLAS_FILE, SELF_LAB]
    isfile(f) || error("Missing required input: $f")
end
mkpath(OUT_DIR)

# ── ROI name lookups ──────────────────────────────────────────────────────────
_atlas_df     = CSV.read(ATLAS_FILE, DataFrame; delim='\t')
nonself_names = Dict(pos => row.ROI_Name
                     for (pos, row) in enumerate(eachrow(_atlas_df)))
glasser_to_pos = Dict(row.ROI_Number => pos
                      for (pos, row) in enumerate(eachrow(_atlas_df)))

self_names = Dict{Int,String}()
for line in eachline(SELF_LAB)
    parts = split(strip(line))
    (isempty(parts) || length(parts) < 2) && continue
    self_names[parse(Int, parts[1])] = join(parts[2:end], " ")
end

get_roi_name(atlas, pos) = atlas == "self" ?
    get(self_names, pos, "ROI_$pos") : get(nonself_names, pos, "ROI_$pos")

# ── Load filtered long_df (Step 7 output, tSNR exclusion already applied) ────
filtered_df = CSV.read(LONG_CSV, DataFrame)
n_filt      = nrow(filtered_df)
n_filt_self    = count(==("self"),    filtered_df.atlas)
n_filt_nonself = count(==("nonself"), filtered_df.atlas)
println("Filtered long_df: $n_filt rows  (self: $n_filt_self, nonself: $n_filt_nonself)")
n_filt == 20650 || @warn "Expected 20650 filtered rows; got $n_filt"

# ── Load tSNR exclusion set ───────────────────────────────────────────────────
excl_df      = CSV.read(EXCL_CSV, DataFrame)
excl_pos_set = Set(excl_df.nonself_pos_id)
println("tSNR exclusion set: $(length(excl_pos_set)) nonself parcels")

# ── Reconstruct unfiltered nonself from JLD2 (no tSNR filtering) ─────────────
println("\nReconstructing unfiltered nonself from JLD2 files...")
ns_rows = NamedTuple{(:subject, :session, :roi_pos_id, :tau),
                     Tuple{String,String,Int,Float64}}[]
n_ns_files = length(SUBJECTS) * length(SESSIONS)
ns_file_idx = 0

for subject in SUBJECTS, session in SESSIONS
    global ns_file_idx
    ns_file_idx += 1
    jld_path = joinpath(ACW_BASE, "nonself", VERSION, "$(subject)_$(session).jld2")
    if !isfile(jld_path)
        println("  [$ns_file_idx/$n_ns_files] MISSING: $jld_path")
        continue
    end
    data        = load(jld_path)
    acw_results = data["acw_results"]
    roi_columns = data["roi_columns"]
    tau_vals    = collect(acw_results[TAU_IDX])
    for (col_name, tau) in zip(roi_columns, tau_vals)
        glasser_id = parse(Int, col_name)
        pos_id     = glasser_to_pos[glasser_id]
        push!(ns_rows, (subject=subject, session=session,
                        roi_pos_id=pos_id, tau=Float64(tau)))
    end
end

ns_unfiltered = DataFrame(ns_rows)
n_ns_unfilt   = nrow(ns_unfiltered)
println("Unfiltered nonself rows: $n_ns_unfilt  [expected 22120]")
n_ns_unfilt == 22120 || @warn "Expected 22120 unfiltered nonself rows; got $n_ns_unfilt"

# Self: tSNR filter was never applied to self → filtered_df already complete
self_df       = filter(r -> r.atlas == "self", filtered_df)
n_self_unfilt = nrow(self_df)
println("Self rows (no filter applied): $n_self_unfilt  [expected 2590]")
n_self_unfilt == 2590 || @warn "Expected 2590 self rows; got $n_self_unfilt"

# ══════════════════════════════════════════════════════════════════════════════
# Analysis — nonself
# ══════════════════════════════════════════════════════════════════════════════
ns_stuck_mask    = is_stuck.(ns_unfiltered.tau)
ns_excluded_mask = [r.roi_pos_id in excl_pos_set for r in eachrow(ns_unfiltered)]

n_ns_stuck       = sum(ns_stuck_mask)
n_self_stuck     = sum(is_stuck.(self_df.tau))

println("\n═══ Sanity checks ═══")
@printf("  Unfiltered nonself stuck: %d\n",  n_ns_stuck)
@printf("  Self stuck:               %d  [expected 33]\n", n_self_stuck)
n_self_stuck == 33 || @warn "Expected 33 self stuck; got $n_self_stuck"

# 2×2 contingency
stuck_excl       = sum(ns_stuck_mask  .& ns_excluded_mask)
stuck_notexcl    = sum(ns_stuck_mask  .& .!ns_excluded_mask)
notstuck_excl    = sum(.!ns_stuck_mask .& ns_excluded_mask)
notstuck_notexcl = sum(.!ns_stuck_mask .& .!ns_excluded_mask)

println()
println("═══ Nonself: stuck × tSNR-exclusion contingency ═══")
println()
@printf("  %-14s  %20s  %22s  %8s\n",
        "", "In tSNR exclusion", "NOT in tSNR exclusion", "Total")
println("  " * "─" ^ 72)
@printf("  %-14s  %20d  %22d  %8d\n",
        "Stuck",     stuck_excl,    stuck_notexcl,    n_ns_stuck)
@printf("  %-14s  %20d  %22d  %8d\n",
        "Not stuck", notstuck_excl, notstuck_notexcl, n_ns_unfilt - n_ns_stuck)
println("  " * "─" ^ 72)
@printf("  %-14s  %20d  %22d  %8d\n",
        "Total",
        sum(ns_excluded_mask), sum(.!ns_excluded_mask), n_ns_unfilt)

# ── Per-p0 breakdown ──────────────────────────────────────────────────────────
println()
println("═══ Nonself: per-p0-value breakdown ═══")
println()
@printf("  %-32s  %8s  %12s  %12s\n",
        "p0 label", "total", "in_excl", "not_in_excl")
println("  " * "─" ^ 70)
for (p0, label) in zip(P0_VALUES, P0_LABELS)
    mask_p = abs.(ns_unfiltered.tau .- p0) .< TOL
    n_p    = sum(mask_p)
    n_in   = sum(mask_p .& ns_excluded_mask)
    n_out  = n_p - n_in
    @printf("  %-32s  %8d  %12d  %12d\n", label, n_p, n_in, n_out)
end

# ══════════════════════════════════════════════════════════════════════════════
# Analysis — self
# ══════════════════════════════════════════════════════════════════════════════
stuck_self_df = filter(r -> is_stuck(r.tau), self_df)
n_distinct_combos = nrow(unique(stuck_self_df, [:subject, :session, :roi_pos_id]))
n_distinct_rois   = length(unique(stuck_self_df.roi_pos_id))

println()
println("═══ Self: stuck observation distribution ═══")
println()
@printf("  Total stuck self observations:            %d\n", n_self_stuck)
@printf("  Distinct (subject, session, ROI) combos: %d\n", n_distinct_combos)
@printf("  Distinct ROIs:                            %d\n", n_distinct_rois)
println()
println("  ROI recurrence (how many subjects have each stuck ROI):")
roi_sub = combine(groupby(stuck_self_df, :roi_pos_id),
                  :subject => (x -> length(unique(x))) => :n_subjects)
sort!(roi_sub, :n_subjects; rev=true)
@printf("  %-8s  %-30s  %s\n", "roi_id", "roi_name", "n_subjects")
println("  " * "─" ^ 55)
for r in eachrow(roi_sub)
    @printf("  %-8d  %-30s  %d\n",
            r.roi_pos_id, get_roi_name("self", r.roi_pos_id), r.n_subjects)
end

# ── Per-p0 breakdown for self ─────────────────────────────────────────────────
println()
println("  Per-p0-value breakdown (self):")
@printf("  %-32s  %8s\n", "p0 label", "count")
println("  " * "─" ^ 44)
for (p0, label) in zip(P0_VALUES, P0_LABELS)
    n_p = count(t -> abs(t - p0) < TOL, self_df.tau)
    @printf("  %-32s  %8d\n", label, n_p)
end

# ══════════════════════════════════════════════════════════════════════════════
# Remaining stuck after tSNR exclusion (= stuck rows in filtered_df)
# ══════════════════════════════════════════════════════════════════════════════
stuck_rem_df     = filter(r -> is_stuck(r.tau), filtered_df)
n_stuck_rem_self    = count(==("self"),    stuck_rem_df.atlas)
n_stuck_rem_nonself = count(==("nonself"), stuck_rem_df.atlas)

println()
println("═══ Stuck observations remaining after tSNR exclusion ═══")
println()
@printf("  Self:    %d / %d  (%.2f%%)\n",
        n_stuck_rem_self,    n_filt_self,
        100.0 * n_stuck_rem_self    / n_filt_self)
@printf("  Nonself: %d / %d  (%.2f%%)\n",
        n_stuck_rem_nonself, n_filt_nonself,
        100.0 * n_stuck_rem_nonself / n_filt_nonself)
println()
println("  *** HEADLINE ***")
@printf("  %d observations remain p0-stuck after tSNR exclusion:\n",
        nrow(stuck_rem_df))
@printf("  %d self (out of %d) + %d nonself (out of %d) = %d total\n",
        n_stuck_rem_self, n_filt_self,
        n_stuck_rem_nonself, n_filt_nonself,
        nrow(stuck_rem_df))

# ── Write CSV ─────────────────────────────────────────────────────────────────
out_rows = []
for r in eachrow(stuck_rem_df)
    idx = which_p0_idx(r.tau)
    push!(out_rows, (
        subject    = r.subject,
        session    = r.session,
        atlas      = r.atlas,
        roi_pos_id = r.roi_pos_id,
        roi_name   = get_roi_name(r.atlas, r.roi_pos_id),
        p0_value   = P0_VALUES[idx],
        p0_label   = P0_LABELS[idx],
    ))
end
out_df = DataFrame(out_rows)
CSV.write(OUT_CSV, out_df)
println("\nWrote: $OUT_CSV  ($(nrow(out_df)) rows)")
println("Done.")
