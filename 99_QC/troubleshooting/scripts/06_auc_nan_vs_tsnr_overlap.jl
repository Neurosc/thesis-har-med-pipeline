# 06_auc_nan_vs_tsnr_overlap.jl — Step 8: AUC NaN vs tSNR exclusion overlap
#
# Checks whether the AUC=NaN nonself observations (360 expected) are already
# covered by the tSNR < 30 exclusion set (58 nonself parcels). If any NaN
# cases fall outside the exclusion set, they are listed and saved as a CSV.
#
# Run from repo root:
#   julia 99_QC/troubleshooting/scripts/06_auc_nan_vs_tsnr_overlap.jl

using CSV, DataFrames, JLD2, Statistics, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT   = normpath(joinpath(@__DIR__, "..", "..", ".."))
const ACW_BASE    = joinpath(REPO_ROOT, "03_intrinsic_neural_metrics", "results", "acw")
const VERSION     = "denoisedNoGSR"
const EXCL_CSV    = joinpath(REPO_ROOT, "04_statistics", "results",
                             "excluded_rois_translated.csv")
const ATLAS_TXT   = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                             "glasser_coordinates_nonself_clean_1mm.txt")
const TSNR_TSV    = joinpath(REPO_ROOT, "99_QC", "03_acw_qc", "results",
                             "tsnr", "excluded_rois_low_tsnr.tsv")  # may not exist
const OUT_DIR     = joinpath(REPO_ROOT, "99_QC", "troubleshooting", "results")
const OUT_CSV     = joinpath(OUT_DIR, "auc_nan_not_in_tsnr_exclusion.csv")

# ── Subjects (40 total; excluded: sub-06, sub-08, sub-12, sub-26, sub-36) ──────
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
const SESSIONS  = ["ses-01", "ses-02"]
const AUC_IDX   = 1  # acw_results[1] = AUC
const TAU_IDX   = 2  # acw_results[2] = τ

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(OUT_CSV)
    println("[skip] output already exists: $OUT_CSV")
    exit()
end

# ── Check required inputs ─────────────────────────────────────────────────────
for f in [EXCL_CSV, ATLAS_TXT]
    isfile(f) || error("Missing required input: $f")
end
mkpath(OUT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Load exclusion set and atlas lookup
# ══════════════════════════════════════════════════════════════════════════════
println("═══ Part 1: Load exclusion set and atlas ═══\n")

excl_df         = CSV.read(EXCL_CSV, DataFrame)
excluded_pos    = Set{Int}(excl_df.nonself_pos_id)
@printf("  tSNR exclusion set: %d nonself positions\n", length(excluded_pos))

# Atlas: row i (1-based) → nonself position i; ROI_Name gives the Glasser label
atlas_df = CSV.read(ATLAS_TXT, DataFrame; delim='\t')
n_rois   = nrow(atlas_df)
@printf("  Atlas rows: %d  (= expected 316 nonself ROIs)\n", n_rois)
n_rois == 316 || @warn "Expected 316 atlas rows; got $n_rois"

# Position → name mapping (row index = position)
pos_to_name = Dict(i => atlas_df[i, :ROI_Name] for i in 1:n_rois)

# tSNR per-position lookup from excluded_rois_translated.csv (for the exclusion set)
pos_to_tsnr = Dict{Int,Float64}(
    row.nonself_pos_id => row.tsnr for row in eachrow(excl_df)
)

has_tsnr_file = isfile(TSNR_TSV)
println("  excluded_rois_low_tsnr.tsv present: $has_tsnr_file")

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Load all nonself JLD2s; classify each (subject, session, roi_pos)
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 2: Classify all nonself observations ═══\n")

# Contingency accumulators
n_A = 0   # NaN AUC  AND in exclusion
n_B = 0   # NaN AUC  AND NOT in exclusion  ← the ones that matter
n_C = 0   # finite   AND in exclusion
n_D = 0   # finite   AND NOT in exclusion

nan_notexcl = []   # will hold (subject, session, roi_pos_id, roi_name, tau)

n_missing_files = 0
file_idx = 0

for subject in SUBJECTS, session in SESSIONS
    global file_idx, n_missing_files, n_A, n_B, n_C, n_D
    file_idx += 1

    jld_path = joinpath(ACW_BASE, "nonself", VERSION, "$(subject)_$(session).jld2")

    if !isfile(jld_path)
        println("  [$file_idx/70] MISSING: $jld_path")
        n_missing_files += 1
        continue
    end

    data        = load(jld_path)
    acw_results = data["acw_results"]
    auc_vals    = collect(acw_results[AUC_IDX])
    tau_vals    = collect(acw_results[TAU_IDX])

    n_pos = length(auc_vals)
    n_pos == n_rois || @warn "[$subject $session] AUC vector length $n_pos ≠ $n_rois"

    for pos in 1:n_pos
        auc_is_nan = isnan(auc_vals[pos])
        in_excl    = pos in excluded_pos

        if auc_is_nan && in_excl
            n_A += 1
        elseif auc_is_nan && !in_excl
            n_B += 1
            push!(nan_notexcl, (
                subject    = subject,
                session    = session,
                roi_pos_id = pos,
                roi_name   = get(pos_to_name, pos, "UNKNOWN"),
                tau        = tau_vals[pos],
            ))
        elseif !auc_is_nan && in_excl
            n_C += 1
        else
            n_D += 1
        end
    end

    n_nan_this = count(isnan, auc_vals)
    println("  [$file_idx/70] $subject $session → $(n_pos) ROIs,  AUC NaN: $n_nan_this")
end

n_total      = n_A + n_B + n_C + n_D
n_nan_total  = n_A + n_B
n_excl_total = n_A + n_C

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Contingency table
# ══════════════════════════════════════════════════════════════════════════════
println()
println("═══ Part 3: Contingency table ═══\n")

@printf("                    │  In tSNR exclusion  │  NOT in tSNR excl.  │   Total\n")
println("  ──────────────────┼─────────────────────┼─────────────────────┼─────────")
@printf("  AUC = NaN         │  %19d  │  %19d  │  %6d\n", n_A, n_B, n_A + n_B)
@printf("  AUC finite        │  %19d  │  %19d  │  %6d\n", n_C, n_D, n_C + n_D)
println("  ──────────────────┼─────────────────────┼─────────────────────┼─────────")
@printf("  Total             │  %19d  │  %19d  │  %6d\n", n_A + n_C, n_B + n_D, n_total)
println()
n_total == 22120 || @warn "Expected 22,120 total observations; got $n_total (missing files: $n_missing_files)"

# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — NaN cases NOT in tSNR exclusion (cell B)
# ══════════════════════════════════════════════════════════════════════════════
println("═══ Part 4: NaN cases outside tSNR exclusion (B = $n_B) ═══\n")

out_df = DataFrame(
    subject    = String[],
    session    = String[],
    roi_pos_id = Int[],
    roi_name   = String[],
    tau        = Float64[],
    tsnr_status = String[],
)

if n_B > 0
    for r in nan_notexcl
        # These ROIs are NOT in the tSNR exclusion set → tSNR was above threshold
        tsnr_status = "above_30"
        push!(out_df, (r.subject, r.session, r.roi_pos_id, r.roi_name,
                       Float64(r.tau), tsnr_status))
    end

    println("  Listing all $n_B observations:\n")
    @printf("  %-10s %-8s  %10s  %-20s  %12s  %s\n",
            "subject", "session", "roi_pos_id", "roi_name", "tau", "tsnr_status")
    println("  " * "─" ^ 76)
    for row in eachrow(out_df)
        tau_str = isnan(row.tau) ? "NaN" : @sprintf("%.6f", row.tau)
        @printf("  %-10s %-8s  %10d  %-20s  %12s  %s\n",
                row.subject, row.session, row.roi_pos_id, row.roi_name,
                tau_str, row.tsnr_status)
    end

    # Per-ROI breakdown
    println()
    println("  Per-ROI breakdown (NaN cases outside exclusion):")
    roi_counts = combine(groupby(out_df, [:roi_pos_id, :roi_name]), nrow => :n_nan_obs)
    sort!(roi_counts, :n_nan_obs, rev=true)
    @printf("  %10s  %-20s  %s\n", "roi_pos_id", "roi_name", "n_nan_obs")
    println("  " * "─" ^ 46)
    for row in eachrow(roi_counts)
        @printf("  %10d  %-20s  %d\n", row.roi_pos_id, row.roi_name, row.n_nan_obs)
    end
    n_distinct_rois = nrow(roi_counts)
    println()
    @printf("  → %d distinct ROIs across %d observations\n", n_distinct_rois, n_B)
    if n_distinct_rois <= 3
        println("    Concentrated in a small set — candidate for targeted exclusion.")
    else
        println("    Scattered across many ROIs — likely data-quality issue, not anatomical.")
    end

    CSV.write(OUT_CSV, out_df)
    println("\n  CSV written: $OUT_CSV  ($(nrow(out_df)) rows)")
else
    println("  B = 0: all AUC=NaN observations are already covered by tSNR exclusion.")
    println("  No CSV output needed.")
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — Console summary
# ══════════════════════════════════════════════════════════════════════════════
println()
println("═══ Part 5: Summary ═══\n")

action = n_B == 0 ? "no action" : "additional exclusion rule"
println("  Of $(n_nan_total) AUC=NaN nonself observations, $n_A were already removed")
println("  by tSNR exclusion. $n_B remain — requiring $action.")
println()
println("Done.")
