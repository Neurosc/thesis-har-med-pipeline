# 02_p0_stuck_diagnostic.jl — Step 5b-iii: Quantify p0-stuck τ values
#
# The IntrinsicTimescales.jl fit initialises τ from ACW-50:
#   p0 = -acw50 / log(0.5)  [or 1.0 if ACW-50 is NaN]
# For our discrete lag grid (TR = 1.8 s), ACW-50 lands at integer multiples
# of 1.8, producing a finite set of deterministic p0 values. If the
# LevenbergMarquardt optimizer does not move from p0 (flat MSE surface),
# τ is returned unchanged. This script counts how many stored τ values
# match these p0 candidates exactly (within 1e-9), per atlas.
#
# Run from repo root:
#   julia 99_QC/troubleshooting/scripts/02_p0_stuck_diagnostic.jl

using CSV, DataFrames, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT  = normpath(joinpath(@__DIR__, "..", "..", ".."))
const LONG_CSV   = joinpath(REPO_ROOT, "04_statistics", "results",
                             "analysis_long_format.csv")
const OUT_DIR    = joinpath(REPO_ROOT, "99_QC", "troubleshooting", "results")
const P0_CSV     = joinpath(OUT_DIR, "p0_match_summary.csv")
const TOP_CSV    = joinpath(OUT_DIR, "top_exact_tau_values.csv")

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(P0_CSV) && isfile(TOP_CSV)
    println("[skip] all outputs already exist; delete to rerun")
    exit()
end

isfile(LONG_CSV) || error("Missing required input: $LONG_CSV")
mkpath(OUT_DIR)

# ── p0 candidates ─────────────────────────────────────────────────────────────
# p0 from ACW-50 = k × TR:  p0 = -(k × TR) / log(0.5) = (k × TR) / log(2)
# k=0 is the NaN fallback (p0 = 1.0, hardcoded in the package source).
const TR      = 1.8
const TOL     = 1e-9
const TOP_N   = 20

p0_candidates = [
    (label = "NaN fallback (hardcoded 1.0)",  value = 1.0, lag = NaN),
]
for k in 1:10
    lag = k * TR
    p0  = lag / log(2)
    push!(p0_candidates, (
        label = "ACW-50 = $(lag) s (lag $(k))",
        value = p0,
        lag   = lag,
    ))
end

# Print computed p0 candidates for visual verification
println("Computed p0 candidates ($(length(p0_candidates)) total):")
for cand in p0_candidates
    @printf("  %22.15f  %s\n", cand.value, cand.label)
end

# ── Load data ─────────────────────────────────────────────────────────────────
long_df    = CSV.read(LONG_CSV, DataFrame)
self_df    = filter(r -> r.atlas == "self",    long_df)
nonself_df = filter(r -> r.atlas == "nonself", long_df)

n_self    = nrow(self_df)
n_nonself = nrow(nonself_df)

println("\nLoaded $(nrow(long_df)) rows  (self: $n_self, nonself: $n_nonself)")

# ── p0 match counts ───────────────────────────────────────────────────────────
function count_matches(tau_vec, p0_val)
    count(t -> abs(t - p0_val) < TOL, tau_vec)
end

p0_rows = []
for cand in p0_candidates
    ns = count_matches(self_df.tau,    cand.value)
    nn = count_matches(nonself_df.tau, cand.value)
    push!(p0_rows, (
        p0_candidate  = cand.value,
        source        = cand.label,
        n_self        = ns,
        n_nonself     = nn,
        pct_self      = 100.0 * ns / n_self,
        pct_nonself   = 100.0 * nn / n_nonself,
    ))
end

p0_df = DataFrame(p0_rows)

# ── Top-N exact τ values ──────────────────────────────────────────────────────
function countmap(v)
    d = Dict{eltype(v), Int}()
    for x in v; d[x] = get(d, x, 0) + 1; end
    d
end

tau_all     = long_df.tau
tau_freq    = sort(collect(countmap(tau_all)), by=x -> -x[2])
top_entries = first(tau_freq, TOP_N)

p0_val_set = Set(c.value for c in p0_candidates)

top_rows = []
for (tau_val, cnt) in top_entries
    ns     = count(==(tau_val), self_df.tau)
    nn     = count(==(tau_val), nonself_df.tau)
    is_p0  = any(abs(tau_val - p) < TOL for p in p0_val_set)
    cand_label = if is_p0
        first(c.label for c in p0_candidates if abs(tau_val - c.value) < TOL)
    else
        ""
    end
    push!(top_rows, (
        tau_value           = tau_val,
        count               = cnt,
        n_self              = ns,
        n_nonself           = nn,
        matches_p0_candidate = is_p0 ? "Y" : "N",
        candidate_label     = cand_label,
    ))
end

top_df = DataFrame(top_rows)

# ══════════════════════════════════════════════════════════════════════════════
# Console output
# ══════════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 78)
println("SELF ATLAS")
println("=" ^ 78)
println("Total observations: $n_self")
println()
@printf("  %-14s  %-36s  %7s  %7s\n", "p0 candidate", "source", "count", "pct")
println("  " * "─" ^ 72)
for r in eachrow(p0_df)
    @printf("  %-14.9f  %-36s  %7d  %6.3f%%\n",
            r.p0_candidate, r.source, r.n_self, r.pct_self)
end
n_stuck_self = sum(p0_df.n_self)
println()
@printf("  TOTAL matching any p0: %d / %d  (%.2f%%)\n",
        n_stuck_self, n_self, 100.0 * n_stuck_self / n_self)
@printf("  NOT matching any p0:   %d / %d  (%.2f%%)\n",
        n_self - n_stuck_self, n_self,
        100.0 * (n_self - n_stuck_self) / n_self)

println()
println("=" ^ 78)
println("NONSELF ATLAS")
println("=" ^ 78)
println("Total observations: $n_nonself")
println()
@printf("  %-14s  %-36s  %7s  %7s\n", "p0 candidate", "source", "count", "pct")
println("  " * "─" ^ 72)
for r in eachrow(p0_df)
    @printf("  %-14.9f  %-36s  %7d  %6.3f%%\n",
            r.p0_candidate, r.source, r.n_nonself, r.pct_nonself)
end
n_stuck_nonself = sum(p0_df.n_nonself)
println()
@printf("  TOTAL matching any p0: %d / %d  (%.2f%%)\n",
        n_stuck_nonself, n_nonself, 100.0 * n_stuck_nonself / n_nonself)
@printf("  NOT matching any p0:   %d / %d  (%.2f%%)\n",
        n_nonself - n_stuck_nonself, n_nonself,
        100.0 * (n_nonself - n_stuck_nonself) / n_nonself)

println()
println("=" ^ 78)
println("TOP $TOP_N MOST COMMON EXACT τ VALUES (both atlases combined)")
println("=" ^ 78)
@printf("  %-22s  %7s  %7s  %7s  %3s  %-36s\n",
        "τ value", "count", "n_self", "n_nself", "p0?", "candidate label")
println("  " * "─" ^ 92)
for r in eachrow(top_df)
    @printf("  %-22.15f  %7d  %7d  %7d  %3s  %-36s\n",
            r.tau_value, r.count, r.n_self, r.n_nonself,
            r.matches_p0_candidate, r.candidate_label)
end

# ── Write CSVs ────────────────────────────────────────────────────────────────
CSV.write(P0_CSV,  p0_df)
println("\nWrote: $P0_CSV")
CSV.write(TOP_CSV, top_df)
println("Wrote: $TOP_CSV")
println("\nDone.")
