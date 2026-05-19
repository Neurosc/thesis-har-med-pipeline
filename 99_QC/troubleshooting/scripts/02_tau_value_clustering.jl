# 02_tau_value_clustering.jl — Spike B τ value clustering check
#
# Tests whether self atlas Spike B cases share a single deterministic τ
# (5.193702147200268 = -3.6 / log(0.5)), indicating a stuck LevenbergMarquardt
# fit at the ACW-50 initial value rather than a true distribution of timescales.
#
# Run from repo root:
#   julia 99_QC/troubleshooting/scripts/02_tau_value_clustering.jl

using CSV, DataFrames, Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const LONG_CSV  = joinpath(REPO_ROOT, "04_statistics", "results", "analysis_long_format.csv")

isfile(LONG_CSV) || error("Missing: $LONG_CSV")

long_df = CSV.read(LONG_CSV, DataFrame)

# ── Reference value ───────────────────────────────────────────────────────────
ref_val = -3.6 / log(0.5)
println()
println("REFERENCE VALUE")
@printf("  -3.6 / log(0.5) = %.15f\n", ref_val)
println()

# ── Helper: analysis for one atlas ───────────────────────────────────────────
function analyse(df, atlas_label)
    rows = filter(r -> 5.0 ≤ r.tau ≤ 5.3, df)

    n_total    = nrow(rows)
    n_distinct = length(unique(rows.tau))
    n_exact    = count(==(5.193702147200268), rows.tau)

    println("$atlas_label Spike B (τ ∈ [5.0, 5.3])")
    println("  Total rows:                         $n_total")
    println("  Distinct τ values (full precision): $n_distinct")
    println("  Rows exactly = 5.193702147200268:   $n_exact")

    # Histogram: round to 6 dp, count, sort by frequency
    rounded = round.(rows.tau; digits=6)
    freq    = sort(collect(countmap(rounded)), by=x -> -x[2])

    println("  Top 10 most common τ values (rounded to 6 dp):")
    @printf("    %-14s  %6s  %7s\n", "τ_value", "count", "percent")
    println("    " * "─" ^ 32)
    for (val, cnt) in first(freq, 10)
        pct = 100.0 * cnt / n_total
        @printf("    %-14.6f  %6d  %6.2f%%\n", val, cnt, pct)
    end
    println()
end

# countmap is in StatsBase; implement a lightweight version here to avoid the
# dependency (the data are small).
function countmap(v)
    d = Dict{eltype(v), Int}()
    for x in v
        d[x] = get(d, x, 0) + 1
    end
    d
end

self_df    = filter(r -> r.atlas == "self",    long_df)
nonself_df = filter(r -> r.atlas == "nonself", long_df)

analyse(self_df,    "SELF ATLAS")
analyse(nonself_df, "NONSELF ATLAS")
