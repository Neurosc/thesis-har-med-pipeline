# 07_auc_distribution.jl — AUC distribution by atlas (denoisedNoGSR)
# Loads all 140 JLD2 files and accumulates raw AUC values (one per ROI per run).
# Produces an overlaid histogram: self vs nonself, all subjects and sessions pooled.
#
# Run from repo root:  julia 04_statistics/scripts/07_auc_distribution.jl

using JLD2, PlotlyJS, Statistics

# ── Paths ──────────────────────────────────────────────────────────────────────
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ACW_BASE  = joinpath(REPO_ROOT, "03_acw_analysis", "results", "acw")
const VERSION   = "denoisedNoGSR"
const FIG_DIR   = joinpath(REPO_ROOT, "04_statistics", "figures")
const FIG_PATH  = joinpath(FIG_DIR, "fig07_auc_distribution.html")

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

const SESSIONS    = ["ses-01", "ses-02"]
const ATLASES     = ["self", "nonself"]
const AUC_IDX     = 1  # ACW_TYPES = [:auc, :tau]; auc is index 1
const TOTAL_FILES = length(SUBJECTS) * length(SESSIONS) * length(ATLASES)  # 140

# ── Palette A ──────────────────────────────────────────────────────────────────
const COLOR_SELF    = "#3D5A6C"  # slate
const COLOR_NONSELF = "#8B7355"  # taupe

# ── Idempotency ────────────────────────────────────────────────────────────────
if isfile(FIG_PATH)
    println("[skip] output already exists at $FIG_PATH")
    exit()
end

mkpath(FIG_DIR)

# ── Load AUC values ────────────────────────────────────────────────────────────
self_auc    = Float64[]
nonself_auc = Float64[]

file_idx = 0

for subject in SUBJECTS, session in SESSIONS, atlas in ATLASES
    global file_idx
    file_idx += 1

    jld_path = joinpath(ACW_BASE, atlas, VERSION, "$(subject)_$(session).jld2")

    if !isfile(jld_path)
        println("[$file_idx/$TOTAL_FILES] MISSING: $jld_path")
        continue
    end

    data        = load(jld_path)
    acw_results = data["acw_results"]
    auc_vals    = collect(acw_results[AUC_IDX])

    println("[$file_idx/$TOTAL_FILES] $subject $session $atlas → $(length(auc_vals)) ROIs")

    if atlas == "self"
        append!(self_auc, auc_vals)
    else
        append!(nonself_auc, auc_vals)
    end
end

# ── Console summary ────────────────────────────────────────────────────────────
fmt(v) = round(v; digits=3)

function atlas_summary(label, vals)
    n_total   = length(vals)
    n_nan     = count(isnan, vals)
    n_inf     = count(isinf, vals)
    finite    = filter(isfinite, vals)
    n_finite  = length(finite)
    n_distinct = length(unique(finite))

    println("\n$label (N=$n_total, expected: $(label == "Self" ? 2590 : 22120))")
    println("  NaN: $n_nan  Inf: $n_inf  Finite: $n_finite  Distinct: $n_distinct")
    if n_finite > 0
        println("  min=$(fmt(minimum(finite)))  median=$(fmt(median(finite)))  max=$(fmt(maximum(finite)))")
    end

    # Top 5 most common finite values
    counts = Dict{Float64,Int}()
    for v in finite
        counts[v] = get(counts, v, 0) + 1
    end
    top5 = sort(collect(counts); by=x -> -x[2])[1:min(5, end)]
    println("  Top 5 most common finite AUC values:")
    for (v, c) in top5
        println("    $(fmt(v))  →  $c occurrences")
    end
end

atlas_summary("Self",    self_auc)
atlas_summary("Nonself", nonself_auc)

# ── Figure ─────────────────────────────────────────────────────────────────────
trace_self = PlotlyJS.histogram(
    x       = filter(isfinite, self_auc),
    name    = "Self (N=$(count(isfinite, self_auc)))",
    opacity = 0.6,
    marker  = attr(color=COLOR_SELF),
)

trace_nonself = PlotlyJS.histogram(
    x       = filter(isfinite, nonself_auc),
    name    = "Nonself (N=$(count(isfinite, nonself_auc)))",
    opacity = 0.6,
    marker  = attr(color=COLOR_NONSELF),
)

layout = PlotlyJS.Layout(
    title         = attr(
        text = "AUC distribution by atlas — denoisedNoGSR, all subjects and sessions pooled",
        font = attr(size=15),
    ),
    barmode       = "overlay",
    xaxis         = attr(title="AUC (seconds)"),
    yaxis         = attr(title="Count"),
    font          = attr(family="Times New Roman", size=13),
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    legend        = attr(x=0.75, y=0.95),
)

fig = PlotlyJS.plot([trace_self, trace_nonself], layout)
PlotlyJS.savefig(fig, FIG_PATH)
println("\n[done] Figure → $FIG_PATH")
