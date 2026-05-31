# 02_tau_distribution.jl — Overall τ distribution by atlas (denoisedNoGSR)
# Loads all 140 JLD2 files and accumulates raw τ values (one per ROI per run).
# Produces an overlaid histogram: self vs nonself, all subjects and sessions pooled.
#
# Run from repo root:  julia 04_statistics/scripts/02_tau_distribution.jl

using JLD2, PlotlyJS, Statistics

# ── Paths ──────────────────────────────────────────────────────────────────────
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ACW_BASE  = joinpath(REPO_ROOT, "03_acw_analysis", "results", "acw")
const VERSION   = "denoisedNoGSR"
const FIG_DIR   = joinpath(REPO_ROOT, "04_statistics", "figures")
const FIG_PATH  = joinpath(FIG_DIR, "fig02_tau_distribution.html")

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

const SESSIONS      = ["ses-01", "ses-02"]
const ATLASES       = ["self", "nonself"]
const TAU_IDX       = 2  # ACW_TYPES = [:auc, :tau]; tau is index 2
const TOTAL_FILES   = length(SUBJECTS) * length(SESSIONS) * length(ATLASES)  # 140

# ── Palette A ──────────────────────────────────────────────────────────────────
const COLOR_SELF    = "#3D5A6C"  # slate
const COLOR_NONSELF = "#8B7355"  # taupe

# ── Idempotency ────────────────────────────────────────────────────────────────
if isfile(FIG_PATH)
    println("[skip] output already exists at $FIG_PATH")
    exit()
end

mkpath(FIG_DIR)

# ── Load τ values ──────────────────────────────────────────────────────────────
self_tau    = Float64[]
nonself_tau = Float64[]

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
    tau_vals    = collect(acw_results[TAU_IDX])

    println("[$file_idx/$TOTAL_FILES] $subject $session $atlas → $(length(tau_vals)) ROIs")

    if atlas == "self"
        append!(self_tau, tau_vals)
    else
        append!(nonself_tau, tau_vals)
    end
end

# ── Console summary ────────────────────────────────────────────────────────────
fmt(v) = round(v; digits=3)

println("\nSelf:    N=$(length(self_tau)),  min=$(fmt(minimum(self_tau))),  " *
        "median=$(fmt(median(self_tau))),  max=$(fmt(maximum(self_tau)))")
println("Nonself: N=$(length(nonself_tau)),  min=$(fmt(minimum(nonself_tau))),  " *
        "median=$(fmt(median(nonself_tau))),  max=$(fmt(maximum(nonself_tau)))")

# ── Figure ─────────────────────────────────────────────────────────────────────
trace_self = PlotlyJS.histogram(
    x         = self_tau,
    name      = "Self (N=$(length(self_tau)))",
    opacity   = 0.6,
    marker    = attr(color=COLOR_SELF),
)

trace_nonself = PlotlyJS.histogram(
    x         = nonself_tau,
    name      = "Nonself (N=$(length(nonself_tau)))",
    opacity   = 0.6,
    marker    = attr(color=COLOR_NONSELF),
)

layout = PlotlyJS.Layout(
    title         = attr(
        text = "τ distribution by atlas — denoisedNoGSR, all subjects and sessions pooled",
        font = attr(size=15),
    ),
    barmode       = "overlay",
    xaxis         = attr(title="τ (seconds)"),
    yaxis         = attr(title="Count"),
    font          = attr(family="Times New Roman", size=13),
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    legend        = attr(x=0.75, y=0.95),
)

fig = PlotlyJS.plot([trace_self, trace_nonself], layout)
PlotlyJS.savefig(fig, FIG_PATH)
println("[done] Figure → $FIG_PATH")
