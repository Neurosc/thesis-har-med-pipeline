# 01_check_missingness.jl — ACW file integrity check (denoisedNoGSR)
# Enumerates all 140 expected JLD2 files (35 subjects × 2 sessions × 2 atlases),
# checks existence, loadability, τ-vector shape, and NaN/Inf content.
# Produces an HTML heatmap and a CSV of non-OK files.
#
# Run from repo root:  julia 04_statistics/scripts/01_check_missingness.jl

using JLD2, PlotlyJS, DataFrames, CSV

# ── Paths ──────────────────────────────────────────────────────────────────────
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ACW_BASE  = joinpath(REPO_ROOT, "03_acw_analysis", "results", "acw")
const VERSION   = "denoisedNoGSR"
const FIG_DIR   = joinpath(REPO_ROOT, "04_statistics", "figures")
const RES_DIR   = joinpath(REPO_ROOT, "04_statistics", "results")
const FIG_PATH  = joinpath(FIG_DIR, "fig01_missingness.html")
const CSV_PATH  = joinpath(RES_DIR, "fig01_missingness_issues.csv")

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

const SESSIONS  = ["ses-01", "ses-02"]
const ATLASES   = ["self", "nonself"]
const TAU_IDX   = 2  # ACW_TYPES = [:auc, :tau]; tau is index 2

const EXPECTED_N_ROIS = Dict("self" => 37, "nonself" => 316)
const TOTAL_EXPECTED  = length(SUBJECTS) * length(SESSIONS) * length(ATLASES)  # 140

# Heatmap columns: atlas varies slowly, session varies fast
const COL_LABELS = ["self / ses-01", "self / ses-02", "nonself / ses-01", "nonself / ses-02"]
const COL_KEYS   = [("self","ses-01"), ("self","ses-02"), ("nonself","ses-01"), ("nonself","ses-02")]

# ── Palette A ──────────────────────────────────────────────────────────────────
const STATUS_COLORS = Dict(
    "OK"          => "#3D5A6C",
    "MISSING"     => "#C0392B",
    "UNLOADABLE"  => "#D98859",
    "WRONG_SHAPE" => "#E2A93B",
    "HAS_NANS"    => "#8B7355",
)
const STATUS_ORDER = ["OK", "MISSING", "UNLOADABLE", "WRONG_SHAPE", "HAS_NANS"]
const STATUS_INT   = Dict(s => Float64(i - 1) for (i, s) in enumerate(STATUS_ORDER))

# ── Idempotency ────────────────────────────────────────────────────────────────
if isfile(FIG_PATH) && isfile(CSV_PATH)
    println("[skip] outputs already exist at $FIG_PATH and $CSV_PATH")
    exit()
end

mkpath(FIG_DIR)
mkpath(RES_DIR)

# ── File-by-file integrity check ───────────────────────────────────────────────
n_sub = length(SUBJECTS)
n_col = length(COL_KEYS)

z_matrix      = zeros(Float64, n_sub, n_col)  # numeric status 0–4 for colorscale
status_matrix = fill("OK", n_sub, n_col)       # status labels for hover text

issue_df = DataFrame(
    subject   = String[],
    session   = String[],
    atlas     = String[],
    status    = String[],
    file_path = String[],
    notes     = String[],
)

file_idx = 0
ok_count = 0

for (row_i, subject) in enumerate(SUBJECTS)
    for (col_j, (atlas, session)) in enumerate(COL_KEYS)
        file_idx += 1

        jld_path   = joinpath(ACW_BASE, atlas, VERSION, "$(subject)_$(session).jld2")
        expected_n = EXPECTED_N_ROIS[atlas]
        status     = "OK"
        notes      = ""

        if !isfile(jld_path)
            status = "MISSING"
        else
            try
                data        = load(jld_path)
                acw_results = data["acw_results"]
                tau_vals    = collect(acw_results[TAU_IDX])
                n           = length(tau_vals)
                if n != expected_n
                    status = "WRONG_SHAPE"
                    notes  = "got $n ROIs, expected $expected_n"
                elseif any(x -> isnan(x) || isinf(x), tau_vals)
                    bad_n  = count(x -> isnan(x) || isinf(x), tau_vals)
                    status = "HAS_NANS"
                    notes  = "$bad_n NaN/Inf values"
                end
            catch e
                status = "UNLOADABLE"
                notes  = string(e)
            end
        end

        println("[$file_idx/$TOTAL_EXPECTED] $subject $session $atlas → $status")

        z_matrix[row_i, col_j]      = STATUS_INT[status]
        status_matrix[row_i, col_j] = status

        if status != "OK"
            push!(issue_df, (
                subject   = subject,
                session   = session,
                atlas     = atlas,
                status    = status,
                file_path = jld_path,
                notes     = notes,
            ))
        else
            ok_count += 1
        end
    end
end

# ── Output B: CSV ──────────────────────────────────────────────────────────────
CSV.write(CSV_PATH, issue_df)

n_issues = nrow(issue_df)
if n_issues == 0
    println("[done] No issues — CSV empty")
end

# ── Output A: PlotlyJS heatmap ─────────────────────────────────────────────────
# Discrete colorscale: 5 equal bands over [0, 4]
colorscale = [
    [0.00, STATUS_COLORS["OK"]],
    [0.20, STATUS_COLORS["OK"]],
    [0.20, STATUS_COLORS["MISSING"]],
    [0.40, STATUS_COLORS["MISSING"]],
    [0.40, STATUS_COLORS["UNLOADABLE"]],
    [0.60, STATUS_COLORS["UNLOADABLE"]],
    [0.60, STATUS_COLORS["WRONG_SHAPE"]],
    [0.80, STATUS_COLORS["WRONG_SHAPE"]],
    [0.80, STATUS_COLORS["HAS_NANS"]],
    [1.00, STATUS_COLORS["HAS_NANS"]],
]

# Pass y = SUBJECTS (sub-01 first) and reverse the y-axis so sub-01 appears at top.
hm = PlotlyJS.heatmap(
    z             = z_matrix,
    x             = COL_LABELS,
    y             = SUBJECTS,
    colorscale    = colorscale,
    zmin          = 0,
    zmax          = 4,
    showscale     = false,
    text          = status_matrix,
    hovertemplate = "Subject: %{y}<br>Condition: %{x}<br>Status: %{text}<extra></extra>",
)

# Legend: one invisible scatter point per status (square marker, labelled)
legend_traces = [
    PlotlyJS.scatter(
        x          = [NaN],
        y          = [NaN],
        mode       = "markers",
        name       = s,
        showlegend = true,
        marker     = attr(color=STATUS_COLORS[s], size=14, symbol="square"),
    )
    for s in STATUS_ORDER
]

layout = PlotlyJS.Layout(
    title         = attr(text="ACW file integrity check — denoisedNoGSR", font=attr(size=16)),
    xaxis         = attr(side="top", tickangle=0),
    yaxis         = attr(autorange="reversed"),
    font          = attr(family="Times New Roman", size=13),
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    legend        = attr(
        title       = attr(text="Status"),
        orientation = "v",
        x           = 1.02,
        xanchor     = "left",
        y           = 1.0,
        yanchor     = "top",
    ),
    margin = attr(l=100, r=180, t=100, b=40),
    width  = 800,
    height = 900,
)

fig = PlotlyJS.plot(vcat([hm], legend_traces), layout)
PlotlyJS.savefig(fig, FIG_PATH)

println("[done] $ok_count/$TOTAL_EXPECTED files OK, $n_issues files with issues")
println("[done] Figure → $FIG_PATH")
println("[done] CSV    → $CSV_PATH")
