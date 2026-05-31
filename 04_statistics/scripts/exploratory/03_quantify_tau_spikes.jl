# 03_quantify_tau_spikes.jl — Step 5a: Quantify τ spike regions (1.0 s and 5.2 s)
# Loads all 140 denoisedNoGSR JLD2 files, identifies ROI-level τ observations
# that fall in two narrow windows, and reports counts + a per-subject figure.
#
# Run from repo root: julia 04_statistics/scripts/03_quantify_tau_spikes.jl

using JLD2, DataFrames, CSV, PlotlyJS, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const ACW_BASE  = joinpath(REPO_ROOT, "03_acw_analysis", "results", "acw")
const VERSION   = "denoisedNoGSR"
const RES_DIR   = joinpath(REPO_ROOT, "04_statistics", "results")
const FIG_DIR   = joinpath(REPO_ROOT, "04_statistics", "figures")
const CSV_PATH  = joinpath(RES_DIR, "fig03_spike_rois.csv")
const FIG_PATH  = joinpath(FIG_DIR, "fig03_spike_subject_counts.html")

# ── Spike windows (tight — do not widen) ──────────────────────────────────────
const SPIKE_A_LO = 0.95; const SPIKE_A_HI = 1.05
const SPIKE_B_LO = 5.00; const SPIKE_B_HI = 5.30

# ── Included subjects (35; excluded: sub-06, sub-08, sub-12, sub-26, sub-36) ──
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
const TAU_IDX     = 2  # ACW_TYPES = [:auc, :tau]; tau is index 2
const TOTAL_FILES = length(SUBJECTS) * length(SESSIONS) * length(ATLASES)  # 140

# ── Colors (Palette A) ────────────────────────────────────────────────────────
const COLOR_SELF    = "#3D5A6C"  # slate
const COLOR_NONSELF = "#8B7355"  # taupe

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(CSV_PATH) && isfile(FIG_PATH)
    println("[skip] all outputs already exist; delete to rerun")
    exit()
end

mkpath(RES_DIR)
mkpath(FIG_DIR)

# ── Load τ values → long-format DataFrame ─────────────────────────────────────
rows = NamedTuple{(:subject, :session, :atlas, :roi_id, :tau),
                  Tuple{String,String,String,Int,Float64}}[]

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

    for (roi_id, tau) in enumerate(tau_vals)
        push!(rows, (subject=subject, session=session, atlas=atlas,
                     roi_id=roi_id, tau=Float64(tau)))
    end
end

df = DataFrame(rows)
println("\nLoaded $(nrow(df)) total ROI observations")

# ── Output 1: Console summary per atlas ───────────────────────────────────────
println()
for atlas in ATLASES
    dfa   = filter(r -> r.atlas == atlas, df)
    total = nrow(dfa)
    n_a   = count(r -> SPIKE_A_LO ≤ r.tau ≤ SPIKE_A_HI, eachrow(dfa))
    n_b   = count(r -> SPIKE_B_LO ≤ r.tau ≤ SPIKE_B_HI, eachrow(dfa))
    @printf("Atlas: %s\n", atlas)
    @printf("  Total observations: %d\n", total)
    @printf("  Spike A (0.95–1.05 s):  N = %d (%.1f%%)\n", n_a, 100n_a / total)
    @printf("  Spike B (5.00–5.30 s):  N = %d (%.1f%%)\n\n", n_b, 100n_b / total)
end

# ── Output 2: CSV of spike observations ───────────────────────────────────────
in_spike_a(t) = SPIKE_A_LO ≤ t ≤ SPIKE_A_HI
in_spike_b(t) = SPIKE_B_LO ≤ t ≤ SPIKE_B_HI

spike_df = filter(r -> in_spike_a(r.tau) || in_spike_b(r.tau), df)
spike_df[!, :spike] = ifelse.(in_spike_a.(spike_df.tau), "A", "B")

CSV.write(CSV_PATH, spike_df[!, [:subject, :session, :atlas, :roi_id, :tau, :spike]])
println("CSV  → $CSV_PATH  ($(nrow(spike_df)) rows)")

# ── Output 3: Per-subject spike count figure ───────────────────────────────────
# Count spike observations per (subject, atlas, spike), pooled across sessions.
cnt = combine(
    groupby(spike_df, [:subject, :atlas, :spike]),
    nrow => :n
)

function get_n(subj, atlas, spike)
    rows_match = filter(r -> r.subject == subj && r.atlas == atlas && r.spike == spike, cnt)
    isempty(rows_match) ? 0 : rows_match[1, :n]
end

self_A    = [get_n(s, "self",    "A") for s in SUBJECTS]
nonself_A = [get_n(s, "nonself", "A") for s in SUBJECTS]
self_B    = [get_n(s, "self",    "B") for s in SUBJECTS]
nonself_B = [get_n(s, "nonself", "B") for s in SUBJECTS]

# Two side-by-side panels via manual axis domains (xaxis/xaxis2, yaxis/yaxis2).
# Self bars form the bottom stack, nonself bars are stacked on top.
trace_self_A = PlotlyJS.bar(
    x=SUBJECTS, y=self_A,
    name="Self", marker_color=COLOR_SELF,
    xaxis="x", yaxis="y",
    showlegend=true,
)
trace_nonself_A = PlotlyJS.bar(
    x=SUBJECTS, y=nonself_A,
    name="Nonself", marker_color=COLOR_NONSELF,
    xaxis="x", yaxis="y",
    showlegend=true,
)
trace_self_B = PlotlyJS.bar(
    x=SUBJECTS, y=self_B,
    name="Self", marker_color=COLOR_SELF,
    xaxis="x2", yaxis="y2",
    showlegend=false,
)
trace_nonself_B = PlotlyJS.bar(
    x=SUBJECTS, y=nonself_B,
    name="Nonself", marker_color=COLOR_NONSELF,
    xaxis="x2", yaxis="y2",
    showlegend=false,
)

layout = PlotlyJS.Layout(
    title         = attr(
        text = "Spike τ values by subject — A: ≈1.0 s, B: ≈5.2 s",
        font = attr(size=16),
    ),
    barmode       = "stack",
    font          = attr(family="Times New Roman", size=13),
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    xaxis  = attr(
        domain    = [0.0, 0.46],
        tickangle = 45,
        tickfont  = attr(size=9),
    ),
    xaxis2 = attr(
        domain    = [0.54, 1.0],
        tickangle = 45,
        tickfont  = attr(size=9),
        anchor    = "y2",
    ),
    yaxis  = attr(title="Count"),
    yaxis2 = attr(title="Count", anchor="x2"),
    annotations = [
        attr(
            x=0.23, y=1.07, xref="paper", yref="paper", showarrow=false,
            text="Spike A (0.95–1.05 s)", font=attr(size=14, family="Times New Roman"),
        ),
        attr(
            x=0.77, y=1.07, xref="paper", yref="paper", showarrow=false,
            text="Spike B (5.00–5.30 s)", font=attr(size=14, family="Times New Roman"),
        ),
    ],
    legend  = attr(x=1.01, y=1.0),
    width   = 1400,
    height  = 600,
    margin  = attr(b=160),
)

fig = PlotlyJS.plot([trace_self_A, trace_nonself_A, trace_self_B, trace_nonself_B], layout)
PlotlyJS.savefig(fig, FIG_PATH)
println("HTML → $FIG_PATH")

println("\nDone.")
