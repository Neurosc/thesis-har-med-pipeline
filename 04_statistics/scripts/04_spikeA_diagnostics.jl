# 04_spikeA_diagnostics.jl — Step 5b-i: Spike A diagnostics
# Part 1: ROI frequency in Spike A (nonself, 0.95–1.05 s)
# Part 2: Motion log cross-reference for high- vs low-Spike-A subjects
#
# Run from repo root: julia 04_statistics/scripts/04_spikeA_diagnostics.jl

using CSV, DataFrames, PlotlyJS, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT    = normpath(joinpath(@__DIR__, "..", ".."))
const SPIKE_CSV    = joinpath(REPO_ROOT, "04_statistics", "results", "fig03_spike_rois.csv")
const BATCH_LOG    = joinpath(REPO_ROOT, "01_preprocessing", "02_denoising", "results", "_batch_log.tsv")
const RES_DIR      = joinpath(REPO_ROOT, "04_statistics", "results")
const FIG_DIR      = joinpath(REPO_ROOT, "04_statistics", "figures")
const ROI_FREQ_CSV = joinpath(RES_DIR, "fig04_spikeA_roi_frequency.csv")
const MOTION_CSV   = joinpath(RES_DIR, "fig04_spikeA_subject_motion.csv")
const FIG_PATH     = joinpath(FIG_DIR, "fig04_spikeA_roi_frequency.html")

# ── Constants ─────────────────────────────────────────────────────────────────
# Excluded subjects hardcoded from CLAUDE.md (criterion: >50% frames censored at FD>0.3mm)
const EXCLUDED = Set(["sub-06", "sub-08", "sub-12", "sub-26", "sub-36"])

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

const HIGH_SPIKE_A  = ["sub-18", "sub-23", "sub-24", "sub-40"]
const COLOR_NONSELF = "#8B7355"  # taupe

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(ROI_FREQ_CSV) && isfile(MOTION_CSV) && isfile(FIG_PATH)
    println("[skip] all outputs already exist; delete to rerun")
    exit()
end

# ── Check inputs ──────────────────────────────────────────────────────────────
for f in [SPIKE_CSV, BATCH_LOG]
    isfile(f) || error("Missing required input: $f")
end

mkpath(RES_DIR)
mkpath(FIG_DIR)

# ── Load inputs ───────────────────────────────────────────────────────────────
spike_df = CSV.read(SPIKE_CSV, DataFrame)
batch_df = CSV.read(BATCH_LOG, DataFrame; delim='\t')

# ═══════════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 64)
println("PART 1 — ROI frequency in Spike A (nonself, 0.95–1.05 s)")
println("=" ^ 64)

spikeA      = filter(r -> r.spike == "A" && r.atlas == "nonself", spike_df)
total_spikeA = nrow(spikeA)

roi_freq = combine(groupby(spikeA, :roi_id), nrow => :count)
sort!(roi_freq, :count, rev=true)
roi_freq[!, :percent_of_spike_A] = round.(100 .* roi_freq.count ./ total_spikeA; digits=1)
roi_freq[!, :cumulative_percent] = round.(cumsum(roi_freq.count) ./ total_spikeA .* 100; digits=1)

n_distinct = nrow(roi_freq)
top20      = first(roi_freq, min(20, n_distinct))

@printf("Total Spike A observations (nonself): %d\n", total_spikeA)
@printf("Distinct ROIs appearing in Spike A:   %d  (of 316 nonself ROIs)\n\n", n_distinct)

println("Top 20 ROIs by frequency:")
@printf("  %6s   %5s   %12s   %13s\n", "roi_id", "count", "% of Spike A", "cumulative %")
println("  " * "-" ^ 46)
for row in eachrow(top20)
    @printf("  %6d   %5d        %5.1f%%         %5.1f%%\n",
            row.roi_id, row.count, row.percent_of_spike_A, row.cumulative_percent)
end

println()
println("Cumulative coverage thresholds:")
for threshold in [50.0, 75.0, 90.0]
    idx = findfirst(x -> x >= threshold, roi_freq.cumulative_percent)
    n   = isnothing(idx) ? n_distinct : idx
    @printf("  Top %2d ROIs account for ≥%.0f%% of all Spike A observations\n", n, threshold)
end

# ── Output 2: ROI frequency CSV ───────────────────────────────────────────────
CSV.write(ROI_FREQ_CSV, roi_freq)
println("\nCSV → $ROI_FREQ_CSV  ($(nrow(roi_freq)) ROIs)")

# ── Output 4: figure ──────────────────────────────────────────────────────────
trace = PlotlyJS.bar(
    x            = string.(top20.roi_id),
    y            = top20.count,
    marker_color = COLOR_NONSELF,
)

layout = PlotlyJS.Layout(
    title         = attr(text="Top 20 nonself ROIs in Spike A (≈1.0 s)", font=attr(size=16)),
    xaxis         = attr(title="ROI ID", tickangle=45, type="category"),
    yaxis         = attr(title="Count (observations in 0.95–1.05 s window)"),
    font          = attr(family="Times New Roman", size=13),
    plot_bgcolor  = "white",
    paper_bgcolor = "white",
    showlegend    = false,
    width         = 900,
    height        = 500,
    margin        = attr(b=100),
)

PlotlyJS.savefig(PlotlyJS.plot([trace], layout), FIG_PATH)
println("HTML → $FIG_PATH")

# ═══════════════════════════════════════════════════════════════════════════════
println()
println("=" ^ 64)
println("PART 2 — Motion cross-reference")
println("=" ^ 64)

# Per-subject Spike A nonself count (pooled across sessions)
subj_counts = combine(groupby(spikeA, :subject), nrow => :spike_A_nonself_count)
count_dict  = Dict(zip(subj_counts.subject, subj_counts.spike_A_nonself_count))

all_counts = DataFrame(
    subject               = SUBJECTS,
    spike_A_nonself_count = [get(count_dict, s, 0) for s in SUBJECTS],
)

# 4 low-Spike-A controls: count ≤ 1, not in HIGH_SPIKE_A, in canonical order
low_pool     = filter(r -> r.spike_A_nonself_count ≤ 1 && !(r.subject in HIGH_SPIKE_A), all_counts)
low_subjects = first(low_pool, 4).subject
focus        = vcat(HIGH_SPIKE_A, low_subjects)

println()
println("High Spike A subjects: $(join(HIGH_SPIKE_A, ", "))")
println("Low Spike A controls:  $(join(low_subjects, ", "))")

# Build output table
out_subjects    = String[]
out_sessions    = String[]
out_spike_count = Int[]
out_n_censored  = Union{Int,Missing}[]
out_pct_cens    = Union{Float64,Missing}[]
out_std_nogsr   = Union{Float64,Missing}[]
out_excluded    = String[]

for subj in focus
    cnt     = get(count_dict, subj, 0)
    excl    = subj in EXCLUDED ? "Y" : "N"
    for session in ["ses-01", "ses-02"]
        log_row = filter(r -> r.subject == subj && r.session == session, batch_df)
        push!(out_subjects,    subj)
        push!(out_sessions,    session)
        push!(out_spike_count, cnt)
        push!(out_excluded,    excl)
        if isempty(log_row)
            push!(out_n_censored, missing)
            push!(out_pct_cens,   missing)
            push!(out_std_nogsr,  missing)
        else
            r = log_row[1, :]
            push!(out_n_censored, r.n_censored)
            push!(out_pct_cens,   r.pct_censored)
            push!(out_std_nogsr,  r.post_std_NoGSR)
        end
    end
end

motion_df = DataFrame(
    subject               = out_subjects,
    session               = out_sessions,
    spike_A_nonself_count = out_spike_count,
    n_censored            = out_n_censored,
    pct_censored          = out_pct_cens,
    post_std_NoGSR        = out_std_nogsr,
    excluded              = out_excluded,
)

# Console display
println()
@printf("  %-10s %-8s  %13s  %10s  %12s  %14s  %s\n",
        "subject", "session", "spike_A_count", "n_censored", "pct_censored",
        "post_std_NoGSR", "excluded")
println("  " * "-" ^ 80)
for (i, subj) in enumerate(focus)
    i > 1 && println()
    for session in ["ses-01", "ses-02"]
        rows = filter(r -> r.subject == subj && r.session == session, motion_df)
        r    = rows[1, :]
        @printf("  %-10s %-8s  %13d  %10s  %11s%%  %14s  %s\n",
                r.subject, r.session, r.spike_A_nonself_count,
                ismissing(r.n_censored)     ? "MISSING" : string(r.n_censored),
                ismissing(r.pct_censored)   ? "MISSING" : @sprintf("%5.2f", r.pct_censored),
                ismissing(r.post_std_NoGSR) ? "MISSING" : @sprintf("%.4f", r.post_std_NoGSR),
                r.excluded)
    end
end

# Exclusion check
println()
flagged = [s for s in HIGH_SPIKE_A if s in EXCLUDED]
if isempty(flagged)
    println("Exclusion check: none of the high-Spike-A subjects are in the exclusion list. ✓")
else
    println("WARNING: high-Spike-A subjects found in exclusion list: $(join(flagged, ", "))")
end

# ── Output 3: motion CSV ──────────────────────────────────────────────────────
CSV.write(MOTION_CSV, motion_df)
println("\nCSV → $MOTION_CSV  ($(nrow(motion_df)) rows)")

println("\nDone.")
