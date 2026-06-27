# 01_spike_b_acf_mse_diagnostic.jl — Step 5b-ii: Spike B ACF and MSE surface diagnostic
#
# Determines whether τ ≈ 5.2 s spike reflects genuine slow dynamics or a
# local-minimum artifact of the scalar-MSE LevenbergMarquardt fit.
#
# Output directory: 99_QC/troubleshooting/
# Run from repo root:
#   julia 99_QC/troubleshooting/scripts/01_spike_b_acf_mse_diagnostic.jl

using CSV, DataFrames, IntrinsicTimescales, PlotlyJS, Random, Statistics, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT  = normpath(joinpath(@__DIR__, "..", "..", ".."))
const SPIKE_CSV  = joinpath(REPO_ROOT, "04_statistics", "results", "fig03_spike_rois.csv")
const LONG_CSV   = joinpath(REPO_ROOT, "04_statistics", "results", "analysis_long_format.csv")
const ATLAS_FILE = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                             "glasser_coordinates_nonself_clean_1mm.txt")
const SELF_LAB   = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases", "self_labels.txt")
const TS_BASE    = joinpath(REPO_ROOT, "02_timeseries_extraction", "results")
const OUT_DIR    = joinpath(REPO_ROOT, "99_QC", "troubleshooting", "results")
const SEL_CSV    = joinpath(OUT_DIR, "selected_rois.csv")
const FIT_CSV    = joinpath(OUT_DIR, "fit_summary.csv")
const FIG_PATH   = joinpath(REPO_ROOT, "99_QC", "troubleshooting", "figures", "fig06_spike_b_acf_mse_grid.html")

# ── ACW / data constants (must match 03_intrinsic_neural_metrics/scripts/01_compute_acw.jl) ───
const FS            = 1.0 / 1.8
const N_LAGS        = 100
const DUMMY_VOLUMES = 6
const RANDOM_SEED   = 42

# Log-spaced τ grid for MSE surface (200 points, 0.1–30 s)
const TAU_GRID = exp.(range(log(0.1), log(30.0); length=200))

# ── Colors (Palette A) ────────────────────────────────────────────────────────
const COL_HIGH   = "#C0392B"   # crimson   — spike_B_high
const COL_LOW    = "#D98859"   # amber-tan — spike_B_low
const COL_NORMAL = "#3D5A6C"   # slate     — normal
const COL_SLATE  = "#3D5A6C"   # recorded-τ overlays
const COL_TAUPE  = "#8B7355"   # grid-search-τ overlays

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(SEL_CSV) && isfile(FIT_CSV) && isfile(FIG_PATH)
    println("[skip] all outputs already exist; delete to rerun")
    exit()
end

mkpath(OUT_DIR)

# ── Verify required inputs ────────────────────────────────────────────────────
for f in [SPIKE_CSV, LONG_CSV, ATLAS_FILE, SELF_LAB]
    isfile(f) || error("Missing required input: $f")
end

# ── ROI name lookup tables ────────────────────────────────────────────────────
_atlas_df     = CSV.read(ATLAS_FILE, DataFrame; delim='\t')
nonself_names = Dict(pos => row.ROI_Name
                     for (pos, row) in enumerate(eachrow(_atlas_df)))

self_names = Dict{Int,String}()
for line in eachline(SELF_LAB)
    parts = split(strip(line))
    (isempty(parts) || length(parts) < 2) && continue
    self_names[parse(Int, parts[1])] = join(parts[2:end], " ")
end

get_roi_name(atlas, pos) = atlas == "self" ?
    get(self_names, pos, "ROI_$pos") : get(nonself_names, pos, "ROI_$pos")

cat_color(cat) = cat == "spike_B_high" ? COL_HIGH :
                 cat == "spike_B_low"  ? COL_LOW  : COL_NORMAL

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Select 12 (subject, session, atlas, roi_pos_id) cases
# ══════════════════════════════════════════════════════════════════════════════
println("═══ Part 1: Case selection ═══\n")

spike_df = CSV.read(SPIKE_CSV, DataFrame)
long_df  = CSV.read(LONG_CSV,  DataFrame)
Random.seed!(RANDOM_SEED)

# Pick up to n rows with distinct subjects, sorted by tau.
# rev=true → highest τ first (for "high" cases); rev=false → lowest first.
function pick_n(df, n; rev=false)
    sorted  = sort(df, :tau; rev=rev)
    seen    = Set{String}()
    indices = Int[]
    for i in 1:nrow(sorted)
        sub = sorted[i, :subject]
        if !(sub in seen)
            push!(indices, i)
            push!(seen, sub)
        end
        length(indices) == n && break
    end
    length(indices) < n &&
        @warn "Only $(length(indices)) distinct-subject rows available; wanted $n"
    return sorted[indices, :]
end

# ── Spike B cases ─────────────────────────────────────────────────────────────
ns_b = filter(r -> r.spike == "B" && r.atlas == "nonself", spike_df)
sf_b = filter(r -> r.spike == "B" && r.atlas == "self",    spike_df)

high_ns = pick_n(ns_b, 2; rev=true)
low_ns  = pick_n(ns_b, 2; rev=false)
high_sf = pick_n(sf_b, 2; rev=true)
low_sf  = pick_n(sf_b, 2; rev=false)

# Flag if all self Spike B τ values are identical (expected from prior data inspection)
ust = unique(sf_b.tau)
if length(ust) == 1
    println("NOTE: All $(nrow(sf_b)) self Spike B cases share τ = $(round(ust[1]; digits=6)).")
    println("      high/low split is arbitrary — both pairs have the same recorded τ.\n")
end

# Verify ≥4 distinct subjects across all 8 Spike B selections
b_subjects = unique(vcat(high_ns.subject, low_ns.subject,
                         high_sf.subject, low_sf.subject))
length(b_subjects) >= 4 ||
    error("Only $(length(b_subjects)) distinct subjects in Spike B selection; " *
          "need ≥4 — STOP")
println("Distinct subjects in Spike B selection: $(length(b_subjects)) [≥4: OK]")

# ── Normal cases: τ ∈ [2.5, 3.0], prefer Spike B subjects ────────────────────
b_sub_set   = Set(b_subjects)
normal_pool = select(filter(r -> 2.5 ≤ r.tau ≤ 3.0, long_df),
                     :subject, :session, :atlas,
                     :roi_pos_id => :roi_id,   # unified column name
                     :tau)

function pick_normal(pool, atlas_name, n)
    sub  = filter(r -> r.atlas == atlas_name, pool)
    isempty(sub) && error("No τ ∈ [2.5,3.0] candidates for atlas=$atlas_name — STOP")
    # Preferred (Spike B) subjects first, then others — fall back seamlessly
    pref    = filter(r -> r.subject in b_sub_set, sub)
    nonpref = filter(r -> !(r.subject in b_sub_set), sub)
    pick_n(vcat(pref, nonpref), n; rev=false)
end

normal_ns = pick_normal(normal_pool, "nonself", 2)
normal_sf = pick_normal(normal_pool, "self",    2)

# ── Build ordered case list (12 entries) ──────────────────────────────────────
# Grid layout (4 logical rows × 3 cols):
#   Row 1: high_ns[1], high_ns[2], high_sf[1]
#   Row 2: high_sf[2], low_ns[1],  low_ns[2]
#   Row 3: low_sf[1],  low_sf[2],  normal_ns[1]
#   Row 4: normal_ns[2], normal_sf[1], normal_sf[2]

function make_cases(df, category)
    [(subject      = df[i, :subject],
      session      = df[i, :session],
      atlas        = df[i, :atlas],
      roi_pos_id   = df[i, :roi_id],
      roi_name     = get_roi_name(df[i, :atlas], df[i, :roi_id]),
      recorded_tau = df[i, :tau],
      category     = category) for i in 1:nrow(df)]
end

cases = [
    make_cases(high_ns,   "spike_B_high")...;
    make_cases(high_sf,   "spike_B_high")...;
    make_cases(low_ns,    "spike_B_low")...;
    make_cases(low_sf,    "spike_B_low")...;
    make_cases(normal_ns, "normal")...;
    make_cases(normal_sf, "normal")...;
]
@assert length(cases) == 12 "Expected 12 cases; got $(length(cases)) — STOP"

# Write selected_rois.csv
CSV.write(SEL_CSV, DataFrame(
    subject      = [c.subject      for c in cases],
    session      = [c.session      for c in cases],
    atlas        = [c.atlas        for c in cases],
    roi_pos_id   = [c.roi_pos_id   for c in cases],
    roi_name     = [c.roi_name     for c in cases],
    recorded_tau = [c.recorded_tau for c in cases],
    category     = [c.category     for c in cases],
))

println("\nSelected 12 cases:")
@printf("  %-10s %-8s %-8s %6s  %-26s  %10s  %-12s\n",
        "subject", "session", "atlas", "roi_id", "roi_name",
        "τ_recorded", "category")
println("  " * "─" ^ 90)
for c in cases
    @printf("  %-10s %-8s %-8s %6d  %-26s  %10.4f  %-12s\n",
            c.subject, c.session, c.atlas, c.roi_pos_id,
            c.roi_name, c.recorded_tau, c.category)
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Reload ACF and τ for each selected ROI
# Part 3 — Compute MSE surface
# (combined into a single pass for efficiency)
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Parts 2–3: ACF reload + MSE surface ═══\n")

# ACWResults struct fields (from IntrinsicTimescales source):
#   .acw_results  — for acwtypes=[:auc,:tau]: Vector{Any}[auc_val, tau_val]
#   .acf          — ACF array; shape (1 × n_lags) for (1 × T) matrix input
#   .lags         — lag vector in seconds: 0, 1.8, 3.6, ..., (n_lags-1)×1.8

case_data    = Any[]
any_mismatch = false

for (k, c) in enumerate(cases)
    ts_path = joinpath(TS_BASE, "timeseries_$(c.atlas)", "denoisedNoGSR",
                       "$(c.subject)_$(c.session)_$(c.atlas)_timeseries.csv")
    isfile(ts_path) || error("[$k/12] Missing timeseries file: $ts_path — STOP")

    ts_df = CSV.read(ts_path, DataFrame)

    # Column at position roi_pos_id:
    #   names(ts_df)[1]           = "Timepoint"
    #   names(ts_df)[pos + 1]     = column for nonself position pos
    # For self:   column name = string(pos), e.g. "5"  → position 5
    # For nonself: column name = Glasser ID string, e.g. "18" → position 17
    # Either way, the index pos+1 into names() gives the correct column.
    col    = names(ts_df)[c.roi_pos_id + 1]
    ts_vec = Float64.(ts_df[!, col])

    # Discard first DUMMY_VOLUMES volumes; reshape to (1 × T) to exactly
    # replicate 01_compute_acw.jl which uses Matrix(df[:,2:end])' → (n_rois × T)
    ts_mat = reshape(ts_vec[(DUMMY_VOLUMES + 1):end], 1, :)

    acw_obj = acw(ts_mat, FS;
                  dims          = 2,
                  acwtypes      = [:auc, :tau],
                  n_lags        = N_LAGS,
                  skip_zero_lag = false)

    # Extract τ — acw_results[2] because acwtypes = [:auc, :tau] and tau is index 2
    tau_fresh = Float64(acw_obj.acw_results[2])
    # Extract ACF: shape (1 × n_lags) for matrix input; vec() flattens to length n_lags
    acf_vec   = vec(acw_obj.acf)
    # Extract lags: StepRangeLen in seconds; collect() to plain Vector{Float64}
    lags_vec  = collect(acw_obj.lags)

    # Sanity check: fresh τ must match stored τ within 1e-6
    delta = abs(tau_fresh - c.recorded_tau)
    ok    = delta ≤ 1e-6
    @printf("  [%2d/12] %s %s %s ROI%d  τ_stored=%9.6f  τ_fresh=%9.6f  Δ=%.2e  %s\n",
            k, c.subject, c.session, c.atlas, c.roi_pos_id,
            c.recorded_tau, tau_fresh, delta, ok ? "OK" : "*** MISMATCH ***")
    ok || (global any_mismatch = true)

    # MSE surface: 200 τ values log-spaced from 0.1 to 30 s
    # Model: ACF(lag) = exp(-lag / τ)  [same as IntrinsicTimescales fit_expdecay]
    mse_vals     = [mean((exp.(-(lags_vec ./ τ)) .- acf_vec).^2) for τ in TAU_GRID]
    best_idx     = argmin(mse_vals)
    tau_grid_opt = TAU_GRID[best_idx]
    mse_at_grid  = mse_vals[best_idx]
    mse_at_rec   = mean((exp.(-(lags_vec ./ c.recorded_tau)) .- acf_vec).^2)

    push!(case_data, (
        c            = c,
        tau_fresh    = tau_fresh,
        acf          = acf_vec,
        lags         = lags_vec,
        mse_vals     = mse_vals,
        tau_grid_opt = tau_grid_opt,
        mse_at_grid  = mse_at_grid,
        mse_at_rec   = mse_at_rec,
    ))
end

if any_mismatch
    error("One or more recomputed τ values differ from stored values by >1e-6. " *
          "See MISMATCH lines above — STOP before writing outputs.")
end
println("\n✓ All 12 recomputed τ values match stored values within 1e-6")

# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Fit summary CSV
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 4: Fit summary ═══\n")

fit_df = DataFrame(
    subject             = [d.c.subject      for d in case_data],
    session             = [d.c.session      for d in case_data],
    atlas               = [d.c.atlas        for d in case_data],
    roi_pos_id          = [d.c.roi_pos_id   for d in case_data],
    category            = [d.c.category     for d in case_data],
    recorded_tau        = [d.c.recorded_tau for d in case_data],
    recomputed_tau      = [d.tau_fresh      for d in case_data],
    grid_search_tau     = [d.tau_grid_opt   for d in case_data],
    mse_at_recorded_tau = [d.mse_at_rec     for d in case_data],
    mse_at_grid_optimum = [d.mse_at_grid    for d in case_data],
    ratio_mse           = [d.mse_at_rec / max(d.mse_at_grid, 1e-15)
                           for d in case_data],
)

CSV.write(FIT_CSV, fit_df)

println("Fit summary (fit_summary.csv):")
@printf("  %-10s %-8s %-8s %5s  %-12s  %7s  %7s  %7s  %9s  %9s  %6s\n",
        "subject", "session", "atlas", "roi", "category",
        "τ_rec", "τ_fresh", "τ_grid", "MSE_rec", "MSE_grid", "ratio")
println("  " * "─" ^ 102)
for row in eachrow(fit_df)
    @printf("  %-10s %-8s %-8s %5d  %-12s  %7.4f  %7.4f  %7.4f  %9.3e  %9.3e  %6.2f\n",
            row.subject, row.session, row.atlas, row.roi_pos_id, row.category,
            row.recorded_tau, row.recomputed_tau, row.grid_search_tau,
            row.mse_at_recorded_tau, row.mse_at_grid_optimum, row.ratio_mse)
end

# One-line conclusion
b_rows  = filter(r -> startswith(r.category, "spike_B"), eachrow(fit_df))
pct_thr = 10.0
n_b     = length(b_rows)
n_div   = count(r -> 100.0 * abs(r.grid_search_tau - r.recorded_tau) /
                     r.recorded_tau > pct_thr, b_rows)
println("\nConclusion: Of $n_b Spike B cases, $n_div had grid_search_tau differing from " *
        "recorded_tau by more than $(pct_thr)%; " *
        "these are candidates for local-minimum artifacts.")

# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — Figure: 4×3 logical grid, each cell = ACF (top) + MSE (bottom)
# ══════════════════════════════════════════════════════════════════════════════
# Actual subplot grid: 8 rows × 3 cols = 24 subplots.
# Logical row lr (1–4) → actual ACF row = (lr-1)*2+1, MSE row = (lr-1)*2+2.
# Axis index for (actual_row r, col c) = (r-1)*3 + c.
# ──────────────────────────────────────────────────────────────────────────────
println("\n═══ Part 5: Building figure ═══")

NCOLS     = 3
NLOG_ROWS = 4
NACT_ROWS = NLOG_ROWS * 2   # 8

# subplot_titles (24 entries, row-major).
# ACF rows (1,3,5,7) get the case title; MSE rows (2,4,6,8) get "".
subplot_titles = String[]
for lr in 1:NLOG_ROWS
    for lc in 1:NCOLS
        k = (lr - 1) * NCOLS + lc
        if k ≤ 12
            d = case_data[k].c
            push!(subplot_titles,
                  "$(d.subject) $(d.session) $(d.atlas) ROI $(d.roi_pos_id) · " *
                  "τ = $(round(d.recorded_tau; digits=3)) s")
        else
            push!(subplot_titles, "")
        end
    end
    append!(subplot_titles, fill("", NCOLS))   # blank titles for MSE row
end

fig = make_subplots(
    rows               = NACT_ROWS,
    cols               = NCOLS,
    subplot_titles     = permutedims(reshape(subplot_titles, NCOLS, NACT_ROWS)),
    vertical_spacing   = 0.04,
    horizontal_spacing = 0.08,
    row_heights        = repeat([2.5, 1.0], NLOG_ROWS),
)

for (k, d) in enumerate(case_data)
    lr      = (k - 1) ÷ NCOLS + 1
    lc      = (k - 1) % NCOLS + 1
    acf_row = (lr - 1) * 2 + 1
    mse_row = (lr - 1) * 2 + 2
    col_cat = cat_color(d.c.category)

    lags      = d.lags
    acf       = d.acf
    tau_rec   = d.c.recorded_tau
    tau_grid  = d.tau_grid_opt
    fit_rec   = exp.(-(lags ./ tau_rec))
    fit_grid  = exp.(-(lags ./ tau_grid))
    max_mse   = maximum(d.mse_vals)

    # ── ACF panel ─────────────────────────────────────────────────────────────
    # Data dots (category color), recorded-τ fit (slate solid), grid-τ fit (taupe dashed)
    add_trace!(fig,
        scatter(x=lags, y=acf, mode="markers",
                marker=attr(color=col_cat, size=4, opacity=0.7),
                showlegend=false, name="ACF"),
        row=acf_row, col=lc)
    add_trace!(fig,
        scatter(x=lags, y=fit_rec, mode="lines",
                line=attr(color=COL_SLATE, width=1.5),
                showlegend=false, name="exp(−lag/τ_rec)"),
        row=acf_row, col=lc)
    add_trace!(fig,
        scatter(x=lags, y=fit_grid, mode="lines",
                line=attr(color=COL_TAUPE, width=1.5, dash="dash"),
                showlegend=false, name="exp(−lag/τ_grid)"),
        row=acf_row, col=lc)

    # ── MSE surface panel ─────────────────────────────────────────────────────
    # MSE curve (gray), recorded-τ vertical (slate solid), grid-τ vertical (taupe dashed)
    add_trace!(fig,
        scatter(x=TAU_GRID, y=d.mse_vals, mode="lines",
                line=attr(color="#888888", width=1.5),
                showlegend=false, name="MSE"),
        row=mse_row, col=lc)
    add_trace!(fig,
        scatter(x=[tau_rec, tau_rec], y=[0.0, max_mse], mode="lines",
                line=attr(color=COL_SLATE, width=1.5),
                showlegend=false, name="τ_rec"),
        row=mse_row, col=lc)
    add_trace!(fig,
        scatter(x=[tau_grid, tau_grid], y=[0.0, max_mse], mode="lines",
                line=attr(color=COL_TAUPE, width=1.5, dash="dash"),
                showlegend=false, name="τ_grid"),
        row=mse_row, col=lc)
end

# Set log-scale x-axis on every MSE panel; add axis labels
axis_updates = Dict{Symbol,Any}()
for k in 1:12
    lr      = (k - 1) ÷ NCOLS + 1
    lc      = (k - 1) % NCOLS + 1
    mse_row = (lr - 1) * 2 + 2
    ax_idx  = (mse_row - 1) * NCOLS + lc
    ax_key  = ax_idx == 1 ? "xaxis" : "xaxis$(ax_idx)"
    axis_updates[Symbol("$(ax_key).type")]       = "log"
    axis_updates[Symbol("$(ax_key).title.text")] = "τ (s)"
end
# Y-axis labels on left-column panels
for lr in 1:NLOG_ROWS
    acf_ax_idx = ((lr - 1) * 2)       * NCOLS + 1
    mse_ax_idx = ((lr - 1) * 2 + 1)   * NCOLS + 1
    acf_y_key  = acf_ax_idx == 1 ? "yaxis" : "yaxis$(acf_ax_idx)"
    mse_y_key  = mse_ax_idx == 1 ? "yaxis" : "yaxis$(mse_ax_idx)"
    axis_updates[Symbol("$(acf_y_key).title.text")] = "ACF"
    axis_updates[Symbol("$(mse_y_key).title.text")] = "MSE"
end
relayout!(fig, axis_updates)

relayout!(fig,
    height        = 2400,
    width         = 1300,
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    font          = attr(family="Times New Roman", size=10),
    title         = attr(
        text = "Spike B diagnostic: ACF and MSE surface (τ ≈ 5.2 s)<br>" *
               "Dots = actual ACF | Slate solid = exp(−lag/τ_recorded) | " *
               "Taupe dashed = exp(−lag/τ_grid_search)",
        font = attr(size=12),
    ),
    margin = attr(t=90, b=40, l=70, r=40),
)

PlotlyJS.savefig(fig, FIG_PATH)
println("  Wrote: $FIG_PATH")

println("\n── Output summary ────────────────────────────────────────────────────────")
println("  $SEL_CSV")
println("  $FIT_CSV")
println("  $FIG_PATH")
println("\nDone.")
