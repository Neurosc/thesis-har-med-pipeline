# 04_recompute_stuck_scrollable.jl — Step 5b-v continued
# Recomputes ACF and MSE surface for 10 cases stuck at τ = 5.193702147200268,
# produces a scrollable 10-panel HTML figure with vivid colors.
#
# Run from repo root:
#   julia 04_statistics/_temp_spike_b_diagnostic/04_recompute_stuck_scrollable.jl

using CSV, DataFrames, IntrinsicTimescales, PlotlyJS, Statistics, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const WIDE_CSV  = joinpath(REPO_ROOT, "04_statistics", "_temp_spike_b_diagnostic",
                            "stuck_5194_timeseries_wide.csv")
const OUT_DIR   = joinpath(REPO_ROOT, "04_statistics", "_temp_spike_b_diagnostic")
const FIG_PATH  = joinpath(OUT_DIR, "fig_stuck_recompute_scrollable.html")

# ── ACW constants (must match 03_acw_analysis/scripts/01_compute_acw.jl) ──────
const FS      = 1.0 / 1.8
const N_LAGS  = 100
const N_CASES = 10

# ── MSE surface τ grid ────────────────────────────────────────────────────────
const TAU_GRID = exp.(range(log(0.1), log(30.0); length=200))

# ── Colors (vivid/saturated) ──────────────────────────────────────────────────
const COL_ACF_DOTS = "#E24B4A"   # vivid red        — actual ACF
const COL_REC_FIT  = "#185FA5"   # bright blue      — exp(−lag/τ_recorded)
const COL_GRID_FIT = "#27500A"   # bright green     — exp(−lag/τ_grid)
const COL_MSE_LINE = "#2C2C2A"   # dark gray        — MSE curve

# ── Reference stuck τ and tolerance ──────────────────────────────────────────
const STUCK_TAU = 5.193702147200268
const TOL       = 1e-6

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(FIG_PATH)
    println("[skip] output already exists; delete to rerun")
    exit()
end

isfile(WIDE_CSV) || error("Missing required input: $WIDE_CSV")
mkpath(OUT_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Load CSV, select first 10 cases
# ══════════════════════════════════════════════════════════════════════════════
println("═══ Part 1: Load CSV and select $N_CASES cases ═══\n")

wide_df  = CSV.read(WIDE_CSV, DataFrame)
n_rows   = nrow(wide_df)
n_cols   = ncol(wide_df)
@printf("  CSV dimensions: %d rows × %d columns\n", n_rows, n_cols)
n_rows == 234 || @warn "Expected 234 rows (post-dummy-discard); got $n_rows"

case_ids = names(wide_df)[1:N_CASES]
println("  Selected cases (first $N_CASES columns):")
for (i, id) in enumerate(case_ids)
    println("    [$i] $id")
end

# ══════════════════════════════════════════════════════════════════════════════
# Parts 2–3 — Recompute ACF, τ, and MSE surface per case
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Parts 2–3: Recompute ACF + MSE surface ═══\n")

case_results = []

for (k, case_id) in enumerate(case_ids)
    ts_vec = Float64.(wide_df[!, case_id])

    # Reshape to (1 × T) to match 01_compute_acw.jl:
    #   Matrix(df[:,2:end])' produces (n_rois × T); here n_rois = 1
    ts_mat = reshape(ts_vec, 1, :)

    acw_obj = acw(ts_mat, FS;
                  dims          = 2,
                  acwtypes      = [:auc, :tau],
                  n_lags        = N_LAGS,
                  skip_zero_lag = false)

    # acw_results[2] = τ  (acwtypes = [:auc, :tau]; tau is index 2)
    # acw_obj.acf    = ACF array, shape (1 × n_lags) for (1 × T) input
    # acw_obj.lags   = lag vector in seconds (StepRangeLen)
    tau_fresh = Float64(acw_obj.acw_results[2])
    acf_vec   = vec(acw_obj.acf)
    lags_vec  = collect(acw_obj.lags)

    ok = abs(tau_fresh - STUCK_TAU) < TOL
    @printf("  [%2d/10] %-48s  τ=%12.8f  %s\n",
            k, case_id, tau_fresh, ok ? "OK" : "*** MISMATCH")
    ok || @warn "τ mismatch for $case_id: got $tau_fresh, expected $STUCK_TAU"

    # MSE surface: mean((exp(−lag/τ) − acf)²) for each candidate τ
    mse_vals    = [Statistics.mean((exp.(-(lags_vec ./ τ)) .- acf_vec).^2)
                   for τ in TAU_GRID]
    best_idx    = argmin(mse_vals)
    tau_grid    = TAU_GRID[best_idx]
    mse_at_rec  = Statistics.mean((exp.(-(lags_vec ./ STUCK_TAU)) .- acf_vec).^2)
    mse_at_grid = mse_vals[best_idx]

    push!(case_results, (
        case_id     = case_id,
        tau_fresh   = tau_fresh,
        acf         = acf_vec,
        lags        = lags_vec,
        mse_vals    = mse_vals,
        tau_grid    = tau_grid,
        mse_at_rec  = mse_at_rec,
        mse_at_grid = mse_at_grid,
        ratio_mse   = mse_at_rec / max(mse_at_grid, 1e-15),
    ))
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Scrollable HTML figure (10 rows × 2 cols)
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 4: Building figure ═══")

NROWS = N_CASES   # 10
NCOLS = 2         # ACF left, MSE right

# subplot_titles: 20 entries row-major [r1_left, r1_right, r2_left, r2_right, ...]
# PlotlyJS requires a (NROWS × NCOLS) Matrix — built via permutedims(reshape(...))
titles_vec = String[]
for d in case_results
    push!(titles_vec,
          "$(d.case_id)  τ = $(round(d.tau_fresh; digits=6)) s")
    push!(titles_vec,
          "MSE surface — τ_grid = $(round(d.tau_grid; digits=4)) s")
end
titles_matrix = permutedims(reshape(titles_vec, NCOLS, NROWS))

fig = make_subplots(
    rows               = NROWS,
    cols               = NCOLS,
    subplot_titles     = titles_matrix,
    vertical_spacing   = 0.03,
    horizontal_spacing = 0.10,
)

for (k, d) in enumerate(case_results)
    lags     = d.lags
    acf      = d.acf
    tau_rec  = d.tau_fresh
    tau_grid = d.tau_grid
    fit_rec  = exp.(-(lags ./ tau_rec))
    fit_grid = exp.(-(lags ./ tau_grid))
    max_mse  = maximum(d.mse_vals)

    # ── ACF panel (col 1) ─────────────────────────────────────────────────────
    add_trace!(fig,
        scatter(x=lags, y=acf, mode="markers",
                marker=attr(color=COL_ACF_DOTS, size=5),
                showlegend=false, name="ACF"),
        row=k, col=1)
    add_trace!(fig,
        scatter(x=lags, y=fit_rec, mode="lines",
                line=attr(color=COL_REC_FIT, width=2),
                showlegend=false, name="exp(−lag/τ_rec)"),
        row=k, col=1)
    add_trace!(fig,
        scatter(x=lags, y=fit_grid, mode="lines",
                line=attr(color=COL_GRID_FIT, width=2, dash="dash"),
                showlegend=false, name="exp(−lag/τ_grid)"),
        row=k, col=1)

    # ── MSE panel (col 2) ─────────────────────────────────────────────────────
    add_trace!(fig,
        scatter(x=TAU_GRID, y=d.mse_vals, mode="lines",
                line=attr(color=COL_MSE_LINE, width=2),
                showlegend=false, name="MSE"),
        row=k, col=2)
    add_trace!(fig,
        scatter(x=[tau_rec, tau_rec], y=[0.0, max_mse], mode="lines",
                line=attr(color=COL_REC_FIT, width=2, dash="dash"),
                showlegend=false, name="τ_rec"),
        row=k, col=2)
    add_trace!(fig,
        scatter(x=[tau_grid, tau_grid], y=[0.0, max_mse], mode="lines",
                line=attr(color=COL_GRID_FIT, width=2),
                showlegend=false, name="τ_grid"),
        row=k, col=2)
end

# Axis styling: fixed ACF range; log-scale MSE x-axis
# Axis index for (row=k, col=c): (k-1)*NCOLS + c
# Index 1 → "xaxis"/"yaxis"; index n>1 → "xaxis{n}"/"yaxis{n}"
axis_updates = Dict{Symbol,Any}()
for k in 1:NROWS
    acf_idx = (k - 1) * NCOLS + 1
    mse_idx = (k - 1) * NCOLS + 2

    acf_xk = acf_idx == 1 ? :xaxis : Symbol("xaxis$(acf_idx)")
    acf_yk = acf_idx == 1 ? :yaxis : Symbol("yaxis$(acf_idx)")
    mse_xk = Symbol("xaxis$(mse_idx)")
    mse_yk = Symbol("yaxis$(mse_idx)")

    # ACF panel
    axis_updates[Symbol("$(acf_xk).title.text")] = "Lag (s)"
    axis_updates[Symbol("$(acf_xk).range")]      = [0, 180]
    axis_updates[Symbol("$(acf_yk).title.text")] = "ACF"
    axis_updates[Symbol("$(acf_yk).range")]      = [-0.3, 1.05]

    # MSE panel (log x)
    axis_updates[Symbol("$(mse_xk).type")]       = "log"
    axis_updates[Symbol("$(mse_xk).title.text")] = "τ (s)"
    axis_updates[Symbol("$(mse_xk).range")]      = [log10(0.1), log10(30.0)]
    axis_updates[Symbol("$(mse_yk).title.text")] = "MSE"
end
relayout!(fig, axis_updates)

relayout!(fig,
    height        = NROWS * 350,
    width         = 1300,
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    font          = attr(family="Times New Roman", size=10),
    title         = attr(
        text = ("Stuck-τ diagnostic (τ ≈ 5.194 s): ACF curves and MSE surfaces — 10 cases<br>" *
                "<span style='color:$(COL_ACF_DOTS)'>● ACF</span>  " *
                "<span style='color:$(COL_REC_FIT)'>— exp(−lag/τ_recorded)</span>  " *
                "<span style='color:$(COL_GRID_FIT)'>- - exp(−lag/τ_grid)</span>"),
        font = attr(size=12),
    ),
    margin = attr(t=80, b=40, l=70, r=40),
)

PlotlyJS.savefig(fig, FIG_PATH)
println("  Wrote: $FIG_PATH")

# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — Console summary
# ══════════════════════════════════════════════════════════════════════════════
println()
println("═══ Part 5: Summary ═══\n")
@printf("  %-48s  %10s  %10s  %9s\n", "case_id", "τ_recorded", "τ_grid", "ratio_mse")
println("  " * "─" ^ 84)
for d in case_results
    @printf("  %-48s  %10.6f  %10.6f  %9.4f\n",
            d.case_id, d.tau_fresh, d.tau_grid, d.ratio_mse)
end
println("\nDone.")
