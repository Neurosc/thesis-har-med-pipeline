# 04_recompute_stuck_scrollable.jl — Step 5b-v continued
# Recomputes ACF and MSE surface for 10 cases stuck at τ = 5.193702147200268,
# produces a scrollable 10-panel HTML figure with vivid colors.
#
# Run from repo root:
#   julia 99_QC/troubleshooting/scripts/04_recompute_stuck_scrollable.jl

using CSV, DataFrames, IntrinsicTimescales, Statistics, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const WIDE_CSV  = joinpath(REPO_ROOT, "99_QC", "troubleshooting", "results",
                            "stuck_5194_timeseries_wide.csv")
const OUT_DIR   = joinpath(REPO_ROOT, "99_QC", "troubleshooting", "figures")
const FIG_PATH  = joinpath(OUT_DIR, "fig_stuck_recompute_scrollable.html")

# ── ACW constants (must match 03_intrinsic_neural_metrics/scripts/01_compute_acw.jl) ──────
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
# Part 4 — Scrollable HTML figure (two separate Plotly divs per case, flex row)
# Each case occupies one <div class="panel-row"> with two independent newPlot
# calls — ACF left, MSE right — guaranteeing structural per-case pairing.
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 4: Building figure ═══")

js(v) = "[" * join(v, ",") * "]"

open(FIG_PATH, "w") do io
    write(io, """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Stuck-τ diagnostic (τ ≈ 5.194 s)</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  body         { font-family: "Times New Roman", serif; background: white; margin: 20px; }
  .global-legend { font-size: 12px; margin-bottom: 14px; }
  .case-block  { margin-bottom: 40px; border-top: 1px solid #ccc; padding-top: 8px; }
  .case-header { font-size: 12px; font-weight: bold; margin-bottom: 6px; }
  .panel-row   { display: flex; flex-direction: row; gap: 10px; }
</style>
</head>
<body>
<h2 style="font-size:15px;">
  Stuck-τ diagnostic (τ ≈ 5.194 s): ACF curves and MSE surfaces — 10 cases
</h2>
<p class="global-legend">
  <span style="color:$(COL_ACF_DOTS);">● ACF</span> &nbsp;
  <span style="color:$(COL_REC_FIT);">&#8212; exp(−lag/τ<sub>recorded</sub>)</span> &nbsp;
  <span style="color:$(COL_GRID_FIT);">- - exp(−lag/τ<sub>grid</sub>)</span>
</p>
""")

    for (k, d) in enumerate(case_results)
        lags     = d.lags
        acf      = d.acf
        tau_rec  = d.tau_fresh
        tau_grid = d.tau_grid
        fit_rec  = exp.(-(lags ./ tau_rec))
        fit_grid = exp.(-(lags ./ tau_grid))
        max_mse  = maximum(d.mse_vals)

        acf_id = "acf_$k"
        mse_id = "mse_$k"

        acf_title = "ACF — τ_recorded = $(round(tau_rec; digits=6)) s, τ_grid = $(round(tau_grid; digits=4)) s"
        mse_title = "MSE surface"

        write(io, """<div class="case-block">
<div class="case-header">$(d.case_id)</div>
<div class="panel-row">
  <div id="$(acf_id)" style="width:630px;height:370px;"></div>
  <div id="$(mse_id)" style="width:630px;height:370px;"></div>
</div>
<script>
(function(){
  var lags         = $(js(lags));
  var acf          = $(js(acf));
  var fit_rec      = $(js(fit_rec));
  var fit_grid     = $(js(fit_grid));
  var tau_grid_arr = $(js(collect(TAU_GRID)));
  var mse_vals     = $(js(d.mse_vals));
  var tau_rec      = $(tau_rec);
  var tau_grid     = $(tau_grid);
  var max_mse      = $(max_mse);

  // ── ACF figure (left) ──────────────────────────────────────────────────────
  Plotly.newPlot('$(acf_id)', [
    { x: lags, y: acf,
      mode: 'markers', marker: { color: '$(COL_ACF_DOTS)', size: 5 },
      name: 'ACF', showlegend: false },
    { x: lags, y: fit_rec,
      mode: 'lines', line: { color: '$(COL_REC_FIT)', width: 2 },
      name: 'exp(−lag/τ_rec)', showlegend: false },
    { x: lags, y: fit_grid,
      mode: 'lines', line: { color: '$(COL_GRID_FIT)', width: 2, dash: 'dash' },
      name: 'exp(−lag/τ_grid)', showlegend: false },
  ], {
    title: { text: '$(acf_title)', font: { size: 11 } },
    width: 630, height: 370,
    paper_bgcolor: 'white', plot_bgcolor: 'white',
    font: { family: 'Times New Roman', size: 10 },
    margin: { t: 55, b: 50, l: 60, r: 20 },
    xaxis: { title: 'Lag (s)', range: [0, 180] },
    yaxis: { title: 'ACF',    range: [-0.3, 1.05] },
  }, { responsive: false });

  // ── MSE figure (right) ─────────────────────────────────────────────────────
  Plotly.newPlot('$(mse_id)', [
    { x: tau_grid_arr, y: mse_vals,
      mode: 'lines', line: { color: '$(COL_MSE_LINE)', width: 2 },
      name: 'MSE', showlegend: false },
    { x: [tau_rec, tau_rec], y: [0, max_mse],
      mode: 'lines', line: { color: '$(COL_REC_FIT)', width: 2, dash: 'dash' },
      name: 'τ_rec', showlegend: false },
    { x: [tau_grid, tau_grid], y: [0, max_mse],
      mode: 'lines', line: { color: '$(COL_GRID_FIT)', width: 2 },
      name: 'τ_grid', showlegend: false },
  ], {
    title: { text: '$(mse_title)', font: { size: 11 } },
    width: 630, height: 370,
    paper_bgcolor: 'white', plot_bgcolor: 'white',
    font: { family: 'Times New Roman', size: 10 },
    margin: { t: 55, b: 50, l: 60, r: 20 },
    xaxis: { title: 'τ (s)', type: 'log',
             range: [Math.log10(0.1), Math.log10(30)] },
    yaxis: { title: 'MSE' },
  }, { responsive: false });
})();
</script>
</div>
""")
    end

    write(io, "</body>\n</html>\n")
end
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
