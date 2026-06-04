# 02_boxplot_session_self_nonself.jl
# Violin plot: ses-01/ses-02 × Self/Nonself with paired Wilcoxon signed-rank tests
# Adapted from 03_01_boxplot.jl (task-med reference):
#   group axis (Control/Meditator) → session axis (ses-01/ses-02)
#   purple/green palette → Palette A (slate/taupe)
#   hardcoded paired DataFrames → load from JLD2 results
#
# Run from repo root:  julia 03_acw_analysis/scripts/02_boxplot_session_self_nonself.jl

using PlotlyJS, DataFrames, JLD2, Statistics, HypothesisTests

# ── Config ────────────────────────────────────────────────────────────────────
include(normpath(joinpath(@__DIR__, "..", "..", "utils", "config_loader.jl")))
let cfg = load_config()
    println("Config: atlas=$(cfg.atlas_method)  denoising=$(cfg.denoising_method)  tag=$(tag())")
end

# ── Subjects (35 included; excluded: sub-06, sub-08, sub-12, sub-26, sub-36) ──
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

const SESSIONS = ["ses-01", "ses-02"]
const VERSIONS = ["raw", "denoisedNoGSR"]
const ATLASES  = ["self", "nonself"]
const TAU_IDX  = 2   # ACW_TYPES = [:auc, :tau]; tau is index 2

# ── Paths (REPO_ROOT provided by config_loader.jl) ───────────────────────────
const ACW_BASE  = joinpath(REPO_ROOT, "03_acw_analysis", "results", tag(), "acw")
const FIG_DIR   = joinpath(REPO_ROOT, "03_acw_analysis", "results", tag(), "figures")
mkpath(FIG_DIR)

# ── Colors (Palette A: slate = self, taupe = nonself) ─────────────────────────
const COLOR_SELF_PRE     = "#3D5A6C"   # darker slate,  ses-01
const COLOR_SELF_POST    = "#7A95A6"   # lighter slate,  ses-02
const COLOR_NONSELF_PRE  = "#8B7355"   # darker taupe,  ses-01
const COLOR_NONSELF_POST = "#BFA888"   # lighter taupe, ses-02

sig_label(p) = p < 0.001 ? "***" : p < 0.01 ? "**" : p < 0.05 ? "*" : "ns"

# ── Load all ACW results → long-form DataFrame ────────────────────────────────
rows = NamedTuple{(:subject, :session, :atlas, :version, :tau_median),
                  Tuple{String,String,String,String,Float64}}[]

for version in VERSIONS, atlas in ATLASES, subject in SUBJECTS, session in SESSIONS
    jld_path = joinpath(ACW_BASE, atlas, version, "$(subject)_$(session).jld2")
    if !isfile(jld_path)
        @warn "Missing JLD2: $jld_path"
        continue
    end
    data        = load(jld_path)
    acw_results = data["acw_results"]
    tau_vals    = collect(acw_results[TAU_IDX])
    tau_clean   = filter(!isnan, tau_vals)
    tau_med     = isempty(tau_clean) ? NaN : median(tau_clean)
    push!(rows, (subject=subject, session=session, atlas=atlas,
                 version=version, tau_median=tau_med))
end

df = DataFrame(rows)
println("Loaded $(nrow(df)) rows (expected 280)")

# ── Per-version figures ───────────────────────────────────────────────────────
println("\n─── Self vs Nonself paired Wilcoxon tests ───\n")

for version in VERSIONS
    dfv = filter(r -> r.version == version, df)

    # Build paired (subject × self/nonself) per session ─────────────────────
    s01_self    = filter(r -> r.session == "ses-01" && r.atlas == "self",    dfv)
    s01_nonself = filter(r -> r.session == "ses-01" && r.atlas == "nonself", dfv)
    s02_self    = filter(r -> r.session == "ses-02" && r.atlas == "self",    dfv)
    s02_nonself = filter(r -> r.session == "ses-02" && r.atlas == "nonself", dfv)

    ses01 = innerjoin(
        rename(s01_self[!,    [:subject, :tau_median]], :tau_median => :self_tau),
        rename(s01_nonself[!, [:subject, :tau_median]], :tau_median => :nonself_tau),
        on = :subject
    )
    filter!(r -> !isnan(r.self_tau) && !isnan(r.nonself_tau), ses01)

    ses02 = innerjoin(
        rename(s02_self[!,    [:subject, :tau_median]], :tau_median => :self_tau),
        rename(s02_nonself[!, [:subject, :tau_median]], :tau_median => :nonself_tau),
        on = :subject
    )
    filter!(r -> !isnan(r.self_tau) && !isnan(r.nonself_tau), ses02)

    p01 = pvalue(SignedRankTest(ses01.self_tau, ses01.nonself_tau))
    p02 = pvalue(SignedRankTest(ses02.self_tau, ses02.nonself_tau))

    println("$version:")
    println("  ses-01: p = $(round(p01; digits=4)) ($(sig_label(p01)))")
    println("  ses-02: p = $(round(p02; digits=4)) ($(sig_label(p02)))")
    println()

    # ── Violin traces: numeric x positions for custom session grouping ──────────
    # Positions 0,1.5 = ses-01 pair; 3.5,5 = ses-02 pair
    # Within-pair gap = 1.5; between-pair gap = 2 → balanced horizontal layout
    n01 = nrow(ses01)
    n02 = nrow(ses02)

    trace_s01s = PlotlyJS.violin(
        y=ses01.self_tau,    x=fill(0.0, n01), name="ses01-Self",
        box_visible=true, points="all", pointpos=0,
        fillcolor="rgba(61, 90, 108, 0.4)", line_color=COLOR_SELF_PRE,
        marker=attr(color="rgba(61, 90, 108, 0.6)", size=6),
        box=attr(fillcolor="rgba(255,255,255,0)", line=attr(color="black", width=2))
    )
    trace_s01n = PlotlyJS.violin(
        y=ses01.nonself_tau, x=fill(1.5, n01), name="ses01-Nonself",
        box_visible=true, points="all", pointpos=0,
        fillcolor="rgba(139, 115, 85, 0.4)", line_color=COLOR_NONSELF_PRE,
        marker=attr(color="rgba(139, 115, 85, 0.6)", size=6),
        box=attr(fillcolor="rgba(255,255,255,0)", line=attr(color="black", width=2))
    )
    trace_s02s = PlotlyJS.violin(
        y=ses02.self_tau,    x=fill(3.5, n02), name="ses02-Self",
        box_visible=true, points="all", pointpos=0,
        fillcolor="rgba(122, 149, 166, 0.4)", line_color=COLOR_SELF_POST,
        marker=attr(color="rgba(122, 149, 166, 0.6)", size=6),
        box=attr(fillcolor="rgba(255,255,255,0)", line=attr(color="black", width=2))
    )
    trace_s02n = PlotlyJS.violin(
        y=ses02.nonself_tau, x=fill(5.0, n02), name="ses02-Nonself",
        box_visible=true, points="all", pointpos=0,
        fillcolor="rgba(191, 168, 136, 0.4)", line_color=COLOR_NONSELF_POST,
        marker=attr(color="rgba(191, 168, 136, 0.6)", size=6),
        box=attr(fillcolor="rgba(255,255,255,0)", line=attr(color="black", width=2))
    )

    bar_y   = 3.30
    tick_sz = 0.04

    layout = PlotlyJS.Layout(
        font          = attr(family="Times New Roman", size=25),
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        showlegend    = false,
        xaxis = attr(
            tickmode = "array",
            tickvals = [0.0, 1.5, 3.5, 5.0],
            ticktext = ["ses01-Self", "ses01-Nonself", "ses02-Self", "ses02-Nonself"],
            range    = [-0.7, 5.7],
        ),
        yaxis = attr(
            title         = "Intrinsic Timescale τ (seconds)",
            range         = [2.0, 3.5],
            tickmode      = "array",
            tickvals      = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4],
            gridcolor     = "rgba(0,0,0,0.05)",
            zerolinecolor = "rgba(0,0,0,0.1)",
        ),
        shapes = [
            # ses-01 bracket: horizontal bar + left tick + right tick
            attr(type="line", x0=0.0, x1=1.5, y0=bar_y,           y1=bar_y,
                 xref="x", yref="y", line=attr(color="black", width=1.5)),
            attr(type="line", x0=0.0, x1=0.0, y0=bar_y,           y1=bar_y - tick_sz,
                 xref="x", yref="y", line=attr(color="black", width=1.5)),
            attr(type="line", x0=1.5, x1=1.5, y0=bar_y,           y1=bar_y - tick_sz,
                 xref="x", yref="y", line=attr(color="black", width=1.5)),
            # ses-02 bracket
            attr(type="line", x0=3.5, x1=5.0, y0=bar_y,           y1=bar_y,
                 xref="x", yref="y", line=attr(color="black", width=1.5)),
            attr(type="line", x0=3.5, x1=3.5, y0=bar_y,           y1=bar_y - tick_sz,
                 xref="x", yref="y", line=attr(color="black", width=1.5)),
            attr(type="line", x0=5.0, x1=5.0, y0=bar_y,           y1=bar_y - tick_sz,
                 xref="x", yref="y", line=attr(color="black", width=1.5)),
        ],
        # Significance asterisks above each bracket midpoint
        # ses01 midpoint x = (0.0+1.5)/2 = 0.75; ses02 midpoint x = (3.5+5.0)/2 = 4.25
        annotations = [
            attr(x=0.75, y=bar_y + 0.05, xref="x", yref="y",
                 text=sig_label(p01), showarrow=false, font=attr(size=22)),
            attr(x=4.25, y=bar_y + 0.05, xref="x", yref="y",
                 text=sig_label(p02), showarrow=false, font=attr(size=22)),
        ]
    )

    fig = PlotlyJS.plot([trace_s01s, trace_s01n, trace_s02s, trace_s02n], layout)

    png_path = joinpath(FIG_DIR, "boxplot_session_self_nonself_$(version).png")
    PlotlyJS.savefig(fig, png_path, width=1200, height=800, scale=2)
    println("Saved: $png_path")

    if version == "raw"
        html_path = joinpath(FIG_DIR, "boxplot_session_self_nonself_raw.html")
        PlotlyJS.savefig(fig, html_path)
        println("Saved: $html_path")
    end
end

println("\nDone.")
