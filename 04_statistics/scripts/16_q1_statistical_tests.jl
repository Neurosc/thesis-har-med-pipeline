# 16_q1_statistical_tests.jl — Step 13: Q1 permutation test + LMM
#
# Tests whether the sensory → higher-order gradient in drug×session interaction
# is statistically robust in both atlases.
#
# Nonself: Sensory    = Visual1, Visual2, Somatomotor, Auditory        (99 ROIs)
#          Higher-order = Frontoparietal, Cingulo-Opercular, Default, Language (132 ROIs)
#          Excluded:    Dorsal-Attention, Posterior-Multimodal, Ventral-Multimodal, Orbito-Affective
# Self:    Sensory    = Exteroception  (14 ROIs)
#          Higher-order = Cognition     (12 ROIs)
#          Excluded:    Interoception   (11 ROIs, ambiguous sensory/higher-order)
#
# Tests (run separately for nonself and self):
#   1. Permutation test — shuffle drug_group labels across subjects (preserving
#      n_verum=17 / n_placebo=18), 10 000 permutations, seed=12345.
#      Test statistic: gradient = drug_interaction[higher_order] − drug_interaction[sensory]
#        where drug_interaction[type] = mean_verum(Δ) − mean_placebo(Δ)
#        and Δ = mean AUC (ses-02) − mean AUC (ses-01) per subject × network type.
#      Two-tailed p = (|null| >= |obs| + 1) / (N + 1)
#
#   2. Linear Mixed Model (MixedModels.jl)
#      Full:    auc ~ sess * drug * nettype + (1|subj) + (1|roi)
#      Reduced: auc ~ sess * drug + sess * nettype + drug * nettype + (1|subj) + (1|roi)
#      Likelihood-ratio test: χ²(1)
#
# Outputs:
#   04_statistics/figures/fig14_permutation_null.html
#   04_statistics/results/q1_statistical_tests.csv
#
# Run from repo root:
#   julia 04_statistics/scripts/16_q1_statistical_tests.jl

using CSV, DataFrames, Statistics, Printf, Random
using Distributions: Chisq, ccdf
using MixedModels
using CategoricalArrays
using PlotlyJS

# ── Paths ──────────────────────────────────────────────────────────────────────
const REPO_ROOT      = normpath(joinpath(@__DIR__, "..", ".."))
const AUC_CSV        = joinpath(REPO_ROOT, "04_statistics", "results",
                                "analysis_long_format_auc.csv")
const SELF_COORDS    = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                                "self_coordinates.txt")
const NONSELF_COORDS = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                                "glasser_coordinates_nonself_clean_1mm.txt")
const CABNP_KEY      = joinpath(REPO_ROOT,
    "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey (1).txt")
const FIG_DIR        = joinpath(REPO_ROOT, "04_statistics", "figures")
const OUT_FIG        = joinpath(FIG_DIR, "fig14_permutation_null.html")
const OUT_CSV        = joinpath(REPO_ROOT, "04_statistics", "results",
                                "q1_statistical_tests.csv")

# ── Idempotency ────────────────────────────────────────────────────────────────
if isfile(OUT_FIG) && isfile(OUT_CSV)
    println("[skip] outputs already exist")
    exit()
end

for f in [AUC_CSV, SELF_COORDS, NONSELF_COORDS, CABNP_KEY]
    isfile(f) || error("Missing required input: $f")
end
mkpath(FIG_DIR)

# ── Network / layer category definitions ──────────────────────────────────────
const SENSORY_NETS = Set(["Visual1", "Visual2", "Somatomotor", "Auditory"])
const HIGHER_NETS  = Set(["Frontoparietal", "Cingulo-Opercular", "Default", "Language"])
const N_PERMS      = 10_000
const SEED         = 12345

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Load data and attach atlas labels
# ══════════════════════════════════════════════════════════════════════════════
println("Loading atlas metadata and AUC data …")

self_coords    = CSV.read(SELF_COORDS,    DataFrame; delim='\t')
nonself_coords = CSV.read(NONSELF_COORDS, DataFrame; delim='\t')
nonself_coords[!, :nonself_pos_id] = 1:nrow(nonself_coords)

cabnp          = CSV.read(CABNP_KEY, DataFrame; delim='\t')
cabnp_cortical = filter(r -> r.KEYVALUE <= 360, cabnp)
cabnp_info     = select(cabnp_cortical, :GLASSERLABELNAME => :ROI_Name, :NETWORK)

nonself_meta = innerjoin(
    select(nonself_coords, :nonself_pos_id => :roi_pos_id, :ROI_Name),
    cabnp_info,
    on = :ROI_Name,
)

auc_df = CSV.read(AUC_CSV, DataFrame)
@printf("  AUC data: %d rows × %d columns\n", nrow(auc_df), ncol(auc_df))

# Attach layer / network labels
self_meta  = select(self_coords, :ROI_Number => :roi_pos_id, :Layer)
self_raw   = innerjoin(filter(r -> r.atlas == "self",    auc_df), self_meta,    on = :roi_pos_id)
nonself_raw = innerjoin(filter(r -> r.atlas == "nonself", auc_df), nonself_meta, on = :roi_pos_id)

# Assign network_type
ntype_nonself(net)   = net in SENSORY_NETS ? "sensory" : net in HIGHER_NETS ? "higher_order" : missing
ntype_self(layer)    = layer == "Exteroception" ? "sensory" : layer == "Cognition" ? "higher_order" : missing

nonself_raw[!, :network_type] = [ntype_nonself(n) for n in nonself_raw.NETWORK]
self_raw[!,    :network_type] = [ntype_self(l)    for l in self_raw.Layer]

nonself_test = filter(r -> !ismissing(r.network_type), nonself_raw)
self_test    = filter(r -> !ismissing(r.network_type), self_raw)

nonself_test[!, :network_type] = String.(nonself_test.network_type)
self_test[!,    :network_type] = String.(self_test.network_type)

@printf("  Nonself: %d rows | sensory ROIs=%d, higher-order ROIs=%d\n",
    nrow(nonself_test),
    length(unique(filter(r -> r.network_type == "sensory",      nonself_test).roi_pos_id)),
    length(unique(filter(r -> r.network_type == "higher_order", nonself_test).roi_pos_id)))
@printf("  Self:    %d rows | sensory ROIs=%d, higher-order ROIs=%d\n",
    nrow(self_test),
    length(unique(filter(r -> r.network_type == "sensory",      self_test).roi_pos_id)),
    length(unique(filter(r -> r.network_type == "higher_order", self_test).roi_pos_id)))

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Permutation test helpers
# ══════════════════════════════════════════════════════════════════════════════

# Pre-compute per-subject pre/post mean AUC by network type, return sorted vectors.
function prep_permdata(df)
    agg = combine(
        groupby(df, [:subject, :session, :network_type]),
        :auc => mean => :mean_auc,
    )
    pre  = filter(r -> r.session == "ses-01", agg)
    post = filter(r -> r.session == "ses-02", agg)
    wide = innerjoin(
        select(pre,  :subject, :network_type, :mean_auc => :auc_pre),
        select(post, :subject, :network_type, :mean_auc => :auc_post),
        on = [:subject, :network_type],
    )
    wide[!, :delta] = wide.auc_post .- wide.auc_pre

    subj_drug = unique(select(df, :subject, :drug_group))
    wide = innerjoin(wide, subj_drug, on = :subject)

    s_df = sort(filter(r -> r.network_type == "sensory",      wide), :subject)
    h_df = sort(filter(r -> r.network_type == "higher_order", wide), :subject)

    @assert s_df.subject == h_df.subject "Subject set mismatch between network types"

    return (
        subjects   = s_df.subject,
        d_sens     = Vector{Float64}(s_df.delta),
        d_high     = Vector{Float64}(h_df.delta),
        is_verum   = Vector{Bool}(s_df.drug_group .== "verum"),
    )
end

# Test statistic: drug_interaction[higher] − drug_interaction[sensory]
function grad_stat(d_high::Vector{Float64}, d_sens::Vector{Float64}, is_verum::Vector{Bool})
    is_pla = .!is_verum
    (mean(d_high[is_verum]) - mean(d_high[is_pla])) -
    (mean(d_sens[is_verum]) - mean(d_sens[is_pla]))
end

function permutation_test(df, n_perms::Int, seed::Int)
    pd      = prep_permdata(df)
    n_s     = length(pd.subjects)
    n_verum = sum(pd.is_verum)
    obs     = grad_stat(pd.d_high, pd.d_sens, pd.is_verum)

    rng     = MersenneTwister(seed)
    null    = Vector{Float64}(undef, n_perms)
    idx     = collect(1:n_s)
    is_v    = Vector{Bool}(undef, n_s)

    for i in 1:n_perms
        shuffle!(rng, idx)
        fill!(is_v, false)
        is_v[idx[1:n_verum]] .= true
        null[i] = grad_stat(pd.d_high, pd.d_sens, is_v)
    end

    p_val = (sum(abs.(null) .>= abs(obs)) + 1) / (n_perms + 1)
    return (obs_stat=obs, null=null, p_val=p_val,
            n_verum=n_verum, n_placebo=n_s - n_verum)
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — LMM helper
# ══════════════════════════════════════════════════════════════════════════════

function run_lmm(df, atlas_label)
    println("  Fitting LMM for $atlas_label atlas …")

    ldf = DataFrame(
        auc     = Float64.(df.auc),
        sess    = CategoricalArray(df.session),
        drug    = CategoricalArray(df.drug_group),
        nettype = CategoricalArray(df.network_type),
        subj    = CategoricalArray(df.subject),
        roi     = CategoricalArray(string.(df.roi_pos_id)),
    )

    m_full = fit(MixedModel,
        @formula(auc ~ sess * drug * nettype + (1|subj) + (1|roi)),
        ldf; progress=false)

    # Reduced: all main effects + 2-way interactions, no 3-way
    m_red = fit(MixedModel,
        @formula(auc ~ sess * drug + sess * nettype + drug * nettype + (1|subj) + (1|roi)),
        ldf; progress=false)

    χ2  = 2.0 * (loglikelihood(m_full) - loglikelihood(m_red))
    p   = ccdf(Chisq(1), χ2)

    @printf("    LRT χ²(1) = %.4f,  p = %.4f\n", χ2, p)
    return (lrt_chi2=χ2, p_lmm=p, df_lrt=1)
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Run all tests
# ══════════════════════════════════════════════════════════════════════════════
println()
println("Running permutation tests (N=$N_PERMS, seed=$SEED) …")
perm_nonself = permutation_test(nonself_test, N_PERMS, SEED)
perm_self    = permutation_test(self_test,    N_PERMS, SEED)

println()
println("Running LMMs …")
lmm_nonself = run_lmm(nonself_test, "nonself")
lmm_self    = run_lmm(self_test,    "self")

println()
println("═" ^ 72)
println("Summary")
println("═" ^ 72)
@printf("  Nonself — permutation: gradient = %+.4f s,  p (two-tailed) = %.4f\n",
        perm_nonself.obs_stat, perm_nonself.p_val)
@printf("  Nonself — LMM LRT:     χ²(1) = %.4f,  p = %.4f\n",
        lmm_nonself.lrt_chi2, lmm_nonself.p_lmm)
@printf("  Self    — permutation: gradient = %+.4f s,  p (two-tailed) = %.4f\n",
        perm_self.obs_stat, perm_self.p_val)
@printf("  Self    — LMM LRT:     χ²(1) = %.4f,  p = %.4f\n",
        lmm_self.lrt_chi2, lmm_self.p_lmm)

# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — Save CSV
# ══════════════════════════════════════════════════════════════════════════════
results_df = DataFrame(
    atlas         = ["nonself",                "nonself",            "self",                "self"],
    test          = ["permutation",            "lmm_lrt",            "permutation",         "lmm_lrt"],
    comparison    = fill("higher_order_vs_sensory", 4),
    observed_stat = [perm_nonself.obs_stat,    NaN,                  perm_self.obs_stat,    NaN],
    chi2          = [NaN,                      lmm_nonself.lrt_chi2, NaN,                   lmm_self.lrt_chi2],
    df_lrt        = [0,                        1,                    0,                     1],
    p_value       = [perm_nonself.p_val,       lmm_nonself.p_lmm,   perm_self.p_val,       lmm_self.p_lmm],
    n_perms       = [N_PERMS,                  0,                    N_PERMS,               0],
    n_verum       = [perm_nonself.n_verum,     0,                    perm_self.n_verum,     0],
    n_placebo     = [perm_nonself.n_placebo,   0,                    perm_self.n_placebo,   0],
)
CSV.write(OUT_CSV, results_df)
@printf("\nResults written: %s\n", OUT_CSV)

# ══════════════════════════════════════════════════════════════════════════════
# Part 6 — Figure: permutation null distributions
# ══════════════════════════════════════════════════════════════════════════════
println()
println("Building figure …")

fmt_p(p) = p < 0.001 ? "p < 0.001" : @sprintf("p = %.3f", p)

fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.12)

for (col, perm, label, color) in [
        (1, perm_nonself, "Nonself (higher − sensory)", "#3D5A6C"),
        (2, perm_self,    "Self (Cognition − Exteroception)", "#8B7355"),
    ]
    add_trace!(fig,
        histogram(
            x      = perm.null,
            name   = "$label null",
            marker = attr(color=color, opacity=0.75),
            nbinsx = 60,
        ),
        row=1, col=col)
end

# Observed-statistic vertical lines
shapes = [
    attr(
        type="line", xref="x",  yref="paper",
        x0=perm_nonself.obs_stat, x1=perm_nonself.obs_stat, y0=0, y1=1,
        line=attr(color="red", width=2, dash="dash"),
    ),
    attr(
        type="line", xref="x2", yref="paper",
        x0=perm_self.obs_stat, x1=perm_self.obs_stat, y0=0, y1=1,
        line=attr(color="red", width=2, dash="dash"),
    ),
]

p_ns = fmt_p(perm_nonself.p_val)
p_s  = fmt_p(perm_self.p_val)

panel_annotations = [
    attr(
        text      = "<b>Nonself: higher-order − sensory ($p_ns)</b>",
        xref="paper", yref="paper",
        x=0.22, y=1.04,
        xanchor="center", yanchor="bottom",
        showarrow=false,
        font=attr(size=12, family="Times New Roman"),
    ),
    attr(
        text      = "<b>Self: Cognition − Exteroception ($p_s)</b>",
        xref="paper", yref="paper",
        x=0.78, y=1.04,
        xanchor="center", yanchor="bottom",
        showarrow=false,
        font=attr(size=12, family="Times New Roman"),
    ),
]

relayout!(fig,
    title         = attr(
        text = "Question 1: Permutation null — sensory vs higher-order gradient",
        font = attr(size=15, family="Times New Roman"),
        x    = 0.5,
    ),
    annotations   = panel_annotations,
    shapes        = shapes,
    height        = 520,
    width         = 1200,
    paper_bgcolor = "white",
    plot_bgcolor  = "white",
    font          = attr(family="Times New Roman", size=11),
    barmode       = "overlay",
    xaxis         = attr(title="Gradient statistic (AUC, s)"),
    xaxis2        = attr(title="Gradient statistic (AUC, s)"),
    yaxis         = attr(title="Count"),
    yaxis2        = attr(title="Count"),
    showlegend    = false,
    margin        = attr(t=90, b=60, l=70, r=60),
)

PlotlyJS.savefig(fig, OUT_FIG)
@printf("  Figure written: %s\n", OUT_FIG)
println()
println("Done.")
