# 17_q1_aggregated_lmm.jl — Step 13b: Aggregated subject-level LMM
#
# Resolves the discrepancy between permutation (p ≈ 0.14 / 0.08) and the
# full-observation LMM (p ≪ 0.0001 / 0.10) from Step 13.  The full LMM
# treats ~16 000 non-independent AUC observations as exchangeable, inflating
# power.  This script collapses within-subject ROI dependency first, then models.
#
# Aggregation: for each subject × session × network_type, compute
#   mean_auc = mean AUC across all ROIs in that category.
# Results in 140 rows per atlas (35 subjects × 2 sessions × 2 network types).
#
# Models (REML=false, ML throughout for valid LRT):
#   Full:    mean_auc ~ session * drug_group * network_type + (1|subject)
#   Reduced: mean_auc ~ session * drug_group + session * network_type
#                     + drug_group * network_type + (1|subject)
#   LRT: χ²(1) for the three-way interaction term
#
# Network categories (same as Step 13, script 16):
#   Nonself sensory:       Visual1, Visual2, Somatomotor, Auditory
#   Nonself higher-order:  Frontoparietal, Cingulo-Opercular, Default, Language
#   Self    sensory:       Exteroception
#   Self    higher-order:  Cognition  (Interoception excluded — ambiguous)
#
# Outputs:
#   04_statistics/results/q1_aggregated_lmm.csv
#
# Run from repo root:
#   julia 04_statistics/scripts/17_q1_aggregated_lmm.jl

using CSV, DataFrames, Statistics, Printf
using Distributions: Chisq, ccdf
using MixedModels
using CategoricalArrays

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
const STEP13_CSV     = joinpath(REPO_ROOT, "04_statistics", "results",
                                "q1_statistical_tests.csv")
const OUT_CSV        = joinpath(REPO_ROOT, "04_statistics", "results",
                                "q1_aggregated_lmm.csv")

# ── Idempotency ────────────────────────────────────────────────────────────────
if isfile(OUT_CSV)
    println("[skip] output already exists: $OUT_CSV")
    exit()
end

for f in [AUC_CSV, SELF_COORDS, NONSELF_COORDS, CABNP_KEY]
    isfile(f) || error("Missing required input: $f")
end

# ── Network category definitions ──────────────────────────────────────────────
const SENSORY_NETS = Set(["Visual1", "Visual2", "Somatomotor", "Auditory"])
const HIGHER_NETS  = Set(["Frontoparietal", "Cingulo-Opercular", "Default", "Language"])

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
@printf("  Loaded %d rows × %d columns\n", nrow(auc_df), ncol(auc_df))

self_meta   = select(self_coords, :ROI_Number => :roi_pos_id, :Layer)
self_raw    = innerjoin(filter(r -> r.atlas == "self",    auc_df), self_meta,    on = :roi_pos_id)
nonself_raw = innerjoin(filter(r -> r.atlas == "nonself", auc_df), nonself_meta, on = :roi_pos_id)

ntype_ns(net)   = net in SENSORY_NETS ? "sensory" : net in HIGHER_NETS ? "higher_order" : missing
ntype_se(layer) = layer == "Exteroception" ? "sensory" : layer == "Cognition" ? "higher_order" : missing

nonself_raw[!, :network_type] = [ntype_ns(n) for n in nonself_raw.NETWORK]
self_raw[!,    :network_type] = [ntype_se(l) for l in self_raw.Layer]

nonself_fil = filter(r -> !ismissing(r.network_type), nonself_raw)
self_fil    = filter(r -> !ismissing(r.network_type), self_raw)

nonself_fil[!, :network_type] = String.(nonself_fil.network_type)
self_fil[!,    :network_type] = String.(self_fil.network_type)

@printf("  Nonself rows in test set: %d\n", nrow(nonself_fil))
@printf("  Self    rows in test set: %d\n", nrow(self_fil))

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Aggregate to subject × session × network_type means
# ══════════════════════════════════════════════════════════════════════════════
println()
println("Aggregating to subject × session × network_type …")

function aggregate_auc(df)
    agg = combine(
        groupby(df, [:subject, :session, :drug_group, :network_type]),
        :auc => mean => :mean_auc,
    )
    sort!(agg, [:subject, :session, :network_type])
    return agg
end

nonself_agg = aggregate_auc(nonself_fil)
self_agg    = aggregate_auc(self_fil)

@printf("  Nonself aggregated: %d rows (expected 140)\n", nrow(nonself_agg))
@printf("  Self    aggregated: %d rows (expected 140)\n", nrow(self_agg))

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Cell means summary
# ══════════════════════════════════════════════════════════════════════════════

function print_cell_means(label, df)
    println()
    @printf("  Cell means — %s atlas\n", label)
    @printf("  %-9s  %-7s  %-14s  %8s ± %-8s  N\n",
            "drug",  "session", "network_type", "mean", "SE")
    cells = combine(
        groupby(df, [:drug_group, :session, :network_type]),
        :mean_auc => mean => :m,
        :mean_auc => (x -> std(x) / sqrt(length(x))) => :se,
        nrow => :n,
    )
    sort!(cells, [:drug_group, :session, :network_type])
    for r in eachrow(cells)
        @printf("  %-9s  %-7s  %-14s  %8.4f ± %-8.4f  %d\n",
                r.drug_group, r.session, r.network_type, r.m, r.se, r.n)
    end
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Aggregated LMM
# ══════════════════════════════════════════════════════════════════════════════

function run_aggregated_lmm(df, atlas_label)
    println()
    println("═" ^ 72)
    println("Atlas: $atlas_label")
    println("═" ^ 72)

    print_cell_means(atlas_label, df)

    ldf = DataFrame(
        mean_auc = Float64.(df.mean_auc),
        sess     = CategoricalArray(df.session),
        drug     = CategoricalArray(df.drug_group),
        nettype  = CategoricalArray(df.network_type),
        subj     = CategoricalArray(df.subject),
    )

    println()
    println("  Fitting full model (ML) …")
    m_full = fit(MixedModel,
        @formula(mean_auc ~ sess * drug * nettype + (1|subj)),
        ldf; REML=false, progress=false)

    println("  Fitting reduced model (ML, no 3-way interaction) …")
    m_red = fit(MixedModel,
        @formula(mean_auc ~ sess * drug + sess * nettype + drug * nettype + (1|subj)),
        ldf; REML=false, progress=false)

    cv_full = string(m_full.optsum.returnvalue)
    cv_red  = string(m_red.optsum.returnvalue)

    println()
    println("  Fixed-effects table (full model):")
    show(stdout, coeftable(m_full))
    println()

    @printf("  Convergence: full=%s,  reduced=%s\n", cv_full, cv_red)

    χ2 = 2.0 * (loglikelihood(m_full) - loglikelihood(m_red))
    p  = ccdf(Chisq(1), χ2)

    println()
    @printf("  LRT — full vs reduced (3-way interaction):\n")
    @printf("    log-likelihood full    = %.4f\n", loglikelihood(m_full))
    @printf("    log-likelihood reduced = %.4f\n", loglikelihood(m_red))
    @printf("    χ²(1) = %.4f,  p = %.4f\n", χ2, p)

    good_cv = Set(["FTOL_REACHED", "XTOL_REACHED", "SUCCESS"])
    if cv_full ∉ good_cv
        @warn "Full model convergence status: $cv_full — interpret with caution"
    end
    if cv_red ∉ good_cv
        @warn "Reduced model convergence status: $cv_red — interpret with caution"
    end

    return (
        lrt_chi2       = χ2,
        p_lmm          = p,
        n_obs          = nrow(df),
        loglik_full    = loglikelihood(m_full),
        loglik_reduced = loglikelihood(m_red),
        converged_full = cv_full,
        converged_red  = cv_red,
    )
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 5 — Run
# ══════════════════════════════════════════════════════════════════════════════
res_nonself = run_aggregated_lmm(nonself_agg, "nonself")
res_self    = run_aggregated_lmm(self_agg,    "self")

# ══════════════════════════════════════════════════════════════════════════════
# Part 6 — Save CSV
# ══════════════════════════════════════════════════════════════════════════════
out_df = DataFrame(
    atlas          = ["nonself",             "self"],
    test           = fill("aggregated_lmm_lrt", 2),
    comparison     = fill("higher_order_vs_sensory_3way", 2),
    lrt_chi2       = [res_nonself.lrt_chi2,       res_self.lrt_chi2],
    df_lrt         = [1,                           1],
    p_value        = [res_nonself.p_lmm,           res_self.p_lmm],
    n_obs          = [res_nonself.n_obs,            res_self.n_obs],
    loglik_full    = [res_nonself.loglik_full,      res_self.loglik_full],
    loglik_reduced = [res_nonself.loglik_reduced,   res_self.loglik_reduced],
    converged_full = [res_nonself.converged_full,   res_self.converged_full],
    converged_red  = [res_nonself.converged_red,    res_self.converged_red],
)
CSV.write(OUT_CSV, out_df)
@printf("\nResults written: %s\n", OUT_CSV)

# ══════════════════════════════════════════════════════════════════════════════
# Part 7 — Cross-test comparison table
# ══════════════════════════════════════════════════════════════════════════════
println()
println("═" ^ 72)
println("Step 13 / 13b comparison table")
println("═" ^ 72)

perm_ns = "N/A"; perm_s = "N/A"
lmm_ns  = "N/A"; lmm_s  = "N/A"

if isfile(STEP13_CSV)
    s13 = CSV.read(STEP13_CSV, DataFrame)
    function lookup_p(df, atl, tst)
        rows = filter(r -> r.atlas == atl && r.test == tst, df)
        nrow(rows) == 0 && return "N/A"
        p = rows[1, :p_value]
        p < 0.001 ? "<0.001" : @sprintf("%.3f", p)
    end
    perm_ns = lookup_p(s13, "nonself", "permutation")
    perm_s  = lookup_p(s13, "self",    "permutation")
    lmm_ns  = lookup_p(s13, "nonself", "lmm_lrt")
    lmm_s   = lookup_p(s13, "self",    "lmm_lrt")
end

agg_ns = res_nonself.p_lmm < 0.001 ? "<0.001" : @sprintf("%.3f", res_nonself.p_lmm)
agg_s  = res_self.p_lmm    < 0.001 ? "<0.001" : @sprintf("%.3f", res_self.p_lmm)

println()
@printf("  %-32s  %10s  %10s\n", "",                              "Nonself",  "Self")
@printf("  %-32s  %10s  %10s\n", "─"^32,                         "─"^10,     "─"^10)
@printf("  %-32s  %10s  %10s\n", "Permutation (Step 13)",         perm_ns,    perm_s)
@printf("  %-32s  %10s  %10s\n", "Full LMM (Step 13)",            lmm_ns,     lmm_s)
@printf("  %-32s  %10s  %10s\n", "Aggregated LMM (Step 13b)",     agg_ns,     agg_s)

println()
println("  Interpretation:")
for (label, p_agg) in [("Nonself", res_nonself.p_lmm), ("Self", res_self.p_lmm)]
    if p_agg < 0.05
        println("  [$label] p < 0.05 → gradient robust at subject level; full LMM result consistent")
    elseif p_agg < 0.20
        println("  [$label] p 0.05–0.20 → full LMM was inflated by pseudoreplication; permutation more accurate")
    else
        println("  [$label] p > 0.20 → no reliable gradient at subject level")
    end
end

println()
println("Done.")
