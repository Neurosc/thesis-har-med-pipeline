# 11_hybrid_anova_drug_session.jl — Step 9: two-way mixed ANOVA (drug × session)
#
# Between-subject factor : drug_group (placebo / verum)
# Within-subject  factor : session    (ses-01 / ses-02)
# Outcome variables      : self_median, nonself_median, diff_median (= self − nonself)
#
# Analysis:
#   1. Aggregate analysis_long_format_auc.csv → per-subject × per-session medians + means
#   2. For each outcome, 2×2 mixed ANOVA (manually implemented from first principles)
#   3. Shapiro-Wilk normality check on within-cell residuals
#   4. Cohen's d at ses-02 between drug groups
#   5. Multiple-comparison correction on Drug×Session p-values: Bonferroni (×3) + BH-FDR
#   6. Sensitivity analysis: repeat with mean aggregation
#   7. Save: aggregated_subject_session.csv, anova_results_median.csv, anova_results_mean.csv
#
# Sphericity assumption is trivially satisfied (b = 2 within-levels → df = 1).
#
# Run from repo root:
#   julia 04_statistics/scripts/11_hybrid_anova_drug_session.jl

using CSV, DataFrames, Distributions, HypothesisTests, MultipleTesting, Statistics, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))
const IN_CSV    = joinpath(REPO_ROOT, "04_statistics", "results",
                           "analysis_long_format_auc.csv")
const RES_DIR   = joinpath(REPO_ROOT, "04_statistics", "results")
const AGG_CSV   = joinpath(RES_DIR, "aggregated_subject_session.csv")
const RES_MED   = joinpath(RES_DIR, "anova_results_median.csv")
const RES_MEAN  = joinpath(RES_DIR, "anova_results_mean.csv")

# ── Idempotency ───────────────────────────────────────────────────────────────
if isfile(AGG_CSV) && isfile(RES_MED) && isfile(RES_MEAN)
    println("[skip] all outputs already exist; delete to rerun")
    exit()
end

isfile(IN_CSV) || error("Missing required input: $IN_CSV")
mkpath(RES_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Aggregate to subject × session level
# ══════════════════════════════════════════════════════════════════════════════
println("═══ Part 1: Aggregate to subject × session level ═══\n")

println("Loading $IN_CSV …")
long_df = CSV.read(IN_CSV, DataFrame)
@printf("  Loaded %d rows × %d columns\n", nrow(long_df), ncol(long_df))

# Per (subject, session, drug_group, atlas)
agg = combine(
    groupby(long_df, [:subject, :session, :drug_group, :atlas]),
    :auc => median => :auc_median,
    :auc => mean   => :auc_mean,
)

self_agg    = filter(r -> r.atlas == "self",    agg)
nonself_agg = filter(r -> r.atlas == "nonself", agg)

wide = innerjoin(
    select(self_agg,
           :subject, :session, :drug_group,
           :auc_median => :self_median,
           :auc_mean   => :self_mean),
    select(nonself_agg,
           :subject, :session,
           :auc_median => :nonself_median,
           :auc_mean   => :nonself_mean),
    on = [:subject, :session],
)
sort!(wide, [:subject, :session])

wide.diff_median = wide.self_median .- wide.nonself_median
wide.diff_mean   = wide.self_mean   .- wide.nonself_mean

ses01_rows = filter(r -> r.session == "ses-01", wide)
n_placebo  = count(==("placebo"), ses01_rows.drug_group)
n_verum    = count(==("verum"),   ses01_rows.drug_group)

@printf("  Aggregated: %d rows × %d columns\n", nrow(wide), ncol(wide))
@printf("  Subjects: %d  (placebo=%d, verum=%d) × 2 sessions\n",
        nrow(ses01_rows), n_placebo, n_verum)

println("\n  Cell means (median AUC) — atlas × session × drug_group:")
@printf("  %-10s  %-8s  %-10s  %12s  %14s  %12s\n",
        "drug_group", "session", "n", "self_median", "nonself_median", "diff_median")
println("  " * "─" ^ 70)
for sess in ["ses-01", "ses-02"]
    for grp in ["placebo", "verum"]
        sub = filter(r -> r.session == sess && r.drug_group == grp, wide)
        @printf("  %-10s  %-8s  %-10d  %12.4f  %14.4f  %12.4f\n",
                grp, sess, nrow(sub),
                median(sub.self_median),
                median(sub.nonself_median),
                median(sub.diff_median))
    end
end

CSV.write(AGG_CSV, select(wide, :subject, :session, :drug_group,
                          :self_median, :nonself_median, :diff_median,
                          :self_mean,   :nonself_mean,   :diff_mean))
println("\n  Saved: $AGG_CSV  ($(nrow(wide)) rows)")

# ══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ══════════════════════════════════════════════════════════════════════════════

"""
    mixed_anova_2x2(df, outcome)

Two-way mixed ANOVA.
  A (between-subjects) : drug_group  (a = 2 levels: "placebo", "verum")
  B (within-subjects)  : session     (b = 2 levels: "ses-01",  "ses-02")

SS decomposition (standard textbook formulas):
  Between-subjects strip:
    SS_A   = b × Σ_g n_g × (Ȳ_g. − Ȳ..)²          df = a − 1
    SS_S(A)= b × Σ_i  (Ȳ_i. − Ȳ_{g(i).})²          df = N − a
  Within-subjects strip:
    SS_B   = N × Σ_j (Ȳ_.j − Ȳ..)²                 df = b − 1
    SS_AB  = Σ_g Σ_j n_g × (Ȳ_gj − Ȳ_g. − Ȳ_.j + Ȳ..)²  df = (a−1)(b−1)
    SS_B×S(A) = Σ_i Σ_j (Y_ij − Ȳ_i. − Ȳ_{g(i)j} + Ȳ_{g(i).})²  df = (b−1)(N−a)
"""
function mixed_anova_2x2(df::DataFrame, outcome::Symbol)
    SESSIONS = ["ses-01", "ses-02"]
    GROUPS   = ["placebo", "verum"]
    J = length(SESSIONS)   # 2
    G = length(GROUPS)     # 2

    subjects = sort(unique(df.subject))
    N        = length(subjects)

    subj_idx = Dict(s => i for (i, s) in enumerate(subjects))
    sess_idx = Dict(s => j for (j, s) in enumerate(SESSIONS))
    grp_idx  = Dict(g => k for (k, g) in enumerate(GROUPS))

    Y       = zeros(N, J)
    grp_of  = zeros(Int, N)

    for row in eachrow(df)
        i = subj_idx[row.subject]
        j = sess_idx[row.session]
        Y[i, j]   = row[outcome]
        grp_of[i] = grp_idx[row.drug_group]
    end

    n_g        = [count(==(k), grp_of) for k in 1:G]
    grand      = mean(Y)
    grp_means  = [mean(Y[grp_of .== k, :]) for k in 1:G]
    sess_means = [mean(Y[:, j]) for j in 1:J]
    cell_means = [mean(Y[grp_of .== k, j]) for k in 1:G, j in 1:J]
    subj_means = [mean(Y[i, :])            for i in 1:N]

    # ── Between-subjects ────────────────────────────────────────────────────
    SS_A  = J * sum(n_g[k] * (grp_means[k] - grand)^2 for k in 1:G)
    SS_SA = J * sum((subj_means[i] - grp_means[grp_of[i]])^2 for i in 1:N)
    df_A  = G - 1        # 1
    df_SA = N - G        # 33

    # ── Within-subjects ─────────────────────────────────────────────────────
    SS_B = N * sum((sess_means[j] - grand)^2 for j in 1:J)
    SS_AB = sum(
        n_g[k] * (cell_means[k, j] - grp_means[k] - sess_means[j] + grand)^2
        for k in 1:G, j in 1:J
    )
    SS_BxSA = sum(
        (Y[i, j] - subj_means[i] - cell_means[grp_of[i], j] + grp_means[grp_of[i]])^2
        for i in 1:N, j in 1:J
    )
    df_B    = J - 1              # 1
    df_AB   = (G - 1) * (J - 1) # 1
    df_BxSA = (J - 1) * (N - G) # 33

    MS_A    = SS_A    / df_A
    MS_SA   = SS_SA   / df_SA
    MS_B    = SS_B    / df_B
    MS_AB   = SS_AB   / df_AB
    MS_BxSA = SS_BxSA / df_BxSA

    F_A  = MS_A  / MS_SA
    F_B  = MS_B  / MS_BxSA
    F_AB = MS_AB / MS_BxSA

    p_A  = ccdf(FDist(df_A,  df_SA),   max(F_A,  0.0))
    p_B  = ccdf(FDist(df_B,  df_BxSA), max(F_B,  0.0))
    p_AB = ccdf(FDist(df_AB, df_BxSA), max(F_AB, 0.0))

    eta2_A  = SS_A  / (SS_A  + SS_SA)
    eta2_B  = SS_B  / (SS_B  + SS_BxSA)
    eta2_AB = SS_AB / (SS_AB + SS_BxSA)

    # Within-cell residuals for normality check
    residuals = Float64[
        Y[i, j] - cell_means[grp_of[i], j]
        for i in 1:N, j in 1:J
    ][:]

    return (
        N=N, n_g=n_g, GROUPS=GROUPS, SESSIONS=SESSIONS,
        grand=grand, grp_means=grp_means, sess_means=sess_means, cell_means=cell_means,
        Y=Y, grp_of=grp_of,
        SS_A=SS_A,   df_A=df_A,   MS_A=MS_A,   F_A=F_A,   p_A=p_A,   eta2_A=eta2_A,
        SS_SA=SS_SA, df_SA=df_SA, MS_SA=MS_SA,
        SS_B=SS_B,   df_B=df_B,   MS_B=MS_B,   F_B=F_B,   p_B=p_B,   eta2_B=eta2_B,
        SS_AB=SS_AB, df_AB=df_AB, MS_AB=MS_AB, F_AB=F_AB, p_AB=p_AB, eta2_AB=eta2_AB,
        SS_BxSA=SS_BxSA, df_BxSA=df_BxSA, MS_BxSA=MS_BxSA,
        residuals=residuals,
    )
end

function cohens_d(x, y)
    nx, ny = length(x), length(y)
    s_pool = sqrt(((nx - 1) * var(x) + (ny - 1) * var(y)) / (nx + ny - 2))
    s_pool ≈ 0 && return 0.0
    (mean(x) - mean(y)) / s_pool
end

function d_label(d)
    a = abs(d)
    a < 0.2 ? "negligible" : a < 0.5 ? "small" : a < 0.8 ? "medium" : "large"
end

function print_anova_table(r, outcome_label)
    println()
    println("  ─── Outcome: $outcome_label ───────────────────────────────────────")

    # Cell means
    println("  Cell means:")
    @printf("  %-10s  %12s  %12s\n", "drug_group", "ses-01", "ses-02")
    println("  " * "─" ^ 36)
    for (k, grp) in enumerate(r.GROUPS)
        @printf("  %-10s  %12.4f  %12.4f\n", grp, r.cell_means[k, 1], r.cell_means[k, 2])
    end

    # ANOVA table
    println()
    @printf("  %-24s  %10s  %3s  %10s  %8s  %8s  %8s\n",
            "Effect", "SS", "df", "MS", "F", "p", "η²_p")
    println("  " * "─" ^ 80)
    @printf("  %-24s  %10.4f  %3d  %10.4f  %8.4f  %8.4f  %8.4f\n",
            "Drug (between)",
            r.SS_A, r.df_A, r.MS_A, r.F_A, r.p_A, r.eta2_A)
    @printf("  %-24s  %10.4f  %3d  %10.4f  %8s  %8s  %8s\n",
            "  Error: S(Drug)",
            r.SS_SA, r.df_SA, r.MS_SA, "—", "—", "—")
    @printf("  %-24s  %10.4f  %3d  %10.4f  %8.4f  %8.4f  %8.4f\n",
            "Session (within)",
            r.SS_B, r.df_B, r.MS_B, r.F_B, r.p_B, r.eta2_B)
    @printf("  %-24s  %10.4f  %3d  %10.4f  %8.4f  %8.4f  %8.4f\n",
            "Drug × Session",
            r.SS_AB, r.df_AB, r.MS_AB, r.F_AB, r.p_AB, r.eta2_AB)
    @printf("  %-24s  %10.4f  %3d  %10.4f  %8s  %8s  %8s\n",
            "  Error: B×S(Drug)",
            r.SS_BxSA, r.df_BxSA, r.MS_BxSA, "—", "—", "—")
end

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Run ANOVAs: primary (median aggregation)
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 2: Two-way mixed ANOVA (primary — median aggregation) ═══")

OUTCOMES_MED = [
    (:self_median,    "Self"),
    (:nonself_median, "Nonself"),
    (:diff_median,    "Self − Nonself"),
]

results_med = Dict{Symbol, Any}()

res_df_med = DataFrame(
    outcome      = String[],
    effect       = String[],
    SS           = Float64[],
    df           = Int[],
    MS           = Float64[],
    F            = Float64[],
    p            = Float64[],
    eta2_partial = Float64[],
)

for (outcome, label) in OUTCOMES_MED
    r = mixed_anova_2x2(wide, outcome)
    results_med[outcome] = r
    print_anova_table(r, label)

    # Shapiro-Wilk on within-cell residuals
    sw   = ShapiroWilkTest(r.residuals)
    sw_p = pvalue(sw)
    @printf("\n  Shapiro-Wilk (within-cell residuals, n=%d):  W=%.4f  p=%.4f  (%s)\n",
            length(r.residuals), sw.W, sw_p,
            sw_p >= 0.05 ? "normality supported" : "normality VIOLATED")

    # Cohen's d at ses-02
    j2    = 2  # "ses-02" is index 2 in SESSIONS
    p_idx = findfirst(==("placebo"), r.GROUPS)
    v_idx = findfirst(==("verum"),   r.GROUPS)
    d     = cohens_d(r.Y[r.grp_of .== p_idx, j2], r.Y[r.grp_of .== v_idx, j2])
    @printf("  Cohen's d (placebo vs verum at ses-02): d = %+.4f  (%s)\n",
            d, d_label(d))

    for (eff, ss, df_e, ms, F, p, eta2) in [
            ("drug",           r.SS_A,  r.df_A,  r.MS_A,  r.F_A,  r.p_A,  r.eta2_A),
            ("session",        r.SS_B,  r.df_B,  r.MS_B,  r.F_B,  r.p_B,  r.eta2_B),
            ("drug_x_session", r.SS_AB, r.df_AB, r.MS_AB, r.F_AB, r.p_AB, r.eta2_AB),
        ]
        push!(res_df_med, (String(outcome), eff, ss, df_e, ms, F, p, eta2))
    end
end

# ── Multiple-comparison correction on Drug×Session p-values ──────────────────
println("\n  ─── Multiple-comparison correction (Drug × Session, 3 outcomes) ───")

int_mask  = res_df_med.effect .== "drug_x_session"
int_p_med = res_df_med.p[int_mask]

p_bonf_med = clamp.(int_p_med .* 3, 0.0, 1.0)
p_fdr_med  = MultipleTesting.adjust(int_p_med, BenjaminiHochberg())

res_df_med.p_bonferroni = fill(NaN, nrow(res_df_med))
res_df_med.p_fdr        = fill(NaN, nrow(res_df_med))
res_df_med.p_bonferroni[int_mask] = p_bonf_med
res_df_med.p_fdr[int_mask]        = p_fdr_med

@printf("\n  %-20s  %8s  %12s  %8s\n", "outcome", "raw_p", "bonferroni_p", "fdr_p")
println("  " * "─" ^ 52)
for (i, row) in enumerate(eachrow(filter(r -> r.effect == "drug_x_session", res_df_med)))
    @printf("  %-20s  %8.4f  %12.4f  %8.4f\n",
            row.outcome, row.p, row.p_bonferroni, row.p_fdr)
end

CSV.write(RES_MED, res_df_med)
println("\n  Saved: $RES_MED  ($(nrow(res_df_med)) rows)")

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Sensitivity analysis (mean aggregation)
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 3: Sensitivity analysis — mean aggregation ═══")

OUTCOMES_MEAN = [
    (:self_mean,    "Self (mean agg.)"),
    (:nonself_mean, "Nonself (mean agg.)"),
    (:diff_mean,    "Self − Nonself (mean agg.)"),
]

results_mean = Dict{Symbol, Any}()

res_df_mean = DataFrame(
    outcome      = String[],
    effect       = String[],
    SS           = Float64[],
    df           = Int[],
    MS           = Float64[],
    F            = Float64[],
    p            = Float64[],
    eta2_partial = Float64[],
)

for (outcome, label) in OUTCOMES_MEAN
    r = mixed_anova_2x2(wide, outcome)
    results_mean[outcome] = r
    print_anova_table(r, label)

    for (eff, ss, df_e, ms, F, p, eta2) in [
            ("drug",           r.SS_A,  r.df_A,  r.MS_A,  r.F_A,  r.p_A,  r.eta2_A),
            ("session",        r.SS_B,  r.df_B,  r.MS_B,  r.F_B,  r.p_B,  r.eta2_B),
            ("drug_x_session", r.SS_AB, r.df_AB, r.MS_AB, r.F_AB, r.p_AB, r.eta2_AB),
        ]
        push!(res_df_mean, (String(outcome), eff, ss, df_e, ms, F, p, eta2))
    end
end

int_mask_mean   = res_df_mean.effect .== "drug_x_session"
int_p_mean_vals = res_df_mean.p[int_mask_mean]

p_bonf_mean = clamp.(int_p_mean_vals .* 3, 0.0, 1.0)
p_fdr_mean  = MultipleTesting.adjust(int_p_mean_vals, BenjaminiHochberg())

res_df_mean.p_bonferroni = fill(NaN, nrow(res_df_mean))
res_df_mean.p_fdr        = fill(NaN, nrow(res_df_mean))
res_df_mean.p_bonferroni[int_mask_mean] = p_bonf_mean
res_df_mean.p_fdr[int_mask_mean]        = p_fdr_mean

println("\n  ─── Multiple-comparison correction (mean, Drug × Session) ───")
@printf("\n  %-20s  %8s  %12s  %8s\n", "outcome", "raw_p", "bonferroni_p", "fdr_p")
println("  " * "─" ^ 52)
for row in eachrow(filter(r -> r.effect == "drug_x_session", res_df_mean))
    @printf("  %-20s  %8.4f  %12.4f  %8.4f\n",
            row.outcome, row.p, row.p_bonferroni, row.p_fdr)
end

CSV.write(RES_MEAN, res_df_mean)
println("\n  Saved: $RES_MEAN  ($(nrow(res_df_mean)) rows)")

# ── Convergence check: median vs mean ────────────────────────────────────────
println("\n═══ Convergence check: median vs mean Drug × Session ═══\n")
@printf("  %-20s  %10s  %10s  %10s\n", "outcome_pair", "p_med", "p_mean", "converge?")
println("  " * "─" ^ 56)
for ((om, _), (on, _)) in zip(OUTCOMES_MED, OUTCOMES_MEAN)
    pm  = results_med[om].p_AB
    pmn = results_mean[on].p_AB
    converge = (pm < 0.05) == (pmn < 0.05) ? "yes" : "NO"
    @printf("  %-20s  %10.4f  %10.4f  %10s\n",
            String(om), pm, pmn, converge)
end

println("\nDone.")
