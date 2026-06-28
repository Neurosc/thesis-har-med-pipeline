# 01_build_df.jl — Extract qinspheres ACW-AUC into a flat CSV, per denoising pipeline.
# Reads JLD2s from 03_intrinsic_neural_metrics/results/acw/{PIPELINE}/{category}/{sub}_{ses}.jld2
# (PIPELINE in detrend/glm/maximal) and writes one row per (subject, session, category, roi) to:
#   04_statistics/results/qinspheres/{PIPELINE}/tables/qinspheres_auc.csv
#
# Run from repo root:
#   julia 04_statistics/scripts/qinspheres/01_build_df.jl

using JLD2, CSV, DataFrames, Printf

const REPO_ROOT  = normpath(joinpath(@__DIR__, "..", "..", ".."))
const PIPELINES  = ["detrend", "glm", "maximal"]
const PARTS_TSV  = joinpath(REPO_ROOT, "participants.tsv")
const CATEGORIES = ["intero", "extero", "mental", "auditory", "motor", "visual"]
const SUBJECTS   = ["sub-$(lpad(i,2,'0'))" for i in 1:40 if !(i in (12,))]   # n=39
const SESSIONS   = ["ses-01", "ses-02"]
const AUC_IDX    = 1   # acw_results[1] = AUC

parts = CSV.read(PARTS_TSV, DataFrame; delim = '\t')
drug  = Dict(string(r.participant_id) => string(r.condition) for r in eachrow(parts))

Row = NamedTuple{(:subject,:session,:drug_group,:category,:roi_id,:auc),
                 Tuple{String,String,String,String,String,Float64}}

function build_pipeline(pipeline)
    jld2_dir = joinpath(REPO_ROOT, "03_intrinsic_neural_metrics", "results", "acw", pipeline)
    out_dir  = joinpath(REPO_ROOT, "04_statistics", "results", "qinspheres", pipeline, "tables")
    out_csv  = joinpath(out_dir, "qinspheres_auc.csv")

    rows = Row[]
    missing_n = 0
    for cat in CATEGORIES, subj in SUBJECTS, ses in SESSIONS
        p = joinpath(jld2_dir, cat, "$(subj)_$(ses).jld2")
        if !isfile(p)
            missing_n += 1
            continue
        end
        d = load(p)
        ids  = d["parcel_ids"]
        aucs = collect(d["acw_results"][AUC_IDX])
        for (rid, a) in zip(ids, aucs)
            push!(rows, (subject    = subj,
                         session    = ses,
                         drug_group = get(drug, subj, "unknown"),
                         category   = cat,
                         roi_id     = String(rid),
                         auc        = Float64(a)))
        end
    end

    df = DataFrame(rows)
    @printf("\n── %s ──  (%d missing JLD2)\n", pipeline, missing_n)
    for cat in CATEGORIES
        sub_df = df[df.category .== cat, :]
        n_rois = isempty(sub_df) ? 0 : length(unique(sub_df.roi_id))
        @printf("  %-10s  %d ROIs  %d rows\n", cat, n_rois, nrow(sub_df))
    end
    mkpath(out_dir)
    CSV.write(out_csv, df)
    @printf("Saved: %s (%d rows)\n", out_csv, nrow(df))
end

for pipeline in PIPELINES
    build_pipeline(pipeline)
end
