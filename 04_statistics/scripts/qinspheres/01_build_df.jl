# 01_build_df.jl — Extract qinspheres ACW metrics into flat CSVs, per metric × pipeline.
# Reads JLD2s from 03_intrinsic_neural_metrics/results/acw/{PIPELINE}/{category}/{sub}_{ses}.jld2
# where acw_results[1] = AUC, acw_results[2] = ACW-50.
#
# Output layout (AUC kept at the original top-level location; acw50 nested):
#   auc   -> 04_statistics/results/qinspheres/{PIPELINE}/tables/qinspheres_auc.csv
#   acw50 -> 04_statistics/results/qinspheres/acw50/{PIPELINE}/tables/qinspheres_acw50.csv
# The value column is named "auc" for both metrics (a carryover); the file/folder name
# identifies the metric and downstream R scripts read the "auc" column generically.
#
# Run from repo root:
#   julia 04_statistics/scripts/qinspheres/01_build_df.jl

using JLD2, CSV, DataFrames, Printf

const REPO_ROOT  = normpath(joinpath(@__DIR__, "..", "..", ".."))
const PIPELINES  = ["detrend", "glm", "maximal"]
const METRICS    = [("auc", 1), ("acw50", 2)]   # name => index into acw_results
const ATLAS      = length(ARGS) >= 1 ? ARGS[1] : "qinspheres"   # qinspheres | qinparcels
const ACW_DIR    = ATLAS == "qinspheres" ? "acw" : "acw_parcels"
const PARTS_TSV  = joinpath(REPO_ROOT, "participants.tsv")
const CATEGORIES = ["intero", "extero", "mental", "auditory", "motor", "visual"]
const SUBJECTS   = ["sub-$(lpad(i,2,'0'))" for i in 1:40 if !(i in (6,8,12,26,36))]   # n=35 (get_included_subjects)
const SESSIONS   = ["ses-01", "ses-02"]

parts = CSV.read(PARTS_TSV, DataFrame; delim = '\t')
drug  = Dict(string(r.participant_id) => string(r.condition) for r in eachrow(parts))

Row = NamedTuple{(:subject,:session,:drug_group,:category,:roi_id,:auc),
                 Tuple{String,String,String,String,String,Float64}}

# AUC stays at qinspheres/{pipeline}/...; other metrics nest under qinspheres/{metric}/{pipeline}/...
metric_dir(metric, pipeline) = metric == "auc" ?
    joinpath(REPO_ROOT, "04_statistics", "results", ATLAS, pipeline, "tables") :
    joinpath(REPO_ROOT, "04_statistics", "results", ATLAS, metric, pipeline, "tables")

function build(metric, midx, pipeline)
    jld2_dir = joinpath(REPO_ROOT, "03_intrinsic_neural_metrics", "results", ACW_DIR, pipeline)
    out_dir  = metric_dir(metric, pipeline)
    out_csv  = joinpath(out_dir, "qinspheres_$(metric).csv")

    rows = Row[]; missing_n = 0
    for cat in CATEGORIES, subj in SUBJECTS, ses in SESSIONS
        p = joinpath(jld2_dir, cat, "$(subj)_$(ses).jld2")
        if !isfile(p); missing_n += 1; continue; end
        d = load(p)
        ids  = d["parcel_ids"]
        vals = collect(d["acw_results"][midx])
        for (rid, a) in zip(ids, vals)
            push!(rows, (subject=subj, session=ses, drug_group=get(drug, subj, "unknown"),
                         category=cat, roi_id=String(rid), auc=Float64(a)))
        end
    end
    df = DataFrame(rows)
    n0 = nrow(df)
    # Drop degenerate non-positive AUC (broadband ACF collapse — only affects the
    # no-bandpass detrend/glm pipelines, concentrated in motor; maximal has none).
    # A non-positive area-under-ACF is not a valid intrinsic timescale.
    if metric == "auc"
        df = df[df.auc .> 0.0, :]
    end
    mkpath(out_dir); CSV.write(out_csv, df)
    @printf("  %-6s %-8s  %d rows (dropped %d non-positive AUC)  (%d missing)\n",
            metric, pipeline, nrow(df), n0 - nrow(df), missing_n)
end

for (metric, midx) in METRICS, pipeline in PIPELINES
    build(metric, midx, pipeline)
end
println("Done.")
