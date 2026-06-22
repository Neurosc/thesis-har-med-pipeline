# 01_build_df.jl — Extract qinspheres ACW-AUC into a flat CSV.
# Reads JLD2s from 03_acw_analysis/results/qinspheres_NoGSR/{category}/{sub}_{ses}.jld2
# and writes one row per (subject, session, category, roi) to:
#   03_acw_analysis/results/qinspheres_NoGSR/qinspheres_auc.csv
#
# Run from repo root:
#   julia 04_statistics/scripts/qinspheres/01_build_df.jl

using JLD2, CSV, DataFrames, Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", ".."))
const JLD2_DIR  = joinpath(REPO_ROOT, "03_acw_analysis", "results", "qinspheres_NoGSR")
const PARTS_TSV = joinpath(REPO_ROOT, "participants.tsv")
const OUT_DIR   = joinpath(REPO_ROOT, "04_statistics", "results", "qinspheres", "tables")
const OUT_CSV   = joinpath(OUT_DIR, "qinspheres_auc.csv")

const CATEGORIES = ["intero", "extero", "mental", "auditory", "motor", "visual"]
const SUBJECTS   = ["sub-$(lpad(i,2,'0'))" for i in 1:40 if !(i in (6,8,12,26,36))]
const SESSIONS   = ["ses-01", "ses-02"]
const AUC_IDX    = 1   # acw_results[1] = AUC

parts = CSV.read(PARTS_TSV, DataFrame; delim = '\t')
drug  = Dict(string(r.participant_id) => string(r.condition) for r in eachrow(parts))

Row = NamedTuple{(:subject,:session,:drug_group,:category,:roi_id,:auc),
                 Tuple{String,String,String,String,String,Float64}}
rows = Row[]
missing_n = 0

for cat in CATEGORIES, subj in SUBJECTS, ses in SESSIONS
    p = joinpath(JLD2_DIR, cat, "$(subj)_$(ses).jld2")
    if !isfile(p)
        global missing_n += 1
        continue
    end
    d = load(p)
    ids = d["parcel_ids"]
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

missing_n > 0 && @printf("Warning: %d missing JLD2 files\n", missing_n)

df = DataFrame(rows)

# Summary
for cat in CATEGORIES
    sub_df = df[df.category .== cat, :]
    n_rois = isempty(sub_df) ? 0 : length(unique(sub_df.roi_id))
    @printf("  %-10s  %d ROIs  %d rows\n", cat, n_rois, nrow(sub_df))
end

mkpath(OUT_DIR)
CSV.write(OUT_CSV, df)
@printf("Saved: %s (%d rows)\n", OUT_CSV, nrow(df))
