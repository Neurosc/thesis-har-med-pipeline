# 01_build_auc.jl — Extract ACW metrics into flat CSVs, per metric × pipeline.
# Reads JLD2s from 03_intrinsic_neural_metrics/results/acw/{spheres|parcels/self_regions}/{PIPELINE}/{category}/{sub}_{ses}.jld2
# The JLD2 payload has two shapes: files written with a single acwtype ([:auc]) hold a FLAT
# Vector{Float64} (one AUC per ROI); older files written with [:auc, :acw50] are nested
# ([1]=AUC, [2]=ACW-50). Both are handled below — see the comment at the load site.
#
# Output layout — mirrors 03_intrinsic_neural_metrics/results, one shape for every metric:
#   04_statistics/results/{acw|sampen|fc}/{spheres|parcels/self_regions}/{PIPELINE}/tables/{metric}.csv
# The value column is named "auc" for every metric (a carryover); the path identifies
# the metric and downstream R scripts read the "auc" column generically.
#
# Run from repo root:
#   julia 04_statistics/scripts/qin/01_build_auc.jl

using JLD2, CSV, DataFrames, Printf

const REPO_ROOT  = normpath(joinpath(@__DIR__, "..", "..", ".."))
# Override with env PIPELINES="maximal_nocensor" to build a control pipeline in isolation.
const PIPELINES  = haskey(ENV, "PIPELINES") ? String.(split(ENV["PIPELINES"], ",")) : ["detrend", "maximal"]
const METRICS    = [("auc", 1)]   # name => index into acw_results (ACW-50 removed — defective metric)
const ATLAS      = length(ARGS) >= 1 ? ARGS[1] : "qinspheres"   # qinspheres | qinparcels
const ACW_DIR    = ATLAS == "qinspheres" ? joinpath("acw", "spheres") :
                                           joinpath("acw", "parcels", "self_regions")
const PARTS_TSV  = joinpath(REPO_ROOT, "participants.tsv")
const CATEGORIES = ["intero", "extero", "mental", "auditory", "motor", "visual"]
const SUBJECTS   = ["sub-$(lpad(i,2,'0'))" for i in 1:40 if !(i in (6,8,12,26,36))]   # n=35 (get_included_subjects)
const SESSIONS   = ["ses-01", "ses-02"]

parts = CSV.read(PARTS_TSV, DataFrame; delim = '\t')
drug  = Dict(string(r.participant_id) => string(r.condition) for r in eachrow(parts))

Row = NamedTuple{(:subject,:session,:drug_group,:category,:roi_id,:auc),
                 Tuple{String,String,String,String,String,Float64}}

# One shape for every metric: results/{metric_dir}/{atlas_dir}/{pipeline}/tables
# (metric "auc" lives under "acw" to match 03_intrinsic_neural_metrics/results).
const ATLAS_DIR = ATLAS == "qinspheres" ? "spheres" : joinpath("parcels", "self_regions")
metric_dir(metric, pipeline) = joinpath(
    REPO_ROOT, "04_statistics", "results",
    metric == "auc" ? "acw" : metric, ATLAS_DIR, pipeline, "tables")

function build(metric, midx, pipeline)
    jld2_dir = joinpath(REPO_ROOT, "03_intrinsic_neural_metrics", "results", ACW_DIR, pipeline)
    out_dir  = metric_dir(metric, pipeline)
    out_csv  = joinpath(out_dir, "$(metric).csv")

    rows = Row[]; missing_n = 0
    for cat in CATEGORIES, subj in SUBJECTS, ses in SESSIONS
        p = joinpath(jld2_dir, cat, "$(subj)_$(ses).jld2")
        if !isfile(p); missing_n += 1; continue; end
        d = load(p)
        ids = d["parcel_ids"]
        ar  = d["acw_results"]
        # IntrinsicTimescales.jl unwraps to a flat Vector when a single acwtype is asked for
        # ([:auc], the current setting) and nests when several are. The detrend/maximal JLD2s
        # predate the narrowing and are nested ([1]=AUC, [2]=ACW-50); anything recomputed since
        # is flat. Accept both — indexing [midx] blindly silently yields ROI 1's scalar.
        vals = if ar isa AbstractVector{<:Real}
            midx == 1 || error("$p: flat acw_results holds only AUC; cannot take metric index $midx")
            collect(ar)
        else
            collect(ar[midx])
        end
        length(vals) == length(ids) ||
            error("$p: $(length(vals)) ACW values for $(length(ids)) ROIs")
        for (rid, a) in zip(ids, vals)
            push!(rows, (subject=subj, session=ses, drug_group=get(drug, subj, "unknown"),
                         category=cat, roi_id=String(rid), auc=Float64(a)))
        end
    end
    df = DataFrame(rows)
    n0 = nrow(df)
    # Drop degenerate non-positive AUC (broadband ACF collapse — only affects the
    # no-bandpass detrend pipeline, concentrated in motor; maximal has none).
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
