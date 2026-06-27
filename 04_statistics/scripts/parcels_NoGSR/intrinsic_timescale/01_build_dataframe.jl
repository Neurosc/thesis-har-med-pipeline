# 01_build_dataframe.jl — Build the Glasser-parcel analysis DataFrame (parcels, NoGSR).
# Loads per-category self (interoceptive/exteroceptive/mental) + nonself ACW-AUC JLD2s,
# attaches drug group and parcel names, drops non-finite AUC, and writes
# glasser_full_dataframe.csv.
# Run from repo root:  julia 04_statistics/scripts/parcels_NoGSR/intrinsic_timescale/01_build_dataframe.jl

using JLD2, DataFrames, CSV, Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", "..", "..", ".."))
const PARCELS   = joinpath(REPO_ROOT, "03_intrinsic_neural_metrics", "results", "parcels_NoGSR")
const OUT_DIR   = joinpath(REPO_ROOT, "04_statistics", "results", "parcels_NoGSR", "int_tables")
const PARTS_TSV = joinpath(REPO_ROOT, "participants.tsv")
const LABEL_KEY = joinpath(REPO_ROOT, "02_timeseries_extraction", "data",
                           "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt")

# Nonself restricted to these 4 CAB-NP networks (~99 parcels, Glasser IDs 1-360).
# Matches the somatomotor/Sensory-Motor reference used in the old 02_lmm.R pipeline.
const SOMATOMOTOR_NETS = Set(["Visual1", "Visual2", "Somatomotor", "Auditory"])

# 39 included subjects (excluded: 12)
const SUBJECTS = ["sub-$(lpad(i, 2, '0'))" for i in 1:40 if !(i in (12,))]
const SESSIONS = ["ses-01", "ses-02"]
const AUC_IDX  = 1   # acw_results[1] = AUC, [2] = τ

# parcel subdir => analysis layer (nonself extraction already excludes the self parcels)
const LAYERS = [("interoceptive", "Interoception"), ("exteroceptive", "Exteroception"),
                ("mental", "Cognition"), ("nonself", "nonself")]

# Load (subject, session, parcel_id, auc) rows from one atlas subdir across all runs.
function load_auc(subdir)
    rows = NamedTuple{(:subject, :session, :pid, :auc), Tuple{String,String,Int,Float64}}[]
    missing_n = 0
    for subj in SUBJECTS, ses in SESSIONS
        p = joinpath(PARCELS, subdir, "$(subj)_$(ses).jld2")
        isfile(p) || (missing_n += 1; continue)
        d = load(p)
        for (pid, a) in zip(d["parcel_ids"], collect(d["acw_results"][AUC_IDX]))
            push!(rows, (subject = subj, session = ses, pid = parse(Int, String(pid)), auc = Float64(a)))
        end
    end
    missing_n > 0 && @printf("  [%s] %d missing JLD2 files\n", subdir, missing_n)
    return rows
end

function main()
    mkpath(OUT_DIR)
    for f in (PARTS_TSV, LABEL_KEY)
        isfile(f) || error("Missing required input: $f")
    end

    # drug group (participants.tsv `condition`) + Glasser parcel names (CAB-NP label key)
    parts = CSV.read(PARTS_TSV, DataFrame; delim = '\t')
    drug  = Dict(string(r.participant_id) => string(r.condition) for r in eachrow(parts))
    lk    = CSV.read(LABEL_KEY, DataFrame; delim = '\t')
    pname = Dict(Int(r.INDEX) => string(r.GLASSERLABELNAME) for r in eachrow(lk))
    gname(pid) = get(pname, pid, "parcel_$(pid)")

    # Somatomotor nonself IDs: Visual1 + Visual2 + Somatomotor + Auditory, Glasser 1-360
    sm_ids = Set{Int}(
        Int(r.INDEX) for r in eachrow(lk)
        if 1 <= Int(r.INDEX) <= 360 && string(r.NETWORK) in SOMATOMOTOR_NETS
    )
    @printf("Somatomotor nonself IDs (CAB-NP %s): %d parcels\n",
            join(sort(collect(SOMATOMOTOR_NETS)), "/"), length(sm_ids))

    Row = NamedTuple{(:subject, :session, :drug_group, :roi_pos_id, :roi_name, :auc, :self_layer, :atlas_source),
                     Tuple{String,String,String,Int,String,Float64,String,String}}
    rows = Row[]
    for (subdir, layer) in LAYERS, r in load_auc(subdir)
        layer == "nonself" && !(r.pid in sm_ids) && continue
        push!(rows, Row((r.subject, r.session, get(drug, r.subject, "unknown"),
                         r.pid, gname(r.pid), r.auc, layer, "glasser")))
    end
    df = DataFrame(rows)

    for layer in ["nonself", "Interoception", "Exteroception", "Cognition"]
        n = length(unique(df.roi_pos_id[df.self_layer .== layer]))
        @printf("  %-15s %d unique parcels\n", layer, n)
    end
    ns_count = length(unique(df.roi_pos_id[df.self_layer .== "nonself"]))
    (ns_count < 90 || ns_count > 110) &&
        error("Somatomotor nonself = $ns_count parcels, expected 90-110 — CAB-NP join likely failed")

    n0 = nrow(df); df = df[isfinite.(df.auc), :]
    @printf("Dropped %d non-finite AUC rows (tSNR-excluded parcels arrive as NaN) (%d → %d)\n",
            n0 - nrow(df), n0, nrow(df))

    out = joinpath(OUT_DIR, "glasser_full_dataframe.csv")
    CSV.write(out, df)
    @printf("Saved: %s (%d rows)\n", out, nrow(df))
end

main()
