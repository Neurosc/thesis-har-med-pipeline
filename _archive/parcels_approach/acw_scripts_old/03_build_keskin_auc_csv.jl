# 03_build_keskin_auc_csv.jl
# Extract AUC from Keskin JLD2 files → long-format model-ready CSV.
# Handles the two duplicate parcels (78: Ext+Cog, 291: Int+Cog) by
# creating two rows per observation for those parcels.
#
# Output: 04_statistics/results/keskin_auc_model_ready.csv
# Columns: subject, session, group, glasser_id, roi_name, auc, G1, layer
#
# Run from repo root:
#   julia 03_acw_analysis/scripts/03_build_keskin_auc_csv.jl

using JLD2, CSV, DataFrames, Statistics, Printf

const REPO_ROOT   = normpath(joinpath(@__DIR__, "..", ".."))
const JLD2_BASE   = joinpath(REPO_ROOT, "03_acw_analysis", "results", "acw",
                             "keskin", "denoisedNoGSR")
const COORD_FILE  = joinpath(REPO_ROOT, "02_timeseries_extraction", "results",
                             "atlases", "keskin_self_coordinates.txt")
const G1_FILE     = joinpath(REPO_ROOT, "g1_values.txt")
const LABEL_FILE  = joinpath(REPO_ROOT, "glasser_label_order.txt")
const PARTICIPANTS = joinpath(REPO_ROOT, "participants.tsv")
const OUT_CSV     = joinpath(REPO_ROOT, "04_statistics", "results",
                             "keskin_auc_model_ready.csv")

const AUC_IDX = 1   # acw_results[1] = AUC

const SUBJECTS = [
    "sub-01","sub-02","sub-03","sub-04","sub-05",
    "sub-07","sub-09","sub-10","sub-11",
    "sub-13","sub-14","sub-15","sub-16","sub-17","sub-18",
    "sub-19","sub-20","sub-21","sub-22","sub-23","sub-24","sub-25",
    "sub-27","sub-28","sub-29","sub-30","sub-31","sub-32","sub-33",
    "sub-34","sub-35","sub-37","sub-38","sub-39","sub-40"
]
const SESSIONS = ["ses-01","ses-02"]

SEP = "=" ^ 70
println(SEP); println("03_build_keskin_auc_csv.jl"); println(SEP, "\n")

isfile(OUT_CSV) && (println("[skip] $OUT_CSV exists"); exit())

# ── Load coordinate / layer mapping ───────────────────────────────────────────
println("Loading coordinate file...")
coord_df = CSV.read(COORD_FILE, DataFrame; delim='\t')
@printf("  %d unique Keskin parcels\n", nrow(coord_df))

# Build: glasser_id → (roi_name, [layers])
# Layer column may be "Exteroception + Cognition" for duplicates
id_to_name  = Dict{Int,String}()
id_to_layers = Dict{Int,Vector{String}}()
for row in eachrow(coord_df)
    gid = row.ROI_Number
    id_to_name[gid] = row.ROI_Name
    # Parse layers (may be "Exteroception + Cognition")
    layers = [strip(l) for l in split(row.Layer, "+")]
    id_to_layers[gid] = layers
end
n_duplicates = sum(length(v) > 1 for v in values(id_to_layers))
@printf("  Parcels in multiple layers: %d\n\n", n_duplicates)

# ── G1 values: Glasser ID is the direct 1-based index into g1_values.txt ──────
println("Loading G1 values...")
g1_raw = CSV.read(G1_FILE, DataFrame; header=false, delim='\t')
g1_vec = Float64.(g1_raw[:, 1])
length(g1_vec) == 360 || error("Expected 360 G1 values, got $(length(g1_vec))")

id_to_g1 = Dict{Int,Float64}()
for gid in keys(id_to_name)
    (1 <= gid <= 360) || error("Glasser ID $gid out of range 1-360")
    id_to_g1[gid] = g1_vec[gid]   # direct 1-based index — no name lookup needed
end
@printf("  G1 mapped for %d Keskin parcels\n", length(id_to_g1))
@printf("  G1 range: [%.4f, %.4f]\n\n",
        minimum(values(id_to_g1)), maximum(values(id_to_g1)))

# ── Drug-group metadata ────────────────────────────────────────────────────────
part_df = CSV.read(PARTICIPANTS, DataFrame; delim='\t')
drug_map = Dict{String,String}(row.participant_id => row.condition
                                for row in eachrow(part_df))

# ── Extract AUC from JLD2 files ───────────────────────────────────────────────
println("Reading JLD2 files and extracting AUC...")

rows = NamedTuple{(:subject, :session, :group, :glasser_id, :roi_name, :auc, :G1, :layer),
                  Tuple{String,String,String,Int,String,Float64,Float64,String}}[]

n_files   = 0
n_nan     = 0
n_dup_obs = 0

for subject in SUBJECTS, session in SESSIONS
    global n_files, n_nan, n_dup_obs
    jld_path = joinpath(JLD2_BASE, "$(subject)_$(session).jld2")
    isfile(jld_path) || (println("MISSING: $jld_path"); continue)
    n_files += 1

    data        = load(jld_path)
    acw_results = data["acw_results"]
    roi_columns = data["roi_columns"]          # Glasser ID strings
    auc_vals    = collect(acw_results[AUC_IDX])
    dg          = get(drug_map, subject, "unknown")

    for (col_name, auc) in zip(roi_columns, auc_vals)
        gid = parse(Int, col_name)
        if isnan(auc)
            n_nan += 1
            continue
        end
        g1      = id_to_g1[gid]
        rname   = id_to_name[gid]
        layers  = id_to_layers[gid]

        for lyr in layers
            push!(rows, (subject=subject, session=session, group=dg,
                         glasser_id=gid, roi_name=rname,
                         auc=Float64(auc), G1=g1, layer=lyr))
        end
        if length(layers) > 1
            n_dup_obs += 1
        end
    end
end

@printf("JLD2 files read  : %d\n", n_files)
@printf("NaN AUC dropped  : %d\n", n_nan)
@printf("Duplicate layer rows added: %d\n\n", n_dup_obs)

df = DataFrame(rows)
@printf("Final DataFrame: %d rows × %d cols\n", nrow(df), ncol(df))

# Expected: 35 × 2 × 35 layer-parcel combos = 2450 (minus NaN, plus dup rows)
n_combos = sum(length(v) for v in values(id_to_layers))
@printf("  (35 subjects × 2 sessions × %d layer-parcel combos = %d expected)\n\n",
        n_combos, 35 * 2 * n_combos)

# Print per-layer summary
for lyr in ["Interoception","Exteroception","Cognition"]
    sub = df[df.layer .== lyr, :]
    @printf("  %-15s: %d rows, %d unique parcels\n",
            lyr, nrow(sub), length(unique(sub.glasser_id)))
end

mkpath(dirname(OUT_CSV))
CSV.write(OUT_CSV, df)
println("\nSaved: $OUT_CSV")
println("\nDone.")
