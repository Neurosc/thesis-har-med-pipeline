# 23_build_g1_316_dataframe.jl
# Build G1 model DataFrame with ALL 316 nonself Glasser parcels (no tSNR exclusion).
# Called automatically by 23_lmm_g1_316_parcels.R if the output CSV is missing.
#
# This is a stripped version of 08_build_analysis_dataframe_auc.jl:
#   - nonself only (no self atlas)
#   - tSNR parcel exclusion REMOVED (keep all 316)
#   - AUC NaN observations still dropped
#   - G1 values merged and G1_z computed
#
# Run from repo root:
#   julia 04_statistics/scripts/23_build_g1_316_dataframe.jl

using JLD2, DataFrames, CSV, Statistics, Printf

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT   = normpath(joinpath(@__DIR__, "..", ".."))
const ATLAS_FILE  = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                             "glasser_coordinates_nonself_clean_1mm.txt")
const G1_FILE     = joinpath(REPO_ROOT, "g1_values.txt")
const LABEL_FILE  = joinpath(REPO_ROOT, "glasser_label_order.txt")
const PARTICIPANTS = joinpath(REPO_ROOT, "participants.tsv")
const ACW_BASE    = joinpath(REPO_ROOT, "03_acw_analysis", "results", "acw")
const VERSION     = "denoisedNoGSR"
const OUT_CSV     = joinpath(REPO_ROOT, "04_statistics", "results",
                             "nonself_g1_model_ready_316.csv")

const SUBJECTS = [
    "sub-01","sub-02","sub-03","sub-04","sub-05",
    "sub-07",
    "sub-09","sub-10","sub-11",
    "sub-13","sub-14","sub-15","sub-16","sub-17","sub-18",
    "sub-19","sub-20","sub-21","sub-22","sub-23","sub-24","sub-25",
    "sub-27","sub-28","sub-29","sub-30","sub-31","sub-32","sub-33",
    "sub-34","sub-35",
    "sub-37","sub-38","sub-39","sub-40"
]
const SESSIONS = ["ses-01","ses-02"]
const AUC_IDX  = 1   # acw_results[1] = AUC

# ── Idempotent ────────────────────────────────────────────────────────────────
if isfile(OUT_CSV)
    println("[skip] $(OUT_CSV) already exists.")
    exit()
end

SEP = "=" ^ 70
println(SEP)
println("23_build_g1_316_dataframe.jl")
println(SEP, "\n")

# ── Validate inputs ───────────────────────────────────────────────────────────
for f in [ATLAS_FILE, G1_FILE, LABEL_FILE, PARTICIPANTS]
    isfile(f) || error("Missing required input: $f")
end

# ── Atlas: Glasser ID → nonself position and ROI name ─────────────────────────
println("Loading atlas file...")
atlas_df = CSV.read(ATLAS_FILE, DataFrame; delim='\t')
nrow(atlas_df) == 316 || error("Expected 316 atlas rows, got $(nrow(atlas_df))")

# ROI_Number in the atlas file is the Glasser row index (1-360, verified in script 15)
# Row position in file (1-indexed) = nonself_pos_id
glasser_to_pos = Dict{Int,Int}()
pos_to_name    = Dict{Int,String}()
for (pos, row) in enumerate(eachrow(atlas_df))
    glasser_to_pos[row.ROI_Number] = pos
    pos_to_name[pos]               = row.ROI_Name
end
@printf("  Atlas mapping: %d Glasser IDs → nonself positions\n\n", length(glasser_to_pos))

# ── G1 values: parse label order file and g1_values.txt ───────────────────────
println("Loading G1 values...")
raw_lines     = [strip(ln) for ln in readlines(LABEL_FILE) if !isempty(strip(ln))]
glasser_names = [raw_lines[i] for i in 1:2:length(raw_lines)]
length(glasser_names) == 360 ||
    error("Expected 360 parcel names from label file, got $(length(glasser_names))")
glasser_names[1] == "R_V1_ROI" ||
    error("First parcel name = $(glasser_names[1]); expected R_V1_ROI — check label file")

g1_raw = CSV.read(G1_FILE, DataFrame; header=false, delim='\t')
g1_vec = Float64.(g1_raw[:, 1])
length(g1_vec) == 360 || error("Expected 360 G1 values, got $(length(g1_vec))")

name_to_g1 = Dict{String,Float64}(name => g1_vec[i] for (i, name) in enumerate(glasser_names))
@printf("  G1 range: [%.4f, %.4f]\n", minimum(g1_vec), maximum(g1_vec))

# Map nonself position → G1 (via ROI_Name)
pos_to_g1 = Dict{Int,Float64}()
for (pos, name) in pos_to_name
    haskey(name_to_g1, name) ||
        error("ROI name '$name' (pos $pos) not found in Glasser label order")
    pos_to_g1[pos] = name_to_g1[name]
end
@printf("  G1 mapped for %d nonself positions\n\n", length(pos_to_g1))

# ── Drug-group metadata ────────────────────────────────────────────────────────
part_df = CSV.read(PARTICIPANTS, DataFrame; delim='\t')
drug_group_map = Dict{String,String}(row.participant_id => row.condition
                                     for row in eachrow(part_df))
@printf("Drug-group metadata: %d participants\n\n", length(drug_group_map))

# ── Load JLD2 files and build long-format DataFrame ───────────────────────────
println("Loading $(length(SUBJECTS) * length(SESSIONS)) JLD2 files (nonself/$(VERSION))...")

rows = NamedTuple{(:subject, :session, :group, :roi_pos_id, :roi_name, :auc, :G1),
                  Tuple{String,String,String,Int,String,Float64,Float64}}[]

n_files      = 0
n_nan_drop   = 0
n_total_obs  = 0

for subject in SUBJECTS, session in SESSIONS
    global n_files, n_nan_drop, n_total_obs
    jld_path = joinpath(ACW_BASE, "nonself", VERSION, "$(subject)_$(session).jld2")
    if !isfile(jld_path)
        println("  MISSING: $jld_path")
        continue
    end
    n_files += 1

    data        = load(jld_path)
    acw_results = data["acw_results"]
    roi_columns = data["roi_columns"]   # Glasser ID strings for nonself
    auc_vals    = collect(acw_results[AUC_IDX])
    dg          = get(drug_group_map, subject, "unknown")

    for (col_name, auc) in zip(roi_columns, auc_vals)
        n_total_obs += 1
        glasser_id = parse(Int, col_name)
        pos_id     = glasser_to_pos[glasser_id]

        # NOTE: tSNR exclusion intentionally omitted — keeping all 316 parcels
        if isnan(auc)
            n_nan_drop += 1
            continue
        end

        push!(rows, (subject  = subject,
                     session  = session,
                     group    = dg,
                     roi_pos_id = pos_id,
                     roi_name = pos_to_name[pos_id],
                     auc      = Float64(auc),
                     G1       = pos_to_g1[pos_id]))
    end
end

@printf("JLD2 files processed: %d\n", n_files)
@printf("Total observations before NaN drop: %d (expected %d)\n",
        n_total_obs, length(SUBJECTS) * length(SESSIONS) * 316)
@printf("NaN AUC observations dropped: %d\n\n", n_nan_drop)

df = DataFrame(rows)

# ── G1_z: z-score across all observations ─────────────────────────────────────
μ_g1 = mean(df.G1)
σ_g1 = std(df.G1)
df.G1_z = (df.G1 .- μ_g1) ./ σ_g1
@printf("G1 z-score: mean = %.6f, SD = %.6f\n", μ_g1, σ_g1)
@printf("G1_z range: [%.4f, %.4f]\n\n", minimum(df.G1_z), maximum(df.G1_z))

# ── Summary ───────────────────────────────────────────────────────────────────
n_rois = length(unique(df.roi_pos_id))
@printf("Final DataFrame: %d rows × %d cols\n", nrow(df), ncol(df))
@printf("Unique ROIs    : %d (expected 316 or close)\n", n_rois)
@printf("Unique subjects: %d\n", length(unique(df.subject)))

# Verify drug-group balance
counts = combine(groupby(df, :group), :subject => (x -> length(unique(x))) => :n_subjects)
for row in eachrow(counts)
    @printf("  group=%-10s  %d subjects\n", row.group, row.n_subjects)
end

# ── Save ──────────────────────────────────────────────────────────────────────
mkpath(dirname(OUT_CSV))
CSV.write(OUT_CSV, df)
println("\nSaved: $OUT_CSV")
println("\nDone.")
