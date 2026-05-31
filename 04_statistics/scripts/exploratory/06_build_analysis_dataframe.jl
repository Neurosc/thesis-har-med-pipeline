# 06_build_analysis_dataframe.jl — Step 7: long-format DataFrame with tSNR exclusion
#
# Builds the analysis-ready long-format DataFrame for the mixed-effects model.
# Low-tSNR parcels (tSNR < 30, nonself only) are excluded a priori.
#
# Run from repo root:  julia 04_statistics/scripts/06_build_analysis_dataframe.jl
# Idempotent: skips if both outputs already exist.

using JLD2, DataFrames, CSV

# ── Paths ─────────────────────────────────────────────────────────────────────
const REPO_ROOT      = normpath(joinpath(@__DIR__, "..", ".."))
const ATLAS_FILE     = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases",
                                "glasser_coordinates_nonself_clean_1mm.txt")
const EXCL_FILE      = joinpath(REPO_ROOT, "excluded_rois_low_tsnr.tsv")
const SPIKE_FILE     = joinpath(REPO_ROOT, "04_statistics", "results",
                                "fig04_spikeA_roi_frequency.csv")
const PARTICIPANTS   = joinpath(REPO_ROOT, "participants.tsv")
const ACW_BASE       = joinpath(REPO_ROOT, "03_acw_analysis", "results", "acw")
const VERSION        = "denoisedNoGSR"
const RES_DIR        = joinpath(REPO_ROOT, "04_statistics", "results")
const OUT_LONG       = joinpath(RES_DIR, "analysis_long_format.csv")
const OUT_EXCL_TRANS = joinpath(RES_DIR, "excluded_rois_translated.csv")

# ── Included subjects (35; excluded: sub-06, sub-08, sub-12, sub-26, sub-36) ──
const SUBJECTS = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
    "sub-07",
    "sub-09", "sub-10", "sub-11",
    "sub-13", "sub-14", "sub-15", "sub-16", "sub-17", "sub-18",
    "sub-19", "sub-20", "sub-21", "sub-22", "sub-23", "sub-24", "sub-25",
    "sub-27", "sub-28", "sub-29", "sub-30", "sub-31", "sub-32", "sub-33",
    "sub-34", "sub-35",
    "sub-37", "sub-38", "sub-39", "sub-40"
]
const SESSIONS    = ["ses-01", "ses-02"]
const ATLASES     = ["self", "nonself"]
const TAU_IDX     = 2   # ACW_TYPES = [:auc, :tau]; tau is index 2
const TOTAL_FILES = length(SUBJECTS) * length(SESSIONS) * length(ATLASES)   # 140

# ── Idempotency ────────────────────────────────────────────────────────────────
if isfile(OUT_LONG) && isfile(OUT_EXCL_TRANS)
    println("[skip] all outputs already exist; delete to rerun")
    exit()
end

mkpath(RES_DIR)

# ══════════════════════════════════════════════════════════════════════════════
# Part 1 — Build Glasser→position mapping
# Row order in the coordinate file defines position numbering:
#   row 1 → nonself position 1, row 2 → position 2, etc.
# Note: nonself col names in timeseries CSVs are Glasser IDs (with gaps where
# self-atlas overlap parcels were removed), so roi_columns in JLD2 holds Glasser
# IDs as strings, not sequential position numbers.
# ══════════════════════════════════════════════════════════════════════════════
println("═══ Part 1: Glasser→position mapping ═══")

atlas_df        = CSV.read(ATLAS_FILE, DataFrame; delim='\t')
glasser_to_pos  = Dict{Int,Int}()
pos_to_glasser  = Dict{Int,Int}()
pos_to_name     = Dict{Int,String}()

for (pos, row) in enumerate(eachrow(atlas_df))
    gid = row.ROI_Number
    glasser_to_pos[gid] = pos
    pos_to_glasser[pos] = gid
    pos_to_name[pos]    = row.ROI_Name
end

n_mapping = length(glasser_to_pos)
if n_mapping != 316
    error("Mapping has $n_mapping entries; expected 316 — STOP")
end
println("Mapping size: $n_mapping [expected 316: OK]")

# ══════════════════════════════════════════════════════════════════════════════
# Part 2 — Translate exclusion list (Glasser IDs → nonself position IDs)
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 2: Translate exclusion list ═══")

excl_df = CSV.read(EXCL_FILE, DataFrame; delim='\t')

# Verify every Glasser ID in the TSV is present in the nonself atlas
for row in eachrow(excl_df)
    if !haskey(glasser_to_pos, row.roi_id)
        error("Glasser ID $(row.roi_id) ($(row.roi_name)) not found in " *
              "nonself atlas mapping — exclusion list references a parcel " *
              "outside the nonself atlas — STOP")
    end
end

excl_translated = DataFrame(
    glasser_id     = excl_df.roi_id,
    roi_name       = excl_df.roi_name,
    tsnr           = excl_df.tsnr,
    nonself_pos_id = [glasser_to_pos[id] for id in excl_df.roi_id],
)
sort!(excl_translated, :nonself_pos_id)

excl_pos_set = Set(excl_translated.nonself_pos_id)
excl_pos_vec = sort(collect(excl_pos_set))
n_excl       = length(excl_pos_vec)

print("Translated exclusion list size: $n_excl [expected 58")
if n_excl != 58
    println("]")
    error("Exclusion list size is $n_excl, expected 58 — STOP")
end
println(": OK]")

# ══════════════════════════════════════════════════════════════════════════════
# Part 3 — Sanity check: Spike A ∩ exclusion set
# fig04_spikeA_roi_frequency.csv contains 46 Spike A ROIs in nonself POSITION
# numbering. We expect exactly 35 of them to be in the exclusion set.
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 3: Spike A overlap sanity check ═══")

spike_df      = CSV.read(SPIKE_FILE, DataFrame)
spike_pos_set = Set(spike_df.roi_id)
overlap_count = count(pos -> pos in excl_pos_set, spike_pos_set)

print("Spike A ROIs in exclusion set: $overlap_count / $(length(spike_pos_set))")
print(" [expected 35")
if overlap_count != 35
    println("]")
    error("Spike A overlap is $overlap_count, expected 35 — " *
          "Glasser↔position mapping is wrong; downstream filtering would " *
          "produce silently incorrect results — STOP")
end
println(": OK — mapping verified]")

# ══════════════════════════════════════════════════════════════════════════════
# Part 4 — Build long-format DataFrame
# ══════════════════════════════════════════════════════════════════════════════
println("\n═══ Part 4: Building long-format DataFrame ═══")

# Drug group metadata from participants.tsv (column: condition = verum/placebo)
drug_group_map = Dict{String,String}()
if isfile(PARTICIPANTS)
    part_df = CSV.read(PARTICIPANTS, DataFrame; delim='\t')
    for row in eachrow(part_df)
        drug_group_map[row.participant_id] = row.condition
    end
    println("Loaded drug_group for $(length(drug_group_map)) participants")
else
    println("WARNING: $(basename(PARTICIPANTS)) not found — " *
            "drug_group column will be populated with \"unknown\"")
end

rows = NamedTuple{(:subject, :session, :atlas, :roi_pos_id, :tau, :drug_group),
                  Tuple{String,String,String,Int,Float64,String}}[]

file_idx = 0

for subject in SUBJECTS, session in SESSIONS, atlas in ATLASES
    global file_idx
    file_idx += 1

    jld_path = joinpath(ACW_BASE, atlas, VERSION, "$(subject)_$(session).jld2")
    if !isfile(jld_path)
        println("[$file_idx/$TOTAL_FILES] MISSING: $jld_path")
        continue
    end

    data        = load(jld_path)
    acw_results = data["acw_results"]
    roi_columns = data["roi_columns"]   # nonself: Glasser ID strings; self: position ID strings
    tau_vals    = collect(acw_results[TAU_IDX])

    dg = get(drug_group_map, subject, "unknown")

    n_kept = 0
    for (col_name, tau) in zip(roi_columns, tau_vals)
        if atlas == "nonself"
            # roi_columns stores Glasser IDs; convert to nonself position
            glasser_id = parse(Int, col_name)
            pos_id     = glasser_to_pos[glasser_id]
            pos_id in excl_pos_set && continue   # tSNR exclusion
        else
            pos_id = parse(Int, col_name)
        end
        push!(rows, (subject=subject, session=session, atlas=atlas,
                     roi_pos_id=pos_id, tau=Float64(tau), drug_group=dg))
        n_kept += 1
    end

    println("[$file_idx/$TOTAL_FILES] $subject $session $atlas → $n_kept ROIs kept")
end

long_df = DataFrame(rows)

# ── Console summary ────────────────────────────────────────────────────────────
println("\n═══ DataFrame Summary ═══")
println("Total rows:  $(nrow(long_df))  [expected 20650]")
for at in ["self", "nonself"]
    n = count(==(at), long_df.atlas)
    exp = at == "self" ? 2590 : 18060
    println("  $at:  $n  [expected $exp]")
end
for sess in ["ses-01", "ses-02"]
    n = count(==(sess), long_df.session)
    println("  $sess:  $n")
end

# ── Write outputs ──────────────────────────────────────────────────────────────
CSV.write(OUT_LONG, long_df)
println("\nWrote: $OUT_LONG")

CSV.write(OUT_EXCL_TRANS, excl_translated)
println("Wrote: $OUT_EXCL_TRANS")
