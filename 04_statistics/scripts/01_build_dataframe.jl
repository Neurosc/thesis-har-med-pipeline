# 01_build_dataframe.jl — Build two clean analysis DataFrames (denoisedNoGSR)
#
# Motivation
# ----------
# The previous Glasser analysis risked mixing SPHERE-extracted AUC values with
# GLASSER-PARCEL-extracted AUC values. This rewrite produces two DataFrames whose
# AUC provenance is unambiguous:
#
#   DataFrame 1  (glasser_full_dataframe.csv)   — every AUC comes from Glasser
#                PARCEL extraction. Self-layer parcels and nonself parcels alike.
#   DataFrame 2  (sphere_nonself_dataframe.csv) — self AUC comes from SPHERE
#                extraction (37 Qin spheres); nonself AUC comes from Glasser PARCEL
#                extraction restricted to sensory-motor networks.
#
# Important data-vs-spec reconciliation (see git history / commit message):
#   * The Glasser PARCEL atlas is indexed by CAB-NP IDs (L-first, 1-360). The
#     nonself parcel extraction (parcels_NoGSR/nonself) contains 320 parcels with
#     the 40 Keskin self-parcels REMOVED. Therefore the self-layer AUC cannot be
#     recovered by "relabelling parcels inside the nonself JLD2" — the self parcels
#     are not there. They live in the per-category parcel extractions
#     (parcels_NoGSR/{interoceptive,exteroceptive,mental}), which ARE Glasser parcel
#     extractions and already encode the Keskin layer assignment (incl. the two
#     cross-layer shared parcels: 111 L_AVI = Intero+Mental, 258 R_6r = Extero+Mental).
#     DataFrame 1 is therefore built from per-category self JLD2s + nonself JLD2.
#   * Drug group is taken from participants.tsv `condition` (placebo|verum).
#
# Run from repo root (do NOT run during Windows dev — server only):
#   julia 04_statistics/scripts/01_build_dataframe.jl

using JLD2, DataFrames, CSV, Statistics, Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, "..", ".."))

# ── Fixed configuration (this pipeline is denoisedNoGSR-specific) ───────────────
const DENOISING     = "NoGSR"
const PARCELS_BASE  = joinpath(REPO_ROOT, "03_acw_analysis", "results", "parcels_$(DENOISING)")
const SPHERES_BASE  = joinpath(REPO_ROOT, "03_acw_analysis", "results", "spheres_$(DENOISING)")
const OUT_DIR       = joinpath(REPO_ROOT, "04_statistics", "results")

const PARTS_TSV  = joinpath(REPO_ROOT, "participants.tsv")
const LABEL_KEY  = joinpath(REPO_ROOT, "02_timeseries_extraction", "data",
                            "CortexSubcortex_ColeAnticevic_NetPartition_wSubcorGSR_parcels_LR_LabelKey.txt")
const SELF_COORDS = joinpath(REPO_ROOT, "_old", "Thesis", "01_atlases", "self_coordinates.txt")
const SELF_META   = joinpath(REPO_ROOT, "02_timeseries_extraction", "results", "atlases",
                             "glasser_self_metadata.tsv")

# 35 included subjects (excluded: sub-06, sub-08, sub-12, sub-26, sub-36)
const SUBJECTS = [
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-05",
    "sub-07",
    "sub-09", "sub-10", "sub-11",
    "sub-13", "sub-14", "sub-15", "sub-16", "sub-17", "sub-18",
    "sub-19", "sub-20", "sub-21", "sub-22", "sub-23", "sub-24", "sub-25",
    "sub-27", "sub-28", "sub-29", "sub-30", "sub-31", "sub-32", "sub-33",
    "sub-34", "sub-35",
    "sub-37", "sub-38", "sub-39", "sub-40",
]
const SESSIONS = ["ses-01", "ses-02"]
const AUC_IDX  = 1   # acw_results[1] = AUC; acw_results[2] = τ

# Per-category self parcel dirs → analysis layer label
const CAT_DIRS = [
    ("interoceptive", "Interoception"),
    ("exteroceptive", "Exteroception"),
    ("mental",        "Cognition"),
]

# CAB-NP networks treated as sensory-motor (DataFrame 2 nonself selection)
const SENSORY_MOTOR_NETS = Set(["Visual1", "Visual2", "Somatomotor", "Auditory"])

# ════════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════════

# Load AUC for one atlas subdir across all subjects/sessions.
# Returns (rows, missing) where rows :: Vector of (subject, session, pid::Int, auc::Float64).
function load_dir_auc(base::AbstractString, subdir::AbstractString,
                      subjects::Vector{String}, sessions::Vector{String})
    rows = NamedTuple{(:subject, :session, :pid, :auc),
                      Tuple{String,String,Int,Float64}}[]
    missing_files = String[]
    for subj in subjects, ses in sessions
        p = joinpath(base, subdir, "$(subj)_$(ses).jld2")
        if !isfile(p)
            push!(missing_files, p)
            continue
        end
        d        = load(p)
        auc_vals = collect(d["acw_results"][AUC_IDX])
        pids     = d["parcel_ids"]
        for (pid, a) in zip(pids, auc_vals)
            push!(rows, (subject = subj, session = ses,
                         pid = parse(Int, String(pid)), auc = Float64(a)))
        end
    end
    return rows, missing_files
end

# Parse the comma-separated Keskin `categories` field into analysis layer labels.
function categories_to_layers(cats::AbstractString)
    out = String[]
    for c in split(cats, ",")
        cc = strip(c)
        cc == "Interoceptive"             && push!(out, "Interoception")
        cc == "Exteroceptive"             && push!(out, "Exteroception")
        (cc == "Mental Self" || cc == "Mental") && push!(out, "Cognition")
    end
    return out
end

# Drop NaN / Inf AUC rows; print how many were removed.
function drop_nonfinite!(df::DataFrame, label::AbstractString)
    n0   = nrow(df)
    mask = isfinite.(df.auc)
    df   = df[mask, :]
    @printf("  %s: dropped %d non-finite AUC rows (%d → %d)\n",
            label, n0 - sum(mask), n0, nrow(df))
    return df
end

# Shared sanity checks. `allow_layer_shared_parcels` documents (does not suppress)
# the DataFrame-1 case where one parcel legitimately appears in two layers.
function sanity_checks(df::DataFrame, name::AbstractString)
    println("\n── Sanity checks: $name ──")

    # 1. Every subject has both sessions.
    bad_sessions = String[]
    for subj in unique(df.subject)
        ns = length(unique(df[df.subject .== subj, :session]))
        ns == length(SESSIONS) || push!(bad_sessions, "$subj ($ns ses)")
    end
    if isempty(bad_sessions)
        @printf("  [OK] all %d subjects have both sessions\n", length(unique(df.subject)))
    else
        @printf("  [WARN] subjects without both sessions: %s\n", join(bad_sessions, ", "))
    end

    # 2. Drug group fixed within subject.
    bad_drug = String[]
    for subj in unique(df.subject)
        length(unique(df[df.subject .== subj, :drug_group])) == 1 || push!(bad_drug, subj)
    end
    isempty(bad_drug) ? println("  [OK] drug_group constant within every subject") :
                        @printf("  [WARN] drug_group varies within: %s\n", join(bad_drug, ", "))

    # 3. Duplicate (subject, session, roi_pos_id, self_layer, atlas_source).
    #    (atlas_source added so a sphere id and a parcel id that collide are not
    #     flagged as duplicates — they are genuinely different ROIs.)
    g = combine(groupby(df, [:subject, :session, :roi_pos_id, :self_layer, :atlas_source]),
                nrow => :n)
    ndup = sum(g.n .> 1)
    ndup == 0 ? println("  [OK] no duplicate (subject, session, roi_pos_id, self_layer, atlas_source)") :
                @printf("  [WARN] %d duplicate key combinations\n", ndup)

    # 4. AUC range.
    nz   = count(<=(0.0), df.auc)
    nbig = count(>(10.0), df.auc)
    @printf("  AUC range: min=%.4f  max=%.4f  | <=0: %d %s | >10: %d %s\n",
            minimum(df.auc), maximum(df.auc),
            nz,   nz   == 0 ? "[OK]" : "[CHECK]",
            nbig, nbig == 0 ? "[OK]" : "[CHECK]")

    # 5. Summary stats per self_layer.
    println("  AUC by self_layer (mean / median / SD):")
    for lyr in sort(unique(df.self_layer))
        v = df[df.self_layer .== lyr, :auc]
        @printf("    %-14s  n=%-6d  mean=%.4f  median=%.4f  sd=%.4f\n",
                lyr, length(v), mean(v), median(v), std(v))
    end
end

# ════════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════════
function main()
    mkpath(OUT_DIR)
    println("Denoising:    $DENOISING")
    println("Parcels base: $PARCELS_BASE")
    println("Spheres base: $SPHERES_BASE")
    println("Out dir:      $OUT_DIR\n")

    for f in (PARTS_TSV, LABEL_KEY, SELF_COORDS, SELF_META)
        isfile(f) || error("Missing required input: $f")
    end

    # ── Drug group (participants.tsv `condition`) ────────────────────────────────
    parts = CSV.read(PARTS_TSV, DataFrame; delim = '\t')
    drug_map = Dict{String,String}(string(r.participant_id) => string(r.condition)
                                   for r in eachrow(parts))
    @printf("Drug groups (participants.tsv `condition`): %d participants  [%s]\n",
            length(drug_map), join(sort(unique(values(drug_map))), ", "))
    dg(s) = get(drug_map, s, "unknown")

    # ── Glasser parcel metadata: name + network (CAB-NP label key) ───────────────
    lk = CSV.read(LABEL_KEY, DataFrame; delim = '\t')
    glasser_name = Dict{Int,String}(Int(r.INDEX) => string(r.GLASSERLABELNAME) for r in eachrow(lk))
    glasser_net  = Dict{Int,String}(Int(r.INDEX) => string(r.NETWORK)          for r in eachrow(lk))
    gname(pid)   = get(glasser_name, pid, "parcel_$(pid)")

    # ── Expected Keskin self parcels per layer (for the "not found" diagnostic) ──
    meta = CSV.read(SELF_META, DataFrame; delim = '\t')
    expected_self = Dict{String,Set{Int}}(
        "Interoception" => Set{Int}(), "Exteroception" => Set{Int}(), "Cognition" => Set{Int}())
    for r in eachrow(meta)
        for lyr in categories_to_layers(string(r.categories))
            push!(expected_self[lyr], Int(r.parcel_id))
        end
    end

    # ── Self sphere metadata: layer + name (Qin 2020 coordinates) ────────────────
    sc = CSV.read(SELF_COORDS, DataFrame; delim = '\t')
    sphere_layer = Dict{Int,String}(Int(r.ROI_Number) => string(r.Layer)    for r in eachrow(sc))
    sphere_name  = Dict{Int,String}(Int(r.ROI_Number) => string(r.ROI_Name) for r in eachrow(sc))

    # ════════════════════════════════════════════════════════════════════════════
    # DataFrame 1 — Glasser-only (all AUC from Glasser parcel extraction)
    # ════════════════════════════════════════════════════════════════════════════
    println("\n" * "="^78)
    println("DataFrame 1 — Glasser-only (parcel extraction)")
    println("="^78)

    Row = NamedTuple{(:subject, :session, :drug_group, :roi_pos_id, :roi_name,
                      :auc, :self_layer, :atlas_source),
                     Tuple{String,String,String,Int,String,Float64,String,String}}
    df1_rows = Row[]
    found_self = Dict{String,Set{Int}}(
        "Interoception" => Set{Int}(), "Exteroception" => Set{Int}(), "Cognition" => Set{Int}())

    # Self-layer rows (per-category parcel extractions).
    for (subdir, layer) in CAT_DIRS
        rows, missing_files = load_dir_auc(PARCELS_BASE, subdir, SUBJECTS, SESSIONS)
        isempty(missing_files) || @printf("  [%s] %d missing JLD2 files\n", subdir, length(missing_files))
        for r in rows
            push!(df1_rows, (subject = r.subject, session = r.session, drug_group = dg(r.subject),
                             roi_pos_id = r.pid, roi_name = gname(r.pid), auc = r.auc,
                             self_layer = layer, atlas_source = "glasser"))
            push!(found_self[layer], r.pid)
        end
    end

    # Nonself rows (Glasser parcel extraction, self parcels already excluded).
    ns_rows, ns_missing = load_dir_auc(PARCELS_BASE, "nonself", SUBJECTS, SESSIONS)
    isempty(ns_missing) || @printf("  [nonself] %d missing JLD2 files\n", length(ns_missing))
    for r in ns_rows
        push!(df1_rows, (subject = r.subject, session = r.session, drug_group = dg(r.subject),
                         roi_pos_id = r.pid, roi_name = gname(r.pid), auc = r.auc,
                         self_layer = "nonself", atlas_source = "glasser"))
    end

    df1 = DataFrame(df1_rows)
    println("\nDropping non-finite AUC:")
    df1 = drop_nonfinite!(df1, "DataFrame 1")

    # Keskin self-parcel coverage diagnostic.
    println("\nKeskin self-parcel coverage (expected vs found in parcel extraction):")
    for lyr in ("Interoception", "Exteroception", "Cognition")
        missing_p = sort(collect(setdiff(expected_self[lyr], found_self[lyr])))
        extra_p   = sort(collect(setdiff(found_self[lyr], expected_self[lyr])))
        @printf("  %-14s  expected=%d  found=%d\n", lyr, length(expected_self[lyr]), length(found_self[lyr]))
        isempty(missing_p) || @printf("     [FLAG] expected but NOT found: %s\n", join(missing_p, ", "))
        isempty(extra_p)   || @printf("     [FLAG] found but NOT in Keskin: %s\n", join(extra_p, ", "))
    end
    println("  (Self parcels are absent from the nonself extraction by design; this is expected.)")

    # Save.
    out1 = joinpath(OUT_DIR, "glasser_full_dataframe.csv")
    CSV.write(out1, df1)
    @printf("\nSaved: %s  (%d rows)\n", out1, nrow(df1))

    # Report.
    println("\nDataFrame 1 report:")
    @printf("  Total rows: %d\n", nrow(df1))
    @printf("  Subjects: %d   Sessions: %d\n",
            length(unique(df1.subject)), length(unique(df1.session)))
    println("  Rows / unique ROIs per self_layer:")
    for lyr in ("nonself", "Interoception", "Exteroception", "Cognition")
        sub = df1[df1.self_layer .== lyr, :]
        @printf("    %-14s  rows=%-7d  unique ROIs=%d\n",
                lyr, nrow(sub), length(unique(sub.roi_pos_id)))
    end

    sanity_checks(df1, "glasser_full_dataframe")

    # ════════════════════════════════════════════════════════════════════════════
    # DataFrame 2 — Sphere self + Glasser (sensory-motor) nonself
    # ════════════════════════════════════════════════════════════════════════════
    println("\n" * "="^78)
    println("DataFrame 2 — Sphere self + Glasser sensory-motor nonself")
    println("="^78)

    df2_rows = Row[]

    # Self spheres (37 Qin ROIs).
    self_rows, self_missing = load_dir_auc(SPHERES_BASE, "self", SUBJECTS, SESSIONS)
    isempty(self_missing) || @printf("  [sphere self] %d missing JLD2 files\n", length(self_missing))
    for r in self_rows
        push!(df2_rows, (subject = r.subject, session = r.session, drug_group = dg(r.subject),
                         roi_pos_id = r.pid, roi_name = get(sphere_name, r.pid, "sphere_$(r.pid)"),
                         auc = r.auc, self_layer = get(sphere_layer, r.pid, "unknown"),
                         atlas_source = "sphere"))
    end

    # Nonself Glasser parcels restricted to sensory-motor networks.
    nonself_rows, nonself_missing = load_dir_auc(PARCELS_BASE, "nonself", SUBJECTS, SESSIONS)
    isempty(nonself_missing) || @printf("  [parcel nonself] %d missing JLD2 files\n", length(nonself_missing))
    sm_ids = Set{Int}()
    for r in nonself_rows
        if get(glasser_net, r.pid, "") in SENSORY_MOTOR_NETS
            push!(df2_rows, (subject = r.subject, session = r.session, drug_group = dg(r.subject),
                             roi_pos_id = r.pid, roi_name = gname(r.pid), auc = r.auc,
                             self_layer = "nonself", atlas_source = "parcel"))
            push!(sm_ids, r.pid)
        end
    end
    @printf("  Sensory-motor nonself parcels selected: %d unique ROIs\n", length(sm_ids))

    df2 = DataFrame(df2_rows)
    println("\nDropping non-finite AUC:")
    df2 = drop_nonfinite!(df2, "DataFrame 2")

    out2 = joinpath(OUT_DIR, "sphere_nonself_dataframe.csv")
    CSV.write(out2, df2)
    @printf("\nSaved: %s  (%d rows)\n", out2, nrow(df2))

    # Report.
    println("\nDataFrame 2 report:")
    @printf("  Total rows: %d\n", nrow(df2))
    @printf("  Subjects: %d   Sessions: %d\n",
            length(unique(df2.subject)), length(unique(df2.session)))
    println("  Rows per self_layer:")
    for lyr in sort(unique(df2.self_layer))
        @printf("    %-14s  rows=%d\n", lyr, nrow(df2[df2.self_layer .== lyr, :]))
    end
    println("  Rows per atlas_source:")
    for src in sort(unique(df2.atlas_source))
        sub = df2[df2.atlas_source .== src, :]
        @printf("    %-8s  rows=%-7d  unique ROIs=%d\n", src, nrow(sub), length(unique(sub.roi_pos_id)))
    end
    @printf("  Unique ROIs overall (atlas_source × roi_pos_id): %d\n",
            nrow(unique(df2[:, [:atlas_source, :roi_pos_id]])))

    sanity_checks(df2, "sphere_nonself_dataframe")

    println("\nDone.")
end

main()
