"""
15_build_g1_model_dataframe.py

Build analysis-ready nonself DataFrame with G1 principal gradient for
mixed-effects modeling of drug × session × cortical hierarchy.

Run from repo root:
    python 04_statistics/scripts/15_build_g1_model_dataframe.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats

# ── Paths ──────────────────────────────────────────────────────────────────────
REPO_ROOT  = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))

AUC_CSV    = os.path.join(REPO_ROOT, "04_statistics", "results", "analysis_long_format_auc.csv")
COORD_FILE = os.path.join(REPO_ROOT, "_old", "Thesis", "01_atlases",
                          "glasser_coordinates_nonself_clean_1mm.txt")
G1_FILE    = os.path.join(REPO_ROOT, "g1_values.txt")
LABEL_FILE = os.path.join(REPO_ROOT, "glasser_label_order.txt")
OUT_MODEL  = os.path.join(REPO_ROOT, "04_statistics", "results", "nonself_g1_model_ready.csv")
OUT_LOOKUP = os.path.join(REPO_ROOT, "04_statistics", "results", "glasser_g1_lookup.csv")

# ── Idempotent guard ───────────────────────────────────────────────────────────
if os.path.exists(OUT_MODEL) and os.path.exists(OUT_LOOKUP):
    print("Outputs already exist — nothing to do. Delete to rerun:")
    print(f"  {OUT_MODEL}\n  {OUT_LOOKUP}")
    sys.exit(0)

SEP = "=" * 70
print(SEP)
print("15_build_g1_model_dataframe.py")
print(SEP)

# ── Step 1: Load long-format AUC data ─────────────────────────────────────────
print("\n─── Step 1: Load long-format AUC data ───")
df = pd.read_csv(AUC_CSV)
print(f"Shape  : {df.shape}")
print(f"Columns: {list(df.columns)}")
print(df.head())
if df.shape[0] != 20632:
    print(f"STOP: expected 20,632 rows, got {df.shape[0]}")
    sys.exit(1)
required_cols = {"subject", "session", "atlas", "roi_pos_id", "auc", "drug_group"}
if not required_cols.issubset(df.columns):
    print(f"STOP: missing columns — {required_cols - set(df.columns)}")
    sys.exit(1)
print("Step 1 OK")

# ── Step 2: Sanity-check structure ─────────────────────────────────────────────
print("\n─── Step 2: Sanity-check structure ───")
ok = True

# Every subject must have exactly 2 sessions
sess_per_subj = df.groupby("subject")["session"].nunique()
bad_sess = sess_per_subj[sess_per_subj != 2]
if len(bad_sess):
    print(f"STOP: subjects missing a session:\n{bad_sess.to_string()}")
    ok = False

# drug_group is fixed within subject (no cross-over contamination)
grp_per_subj = df.groupby("subject")["drug_group"].nunique()
multi_grp = grp_per_subj[grp_per_subj > 1]
if len(multi_grp):
    print(f"STOP: subjects appearing in multiple groups:\n{multi_grp.to_string()}")
    ok = False

# Group sizes
n_per_group = df.groupby("drug_group")["subject"].nunique()
print(f"Subjects per group:\n{n_per_group.to_string()}")
if n_per_group.get("placebo", 0) != 18:
    print(f"STOP: expected 18 placebo subjects, got {n_per_group.get('placebo', 0)}")
    ok = False
if n_per_group.get("verum", 0) != 17:
    print(f"STOP: expected 17 verum subjects, got {n_per_group.get('verum', 0)}")
    ok = False

# Atlas row counts
atlas_n = df.groupby("atlas").size()
print(f"Rows per atlas:\n{atlas_n.to_string()}")
self_n    = atlas_n.get("self", 0)
nonself_n = atlas_n.get("nonself", 0)
if self_n != 2590:
    print(f"STOP: expected 2,590 self rows, got {self_n}")
    ok = False
if abs(nonself_n - 18042) > 100:
    print(f"WARNING: expected ~18,042 nonself rows, got {nonself_n}")

if not ok:
    print("\nSTOP: sanity checks failed — see details above.")
    sys.exit(1)
print("Step 2 OK")

# ── Step 3: Subset to nonself atlas ───────────────────────────────────────────
print("\n─── Step 3: Subset to nonself atlas ───")
df_ns = df[df["atlas"] == "nonself"].copy()
n_rows = len(df_ns)
n_rois = df_ns["roi_pos_id"].nunique()
print(f"Nonself rows : {n_rows}")
print(f"Distinct ROIs: {n_rois}")
if n_rois != 258:
    print(f"STOP: expected 258 distinct nonself ROIs, got {n_rois}")
    sys.exit(1)
print("Step 3 OK")

# ── Step 4: Load G1 values ─────────────────────────────────────────────────────
print("\n─── Step 4: Load G1 values ───")
g1_raw = pd.read_csv(G1_FILE, sep="\t", header=None)
g1_vec = g1_raw.iloc[:, 0].values
if len(g1_vec) != 360:
    print(f"STOP: expected 360 G1 values, got {len(g1_vec)}")
    sys.exit(1)
print(f"G1 length: {len(g1_vec)}")
print(f"G1 min={g1_vec.min():.4f}  max={g1_vec.max():.4f}  mean={g1_vec.mean():.4f}")
print("Step 4 OK")

# ── Step 5: Build Glasser-to-G1 lookup table ──────────────────────────────────
print("\n─── Step 5: Build Glasser-to-G1 lookup table ───")

# a. Parse label order file — even-indexed lines are parcel names, odd lines are colour codes
with open(LABEL_FILE, "r") as fh:
    raw_lines = [ln.strip() for ln in fh if ln.strip()]
glasser_names = [raw_lines[i] for i in range(0, len(raw_lines), 2)]

if len(glasser_names) != 360:
    print(f"STOP: expected 360 Glasser parcel names, got {len(glasser_names)}")
    sys.exit(1)
print(f"Parsed {len(glasser_names)} Glasser parcel names")
print(f"  Row 1 : {glasser_names[0]}  (expected R_V1_ROI)")
print(f"  Row 2 : {glasser_names[1]}  (expected R_MST_ROI)")
if glasser_names[0] != "R_V1_ROI":
    print(f"STOP: first Glasser name is {glasser_names[0]!r}, expected R_V1_ROI")
    sys.exit(1)

# b–c. name → G1 value and name → 1-indexed Glasser row
name_to_g1  = {n: float(g1_vec[i]) for i, n in enumerate(glasser_names)}
name_to_row = {n: i + 1            for i, n in enumerate(glasser_names)}

# d–e. Load coordinate file; assign nonself_pos_id = 1-indexed row position
coords = pd.read_csv(COORD_FILE, sep="\t").reset_index(drop=True)
if len(coords) != 316:
    print(f"STOP: expected 316 nonself coordinate rows, got {len(coords)}")
    sys.exit(1)
if coords.iloc[0]["ROI_Name"] != "R_V1_ROI":
    print(f"STOP: first coord row is {coords.iloc[0]['ROI_Name']!r}, expected R_V1_ROI")
    sys.exit(1)

coords["nonself_pos_id"]    = coords.index + 1          # 1-indexed file position
coords["G1"]                = coords["ROI_Name"].map(name_to_g1)
coords["glasser_row_index"] = coords["ROI_Name"].map(name_to_row)

# f. Verify every name was found
n_missing_names = coords["G1"].isna().sum()
if n_missing_names > 0:
    bad = coords[coords["G1"].isna()]["ROI_Name"].tolist()
    print(f"STOP: {n_missing_names} ROI names not found in Glasser label order: {bad}")
    sys.exit(1)

# Cross-check: ROI_Number (Glasser standard number) must match glasser_row_index
mismatch = (coords["ROI_Number"] != coords["glasser_row_index"]).sum()
if mismatch > 0:
    print(f"STOP: {mismatch} rows have ROI_Number ≠ glasser_row_index — file ordering mismatch!")
    print(coords[coords["ROI_Number"] != coords["glasser_row_index"]][
        ["nonself_pos_id", "ROI_Number", "ROI_Name", "glasser_row_index"]].head(10).to_string())
    sys.exit(1)

lookup_df = (coords[["nonself_pos_id", "ROI_Name", "G1", "glasser_row_index"]]
             .rename(columns={"ROI_Name": "roi_name"})
             .copy())
print(f"Lookup built: {len(lookup_df)} entries, all with G1 values")
print("ROI_Number == glasser_row_index for all 316 rows: VERIFIED")
print("Step 5 OK")

# ── Step 6: Account for tSNR exclusions ───────────────────────────────────────
print("\n─── Step 6: Account for tSNR exclusions ───")
data_pos_ids   = set(df_ns["roi_pos_id"].unique())
lookup_pos_ids = set(lookup_df["nonself_pos_id"].values)
orphan_ids = data_pos_ids - lookup_pos_ids
if orphan_ids:
    print(f"STOP: {len(orphan_ids)} roi_pos_id values in data have no G1 entry: "
          f"{sorted(orphan_ids)}")
    sys.exit(1)
n_covered = len(data_pos_ids)
status = "OK" if n_covered == 258 else f"MISMATCH — expected 258"
print(f"{n_covered} of 258 nonself ROIs have G1 values: {status}")
if n_covered != 258:
    sys.exit(1)

# Keep only the 258 parcels present in the data (drop the 58 tSNR-excluded ones)
lookup_active = lookup_df[lookup_df["nonself_pos_id"].isin(data_pos_ids)].copy()
print(f"Active lookup entries: {len(lookup_active)} (expected 258)")
print("Step 6 OK")

# ── Step 7: Merge G1 onto nonself data ────────────────────────────────────────
print("\n─── Step 7: Merge G1 onto nonself data ───")
g1_merge = (lookup_df[["nonself_pos_id", "roi_name", "G1"]]
            .rename(columns={"nonself_pos_id": "roi_pos_id"}))
df_merged = df_ns.merge(g1_merge, on="roi_pos_id", how="left")

if len(df_merged) != len(df_ns):
    print(f"STOP: row count changed {len(df_ns)} → {len(df_merged)} after merge")
    sys.exit(1)
n_missing_g1 = df_merged["G1"].isna().sum()
if n_missing_g1 > 0:
    print(f"STOP: {n_missing_g1} rows have missing G1 after merge")
    sys.exit(1)
print(f"Rows after merge : {len(df_merged)} (unchanged)")
print(f"Missing G1 values: 0")
print("Step 7 OK")

# ── Step 8: Set factor variables and reference levels ──────────────────────────
print("\n─── Step 8: Categorical variables and reference levels ───")
df_merged = df_merged.rename(columns={"drug_group": "group"})

df_merged["group"]      = pd.Categorical(df_merged["group"],
                                          categories=["placebo", "verum"], ordered=False)
df_merged["session"]    = pd.Categorical(df_merged["session"],
                                          categories=["ses-01", "ses-02"], ordered=False)
df_merged["subject"]    = df_merged["subject"].astype("category")
df_merged["roi_pos_id"] = df_merged["roi_pos_id"].astype("category")

print(f"group      reference level : {df_merged['group'].cat.categories[0]!r}")
print(f"session    reference level : {df_merged['session'].cat.categories[0]!r}")
print(f"subject    categories      : {df_merged['subject'].nunique()} subjects")
print(f"roi_pos_id categories      : {df_merged['roi_pos_id'].nunique()} ROIs")
print("Step 8 OK")

# ── Step 9: Center and scale G1 ───────────────────────────────────────────────
print("\n─── Step 9: Center and scale G1 ───")
g1_mean = df_merged["G1"].mean()
g1_sd   = df_merged["G1"].std()
df_merged["G1_z"] = (df_merged["G1"] - g1_mean) / g1_sd

print(f"G1_z mean : {df_merged['G1_z'].mean():.8f}  (should be ~0)")
print(f"G1_z SD   : {df_merged['G1_z'].std():.8f}  (should be ~1)")
print(f"G1_z range: {df_merged['G1_z'].min():.4f}  –  {df_merged['G1_z'].max():.4f}")
print("Step 9 OK")

# ── Step 10: Final inspection ──────────────────────────────────────────────────
print("\n─── Step 10: Final inspection ───")

final_cols = ["subject", "session", "group", "roi_pos_id", "roi_name", "auc", "G1", "G1_z"]
df_out = df_merged[final_cols].copy()

print("\nFirst 10 rows:")
print(df_out.head(10).to_string())

def _dist_summary(series, label):
    d = series.describe()
    try:
        sk = stats.skew(series.dropna())
        sk_str = f"  skewness={sk:.4f}"
    except Exception:
        sk_str = ""
    print(f"{label}: min={d['min']:.4f}  Q1={d['25%']:.4f}  median={d['50%']:.4f}"
          f"  Q3={d['75%']:.4f}  max={d['max']:.4f}{sk_str}")

print()
_dist_summary(df_out["auc"],  "AUC ")
_dist_summary(df_out["G1_z"], "G1_z")

print("\nMissing values per column:")
missing = df_out.isnull().sum()
print(missing.to_string())
if missing.sum() > 0:
    print("STOP: unexpected missing values found")
    sys.exit(1)

print("\nColumn dtypes:")
print(df_out.dtypes.to_string())
print("Step 10 OK")

# ── Save outputs ───────────────────────────────────────────────────────────────
print("\n─── Saving outputs ───")

df_out.to_csv(OUT_MODEL, index=False)
print(f"Saved model-ready DataFrame : {OUT_MODEL}")
print(f"  Shape: {df_out.shape}")

lookup_out = lookup_active.copy()
lookup_out["G1_z"] = (lookup_out["G1"] - g1_mean) / g1_sd
lookup_out = (lookup_out[["nonself_pos_id", "roi_name", "G1", "G1_z", "glasser_row_index"]]
              .sort_values("nonself_pos_id")
              .reset_index(drop=True))
lookup_out.to_csv(OUT_LOOKUP, index=False)
print(f"Saved Glasser-G1 lookup     : {OUT_LOOKUP}")
print(f"  Shape: {lookup_out.shape}")

print(f"\n{SEP}")
print("ALL STEPS COMPLETE")
print(SEP)
