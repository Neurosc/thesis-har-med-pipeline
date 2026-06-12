#!/usr/bin/env python3
"""
debug_auc_comparison.py — Compare AUC values between old and new DataFrames
for 5 Keskin self-mapped Glasser parcels.

Goal: determine whether AUC values in the old parcels_NoGSR extraction and the
new glasser_full_dataframe are identical (labeling/assembly change only) or
different (extraction itself changed).

Files compared:
  OLD : 04_statistics/results/parcels_NoGSR/tables/analysis_long_format_auc.csv
        atlas="self"    → Glasser parcel-extracted self AUC (CAB-NP IDs)
        atlas="nonself" → Glasser parcel-extracted nonself AUC

  NEW : 04_statistics/results/glasser_full_dataframe.csv
        self_layer in (Interoception, Exteroception, Cognition) → self AUC
        self_layer == "nonself" → nonself AUC
        (generate by running: julia 04_statistics/scripts/01_build_dataframe.jl)

  SPHERE: 04_statistics/results/spheres_NoGSR/tables/analysis_long_format_auc.csv
          atlas="self" → sphere-extracted self AUC (Qin 2020 ROI IDs 1–37)

Parcel ID note:
  The old and new DataFrames use CAB-NP L-first IDs (1–360, L hemisphere 1–180).
  Self2Glasser.csv uses R-first Glasser IDs (R hemisphere 1–180, L hemisphere 181–360).
  Conversion: parcel_index ≤ 180  →  CAB-NP ID = parcel_index + 180  (R hemisphere)
              parcel_index >  180  →  CAB-NP ID = parcel_index - 180  (L hemisphere)

Run from repo root:
  python 04_statistics/scripts/debug_auc_comparison.py
"""

import os
import sys
import pandas as pd

REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

OLD_CSV    = os.path.join(REPO_ROOT, "04_statistics", "results",
                          "parcels_NoGSR", "tables", "analysis_long_format_auc.csv")
NEW_CSV    = os.path.join(REPO_ROOT, "04_statistics", "results",
                          "glasser_full_dataframe.csv")
SPHERE_CSV = os.path.join(REPO_ROOT, "04_statistics", "results",
                          "spheres_NoGSR", "tables", "analysis_long_format_auc.csv")

SUBJECT = "sub-01"
SESSION = "ses-01"

# 5 Keskin self-mapped Glasser parcels chosen from Self2Glasser.csv entries
# with a valid parcel_index, verified against glasser_self_metadata.tsv.
#
# Columns: (CAB-NP ID, parcel name, Self2Glasser parcel_index [R-first],
#           layer, sphere ROI ID [1-indexed within self_coordinates.txt],
#           description)
#
# Sphere ROI mapping (Qin 2020 → sphere ROI number in self_coordinates.txt):
#   Interoception rows  1–11 → sphere ROIs  1–11
#   Exteroception rows  1–14 → sphere ROIs 12–25
#   Cognition     rows  1–12 → sphere ROIs 26–37
PARCELS = [
    # CAB-NP  parcel_name   rf_idx  layer            sph_roi  description
    (288,    "R_FOP4",       108,   "Interoception",   1,   "Interoceptive-1 / R Insula"),
    ( 40,    "L_24dd",       220,   "Interoception",   2,   "Interoceptive-2 / L dACC"),
    (300,    "R_H",          120,   "Interoception",   4,   "Interoceptive-4 / R Parahippocampal"),
    (138,    "L_PH",         318,   "Exteroception",  16,   "Exteroceptive-5 / L Fusiform"),
    (330,    "R_PGi",        150,   "Cognition",      34,   "Cognition-9 / R MTG"),
]


def fmt_auc(v):
    if v is None:
        return "  [file N/A]"
    if isinstance(v, float):
        return f"{v:13.6f}"
    return f"{str(v):>13}"


def lookup_old(df, glasser_id):
    """AUC from old file: atlas='self', matching glasser_id."""
    rows = df[(df["atlas"] == "self") & (df["glasser_id"] == glasser_id)]
    if len(rows) == 1:
        return float(rows["auc"].values[0])
    if len(rows) > 1:
        return f"n={len(rows)}"
    return "NOT FOUND"


def lookup_new(df, roi_pos_id):
    """AUC from new file: self_layer != 'nonself', matching roi_pos_id.
    If a parcel is shared across two layers (e.g. parcel 111, 258) there
    will be two rows; we return both values as a tuple for flagging."""
    rows = df[(df["self_layer"] != "nonself") & (df["roi_pos_id"] == roi_pos_id)]
    if len(rows) == 1:
        return float(rows["auc"].values[0])
    if len(rows) == 2:
        vals = rows["auc"].values
        if abs(vals[0] - vals[1]) < 1e-10:
            return float(vals[0])  # identical duplicates (shared parcel, both layers)
        return f"dup≠({vals[0]:.4f},{vals[1]:.4f})"
    if len(rows) > 2:
        return f"n={len(rows)}"
    return "NOT FOUND"


def lookup_sphere(df, roi_id):
    """AUC from sphere file: atlas='self', matching roi_id."""
    rows = df[(df["atlas"] == "self") & (df["roi_id"] == roi_id)]
    if len(rows) == 1:
        return float(rows["auc"].values[0])
    if len(rows) > 1:
        return f"n={len(rows)}"
    return "NOT FOUND"


def match_label(old_val, new_val):
    if not isinstance(old_val, float) or not isinstance(new_val, float):
        return "—"
    diff = abs(old_val - new_val)
    if diff < 1e-8:
        return "IDENTICAL"
    if diff < 1e-3:
        return f"≈diff={diff:.2e}"
    return f"DIFFER({diff:.4f})"


def main():
    SEP = "=" * 90
    print(SEP)
    print("AUC COMPARISON  |  old parcels_NoGSR  vs  new glasser_full_dataframe")
    print(f"Subject: {SUBJECT}   Session: {SESSION}")
    print(SEP)

    # ── Load files ───────────────────────────────────────────────────────────────
    missing = set()
    for label, path in [
        ("OLD (parcels_NoGSR/tables/analysis_long_format_auc.csv)", OLD_CSV),
        ("NEW (glasser_full_dataframe.csv)",                        NEW_CSV),
        ("SPHERE (spheres_NoGSR/tables/analysis_long_format_auc.csv)", SPHERE_CSV),
    ]:
        status = "[OK]     " if os.path.exists(path) else "[MISSING]"
        print(f"  {status} {label}")
        if not os.path.exists(path):
            missing.add(path)
    print()

    if OLD_CSV in missing and NEW_CSV in missing:
        print("Both main files missing — run 01_build_dataframe.jl first.")
        sys.exit(1)

    old_df = pd.read_csv(OLD_CSV) if OLD_CSV not in missing else None
    new_df = pd.read_csv(NEW_CSV) if NEW_CSV not in missing else None
    sph_df = pd.read_csv(SPHERE_CSV) if SPHERE_CSV not in missing else None

    # Filter to target subject/session
    def filt(df, sub, ses):
        return df[(df["subject"] == sub) & (df["session"] == ses)].copy() if df is not None else None

    old_sub = filt(old_df, SUBJECT, SESSION)
    new_sub = filt(new_df, SUBJECT, SESSION)
    sph_sub = filt(sph_df, SUBJECT, SESSION)

    # ── Per-ROI comparison table ─────────────────────────────────────────────────
    hdr = (f"{'Description':<38} {'ID':>4} {'Layer':<14} "
           f"{'OLD self AUC':>13} {'NEW self AUC':>13} {'SPHERE AUC':>12}  {'Match?'}")
    print(hdr)
    print("-" * len(hdr))

    for cab_id, name, rf_idx, layer, sph_roi, desc in PARCELS:
        old_val = lookup_old(old_sub, cab_id)   if old_sub is not None else None
        new_val = lookup_new(new_sub, cab_id)   if new_sub is not None else None
        sph_val = lookup_sphere(sph_sub, sph_roi) if sph_sub is not None else None

        match = match_label(old_val, new_val)
        print(f"  {desc:<36} {cab_id:>4} {layer:<14}"
              f" {fmt_auc(old_val)} {fmt_auc(new_val)} {fmt_auc(sph_val)}  {match}")

    # ── Sanity: Keskin parcel IDs in old atlas='nonself' (should be empty) ───────
    print()
    print("─" * 70)
    print("SANITY: Keskin CAB-NP IDs in old atlas='nonself' (should be zero):")
    if old_sub is not None:
        keskin_ids = [p[0] for p in PARCELS]
        leaked = old_sub[(old_sub["atlas"] == "nonself") &
                         (old_sub["glasser_id"].isin(keskin_ids))]
        if len(leaked) == 0:
            print("  [OK] None — self parcels correctly excluded from old nonself atlas.")
        else:
            print(f"  [WARN] {len(leaked)} rows leaked into nonself:")
            print(leaked.to_string(index=False))

    # ── All-subjects summary ─────────────────────────────────────────────────────
    print()
    print("─" * 70)
    print("SUMMARY across ALL subjects/sessions (5 Keskin parcel IDs):")
    if old_df is not None and new_df is not None:
        keskin_ids = [p[0] for p in PARCELS]

        old_all = old_df[
            (old_df["atlas"] == "self") & (old_df["glasser_id"].isin(keskin_ids))
        ][["subject", "session", "glasser_id", "auc"]].rename(
            columns={"glasser_id": "parcel_id", "auc": "auc_old"})

        new_all = new_df[
            (new_df["self_layer"] != "nonself") & (new_df["roi_pos_id"].isin(keskin_ids))
        ][["subject", "session", "roi_pos_id", "auc"]].rename(
            columns={"roi_pos_id": "parcel_id", "auc": "auc_new"})

        # Deduplicate new_all for shared parcels (identical AUC, different layer)
        new_all = new_all.drop_duplicates(subset=["subject", "session", "parcel_id"])

        merged = pd.merge(old_all, new_all, on=["subject", "session", "parcel_id"])
        merged["diff"] = (merged["auc_old"] - merged["auc_new"]).abs()

        n_matched   = len(merged)
        n_identical = int((merged["diff"] < 1e-8).sum())
        n_differ    = int((merged["diff"] >= 1e-8).sum())
        max_diff    = float(merged["diff"].max()) if n_matched > 0 else float("nan")

        print(f"  Matched row pairs : {n_matched}")
        print(f"  Identical (< 1e-8): {n_identical}")
        print(f"  Different (≥ 1e-8): {n_differ}  (max diff = {max_diff:.2e})")

        if n_differ == 0 and n_matched > 0:
            print()
            print("  → OLD and NEW AUC values are IDENTICAL for all 5 parcels.")
            print("    Any discrepancy between analyses is in labeling / assembly,")
            print("    NOT in the underlying Glasser parcel extraction.")
        elif n_differ > 0:
            print()
            print("  → AUC values DIFFER — the extraction itself changed between old and new.")
            worst = merged.nlargest(3, "diff")[["subject", "session", "parcel_id",
                                               "auc_old", "auc_new", "diff"]]
            print("  Top 3 largest differences:")
            print(worst.to_string(index=False))
        else:
            print("  → No matched pairs (check file contents).")
    else:
        missing_labels = []
        if old_df is None:
            missing_labels.append("OLD")
        if new_df is None:
            missing_labels.append("NEW (run 01_build_dataframe.jl first)")
        print(f"  Skipped — missing: {', '.join(missing_labels)}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
