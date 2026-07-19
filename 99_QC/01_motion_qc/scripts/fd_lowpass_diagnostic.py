"""
FD low-pass diagnostic — DIAGNOSTIC ONLY, changes nothing in the pipeline.

Question: our FD is computed from UNFILTERED motion parameters. At TR = 1.8 s
respiration (~0.2-0.5 Hz) is aliased and can masquerade as head motion,
inflating FD and therefore the censoring rate. Goldberg et al. 2024 notch-filter
the motion parameters (0.2-0.5 Hz stop band, Power et al. 2019) before computing
FD, but their TR is 0.72 s. At TR = 1.8 s the Nyquist frequency is 0.278 Hz, so
most of the respiratory band sits at or above Nyquist and a notch is not
feasible — a low-pass filter is the available approximation.

This script recomputes FD from low-pass-filtered motion parameters and compares
it against the current unfiltered FD. It writes a per-run CSV, a group summary,
two figures and a log. It does NOT touch the denoising, the censoring threshold,
the stats, or any config, and the filtered FD is not applied anywhere.

WHAT IS HELD FIXED (the existing FD formula, unchanged)
    Power et al. 2012 framewise displacement, computed by the existing
    utils.motion_qc.compute_custom_fd():
      - 1-TR backward differences
      - rotations converted to mm on a 50 mm radius sphere
      - FD_t = sum of the 6 absolute differences
      - TR = 1.8 s, all frames used (no dummy-volume trimming, matching
        fd_all_runs_grid.py and the censoring that feeds the pipeline)
    The ONLY thing this script varies is whether the 6 motion parameters are
    low-pass filtered before that function differences them. Filtering is
    applied to the parameters, then the unmodified compute_custom_fd() is
    called with no bandstop — so the formula cannot drift from the pipeline's.

WHAT IS VARIED
    Low-pass filter: Butterworth, order 4, applied zero-phase with filtfilt
    (padtype="odd") — the same design and call convention as the bandstop
    branch already in compute_custom_fd(). Zero-phase filtfilt applies the
    filter forward and backward, so the effective magnitude response is order 8
    and there is no phase distortion (no temporal smearing of spikes).
    Cutoffs: 0.10 Hz (primary), 0.08 Hz and 0.20 Hz (sensitivity).

SAMPLE
    Full original cohort — every sub-*/ses-* found under the fMRIPrep
    derivatives, including the five subjects excluded from the analysis sample
    (sub-06, 08, 12, 26, 36), which are flagged in an `excluded` column.

FAILURE BEHAVIOUR
    Fails loudly and stops. No run is silently skipped: if any discovered
    session lacks a confounds file, or any subject is missing from
    participants.tsv, or motion columns are missing or contain NaN, the script
    raises before computing anything.

Runs on the server (reads fMRIPrep confounds TSVs). Activate env: conda activate fmri.

    python 99_QC/01_motion_qc/scripts/fd_lowpass_diagnostic.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from utils.motion_qc import MOTION_COLS, compute_custom_fd
from utils.subject_filter import _EXCLUDED

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parents[3]
FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
PARTICIPANTS  = REPO_ROOT / "participants.tsv"
OUT_DIR       = REPO_ROOT / "99_QC" / "01_motion_qc" / "results" / "fd_lowpass_diagnostic"

TASK            = "rest"
TR              = 1.8
BACKWARD_DIFF_N = 1        # unchanged — Power 2012, 1-TR backward difference
SPHERE_RADIUS   = 50.0     # unchanged — 50 mm rotation sphere
FILTER_ORDER    = 4        # Butterworth order, applied zero-phase via filtfilt
CUTOFFS_HZ      = {"lp008": 0.08, "lp010": 0.10, "lp020": 0.20}
PRIMARY         = "lp010"
FD_THRESHOLDS   = {"gt03": 0.3, "gt02": 0.2}   # 0.3 mm = the pipeline threshold
N_EXAMPLES      = 6        # example traces (half collapsed, half retained)
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "fd_lowpass_diagnostic_log.txt"   # .txt — *.log is gitignored


class _Tee:
    """Mirror stdout to the log file so the printed summary is committed too."""

    def __init__(self, stream, path):
        self.stream = stream
        self.fh = open(path, "w", encoding="utf-8")

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


sys.stdout = _Tee(sys.__stdout__, LOG_PATH)

NYQUIST = 1.0 / (2.0 * TR)

print("=" * 78)
print("FD LOW-PASS DIAGNOSTIC — comparison only, nothing in the pipeline changes")
print("=" * 78)
print(f"FD formula (UNCHANGED, from utils.motion_qc.compute_custom_fd):")
print(f"    Power et al. 2012 — {BACKWARD_DIFF_N}-TR backward differences,")
print(f"    rotations converted with a {SPHERE_RADIUS:.0f} mm radius sphere,")
print(f"    FD_t = sum of 6 absolute differences, all frames, TR = {TR} s.")
print(f"Filter applied to the 6 motion parameters BEFORE differencing:")
print(f"    Butterworth low-pass, order {FILTER_ORDER}, zero-phase via")
print(f"    scipy.signal.filtfilt(padtype='odd') -> effective magnitude order")
print(f"    {FILTER_ORDER * 2}, no phase distortion.")
print(f"    Cutoffs: " + ", ".join(f"{k} = {v} Hz" for k, v in CUTOFFS_HZ.items())
      + f"   (primary = {CUTOFFS_HZ[PRIMARY]} Hz)")
print(f"Sampling: fs = {1.0 / TR:.4f} Hz, Nyquist = {NYQUIST:.4f} Hz.")
print(f"    Note: the respiratory band (~0.2-0.5 Hz) lies largely AT OR ABOVE")
print(f"    Nyquist at this TR, hence low-pass rather than a Power 2019 notch.")
print(f"Output directory: {OUT_DIR}")
print()

# ── Guard: cutoffs must be below Nyquist ──────────────────────────────────────
for name, hz in CUTOFFS_HZ.items():
    if hz >= NYQUIST:
        raise ValueError(
            f"Cutoff {name} = {hz} Hz is >= Nyquist ({NYQUIST:.4f} Hz) at TR = {TR} s. "
            f"Refusing to clip silently — choose a lower cutoff."
        )

# ── Step 1: participants.tsv -> arm ───────────────────────────────────────────
if not PARTICIPANTS.is_file():
    raise FileNotFoundError(f"participants.tsv not found: {PARTICIPANTS}")

participants = pd.read_csv(PARTICIPANTS, sep="\t")
for col in ("participant_id", "condition"):
    if col not in participants.columns:
        raise ValueError(
            f"participants.tsv is missing the '{col}' column. "
            f"Columns present: {list(participants.columns)}"
        )
ARM = dict(zip(participants["participant_id"], participants["condition"]))
print(f"participants.tsv: {len(ARM)} subjects "
      f"({sum(v == 'verum' for v in ARM.values())} verum / "
      f"{sum(v == 'placebo' for v in ARM.values())} placebo)")

# ── Step 2: discover every run, fail if any confounds file is missing ─────────
if not FMRIPREP_ROOT.is_dir():
    raise FileNotFoundError(f"fMRIPrep derivatives not found: {FMRIPREP_ROOT}")

# is_dir() matters: fMRIPrep also writes a sub-XX.html report per subject at the
# derivatives root, and a bare glob would count those as subjects.
subject_dirs = sorted(d for d in FMRIPREP_ROOT.glob("sub-*") if d.is_dir())
if not subject_dirs:
    raise RuntimeError(f"No sub-* directories found under {FMRIPREP_ROOT}")

runs, missing = [], []
for sub_dir in subject_dirs:
    subject_id = sub_dir.name
    for ses_dir in sorted(sub_dir.glob("ses-*")):
        session_id = ses_dir.name
        pattern = f"{subject_id}_{session_id}_task-{TASK}_desc-confounds_timeseries.tsv.gz"
        matches = list((ses_dir / "func").glob(pattern))
        if not matches:
            missing.append(f"{subject_id}/{session_id}  (expected {pattern})")
        else:
            runs.append((subject_id, session_id, matches[0]))

if missing:
    raise FileNotFoundError(
        "Confounds file missing for "
        f"{len(missing)} discovered session(s) — refusing to skip them:\n  "
        + "\n  ".join(missing)
    )

unknown_arm = sorted({s for s, _, _ in runs} - set(ARM))
if unknown_arm:
    raise ValueError(
        f"Subjects present in the fMRIPrep derivatives but absent from "
        f"participants.tsv: {unknown_arm}"
    )

print(f"Discovered {len(subject_dirs)} subject(s), {len(runs)} run(s). "
      f"All confounds files present.\n")


def lowpass_motion(params, cutoff_hz):
    """Zero-phase Butterworth low-pass of each motion parameter column."""
    b, a = signal.butter(FILTER_ORDER, cutoff_hz / NYQUIST, btype="lowpass")
    padlen = 3 * max(len(a), len(b))
    if params.shape[0] <= padlen:
        raise ValueError(
            f"Run has {params.shape[0]} frames, too short for filtfilt "
            f"(needs > {padlen})."
        )
    out = np.empty_like(params)
    for i in range(params.shape[1]):
        out[:, i] = signal.filtfilt(b, a, params[:, i], padtype="odd")
    return out


def pct_above(fd, thresh):
    n_frames = len(fd)
    n_above = int(np.sum(fd > thresh, where=~np.isnan(fd)))
    return 100.0 * n_above / n_frames


# ── Step 3: per-run FD, unfiltered and at each cutoff ─────────────────────────
rows, fd_traces = [], {}

for subject_id, session_id, confounds_path in runs:
    df = pd.read_csv(confounds_path, sep="\t")

    missing_cols = [c for c in MOTION_COLS if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"{subject_id}/{session_id}: motion columns missing {missing_cols} "
            f"in {confounds_path}"
        )
    params = df[MOTION_COLS].to_numpy(dtype=float)
    if np.isnan(params).any():
        n_bad = int(np.isnan(params).sum())
        raise ValueError(
            f"{subject_id}/{session_id}: {n_bad} NaN value(s) in the motion "
            f"parameters. Refusing to interpolate or drop — inspect "
            f"{confounds_path}"
        )

    # Reference: the FD the pipeline currently uses (unfiltered parameters).
    fd_unfilt = compute_custom_fd(
        df, tr=TR, backward_diff_n=BACKWARD_DIFF_N,
        sphere_radius_mm=SPHERE_RADIUS, verbose=False,
    )

    row = {
        "subject":   subject_id,
        "session":   session_id,
        "arm":       ARM[subject_id],
        "excluded":  subject_id in _EXCLUDED,
        "n_frames":  len(fd_unfilt),
        "mean_fd_unfilt": float(np.nanmean(fd_unfilt)),
        "max_fd_unfilt":  float(np.nanmax(fd_unfilt)),
    }
    for tname, thresh in FD_THRESHOLDS.items():
        row[f"pct_{tname}_unfilt"] = pct_above(fd_unfilt, thresh)

    traces = {"unfilt": fd_unfilt}
    for cname, hz in CUTOFFS_HZ.items():
        # Same DataFrame, motion columns replaced by their filtered version, so
        # the identical compute_custom_fd() code path produces the filtered FD.
        df_f = df.copy()
        df_f[MOTION_COLS] = lowpass_motion(params, hz)
        fd_f = compute_custom_fd(
            df_f, tr=TR, backward_diff_n=BACKWARD_DIFF_N,
            sphere_radius_mm=SPHERE_RADIUS, verbose=False,
        )
        traces[cname] = fd_f
        row[f"mean_fd_{cname}"] = float(np.nanmean(fd_f))
        row[f"max_fd_{cname}"]  = float(np.nanmax(fd_f))
        for tname, thresh in FD_THRESHOLDS.items():
            row[f"pct_{tname}_{cname}"] = pct_above(fd_f, thresh)

    rows.append(row)
    fd_traces[(subject_id, session_id)] = traces
    print(f"  {subject_id} {session_id}  "
          f"meanFD {row['mean_fd_unfilt']:.3f} -> {row[f'mean_fd_{PRIMARY}']:.3f} mm   "
          f"%cens(0.3) {row['pct_gt03_unfilt']:.1f} -> {row[f'pct_gt03_{PRIMARY}']:.1f}")

run_df = pd.DataFrame(rows).sort_values(["subject", "session"]).reset_index(drop=True)

col_order = ["subject", "session", "arm", "excluded", "n_frames"]
for stat in ("mean_fd", "max_fd"):
    col_order += [f"{stat}_unfilt"] + [f"{stat}_{c}" for c in CUTOFFS_HZ]
for tname in FD_THRESHOLDS:
    col_order += [f"pct_{tname}_unfilt"] + [f"pct_{tname}_{c}" for c in CUTOFFS_HZ]
run_df = run_df[col_order]

csv_path = OUT_DIR / "fd_lowpass_per_run.csv"
run_df.to_csv(csv_path, index=False)
print(f"\nPer-run CSV saved: {csv_path}  ({len(run_df)} runs)\n")

# ── Step 4: group summary ─────────────────────────────────────────────────────
conditions = ["unfilt"] + list(CUTOFFS_HZ)
label_of = {"unfilt": "unfiltered"}
label_of.update({c: f"low-pass {hz} Hz" for c, hz in CUTOFFS_HZ.items()})

summary_rows = []
for cond in conditions:
    pct = run_df[f"pct_gt03_{cond}"]
    summary_rows.append({
        "condition":        label_of[cond],
        "n_runs":           len(pct),
        "median_pct_cens":  float(pct.median()),
        "mean_pct_cens":    float(pct.mean()),
        "min_pct_cens":     float(pct.min()),
        "max_pct_cens":     float(pct.max()),
        "n_runs_gt30pct":   int((pct > 30).sum()),
        "n_runs_gt50pct":   int((pct > 50).sum()),
        "mean_fd_mm":       float(run_df[f"mean_fd_{cond}"].mean()),
    })
summary_df = pd.DataFrame(summary_rows)
summary_path = OUT_DIR / "fd_lowpass_group_summary.csv"
summary_df.to_csv(summary_path, index=False)

print("── Group summary: percent frames censored at FD > 0.3 mm ──")
print(f"{'condition':<20}{'median':>8}{'mean':>8}{'min':>8}{'max':>8}"
      f"{'>30%':>7}{'>50%':>7}{'meanFD':>9}")
for r in summary_rows:
    print(f"{r['condition']:<20}{r['median_pct_cens']:>8.1f}{r['mean_pct_cens']:>8.1f}"
          f"{r['min_pct_cens']:>8.1f}{r['max_pct_cens']:>8.1f}"
          f"{r['n_runs_gt30pct']:>7d}{r['n_runs_gt50pct']:>7d}{r['mean_fd_mm']:>9.3f}")
print(f"\nGroup summary saved: {summary_path}")

print("\n── Percent frames above 0.2 mm (secondary threshold) ──")
for cond in conditions:
    pct = run_df[f"pct_gt02_{cond}"]
    print(f"  {label_of[cond]:<20} median {pct.median():>6.1f}   mean {pct.mean():>6.1f}")

# ── Step 5: figure 1 — example traces spanning low to high motion ─────────────
# Selection rule (deterministic, no randomness). Runs are shown from BOTH
# behaviours, because picking only the biggest changes would flatter the filter
# and hide the runs where motion survives it:
#   "collapsed" — the largest reduction in %censored, one per motion tertile so
#                 the panel still spans low -> high motion.
#   "retained"  — the runs with the most %censored still REMAINING after
#                 filtering, i.e. where the filter did not erase the motion.
# Both groups are needed to judge whether the filter is removing rhythmic
# oscillation (respiration) or flattening isolated real-motion spikes.
sel = run_df.copy()
sel["delta_pct"] = sel["pct_gt03_unfilt"] - sel[f"pct_gt03_{PRIMARY}"]
sel["tertile"] = pd.qcut(sel["pct_gt03_unfilt"].rank(method="first"), 3,
                         labels=["low", "mid", "high"])

collapsed = (sel.sort_values("delta_pct", ascending=False)
                .groupby("tertile", observed=True)
                .head(1)
                .assign(group="collapsed"))
retained = (sel.drop(index=collapsed.index)
               .sort_values(f"pct_gt03_{PRIMARY}", ascending=False)
               .head(max(0, N_EXAMPLES - len(collapsed)))
               .assign(group="retained"))
examples = (pd.concat([collapsed, retained])
              .sort_values(["group", "pct_gt03_unfilt"])
              .reset_index(drop=True))

print(f"\n── Example runs for the trace figure "
      f"({len(collapsed)} collapsed + {len(retained)} retained) ──")
for r in examples.itertuples(index=False):
    print(f"  {r.subject} {r.session}  [{r.tertile:>4} motion, {r.group:>9}]  "
          f"%cens {r.pct_gt03_unfilt:.1f} -> {getattr(r, f'pct_gt03_{PRIMARY}'):.1f}  "
          f"(-{r.delta_pct:.1f} pts)")

n_ex = len(examples)
ncols = 2
nrows = int(np.ceil(n_ex / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(13, 2.6 * nrows), squeeze=False)
axes = axes.ravel()

for ax, r in zip(axes, examples.itertuples(index=False)):
    tr_ = fd_traces[(r.subject, r.session)]
    frames = np.arange(len(tr_["unfilt"]))
    ax.plot(frames, tr_["unfilt"], color="#999999", linewidth=0.9,
            label="unfiltered FD (current)")
    ax.plot(frames, tr_[PRIMARY], color="#4878CF", linewidth=1.2,
            label=f"low-pass {CUTOFFS_HZ[PRIMARY]} Hz FD")
    ax.axhline(FD_THRESHOLDS["gt03"], color="black", linestyle="--", linewidth=0.9,
               label="0.3 mm threshold")
    ax.set_xlim(0, len(frames) - 1)
    ax.set_ylim(0, max(0.45, float(r.max_fd_unfilt) * 1.08))
    title = f"{r.subject} {r.session}  [{r.tertile} motion — {r.group}]"
    if r.excluded:
        title += "  [EXCL]"
    ax.set_title(title, fontsize=9,
                 color="#b03030" if r.group == "retained" else "black")
    ax.annotate(
        f"%cens 0.3mm: {r.pct_gt03_unfilt:.1f} -> {getattr(r, f'pct_gt03_{PRIMARY}'):.1f}\n"
        f"max FD: {r.max_fd_unfilt:.2f} -> {getattr(r, f'max_fd_{PRIMARY}'):.2f} mm",
        xy=(0.985, 0.93), xycoords="axes fraction", ha="right", va="top",
        fontsize=6.5, color="#333333",
    )
    ax.tick_params(labelsize=7)

for ax in axes[n_ex:]:
    ax.axis("off")

axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle(
    f"FD before and after low-pass filtering the motion parameters "
    f"(Butterworth order {FILTER_ORDER}, zero-phase, {CUTOFFS_HZ[PRIMARY]} Hz)\n"
    f"Rhythmic oscillation removed = respiration (good); tall isolated spikes "
    f"flattened = real motion (bad)\n"
    f"Top rows: 'collapsed' runs (largest drop, one per motion tertile).  "
    f"Bottom rows in red: 'retained' runs (most censoring surviving the filter).",
    fontsize=11,
)
fig.supxlabel("Frame index", fontsize=9)
fig.supylabel("FD (mm)", fontsize=9)
plt.tight_layout(rect=[0.01, 0.01, 1, 0.94])
trace_path = OUT_DIR / "fd_lowpass_example_traces.png"
plt.savefig(trace_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"\nTrace figure saved: {trace_path}")

# ── Step 6: figure 2 — unfiltered vs filtered %censored, one point per run ────
fig, axes = plt.subplots(1, len(CUTOFFS_HZ), figsize=(5 * len(CUTOFFS_HZ), 5),
                         squeeze=False)
axes = axes.ravel()
lim = float(max(run_df["pct_gt03_unfilt"].max(),
                *[run_df[f"pct_gt03_{c}"].max() for c in CUTOFFS_HZ])) * 1.08
lim = max(lim, 5.0)

for ax, (cname, hz) in zip(axes, CUTOFFS_HZ.items()):
    keep = ~run_df["excluded"]
    ax.scatter(run_df.loc[keep, "pct_gt03_unfilt"], run_df.loc[keep, f"pct_gt03_{cname}"],
               s=26, color="#4878CF", alpha=0.75, edgecolor="none", label="analysis sample")
    ax.scatter(run_df.loc[~keep, "pct_gt03_unfilt"], run_df.loc[~keep, f"pct_gt03_{cname}"],
               s=30, color="#cc2222", alpha=0.8, marker="^", edgecolor="none",
               label="FD-excluded subject")
    ax.plot([0, lim], [0, lim], color="black", linestyle="--", linewidth=1,
            label="identity (no change)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_aspect("equal")
    ax.set_xlabel("% frames censored — unfiltered (current)", fontsize=9)
    ax.set_ylabel(f"% frames censored — low-pass {hz} Hz", fontsize=9)
    ax.set_title(f"cutoff {hz} Hz"
                 + ("   (primary)" if cname == PRIMARY else ""), fontsize=10)
    ax.tick_params(labelsize=8)

axes[0].legend(fontsize=7, loc="upper left")
fig.suptitle("Censoring rate (FD > 0.3 mm) before vs after low-pass filtering — one point per run",
             fontsize=12)
plt.tight_layout(rect=[0.01, 0.01, 1, 0.95])
scatter_path = OUT_DIR / "fd_lowpass_pct_censored_scatter.png"
plt.savefig(scatter_path, dpi=300, bbox_inches="tight")
plt.close(fig)
print(f"Scatter figure saved: {scatter_path}")

# ── Step 7: closing note ──────────────────────────────────────────────────────
print("\n── Done ──")
print(f"  Runs processed : {len(run_df)}  "
      f"({int(run_df['excluded'].sum())} from FD-excluded subjects)")
print("  Nothing in the pipeline was modified. The filtered FD is not applied")
print("  anywhere — this is a comparison only.")
print(f"\n  Outputs in {OUT_DIR}:")
for p in (csv_path, summary_path, trace_path, scatter_path, LOG_PATH):
    print(f"    {p.name}")

sys.stdout.flush()
