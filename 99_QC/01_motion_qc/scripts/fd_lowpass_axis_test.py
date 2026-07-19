"""
FD low-pass axis test — is the FD removed by a 0.1 Hz low-pass respiration or
real head motion? DIAGNOSTIC ONLY, changes nothing in the pipeline.

The companion script fd_lowpass_diagnostic.py showed that a 0.1 Hz low-pass of
the motion parameters collapses censoring from a median 19% to 0%. That collapse
only proves the removed signal is HIGH-FREQUENCY. It does not identify the
source. This script applies the phase-encode-axis test to decide between the two
candidates.

THE LOGIC
    Respiratory pseudomotion is a B0 effect: chest expansion perturbs the field,
    and the resulting apparent displacement is concentrated along the PHASE-ENCODE
    axis (translation, plus pitch), because that is the direction in which off-
    resonance maps to spatial shift. Real head motion has no such preference and
    is broadband across all six parameters.
    So: if the removed FD is concentrated in trans_y (+ rot_x), it is respiration.
    If it is spread roughly evenly across the six axes, the filter deflated real
    motion and low-pass filtering should be dropped.

ACQUISITION (from CLAUDE.md, not assumed)
    TR = 1.8 s  ->  fs = 0.5556 Hz, Nyquist = 0.2778 Hz
    Phase-encode direction j (AP)  ->  phase-encode axis = trans_y
    Pitch (nodding, rotation about the L-R axis) = rot_x

ALIASING — NOTE THE BAND IS COMPUTED, NOT ASSUMED
    At TR = 1.8 s, respiration is only partly above Nyquist. Frequencies above
    Nyquist fold back as f_alias = |f - fs|. For a respiratory range of
    0.15-0.40 Hz this maps to roughly 0.156-0.278 Hz — the TOP of the observable
    band, pressed against Nyquist, NOT 0.1-0.2 Hz. (0.1-0.2 Hz would be the
    answer at TR = 2.0 s.) The band is folded programmatically below and the
    arithmetic is printed, because shading the wrong band would make a genuine
    respiratory signature look like a miss.

FILTER — TWO ORDERS, DELIBERATELY
    Specified for this test: first-order Butterworth, 0.1 Hz, zero-phase
    (filtfilt), per Gratton et al. 2020. But the censoring collapse being
    explained was produced by the ORDER-4 filter in fd_lowpass_diagnostic.py.
    These are not interchangeable — order 1 rolls off far more gently. Both are
    computed and reported for every run and every axis. If the two orders agree
    on which axis the removed FD came from, the verdict is robust to the choice;
    if they disagree, that is itself the finding.

FD FORMULA (unchanged, Power et al. 2012)
    Backward differences (1 TR), rotations converted to arc length on a 50 mm
    radius sphere, FD = sum of the 6 absolute differences. Here FD is decomposed:
    each axis's contribution is the sum over time of its own |backward difference|,
    so the 6 contributions sum exactly to total FD.

NO THRESHOLD IS APPLIED
    The 0.3 mm censoring threshold is deliberately NOT carried onto filtered FD
    anywhere in this script. The question is the SOURCE of the removed signal,
    not where to set a threshold.

RUNS
    Collapsed (censoring nearly vanished after filtering): sub-34 ses-02,
    sub-22 ses-02, sub-12 ses-02.
    Retained (high motion surviving the filter), as a contrast control: sub-08
    ses-02 — its residual FD must NOT show phase-encode dominance, or the test
    is labelling everything respiratory and discriminates nothing.

Runs on the server (reads fMRIPrep confounds TSVs). Activate env: conda activate fmri.

    python 99_QC/01_motion_qc/scripts/fd_lowpass_axis_test.py
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
from utils.motion_qc import MOTION_COLS

# ── Configuration ─────────────────────────────────────────────────────────────
REPO_ROOT     = Path(__file__).resolve().parents[3]
FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
OUT_DIR       = REPO_ROOT / "99_QC" / "01_motion_qc" / "results" / "fd_lowpass_axis_test"

TASK          = "rest"
TR            = 1.8
SPHERE_RADIUS = 50.0          # mm — rotations to arc length (Power et al. 2012)
CUTOFF_HZ     = 0.10          # low-pass cutoff under test
FILTER_ORDERS = {1: "order 1 (Gratton et al. 2020 spec)",
                 4: "order 4 (what produced the censoring collapse)"}
PRIMARY_ORDER = 1

RESP_RANGE_HZ = (0.15, 0.40)  # plausible adult respiration, folded below

PE_AXIS    = "trans_y"        # phase-encode direction j (AP)
PITCH_AXIS = "rot_x"          # pitch = rotation about the L-R axis

RUNS = [
    ("sub-34", "ses-02", "collapsed"),
    ("sub-22", "ses-02", "collapsed"),
    ("sub-12", "ses-02", "collapsed"),
    ("sub-08", "ses-02", "retained"),   # contrast control
]

# Decision-rule thresholds (explicit, printed with the verdict)
CRIT_A_MIN_BAND_FRAC = 0.50   # >=50% of removed power inside the aliased band
CRIT_B_MIN_PE        = 0.50   # >=50% of removed FD from the phase-encode axis
CRIT_B_MIN_PE_PITCH  = 0.60   # ...or >=60% from phase-encode + pitch together
# Rhythmicity threshold, calibrated against a null rather than picked by eye.
# Over 300 simulated broadband/spiky runs (white noise + 4 random step spikes,
# n=240) the peak/median statistic had median 2.66, p99 5.90, max 9.29 — so a
# threshold of 3 would pass almost any run. A genuine 0.30 Hz oscillation scores
# ~190 at 0.05 mm amplitude and ~11000 at 0.4 mm. 10.0 clears the entire null
# while sitting well over an order of magnitude below the weakest real rhythm.
CRIT_C_MIN_RHYTHM    = 10.0   # spectral peak / median power in band
# ──────────────────────────────────────────────────────────────────────────────

OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "fd_lowpass_axis_test_log.txt"   # .txt — *.log is gitignored


class _Tee:
    """Mirror stdout to the log file so the printed verdicts are committed too."""

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

FS      = 1.0 / TR
NYQUIST = FS / 2.0

# ── Fold the respiratory range through Nyquist ────────────────────────────────
def fold(f, fs):
    """Alias a frequency into [0, fs/2] by reflecting about Nyquist."""
    f = f % fs
    return fs - f if f > fs / 2.0 else f


_folded = [fold(f, FS) for f in np.linspace(*RESP_RANGE_HZ, 400)]
RESP_BAND = (float(min(_folded)), float(max(_folded)))

print("=" * 78)
print("FD LOW-PASS AXIS TEST — respiration or real motion?")
print("=" * 78)
print(f"TR = {TR} s  ->  fs = {FS:.4f} Hz, Nyquist = {NYQUIST:.4f} Hz")
print(f"Phase-encode axis = {PE_AXIS} (dir j, AP);  pitch = {PITCH_AXIS}")
print(f"Low-pass cutoff under test = {CUTOFF_HZ} Hz, zero-phase (filtfilt)")
for o, desc in FILTER_ORDERS.items():
    print(f"    Butterworth {desc}")
print()
print(f"Aliasing: respiration {RESP_RANGE_HZ[0]}-{RESP_RANGE_HZ[1]} Hz folds "
      f"(f_alias = |f - fs|) into")
print(f"    {RESP_BAND[0]:.3f}-{RESP_BAND[1]:.3f} Hz  <- the shaded band, computed not assumed.")
print(f"    Worked example: {RESP_RANGE_HZ[1]} Hz -> |{RESP_RANGE_HZ[1]} - {FS:.4f}| "
      f"= {fold(RESP_RANGE_HZ[1], FS):.3f} Hz")
print(f"    Note this sits ABOVE the {CUTOFF_HZ} Hz cutoff and near Nyquist — at")
print(f"    TR = 2.0 s it would instead land near 0.1-0.2 Hz.")
print()
print("FD is NOT thresholded anywhere in this script — source attribution only.")
print(f"Output directory: {OUT_DIR}")
print()

if RESP_BAND[0] <= CUTOFF_HZ:
    print(f"  NOTE: the aliased band starts at {RESP_BAND[0]:.3f} Hz, at or below the")
    print(f"  {CUTOFF_HZ} Hz cutoff — the filter cannot separate the low edge of the")
    print(f"  respiratory band from genuine low-frequency motion.\n")

AXIS_UNITS = {c: ("mm" if c.startswith("trans") else "mm arc") for c in MOTION_COLS}


def load_params_mm(subject_id, session_id):
    """Load the 6 motion parameters, rotations converted to mm arc length."""
    func_dir = FMRIPREP_ROOT / subject_id / session_id / "func"
    pattern = f"{subject_id}_{session_id}_task-{TASK}_desc-confounds_timeseries.tsv.gz"
    matches = list(func_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"Confounds file not found for {subject_id}/{session_id}\n"
            f"  looked in : {func_dir}\n"
            f"  pattern   : {pattern}"
        )
    df = pd.read_csv(matches[0], sep="\t")
    missing = [c for c in MOTION_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{subject_id}/{session_id}: motion columns missing {missing} in {matches[0]}"
        )
    params = df[MOTION_COLS].to_numpy(dtype=float)
    if np.isnan(params).any():
        raise ValueError(
            f"{subject_id}/{session_id}: NaN in motion parameters. Refusing to "
            f"interpolate or drop — inspect {matches[0]}"
        )
    # Rotations (radians) -> mm of arc on a 50 mm sphere, so all six are comparable.
    params = params.copy()
    params[:, 3:] *= SPHERE_RADIUS
    return params, matches[0]


def lowpass(params, order):
    """Zero-phase Butterworth low-pass of each column."""
    b, a = signal.butter(order, CUTOFF_HZ / NYQUIST, btype="lowpass")
    padlen = 3 * max(len(a), len(b))
    if params.shape[0] <= padlen:
        raise ValueError(f"Run too short for filtfilt: {params.shape[0]} <= {padlen}")
    out = np.empty_like(params)
    for i in range(params.shape[1]):
        out[:, i] = signal.filtfilt(b, a, params[:, i], padtype="odd")
    return out


def axis_fd_contributions(params):
    """Per-axis FD contribution = sum over time of |backward difference|.

    The 6 values sum exactly to total FD (Power et al. 2012), so they partition
    FD without approximation.
    """
    diffs = np.diff(params, axis=0)          # frame t minus frame t-1
    return np.abs(diffs).sum(axis=0)         # one scalar per axis


def welch_psd(x):
    nperseg = min(128, len(x))
    return signal.welch(x, fs=FS, nperseg=nperseg,
                        noverlap=nperseg // 2, detrend="linear")


def rhythmicity(x):
    """Peak-to-median PSD ratio of the DIFFERENCED signal inside the aliased band.

    Differenced because FD is built from differences. A rhythmic oscillation
    gives one dominant narrow peak (high ratio); broadband transients give a
    flat spectrum (ratio near 1).
    """
    f, p = welch_psd(np.diff(x))
    band = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
    hi = f >= CUTOFF_HZ
    if band.sum() < 2 or hi.sum() < 2:
        raise ValueError("Too few spectral bins to assess rhythmicity.")
    return float(p[band].max() / np.median(p[hi]))


def removed_power_band_fraction(raw, filt):
    """Fraction of the power removed by the filter that lies in the aliased band.

    Summed across all six axes.
    """
    num = den = 0.0
    for i in range(raw.shape[1]):
        f, p_raw = welch_psd(raw[:, i])
        _, p_filt = welch_psd(filt[:, i])
        removed = np.clip(p_raw - p_filt, 0, None)
        band = (f >= RESP_BAND[0]) & (f <= RESP_BAND[1])
        num += removed[band].sum()
        den += removed.sum()
    if den <= 0:
        raise ValueError("Filter removed no power — cannot attribute it.")
    return float(num / den)


# ── Per-run analysis ──────────────────────────────────────────────────────────
summary_rows, axis_rows = [], []

for subject_id, session_id, group in RUNS:
    label = f"{subject_id}_{session_id}"
    print("=" * 78)
    print(f"{label}   [{group}]")
    print("=" * 78)

    raw, src = load_params_mm(subject_id, session_id)
    print(f"  source: {src}")
    print(f"  frames: {raw.shape[0]}")

    raw_contrib = axis_fd_contributions(raw)
    filt, filt_contrib, removed_frac = {}, {}, {}

    for order in FILTER_ORDERS:
        filt[order] = lowpass(raw, order)
        filt_contrib[order] = axis_fd_contributions(filt[order])
        removed = raw_contrib - filt_contrib[order]
        total_removed = removed.sum()
        if total_removed <= 0:
            raise ValueError(
                f"{label}: order {order} filter removed no FD — nothing to attribute."
            )
        removed_frac[order] = removed / total_removed

    # ── per-axis table ────────────────────────────────────────────────────────
    print(f"\n  Per-axis FD contribution (mm, summed over run) and share of REMOVED FD:")
    print(f"    {'axis':<10}{'raw':>10}" +
          "".join(f"{'ord' + str(o):>10}{'%rm':>8}" for o in FILTER_ORDERS))
    for i, ax_name in enumerate(MOTION_COLS):
        line = f"    {ax_name:<10}{raw_contrib[i]:>10.2f}"
        for o in FILTER_ORDERS:
            line += f"{filt_contrib[o][i]:>10.2f}{100 * removed_frac[o][i]:>7.1f}%"
        print(line)
        axis_rows.append({
            "subject": subject_id, "session": session_id, "group": group,
            "axis": ax_name, "unit": AXIS_UNITS[ax_name],
            "fd_contrib_raw": float(raw_contrib[i]),
            **{f"fd_contrib_order{o}": float(filt_contrib[o][i]) for o in FILTER_ORDERS},
            **{f"pct_removed_fd_order{o}": float(100 * removed_frac[o][i])
               for o in FILTER_ORDERS},
        })

    pe_i, pitch_i = MOTION_COLS.index(PE_AXIS), MOTION_COLS.index(PITCH_AXIS)
    print(f"\n    (chance level with 6 axes = 16.7% each)")

    # ── decision rule, per filter order ───────────────────────────────────────
    for order in FILTER_ORDERS:
        pe_share    = removed_frac[order][pe_i]
        pitch_share = removed_frac[order][pitch_i]
        combined    = pe_share + pitch_share
        band_frac   = removed_power_band_fraction(raw, filt[order])
        rhythm      = rhythmicity(raw[:, pe_i])

        crit_a = band_frac >= CRIT_A_MIN_BAND_FRAC
        crit_b = (pe_share >= CRIT_B_MIN_PE) or (combined >= CRIT_B_MIN_PE_PITCH)
        crit_c = rhythm >= CRIT_C_MIN_RHYTHM

        n_met = sum((crit_a, crit_b, crit_c))
        if n_met == 3:
            verdict = "RESPIRATION"
        elif n_met == 0:
            verdict = "REAL MOTION"
        else:
            verdict = "AMBIGUOUS"

        print(f"\n  ── Decision rule, Butterworth {FILTER_ORDERS[order]} ──")
        print(f"    (a) removed power in aliased band "
              f"{RESP_BAND[0]:.2f}-{RESP_BAND[1]:.2f} Hz : {100 * band_frac:5.1f}%  "
              f"(>= {100 * CRIT_A_MIN_BAND_FRAC:.0f}%)  {'PASS' if crit_a else 'FAIL'}")
        print(f"    (b) removed FD from {PE_AXIS:<8}                  : "
              f"{100 * pe_share:5.1f}%  (>= {100 * CRIT_B_MIN_PE:.0f}%)   "
              f"{'PASS' if pe_share >= CRIT_B_MIN_PE else 'fail'}")
        print(f"        removed FD from {PE_AXIS} + {PITCH_AXIS}           : "
              f"{100 * combined:5.1f}%  (>= {100 * CRIT_B_MIN_PE_PITCH:.0f}%)   "
              f"{'PASS' if combined >= CRIT_B_MIN_PE_PITCH else 'fail'}")
        print(f"    (c) rhythmicity of {PE_AXIS} (peak/median)      : "
              f"{rhythm:5.2f}   (>= {CRIT_C_MIN_RHYTHM:.1f})   {'PASS' if crit_c else 'FAIL'}")
        print(f"    VERDICT: {verdict}   ({n_met}/3 criteria met)")
        print(f"    HEADLINE: {100 * pe_share:.1f}% of removed FD from the "
              f"phase-encode axis")

        summary_rows.append({
            "subject": subject_id, "session": session_id, "group": group,
            "filter_order": order,
            "pct_removed_fd_phase_encode": float(100 * pe_share),
            "pct_removed_fd_pe_plus_pitch": float(100 * combined),
            "pct_removed_power_in_aliased_band": float(100 * band_frac),
            "rhythmicity_peak_over_median": rhythm,
            "aliased_band_power_present": "y" if crit_a else "n",
            "rhythmic": "y" if crit_c else "n",
            "phase_encode_dominant": "y" if crit_b else "n",
            "verdict": verdict,
        })

    # ── contrast check on the retained run ────────────────────────────────────
    if group == "retained":
        resid = filt_contrib[PRIMARY_ORDER]
        resid_pe = resid[pe_i] / resid.sum()
        print(f"\n  ── Contrast check (residual FD after order-{PRIMARY_ORDER} filter) ──")
        print(f"    {PE_AXIS} share of SURVIVING FD: {100 * resid_pe:.1f}%  "
              f"(chance 16.7%)")
        print(f"    Expected for real motion: broadband, no phase-encode dominance.")
        print(f"    -> {'NOT phase-encode dominant (test discriminates)' if resid_pe < CRIT_B_MIN_PE else 'PHASE-ENCODE DOMINANT — test may not discriminate'}")

    # ── figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    # Panel 1: PSD of all six parameters
    for i, ax_name in enumerate(MOTION_COLS):
        f, p = welch_psd(raw[:, i])
        style = "-" if ax_name.startswith("trans") else "--"
        width = 2.0 if ax_name in (PE_AXIS, PITCH_AXIS) else 1.0
        axes[0].semilogy(f, p, style, linewidth=width, label=ax_name)
    axes[0].axvspan(*RESP_BAND, color="#f0c000", alpha=0.20,
                    label=f"aliased respiration {RESP_BAND[0]:.2f}-{RESP_BAND[1]:.2f} Hz")
    axes[0].axvline(CUTOFF_HZ, color="black", linestyle=":", linewidth=1.4,
                    label=f"{CUTOFF_HZ} Hz cutoff")
    axes[0].set_xlim(0, NYQUIST)
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].set_ylabel("PSD (mm$^2$/Hz)")
    axes[0].set_title(f"Power spectral density of the 6 motion parameters "
                      f"(rotations as mm arc, {SPHERE_RADIUS:.0f} mm sphere)")
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")

    # Panel 2: per-axis FD contribution, raw vs each filter order
    x = np.arange(len(MOTION_COLS))
    nbar = 1 + len(FILTER_ORDERS)
    width = 0.8 / nbar
    axes[1].bar(x - 0.4 + width / 2, raw_contrib, width, label="raw", color="#999999")
    for k, order in enumerate(FILTER_ORDERS, start=1):
        axes[1].bar(x - 0.4 + width * (k + 0.5), filt_contrib[order], width,
                    label=f"low-pass order {order}",
                    color=["#4878CF", "#cc2222"][k - 1])
    for i, ax_name in enumerate(MOTION_COLS):
        axes[1].annotate(f"{100 * removed_frac[PRIMARY_ORDER][i]:.0f}%",
                         xy=(i, raw_contrib[i]), ha="center", va="bottom",
                         fontsize=8, color="#333333")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(
        [c + ("  (PE)" if c == PE_AXIS else "  (pitch)" if c == PITCH_AXIS else "")
         for c in MOTION_COLS])
    axes[1].set_ylabel("FD contribution (mm, summed)")
    axes[1].set_title(f"Per-axis FD contribution before vs after filtering  "
                      f"(% = share of removed FD, order {PRIMARY_ORDER})")
    axes[1].legend(fontsize=8)

    # Panel 3: phase-encode translation time trace
    pe_raw = raw[:, pe_i] - raw[:, pe_i].mean()
    pe_filt = filt[PRIMARY_ORDER][:, pe_i] - filt[PRIMARY_ORDER][:, pe_i].mean()
    frames = np.arange(len(pe_raw))
    axes[2].plot(frames, pe_raw, color="#999999", linewidth=0.9,
                 label=f"{PE_AXIS} raw (demeaned)")
    axes[2].plot(frames, pe_filt, color="#4878CF", linewidth=1.4,
                 label=f"{PE_AXIS} low-pass order {PRIMARY_ORDER}")
    axes[2].set_xlim(0, len(frames) - 1)
    axes[2].set_xlabel("Frame index")
    axes[2].set_ylabel("Displacement (mm)")
    axes[2].set_title(f"Phase-encode translation — rhythmic oscillation = respiration, "
                      f"isolated transients = real motion")
    axes[2].legend(fontsize=8, loc="upper right")

    head = summary_rows[-len(FILTER_ORDERS)]   # primary order row
    fig.suptitle(
        f"{label}  [{group}]   —   {head['pct_removed_fd_phase_encode']:.1f}% of removed FD "
        f"from {PE_AXIS} (order {PRIMARY_ORDER})   —   verdict: {head['verdict']}",
        fontsize=13,
    )
    plt.tight_layout(rect=[0.01, 0.01, 1, 0.97])
    fig_path = OUT_DIR / f"axis_test_{label}.png"
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Figure saved: {fig_path}\n")

# ── Summary table ─────────────────────────────────────────────────────────────
summary_df = pd.DataFrame(summary_rows)
axis_df = pd.DataFrame(axis_rows)
summary_path = OUT_DIR / "axis_test_summary.csv"
axis_path = OUT_DIR / "axis_test_per_axis.csv"
summary_df.to_csv(summary_path, index=False)
axis_df.to_csv(axis_path, index=False)

print("=" * 78)
print("SUMMARY")
print("=" * 78)
print(f"{'run':<18}{'grp':<11}{'ord':>4}{'%rm PE':>9}{'%rm PE+pitch':>14}"
      f"{'band':>7}{'rhythm':>8}  verdict")
for r in summary_rows:
    print(f"{r['subject'] + ' ' + r['session']:<18}{r['group']:<11}"
          f"{r['filter_order']:>4}{r['pct_removed_fd_phase_encode']:>8.1f}%"
          f"{r['pct_removed_fd_pe_plus_pitch']:>13.1f}%"
          f"{r['aliased_band_power_present']:>7}"
          f"{r['rhythmicity_peak_over_median']:>8.2f}  {r['verdict']}")

print(f"\nSummary table saved : {summary_path}")
print(f"Per-axis table saved: {axis_path}")
print("\n── Done ──")
print("  Nothing in the pipeline was modified. No censoring threshold was applied")
print("  to filtered FD — this is source attribution only.")
print(f"\n  Outputs in {OUT_DIR}:")
for p in sorted(OUT_DIR.iterdir()):
    print(f"    {p.name}")

sys.stdout.flush()
