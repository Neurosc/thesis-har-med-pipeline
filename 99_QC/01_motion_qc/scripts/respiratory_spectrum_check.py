import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal

# ── Configuration ─────────────────────────────────────────────────────────────
TR                = 1.8
FS                = 1.0 / TR          # 0.556 Hz
NYQUIST           = FS / 2.0          # 0.278 Hz
PHASE_ENCODE_PARAM = "trans_y"        # BIDS PhaseEncodingDirection: j

LOW_MOTION_SUBJECTS  = ["sub-39", "sub-14", "sub-17", "sub-37", "sub-18"]
HIGH_MOTION_SUBJECTS = ["sub-12", "sub-21", "sub-26", "sub-33", "sub-11"]
SUBJECTS_TO_INSPECT  = LOW_MOTION_SUBJECTS + HIGH_MOTION_SUBJECTS
SESSIONS             = ["ses-01", "ses-02"]
TASK                 = "rest"

FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
REPO_ROOT     = Path(__file__).resolve().parents[3]
OUTPUT_DIR    = REPO_ROOT / "01_preprocessing" / "01_QC" / "figures" / "respiratory_check"

MOTION_COLS   = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
RESP_BAND_LOW = 0.05     # Hz — lower edge of observable respiratory band
PEAK_FACTOR   = 5.0      # PSD must exceed N× rolling-median baseline to count as a peak
BASELINE_BW   = 0.05     # Hz — rolling-median window width for baseline

# Aliased respiratory frequencies for TR=1.8 s (Nyquist=0.278 Hz)
ALIASED_FREQS = {
    "0.20 Hz (direct)":  0.200,
    "0.30 Hz → aliased": abs(0.30 - 2 * NYQUIST),   # 0.256 Hz
    "0.40 Hz → aliased": abs(0.40 - 2 * NYQUIST),   # 0.156 Hz
    "0.50 Hz → aliased": abs(0.50 - 2 * NYQUIST),   # 0.056 Hz
}
# ──────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PARAM_COLORS = [
    "#4878CF", "#D65F5F", "#6ACC65", "#B47CC7", "#C4AD66", "#77BEDB"
]
PARAM_COLOR_MAP = dict(zip(MOTION_COLS, PARAM_COLORS))

print(
    "─── Respiratory Spectral Check (Power 2019 method) ───\n"
    f"TR = {TR} s, Nyquist = {NYQUIST:.3f} Hz\n"
    f"Phase-encode parameter: {PHASE_ENCODE_PARAM} (BIDS PhaseEncodingDirection: j)\n"
    "\nAliased respiratory frequencies in our spectrum:"
)
for label, f in ALIASED_FREQS.items():
    print(f"  Real {label}: observed at {f:.3f} Hz")
print()


def _rolling_median_baseline(freqs, psd, bw_hz):
    """Compute a rolling-median baseline over a frequency window of bw_hz."""
    df = freqs[1] - freqs[0] if len(freqs) > 1 else bw_hz
    half_n = max(1, int(round(bw_hz / (2 * df))))
    baseline = np.empty_like(psd)
    n = len(psd)
    for i in range(n):
        lo = max(0, i - half_n)
        hi = min(n, i + half_n + 1)
        baseline[i] = np.median(psd[lo:hi])
    return baseline


def _detect_peak(freqs, psd, band_low, band_high, factor):
    """
    Return (peak_freq, peak_ratio) if a robust peak is found, else None.
    A peak must be > factor × local baseline AND a local maximum.
    """
    band_mask = (freqs >= band_low) & (freqs <= band_high)
    if not band_mask.any():
        return None

    baseline = _rolling_median_baseline(freqs, psd, BASELINE_BW)
    ratio = np.where(baseline > 0, psd / baseline, 0.0)

    band_freqs = freqs[band_mask]
    band_ratio = ratio[band_mask]
    band_psd   = psd[band_mask]

    # Local maxima within band
    local_max_idx = []
    for i in range(1, len(band_psd) - 1):
        if band_psd[i] > band_psd[i - 1] and band_psd[i] > band_psd[i + 1]:
            local_max_idx.append(i)
    if not local_max_idx:
        # Fallback: global max
        local_max_idx = [int(np.argmax(band_psd))]

    best_ratio = 0.0
    best_freq  = None
    for idx in local_max_idx:
        if band_ratio[idx] > best_ratio:
            best_ratio = band_ratio[idx]
            best_freq  = band_freqs[idx]

    if best_ratio >= factor and best_freq is not None:
        return (float(best_freq), float(best_ratio))
    return None


def _plot_overlay_ax(ax, param_psds, title, xlabel=True):
    """
    Plot all 6 motion parameters as overlaid lines on a single axes.

    param_psds: dict of param → (freqs, psd)
    trans_y (phase-encode) is drawn with linewidth=2.5; all others linewidth=1.0.
    """
    for param, (freqs, psd) in param_psds.items():
        lw = 2.5 if param == PHASE_ENCODE_PARAM else 1.0
        ax.plot(freqs, psd, color=PARAM_COLOR_MAP[param], linewidth=lw,
                label=f"{param} *" if param == PHASE_ENCODE_PARAM else param)

    ax.axvline(NYQUIST, color="black", linestyle="--", linewidth=1.0,
               label=f"Nyquist ({NYQUIST:.3f} Hz)")
    ax.set_yscale("log")
    ax.set_xlim(0, NYQUIST)
    ax.set_title(title, fontsize=10)
    if xlabel:
        ax.set_xlabel("Frequency (Hz)", fontsize=9)
    ax.set_ylabel("Power", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", ncol=2)


# ── Steps 1–4: Per-run computation and figures ────────────────────────────────
# Store PSDs for group figure
group_psds = {
    "low":  {p: [] for p in MOTION_COLS},
    "high": {p: [] for p in MOTION_COLS},
}
freqs_ref = None

# Peak tracking: {subject: {session: {param: (freq, ratio) or None}}}
all_peaks = {sub: {} for sub in SUBJECTS_TO_INSPECT}

for subject_id in SUBJECTS_TO_INSPECT:
    motion_group = "low" if subject_id in LOW_MOTION_SUBJECTS else "high"

    for session_id in SESSIONS:
        label   = f"{subject_id}_{session_id}"
        pattern = (
            f"{subject_id}_{session_id}_task-{TASK}_desc-confounds_timeseries.tsv.gz"
        )
        cfpath = FMRIPREP_ROOT / subject_id / session_id / "func" / pattern

        if not cfpath.exists():
            print(f"  WARNING: {label} — confounds not found, skipping.")
            all_peaks[subject_id][session_id] = None
            continue

        try:
            df = pd.read_csv(cfpath, sep="\t")
            missing = [c for c in MOTION_COLS if c not in df.columns]
            if missing:
                print(f"  WARNING: {label} — missing columns {missing}, skipping.")
                all_peaks[subject_id][session_id] = None
                continue

            run_peak_info = {}
            run_psds      = {}

            for param in MOTION_COLS:
                ts = df[param].to_numpy(dtype=float)
                # Drop leading NaN defensively
                first_valid = 0
                while first_valid < len(ts) and np.isnan(ts[first_valid]):
                    first_valid += 1
                ts = ts[first_valid:]

                ts = signal.detrend(ts, type="linear")

                nperseg = min(128, len(ts))
                freqs_full, psd_full = signal.welch(
                    ts, fs=FS, nperseg=nperseg, window="hann"
                )

                mask = freqs_full <= NYQUIST
                freqs_plot = freqs_full[mask]
                psd_plot   = psd_full[mask]

                if freqs_ref is None:
                    freqs_ref = freqs_plot.copy()

                run_psds[param] = (freqs_plot, psd_plot)
                group_psds[motion_group][param].append(psd_plot)

                peak = _detect_peak(
                    freqs_plot, psd_plot, RESP_BAND_LOW, NYQUIST, PEAK_FACTOR
                )
                run_peak_info[param] = peak

            all_peaks[subject_id][session_id] = run_peak_info

            # Per-run figure — single axes, all 6 parameters overlaid
            fig, ax = plt.subplots(figsize=(8, 5))
            _plot_overlay_ax(
                ax, run_psds,
                title=(
                    f"{label} — motion parameter PSDs  "
                    f"(TR={TR} s, Nyquist={NYQUIST:.3f} Hz)"
                ),
            )
            plt.tight_layout()
            fig_path = OUTPUT_DIR / f"{label}_respiratory.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            ty_peak = run_peak_info.get(PHASE_ENCODE_PARAM)
            peak_str = (
                f"peak @ {ty_peak[0]:.3f} Hz ({ty_peak[1]:.1f}× baseline)"
                if ty_peak else "no peak"
            )
            print(f"  {label}: trans_y → {peak_str}")

        except Exception as exc:
            print(f"  ERROR: {label} — {exc}")
            all_peaks[subject_id][session_id] = None

# ── Step 5: Group summary figure — 2 panels (low / high), overlaid lines ──────
fig_grp, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(14, 5))
fig_grp.suptitle(
    f"Mean PSD by group — all motion parameters overlaid  "
    f"(TR={TR} s, Nyquist={NYQUIST:.3f} Hz)",
    fontsize=12,
)

for ax, group_key, title in [
    (ax_low,  "low",  "Low-motion subjects (n=10 runs)"),
    (ax_high, "high", "High-motion subjects (n=10 runs)"),
]:
    mean_psds = {}
    for param in MOTION_COLS:
        arrays = group_psds[group_key][param]
        if not arrays:
            continue
        min_len = min(len(a) for a in arrays)
        stack = np.stack([a[:min_len] for a in arrays])
        f_plot = freqs_ref[:min_len] if freqs_ref is not None else \
                 np.linspace(0, NYQUIST, min_len)
        mean_psds[param] = (f_plot, np.mean(stack, axis=0))
    _plot_overlay_ax(ax, mean_psds, title=title)

plt.tight_layout()
grp_path = OUTPUT_DIR / "group_summary.png"
plt.savefig(grp_path, dpi=300, bbox_inches="tight")
plt.close(fig_grp)
print(f"\nGroup summary figure saved: {grp_path}")

# ── Step 6: Console summary ───────────────────────────────────────────────────
def _peak_counts(subjects, param):
    total, n_peak = 0, 0
    example = None
    for sub in subjects:
        for ses in SESSIONS:
            entry = all_peaks.get(sub, {}).get(ses)
            if entry is None:
                continue
            total += 1
            pk = entry.get(param)
            if pk is not None:
                n_peak += 1
                if example is None:
                    example = (sub, ses, pk[0], pk[1])
    return n_peak, total, example

print("\n─── Peak detection results (PSD > 5× baseline in 0.05–0.278 Hz band) ───\n")
print(f"trans_y (phase-encode):")
low_n,  low_t,  low_ex  = _peak_counts(LOW_MOTION_SUBJECTS,  PHASE_ENCODE_PARAM)
high_n, high_t, high_ex = _peak_counts(HIGH_MOTION_SUBJECTS, PHASE_ENCODE_PARAM)
print(f"  Low-motion  runs with peak: {low_n}/{low_t}")
print(f"  High-motion runs with peak: {high_n}/{high_t}")
ex = low_ex or high_ex
if ex:
    print(f"  Example peak: {ex[0]}_{ex[1]} at {ex[2]:.3f} Hz ({ex[3]:.1f}× baseline)")

print("\nOther parameters (informational):")
for param in [p for p in MOTION_COLS if p != PHASE_ENCODE_PARAM]:
    n_all, t_all, _ = _peak_counts(SUBJECTS_TO_INSPECT, param)
    print(f"  {param}: {n_all}/{t_all}")

ty_total_peak = low_n + high_n
ty_total_runs = low_t + high_t
pct_peak = 100.0 * ty_total_peak / ty_total_runs if ty_total_runs else 0.0

print("\n─── Recommendation ───")
if pct_peak > 50:
    print(
        f"  Respiratory contamination detected in phase-encode parameter "
        f"({ty_total_peak}/{ty_total_runs} runs, {pct_peak:.0f}%).\n"
        "  Bandstop filtering of motion parameters at 0.2–Nyquist Hz is "
        "recommended before FD computation."
    )
else:
    print(
        f"  No clear respiratory contamination in phase-encode parameter "
        f"({ty_total_peak}/{ty_total_runs} runs, {pct_peak:.0f}%).\n"
        "  Filtering is not empirically justified for this dataset.\n"
        "  Recommend using standard Power 2012 FD (no filter) and a "
        "censoring threshold of FD > 0.3 or 0.5 mm."
    )

print(f"\nSaved files:")
print(f"  Individual figures : {OUTPUT_DIR}/")
print(f"  Group summary      : {grp_path}")
