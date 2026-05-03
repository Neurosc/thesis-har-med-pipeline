import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
REPO_ROOT     = Path(__file__).resolve().parents[1]
OUTPUT_DIR    = REPO_ROOT / "00_QC" / "figures" / "respiratory_check"

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


def _plot_spectrum_ax(ax, freqs, psd, param, peak_info, is_phase_encode):
    """Render a single PSD subplot."""
    color = PARAM_COLOR_MAP[param]

    # Highlight phase-encode panel
    if is_phase_encode:
        for spine in ax.spines.values():
            spine.set_edgecolor("#D65F5F")
            spine.set_linewidth(2.0)
        ax.set_facecolor("#fff5f5")

    # Respiratory band
    ax.axvspan(RESP_BAND_LOW, NYQUIST, color="red", alpha=0.10,
               label="Resp. band (direct + aliased)")

    # Aliased frequency markers
    alias_colors = ["#888888", "#aaaaaa", "#bbbbbb", "#cccccc"]
    for (lbl, af), ac in zip(ALIASED_FREQS.items(), alias_colors):
        if RESP_BAND_LOW <= af <= NYQUIST:
            ax.axvline(af, color=ac, linestyle=":", linewidth=0.9)

    # Nyquist line
    ax.axvline(NYQUIST, color="black", linestyle="--", linewidth=1.0,
               label=f"Nyquist ({NYQUIST:.3f} Hz)")

    ax.plot(freqs, psd, color=color, linewidth=1.2)

    # Mark detected peak
    if peak_info is not None:
        pf, pr = peak_info
        psd_at_peak = float(psd[np.argmin(np.abs(freqs - pf))])
        ax.plot(pf, psd_at_peak, "o", color="red", markersize=5, zorder=5)
        ax.annotate(
            f"{pf:.3f} Hz\n({pr:.1f}×)",
            xy=(pf, psd_at_peak),
            xytext=(pf + 0.01, psd_at_peak * 1.5),
            fontsize=6,
            color="red",
            arrowprops=dict(arrowstyle="-", color="red", lw=0.6),
        )

    ax.set_yscale("log")
    ax.set_xlim(0, NYQUIST)
    ax.set_xlabel("Frequency (Hz)", fontsize=7)
    ax.set_ylabel("Power", fontsize=7)
    ax.tick_params(labelsize=6)

    title = f"{param} *" if is_phase_encode else param
    ax.set_title(title, fontsize=9, fontweight="bold" if is_phase_encode else "normal")


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

            # Per-run figure
            fig, axes = plt.subplots(2, 3, figsize=(13, 7))
            fig.suptitle(
                f"{label} — motion parameter PSDs  "
                f"(TR={TR} s, Nyquist={NYQUIST:.3f} Hz)",
                fontsize=11, y=1.01,
            )

            for ax, param in zip(axes.flat, MOTION_COLS):
                freqs_p, psd_p = run_psds[param]
                _plot_spectrum_ax(
                    ax, freqs_p, psd_p, param,
                    run_peak_info[param],
                    is_phase_encode=(param == PHASE_ENCODE_PARAM),
                )

            # Shared legend on first subplot
            handles = [
                mpatches.Patch(color="red", alpha=0.25, label="Resp. band (direct + aliased)"),
                plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1,
                           label=f"Nyquist ({NYQUIST:.3f} Hz)"),
                plt.Line2D([0], [0], marker="o", color="red", linestyle="None",
                           markersize=5, label=f"Peak (>{PEAK_FACTOR:.0f}× baseline)"),
            ]
            axes.flat[0].legend(handles=handles, fontsize=6, loc="upper right")

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

# ── Step 5: Group summary figure (2 rows × 6 cols) ────────────────────────────
fig_grp, axes_grp = plt.subplots(
    2, 6, figsize=(22, 7), sharey="row",
    gridspec_kw={"hspace": 0.45, "wspace": 0.25},
)
fig_grp.suptitle(
    "Mean PSD by group and parameter — respiratory band shaded  "
    f"(TR={TR} s, Nyquist={NYQUIST:.3f} Hz)",
    fontsize=12,
)

row_labels  = ["Low-motion (n=10 runs)", "High-motion (n=10 runs)"]
group_keys  = ["low", "high"]

for row_idx, (group_key, row_label) in enumerate(zip(group_keys, row_labels)):
    for col_idx, param in enumerate(MOTION_COLS):
        ax = axes_grp[row_idx, col_idx]
        is_pe = param == PHASE_ENCODE_PARAM

        arrays = group_psds[group_key][param]
        if arrays:
            min_len = min(len(a) for a in arrays)
            stack = np.stack([a[:min_len] for a in arrays])
            mean_psd = np.mean(stack, axis=0)
            f_plot   = freqs_ref[:min_len] if freqs_ref is not None else \
                       np.linspace(0, NYQUIST, min_len)
            _plot_spectrum_ax(ax, f_plot, mean_psd, param, None, is_pe)

        if col_idx == 0:
            ax.set_ylabel(f"Power\n{row_label}", fontsize=8)

        # Highlight trans_y column title
        if is_pe:
            ax.set_title(f"{param} *\n(phase-encode)", fontsize=8,
                         color="#D65F5F", fontweight="bold")

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
