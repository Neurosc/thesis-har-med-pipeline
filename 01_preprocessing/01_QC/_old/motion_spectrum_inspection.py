import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal

# ── Configuration ─────────────────────────────────────────────────────────────
FMRIPREP_ROOT = Path("/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep")
_REPO_ROOT    = Path(__file__).resolve().parents[1]
_QC_DIR       = Path(__file__).resolve().parent
SPECTRA_DIR   = _QC_DIR / "figures" / "spectra"
TR            = 1.5
FS            = 1.0 / TR          # 0.667 Hz
NYQUIST       = FS / 2.0          # 0.333 Hz
RESP_LOW      = 0.15              # Hz — lower bound of aliased respiratory region
RESP_HIGH     = NYQUIST           # Hz — Nyquist is the upper bound we can observe
SESSIONS      = ["ses-01", "ses-02"]
TASK          = "rest"
MOTION_COLS   = ["trans_x", "trans_y", "trans_z", "rot_x", "rot_y", "rot_z"]
PEAK_RATIO    = 2.0               # respiratory band power > N× median = peak detected

LOW_MOTION_SUBJECTS  = ["sub-39", "sub-14", "sub-17", "sub-37", "sub-18"]
HIGH_MOTION_SUBJECTS = ["sub-12", "sub-21", "sub-26", "sub-33", "sub-11"]
SUBJECTS_TO_INSPECT  = LOW_MOTION_SUBJECTS + HIGH_MOTION_SUBJECTS
# ──────────────────────────────────────────────────────────────────────────────

SPECTRA_DIR.mkdir(parents=True, exist_ok=True)

PARAM_COLORS = [
    "#4878CF", "#D65F5F", "#6ACC65", "#B47CC7", "#C4AD66", "#77BEDB"
]

print(
    f"Motion parameter power spectrum inspection\n"
    f"  TR = {TR} s  |  fs = {FS:.4f} Hz  |  Nyquist = {NYQUIST:.4f} Hz\n"
    f"  Respiratory region shown: {RESP_LOW}–{RESP_HIGH:.4f} Hz\n"
    f"  Subjects: {len(SUBJECTS_TO_INSPECT)} total "
    f"({len(LOW_MOTION_SUBJECTS)} low-motion, {len(HIGH_MOTION_SUBJECTS)} high-motion)\n"
)

# ── Step 1–3: Per-run spectra ─────────────────────────────────────────────────
# Store PSD results for the group figure
# Structure: {"low": {param: [psd arrays]}, "high": {param: [psd arrays]}}
group_psds = {
    "low":  {p: [] for p in MOTION_COLS},
    "high": {p: [] for p in MOTION_COLS},
}
freqs_ref = None   # will be set on first run

peak_results = {sub: {} for sub in SUBJECTS_TO_INSPECT}

for subject_id in SUBJECTS_TO_INSPECT:
    motion_group = "low" if subject_id in LOW_MOTION_SUBJECTS else "high"

    for session_id in SESSIONS:
        label = f"{subject_id}_{session_id}"
        pattern = (
            f"{subject_id}_{session_id}_task-{TASK}_desc-confounds_timeseries.tsv.gz"
        )
        confounds_path = (
            FMRIPREP_ROOT / subject_id / session_id / "func" / pattern
        )
        if not confounds_path.exists():
            print(f"  WARNING: {label} — confounds not found, skipping.")
            peak_results[subject_id][session_id] = None
            continue

        try:
            df = pd.read_csv(confounds_path, sep="\t")
            missing = [c for c in MOTION_COLS if c not in df.columns]
            if missing:
                print(f"  WARNING: {label} — missing columns {missing}, skipping.")
                peak_results[subject_id][session_id] = None
                continue

            n_frames = df.shape[0]
            nperseg = min(128, n_frames)

            run_psds = {}
            has_resp_peak = False

            fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
            fig.suptitle(
                f"{label} — motion parameter power spectra", fontsize=12, y=1.01
            )

            for ax, param, color in zip(axes.flat, MOTION_COLS, PARAM_COLORS):
                ts = df[param].to_numpy(dtype=float)
                ts = signal.detrend(ts, type="linear")

                freqs, psd = signal.welch(ts, fs=FS, nperseg=nperseg, window="hann")

                # Restrict to 0–Nyquist
                mask = freqs <= NYQUIST
                freqs_plot = freqs[mask]
                psd_plot   = psd[mask]

                if freqs_ref is None:
                    freqs_ref = freqs_plot.copy()

                run_psds[param] = psd_plot
                group_psds[motion_group][param].append(psd_plot)

                # Detect respiratory peak
                resp_mask   = (freqs_plot >= RESP_LOW) & (freqs_plot <= RESP_HIGH)
                if resp_mask.any():
                    peak_power  = float(np.max(psd_plot[resp_mask]))
                    median_power = float(np.median(psd_plot))
                    if median_power > 0 and peak_power > PEAK_RATIO * median_power:
                        has_resp_peak = True

                ax.axvspan(RESP_LOW, RESP_HIGH, color="red", alpha=0.12,
                           label="Resp. region")
                ax.axvline(0.2,    color="gray", linestyle="--", linewidth=0.8)
                ax.axvline(NYQUIST, color="black", linestyle="--", linewidth=0.8,
                           label=f"Nyquist ({NYQUIST:.3f} Hz)")
                ax.plot(freqs_plot, psd_plot, color=color, linewidth=1.1)
                ax.set_yscale("log")
                ax.set_xlim(0, NYQUIST)
                ax.set_title(param, fontsize=10)
                ax.set_xlabel("Frequency (Hz)", fontsize=8)
                ax.set_ylabel("Power", fontsize=8)
                ax.tick_params(labelsize=7)

            # Shared legend on last subplot
            axes.flat[-1].legend(fontsize=7, loc="upper left")

            plt.tight_layout()
            fig_path = SPECTRA_DIR / f"{label}_motion_spectrum.png"
            plt.savefig(fig_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

            peak_results[subject_id][session_id] = has_resp_peak
            peak_str = "PEAK DETECTED" if has_resp_peak else "no peak"
            print(f"  {label}: {peak_str}  →  {fig_path.name}")

        except Exception as exc:
            print(f"  ERROR: {label} — {exc}")
            peak_results[subject_id][session_id] = None

# ── Step 4: Group summary figure ──────────────────────────────────────────────
fig_grp, (ax_low, ax_high) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
fig_grp.suptitle(
    "Average motion parameter PSD — low-motion vs high-motion subjects", fontsize=12
)

for ax, group_key, title_n, title_label in [
    (ax_low,  "low",  len(LOW_MOTION_SUBJECTS),  "Low-motion"),
    (ax_high, "high", len(HIGH_MOTION_SUBJECTS), "High-motion"),
]:
    for param, color in zip(MOTION_COLS, PARAM_COLORS):
        arrays = group_psds[group_key][param]
        if not arrays:
            continue
        min_len = min(len(a) for a in arrays)
        stack = np.stack([a[:min_len] for a in arrays])
        mean_psd = np.mean(stack, axis=0)
        f_plot = freqs_ref[:min_len] if freqs_ref is not None else np.linspace(0, NYQUIST, min_len)
        ax.plot(f_plot, mean_psd, color=color, linewidth=1.2, label=param)

    ax.axvspan(RESP_LOW, RESP_HIGH, color="red", alpha=0.12, label="Resp. region")
    ax.axvline(0.2,     color="gray",  linestyle="--", linewidth=0.8)
    ax.axvline(NYQUIST, color="black", linestyle="--", linewidth=0.8,
               label=f"Nyquist ({NYQUIST:.3f} Hz)")
    ax.set_yscale("log")
    ax.set_xlim(0, NYQUIST)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(f"Average PSD — {title_label} subjects (n={title_n * len(SESSIONS)} runs)")
    ax.legend(fontsize=8, ncol=2, loc="upper left")

plt.tight_layout()
grp_path = _QC_DIR / "figures" / "spectra_group_summary.png"
plt.savefig(grp_path, dpi=300, bbox_inches="tight")
plt.close(fig_grp)
print(f"\nGroup summary figure saved: {grp_path}")

# ── Step 5: Console summary ───────────────────────────────────────────────────
print("\n── Respiratory peak summary ──")

def _count_peaks(subjects):
    n_total = 0
    n_peak  = 0
    for sub in subjects:
        for ses in SESSIONS:
            result = peak_results.get(sub, {}).get(ses)
            if result is None:
                continue
            n_total += 1
            if result:
                n_peak += 1
    return n_peak, n_total

low_peak, low_total   = _count_peaks(LOW_MOTION_SUBJECTS)
high_peak, high_total = _count_peaks(HIGH_MOTION_SUBJECTS)

print(f"  Low-motion  runs with respiratory peak: {low_peak}/{low_total}")
print(f"  High-motion runs with respiratory peak: {high_peak}/{high_total}")

all_peak  = low_peak + high_peak
all_total = low_total + high_total
pct_with_peak = 100.0 * all_peak / all_total if all_total else 0.0

print(f"  Overall: {all_peak}/{all_total} runs ({pct_with_peak:.0f}%) show respiratory-band peaks")

if pct_with_peak > 50:
    conclusion = "Filtering may be justified — respiratory contamination present in >50% of runs."
else:
    conclusion = "Filter likely unnecessary — respiratory peaks present in ≤50% of runs."
print(f"\n  Conclusion: {conclusion}")

print(f"\n  Individual figures: {SPECTRA_DIR}/")
print(f"  Group figure      : {grp_path}")
