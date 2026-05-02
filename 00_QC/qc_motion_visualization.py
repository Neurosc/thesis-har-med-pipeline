import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Configurable paths and parameters ────────────────────────────────────────
CONFOUNDS_PATH = Path(
    "/BICNAS2/group-northoff/jkokino/"
    "sub-01/ses-01/func/"
    "sub-01_ses-01_task-rest_desc-confounds_timeseries.tsv"
)
OUTPUT_DIR = Path("/BICNAS2/group-northoff/jkokino/har_med_codes/figures")
RESULTS_DIR = Path("/BICNAS2/group-northoff/jkokino/har_med_codes/results")
FD_THRESHOLD = 0.3  # mm (Goldberg et al. 2024)
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Extract subject/session label from filename for titles and output names
subject_label = "_".join(CONFOUNDS_PATH.name.split("_")[:2])  # sub-XX_ses-YY

# Step 2: Load confounds
print(f"Loading confounds: {CONFOUNDS_PATH}")
df = pd.read_csv(CONFOUNDS_PATH, sep="\t")
print(f"  Shape: {df.shape[0]} frames x {df.shape[1]} columns")

if "framewise_displacement" not in df.columns or "dvars" not in df.columns:
    fd_cols = [c for c in df.columns if "fd" in c.lower() or "dvars" in c.lower()]
    raise ValueError(
        "Required columns 'framewise_displacement' and/or 'dvars' not found.\n"
        f"Columns containing 'fd' or 'dvars': {fd_cols}"
    )

# Step 3: Extract time series; replace NaN with 0 for plotting only
fd_raw = df["framewise_displacement"].to_numpy(dtype=float)
dvars_raw = df["dvars"].to_numpy(dtype=float)

fd_plot = np.where(np.isnan(fd_raw), 0.0, fd_raw)
dvars_plot = np.where(np.isnan(dvars_raw), 0.0, dvars_raw)

# Step 4: Summary statistics (exclude NaN — frame 0 is not a real displacement)
n_frames = len(fd_raw)
mean_fd = np.nanmean(fd_raw)
median_fd = np.nanmedian(fd_raw)
max_fd = np.nanmax(fd_raw)
n_censored = int(np.sum(fd_raw > FD_THRESHOLD, where=~np.isnan(fd_raw)))
pct_censored = 100.0 * n_censored / n_frames
mean_dvars = np.nanmean(dvars_raw)

print(f"\n── Motion summary: {subject_label} ──")
print(f"  Total frames        : {n_frames}")
print(f"  Mean FD             : {mean_fd:.4f} mm")
print(f"  Median FD           : {median_fd:.4f} mm")
print(f"  Max FD              : {max_fd:.4f} mm")
print(f"  Frames FD > {FD_THRESHOLD} mm : {n_censored}")
print(f"  Censored            : {pct_censored:.1f}%")
print(f"  Mean DVARS          : {mean_dvars:.4f}")

# Step 5: High-motion frame indices
censor_mask = np.where(np.isnan(fd_raw), False, fd_raw > FD_THRESHOLD)
censor_indices = np.where(censor_mask)[0].astype(int)

npy_path = RESULTS_DIR / f"{subject_label}_censored_frame_indices.npy"
np.save(npy_path, censor_indices)

# Step 6: 3-panel figure
fig, axes = plt.subplots(
    3, 1, figsize=(10, 8), sharex=True,
    gridspec_kw={"height_ratios": [3, 3, 1]}
)
fig.suptitle(f"Motion QC — {subject_label}", fontsize=13, y=1.01)

frames = np.arange(n_frames)

# Panel 1: FD
axes[0].plot(frames, fd_plot, color="steelblue", linewidth=0.9)
axes[0].axhline(FD_THRESHOLD, color="black", linestyle="--", linewidth=1,
                label=f"Threshold ({FD_THRESHOLD} mm)")
axes[0].set_ylabel("FD (mm)")
axes[0].set_title("Framewise Displacement")
axes[0].legend(fontsize=8, loc="upper right")

# Panel 2: DVARS
axes[1].plot(frames, dvars_plot, color="darkorange", linewidth=0.9)
axes[1].set_ylabel("DVARS")
axes[1].set_title("DVARS (raw, before denoising)")

# Panel 3: censored frames strip
censor_strip = censor_mask.astype(float).reshape(1, -1)
cmap = matplotlib.colors.ListedColormap(["#d0d0d0", "#cc2222"])
axes[2].imshow(censor_strip, aspect="auto", cmap=cmap,
               vmin=0, vmax=1, interpolation="nearest")
axes[2].set_yticks([])
axes[2].set_xlabel("Frame index")
axes[2].set_title(f"Censored frames (FD > {FD_THRESHOLD} mm)")

plt.tight_layout()

fig_path = OUTPUT_DIR / f"{subject_label}_motion_qc.png"
plt.savefig(fig_path, dpi=300, bbox_inches="tight")
plt.close(fig)

# Step 7: Done
print(f"\n  Figure saved : {fig_path}")
print(f"  Indices saved: {npy_path}")
