import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.motion_qc import run_motion_qc

# ── Configurable paths and parameters ────────────────────────────────────────
CONFOUNDS_PATH = Path(
    "/BICNAS2/group-northoff/jkokino/data/dmt_med/derivatives/fmriprep/"
    "sub-01/ses-01/func/"
    "sub-01_ses-01_task-rest_desc-confounds_timeseries.tsv.gz"
)
_REPO_ROOT  = Path(__file__).resolve().parents[1]  # har_med_codes/, works from any cwd
FIGURES_DIR = _REPO_ROOT / "figures" / "individual"
RESULTS_DIR = _REPO_ROOT / "results" / "individual"
FD_THRESHOLD = 0.3    # mm (Goldberg et al. 2024)
DVARS_THRESHOLD = 1.5  # std_dvars (Power et al. 2014)
USE_CUSTOM_FD = True   # Goldberg/Lynch filtered FD (recommended)
COMPARE_FD = True      # overlay fMRIPrep FD on figure for sanity-check
# ─────────────────────────────────────────────────────────────────────────────

FIGURES_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

parts = CONFOUNDS_PATH.name.split("_")
subject_id = parts[0]
session_id = parts[1]

run_motion_qc(
    confounds_path=CONFOUNDS_PATH,
    subject_id=subject_id,
    session_id=session_id,
    output_dirs={"figures": FIGURES_DIR, "results": RESULTS_DIR},
    fd_thresh=FD_THRESHOLD,
    dvars_thresh=DVARS_THRESHOLD,
    use_custom_fd=USE_CUSTOM_FD,
    compare_fd=COMPARE_FD,
)
