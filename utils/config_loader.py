"""Read active analysis configuration from config.toml at repo root."""
from pathlib import Path
import tomllib  # Python 3.11+

REPO_ROOT   = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config.toml"


def load_config() -> dict:
    with CONFIG_PATH.open("rb") as f:
        config = tomllib.load(f)
    return {
        "denoising_method": config["active"]["denoising_method"],
        "atlas_method":     config["active"]["atlas_method"],
    }


def tag() -> str:
    """Return tag string like 'parcels_NoGSR' for filename/folder use."""
    c = load_config()
    return f"{c['atlas_method']}_{c['denoising_method']}"
