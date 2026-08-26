"""Central path configuration for analysis code.

Data and fig locations are resolved here to avoid hardcoding paths in nbs and modules.
Both default to gitignored folders inside the repo, override w/
environment vars if your data or figures live elsewhere (like in Docker).
"""
import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]   # repo root (src/ is one level below)

# Reconstructed additional-data files live here, populated from DANDI by
# notebooks/reconstruct_additional_data_files.ipynb
DATA_DIR = Path(os.environ.get("SPATIAL_BANDIT_DATA_DIR", _REPO / "data"))

# Figure panels get written here
FIG_DIR = Path(os.environ.get("SPATIAL_BANDIT_FIG_DIR", _REPO / "figures"))

# original DB-stored file paths data were written under, used by to_local()
# to remap a stored path to reconstructed copy under DATA_DIR
_SOURCE_ROOT = "/stelmo/alison"


def to_local(p):
    """map a DB-stored /stelmo/alison/... path to reconstructed copy under DATA_DIR

    Non-matching paths go through unchanged.
    Used for data-sharing / DANDI compatibility, as on the orig machine the file is at
    /stelmo path while for shared use it lives under DATA_DIR after file reconstruction
    """
    p = str(p)
    if p.startswith(_SOURCE_ROOT):
        return str(DATA_DIR / p[len(_SOURCE_ROOT):].lstrip("/"))
    return p

def fig_dir(sub=""):
    """return FIG_DIR/<sub> as a string ending in '/', creating it if needed.
    Use as base path for saving fig panels"""
    d = FIG_DIR / sub
    d.mkdir(parents=True, exist_ok=True)
    return str(d) + "/"