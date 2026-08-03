import csv
from pathlib import Path

import pandas as pd

from storage import sync_down

DATA_LIBRARY_DIR = Path(__file__).resolve().parent.parent / "data_library"
PAIRS_PATH = DATA_LIBRARY_DIR / "groenwold_aave_sae" / "aave_sae_pairs.tsv"


def load_pairs() -> pd.DataFrame:
    """The 2019 intent-equivalent AAVE/SAE tweet pairs -- see
    ../data_library/groenwold_aave_sae/aave_sae_pairs.SOURCE.md
    for where this came from and which papers to cite. Row i here matches
    row i of the source tsv (no header, no reordering).

    Pulled fresh from DO Spaces first if configured (see ../storage)
    -- falls back to whatever's already at PAIRS_PATH if not."""
    sync_down(PAIRS_PATH)
    return pd.read_csv(
        PAIRS_PATH, sep="\t", header=None, names=["aave", "sae"], quoting=csv.QUOTE_NONE
    )
