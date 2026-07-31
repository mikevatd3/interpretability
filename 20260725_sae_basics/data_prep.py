import csv

import pandas as pd

from config import OUTER_PROMPT, PAIRS_PATH


def load_pairs() -> pd.DataFrame:
    """The 2019 intent-equivalent AAVE/SAE tweet pairs -- see
    ../../justhousingnotes/data/groenwold_aave_sae/aave_sae_pairs.SOURCE.md
    for where this came from and which papers to cite."""
    return pd.read_csv(
        PAIRS_PATH, sep="\t", header=None, names=["aave", "sae"], quoting=csv.QUOTE_NONE
    )


def prompt_text(text: str, variant: str) -> str:
    """Apply the judgment-prompt wrapper for variant="outer_prompt"; identity for "plain"."""
    if variant == "outer_prompt":
        return OUTER_PROMPT.format(t=text)
    if variant == "plain":
        return text
    raise ValueError(f"unknown variant: {variant!r}, expected 'plain' or 'outer_prompt'")


def prompted_pairs(variant: str) -> pd.DataFrame:
    """The pairs dataframe with `variant`'s prompt wrapper applied -- ready to feed to the model."""
    pairs = load_pairs()
    return pd.DataFrame(
        {
            "aave": pairs["aave"].map(lambda t: prompt_text(t, variant)),
            "sae": pairs["sae"].map(lambda t: prompt_text(t, variant)),
        }
    )
