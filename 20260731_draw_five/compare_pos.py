"""compare_pos.py: AAVE vs SAE POS-tag distribution comparison over
analyze_grammar.py's tagged completions (grammar_n50.csv).

grammar_n50.csv is completion-level (one row per (template, pair, variant),
pos_sequence a space-joined string of that completion's spaCy tokens' POS
tags) -- explode() splits pos_sequence back into one row per token here,
since the frequency comparison below is inherently token-level. Those
tokens aren't independent draws (they share a continuation and most of
their preceding context), so the two-proportion z-test below is a rough
signal of where the two distributions diverge, not a rigorous significance
test. Treat p-values as descriptive, not confirmatory.
"""

import math

import pandas as pd

from config import DATA_DIR, grammar_csv_path, pos_compare_csv_path
from interp_storage import sync_down, sync_up


def explode_pos(completions: pd.DataFrame) -> pd.DataFrame:
    """One row per spaCy token, from completions' pos_sequence column."""
    return completions.assign(pos=completions["pos_sequence"].str.split()).explode("pos")


def two_proportion_z(x1: float, n1: float, x2: float, n2: float) -> tuple[float, float]:
    """z-stat and two-sided p-value for comparing two sample proportions
    (x1/n1 vs x2/n2), via a pooled-proportion normal approximation."""
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_value


def compare(tagged: pd.DataFrame) -> pd.DataFrame:
    counts = tagged.groupby(["variant", "pos"]).size().unstack("variant", fill_value=0)
    for variant in ("aave", "sae"):
        if variant not in counts.columns:
            counts[variant] = 0

    n_aave, n_sae = counts["aave"].sum(), counts["sae"].sum()
    counts["aave_freq"] = counts["aave"] / n_aave
    counts["sae_freq"] = counts["sae"] / n_sae
    counts["diff"] = counts["sae_freq"] - counts["aave_freq"]  # positive => more common in sae

    z_stats, p_values = zip(
        *(two_proportion_z(row["aave"], n_aave, row["sae"], n_sae) for _, row in counts.iterrows())
    )
    counts["z"] = z_stats
    counts["p_value"] = p_values

    counts = counts.rename(columns={"aave": "aave_count", "sae": "sae_count"}).reset_index()
    return counts.sort_values("diff", key=lambda s: s.abs(), ascending=False)


def main():
    grammar_path = grammar_csv_path()
    sync_down(grammar_path)
    completions = pd.read_csv(grammar_path)
    tagged = explode_pos(completions)
    table = compare(tagged)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = pos_compare_csv_path()
    table.to_csv(out_path, index=False)

    with pd.option_context("display.float_format", "{:.3f}".format, "display.width", 120):
        print(table.to_string(index=False))
    print(f"\nwrote {out_path}")
    sync_up(out_path)


if __name__ == "__main__":
    main()
