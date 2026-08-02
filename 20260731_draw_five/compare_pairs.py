"""compare_pairs.py: paired aave-vs-sae comparison of completions, restricted
to (template, pair) combinations where the completion's lead word -- its
first predicted token, the one that fills the template's predicate-adjective
slot ("...tends to be ___", "...They are ___") -- is an adjective on at
least one side.

Paired, not pooled: for the SAME (template_idx, pair_idx) -- the same
template applied to the aave and sae side of the same underlying tweet --
what word did the model pick on each side? This is the comparison
compare_pos.py's marginal POS-frequency table can't make, since pooling
throws away which aave row goes with which sae row.

"Lead word" is operationalized as the completion's first spaCy token
(analyze_grammar.py's lead_word/lead_pos columns), not spaCy's dependency
ROOT -- the ROOT of the full parse essentially never falls on the trait
adjective here, since the adjective typically attaches as an `acomp`
dependent of the template's own copula ("be"/"is"/"are"), which sits just
outside the completion span in the prompt.
"""

import math

import pandas as pd

from config import DATA_DIR, adj_pairs_csv_path, draws_csv_path, grammar_csv_path
from storage import sync_down, sync_up


def mcnemar(b: int, c: int) -> tuple[float, float]:
    """McNemar's test on a 2x2 paired-binary table's two discordant cells
    (b = aave-yes/sae-no, c = aave-no/sae-yes). Continuity-corrected chi2
    statistic (1 df) and its p-value, computed without scipy."""
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # Yates' continuity correction
    # 1 - CDF of chi2(1 df) == erfc(sqrt(chi2/2))
    p_value = math.erfc(math.sqrt(chi2 / 2))
    return chi2, p_value


def lead_words(grammar: pd.DataFrame) -> pd.DataFrame:
    """Each (template_idx, pair_idx, variant)'s lead word and its POS tag --
    already computed at the completion level by analyze_grammar.py."""
    return grammar[["template_idx", "pair_idx", "variant", "lead_word", "lead_pos"]]


def paired_table(draws: pd.DataFrame, lead: pd.DataFrame) -> pd.DataFrame:
    """One row per (template_idx, pair_idx) -- every combo, not just
    adjective-led ones -- with both sides' lead word/POS side by side."""
    merged = draws.merge(lead, on=["template_idx", "pair_idx", "variant"], how="left")
    wide = merged.pivot(
        index=["template_idx", "pair_idx"],
        columns="variant",
        values=["lead_word", "lead_pos", "completion"],
    )
    wide.columns = [f"{variant}_{field}" for field, variant in wide.columns]
    wide = wide.reset_index()

    wide["both_adj"] = (wide["aave_lead_pos"] == "ADJ") & (wide["sae_lead_pos"] == "ADJ")
    wide["either_adj"] = (wide["aave_lead_pos"] == "ADJ") | (wide["sae_lead_pos"] == "ADJ")
    wide["same_word"] = wide["aave_lead_word"].str.strip().str.lower() == wide[
        "sae_lead_word"
    ].str.strip().str.lower()

    return wide.sort_values(["template_idx", "pair_idx"])


def main():
    draws_path, grammar_path = draws_csv_path(), grammar_csv_path()
    sync_down(draws_path)
    sync_down(grammar_path)
    draws = pd.read_csv(draws_path)
    grammar = pd.read_csv(grammar_path)
    lead = lead_words(grammar)

    table = paired_table(draws, lead)
    n_combos = len(table)

    aave_adj = table["aave_lead_pos"] == "ADJ"
    sae_adj = table["sae_lead_pos"] == "ADJ"
    b = int((aave_adj & ~sae_adj).sum())  # aave leads ADJ, sae doesn't
    c = int((~aave_adj & sae_adj).sum())  # sae leads ADJ, aave doesn't
    chi2, p_value = mcnemar(b, c)

    print(f"marginal adjective-lead rate: aave {aave_adj.sum()}/{n_combos} ({aave_adj.mean():.1%}), "
          f"sae {sae_adj.sum()}/{n_combos} ({sae_adj.mean():.1%})")
    print(f"paired (McNemar): aave-only-ADJ={b}, sae-only-ADJ={c}, chi2={chi2:.2f}, p={p_value:.4f}")

    either = table[table["either_adj"]]
    print(f"\n{len(either)} / {n_combos} (template, pair) combos have an adjective lead word on at least one side")
    print(f"  both sides adjective-led: {int(table['both_adj'].sum())}")
    both = table[table["both_adj"]]
    print(f"  of those, same lead word on both sides: {int(both['same_word'].sum())} / {len(both)}")

    print("\n=== both sides adjective-led, DIFFERENT word ===")
    diff = both[~both["same_word"]]
    for _, row in diff.head(15).iterrows():
        print(f"  [t{row.template_idx},p{row.pair_idx}] aave={row.aave_lead_word!r:>15} sae={row.sae_lead_word!r:>15}")

    print("\n=== only one side adjective-led ===")
    one_side = either[~either["both_adj"]]
    for _, row in one_side.head(15).iterrows():
        which = "aave" if row.aave_lead_pos == "ADJ" else "sae"
        other = "sae" if which == "aave" else "aave"
        print(f"  [t{row.template_idx},p{row.pair_idx}] {which}={row[f'{which}_lead_word']!r} (ADJ), "
              f"{other}={row[f'{other}_lead_word']!r} ({row[f'{other}_lead_pos']})")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = adj_pairs_csv_path()
    table.to_csv(out_path, index=False)
    print(f"\nwrote {len(table)} rows to {out_path}")
    sync_up(out_path)


if __name__ == "__main__":
    main()
