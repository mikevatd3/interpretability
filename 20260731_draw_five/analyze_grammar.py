"""analyze_grammar.py: tag each row's `completion` span with spaCy POS/
dependency info, for grammatical-construction comparison between AAVE- and
SAE-guise completions (see main.py / README.md for how draws_n50.csv,
this script's input, gets built).

Parses `full_text` (prompt + completion) rather than `completion` alone --
spaCy's tagger/parser does much better with sentence context than on a bare
short fragment -- then keeps only the spaCy tokens whose characters fall
within the completion span (found via Doc.char_span with
alignment_mode="expand", since GPT-2 BPE token boundaries don't necessarily
line up with spaCy's word-level tokenization).

Completion level: one row per (template_idx, pair_idx, variant), matching
draws_n50.csv's grain, not one row per spaCy token. The number of spaCy
tokens in a completion span varies (word-level tokenization doesn't line up
1:1 with GPT-2 BPE), so there's no fixed pos_1/pos_2/... column scheme the
way draws_n50.csv's token_1/token_2/... works -- instead each tag type gets
one space-joined *_sequence string column, in span order. `lead_word`/
`lead_pos`/`lead_tag`/`lead_dep` are pulled out as their own scalar columns
too (the span's first token), since that's the specific feature
compare_pairs.py compares -- consumers that want per-token counts (e.g.
compare_pos.py) split a *_sequence column back out with .str.split().
"""

import time

import pandas as pd
import spacy

from config import DATA_DIR, SPACY_MODEL, draws_csv_path, grammar_csv_path
from storage import sync_down, sync_up


def tag_rows(draws: pd.DataFrame, nlp) -> pd.DataFrame:
    records = []
    docs = nlp.pipe(draws["full_text"].tolist())

    for row, doc in zip(draws.itertuples(index=False), docs):
        span = doc.char_span(len(row.prompt), len(row.full_text), alignment_mode="expand")
        assert span is not None, f"couldn't align completion span for row {row.template_idx},{row.pair_idx},{row.variant}"
        tokens = list(span)
        lead = tokens[0]

        records.append(
            {
                "template_idx": row.template_idx,
                "pair_idx": row.pair_idx,
                "variant": row.variant,
                "completion": row.completion,
                "n_span_tokens": len(tokens),
                "lead_word": lead.text,
                "lead_pos": lead.pos_,
                "lead_tag": lead.tag_,
                "lead_dep": lead.dep_,
                "token_text_sequence": " ".join(t.text for t in tokens),
                "pos_sequence": " ".join(t.pos_ for t in tokens),
                "tag_sequence": " ".join(t.tag_ for t in tokens),
                "dep_sequence": " ".join(t.dep_ for t in tokens),
                "head_text_sequence": " ".join(t.head.text for t in tokens),
                "head_pos_sequence": " ".join(t.head.pos_ for t in tokens),
                "is_root_sequence": " ".join(str(t.dep_ == "ROOT") for t in tokens),
            }
        )

    return pd.DataFrame.from_records(records)


def main():
    nlp = spacy.load(SPACY_MODEL)
    draws_path = draws_csv_path()
    sync_down(draws_path)
    draws = pd.read_csv(draws_path)

    t0 = time.time()
    tagged = tag_rows(draws, nlp)
    print(f"tagged {len(draws)} completions in {time.time() - t0:.1f}s")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = grammar_csv_path()
    tagged.to_csv(out_path, index=False)
    print(f"wrote {len(tagged)} rows to {out_path}")
    sync_up(out_path)


if __name__ == "__main__":
    main()
