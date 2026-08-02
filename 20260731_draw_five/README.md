# Draw five

For every `(template, pair, variant)` combination — 9 `TRAIT_PROMPTS`
templates × `N_PAIRS` AAVE/SAE tweet pairs × 2 sides (`aave`/`sae`) —
predicts a **10-token continuation** from GPT-2 and records one row per
combination: the filled-in template plus its predicted continuation,
building a dataset for later analysis of how AAVE- vs SAE-phrased prompts
diverge over a short generation, not just at the very next token.

This is a sibling of `../20260725_sae_basics/collect_activations_all_templates.py`
and `../20260731_causal_tracing/`'s "walk every template × pair" pattern, but
instead of extracting SAE activations or a single scalar score, it produces
predicted continuations.

## Method

Each of the 5 tokens is predicted with **temperature + top-p (nucleus)
sampling** — one full forward pass over the tokens so far, softmax the final
position's logits at `TEMPERATURE`, truncate to the smallest set of
highest-probability tokens whose cumulative probability crosses `TOP_P`,
sample from that renormalized set, append, repeat. This approximates
realistic chatbot-style decoding: pure greedy (always the single
highest-probability token) tends to be dull/repetitive, and sampling over
the full untruncated vocabulary can pick bizarre, very-low-probability
tokens — top-p sampling is the standard middle ground most production chat
models actually use.

Two deliberate simplifications, worth knowing about before scaling this up:

- **Batched, no KV cache** — `BATCH_SIZE` prompts share one forward pass per
  step (left-padded; see `batch_predict_continuations`' docstring in
  `main.py` for how the padding/attention-mask handling is verified
  correct), but each step still re-runs the full sequence through the model
  rather than incrementally decoding. A from-scratch KV cache attempt only
  bought a 1.22x speedup on CPU (`transformer_lens`'s per-layer hook
  overhead dominates wall-clock time there far more than the attention
  recompute being cached) -- batching alone got the per-draw rate down
  ~3.2x, and is expected to matter even more on GPU. See `RUNPOD.md`.
- **No early stopping at EOS** — every prompt always predicts exactly
  `N_DRAWS` tokens, even if `<|endoftext|>` gets picked mid-continuation, so
  the output row count is a fixed, checkable `9 * N_PAIRS * 2`.

## Knobs (`config.py`)

- `N_PAIRS = 50` — `None` for the full 2019-pair dataset. Overridable via
  the `N_PAIRS` env var (e.g. `N_PAIRS=all uv run main.py`) without editing
  the file -- see `RUNPOD.md`.
- `N_DRAWS = 10` — tokens predicted per prompt.
- `BATCH_SIZE = 16` — prompts per forward pass in `batch_predict_continuations`.
  16 is CPU-tuned (32 measured *slower* on CPU -- no real parallelism headroom,
  more padding waste); bump this a lot (128+) on GPU. Overridable via the
  `BATCH_SIZE` env var.
- `TEMPERATURE = 0.7`, `TOP_P = 0.9` — standard chat-style decoding defaults.
- `SEED = 76` — fixes `torch.manual_seed` so a re-run on the same
  machine/device reproduces the same predictions. (Cross-device
  reproducibility, e.g. CPU vs MPS, isn't guaranteed by PyTorch.)

## Run

```
uv sync
uv run main.py
```

## DigitalOcean Spaces

Every script here reads/writes through `../storage` (a shared
package, not local to this experiment -- see the root `README.md`'s
"Shared infrastructure" section) if Spaces is configured (silent no-op
otherwise): input pulled fresh from the bucket before each script reads
it, output pushed up after each script writes it. This is how a run on a
different machine (e.g. a rented GPU pod, see `RUNPOD.md`) gets its input
data and returns its output, without a shared filesystem or manual rsync
of data.

See the root `README.md`'s "DigitalOcean Spaces" section for the required
`.env` keys. Before a fresh machine's first run, the bucket needs the one
input file this project reads
(`../data_library/groenwold_aave_sae/aave_sae_pairs.tsv`) already in it --
`sync_down` silently falls back to a local copy if the key isn't there
yet, which won't exist on a fresh pod. Seed it once from a machine that
already has the file locally:
```
uv run load-to-spaces ../data_library/groenwold_aave_sae/aave_sae_pairs.tsv
```

## Output

`data/draws_n50.csv` — one row per `(template, pair, variant)`:

| column | meaning |
|---|---|
| `template_idx` | 0-8, index into `TRAIT_PROMPTS` |
| `template` | the raw `{t}`-templated string |
| `pair_idx` | row index into the pairs dataframe (0..`N_PAIRS`-1) |
| `variant` | `"aave"` or `"sae"` |
| `source_text` | the tweet text substituted for `{t}` |
| `prompt` | the fully formatted prompt before prediction |
| `token_1` … `token_10` | the `N_DRAWS` predicted tokens, in order |
| `prob_1` … `prob_10` | each token's probability under the model at that step (pre-truncation, temperature-scaled) |
| `completion` | `token_1` through `token_10` concatenated |
| `full_text` | `prompt` + `completion` — the template, fully filled in |

`template_idx`, `pair_idx`, and `variant` together uniquely identify each
row and can be used to join back to `TRAIT_PROMPTS`/the source tsv.

See `../data_library/groenwold_aave_sae/aave_sae_pairs.SOURCE.md` for pair
provenance/citation, and the root `README.md`'s TODO for why `TRAIT_PROMPTS`/
`get_device()` are copy-pasted here rather than imported from a sibling.

## Grammatical analysis (`analyze_grammar.py`)

Tags each row's `completion` with spaCy (`en_core_web_sm`) POS/dependency
info, for comparing the grammatical construction of AAVE- vs SAE-guise
completions. Parses `full_text` rather than `completion` alone (spaCy's
tagger/parser does much better with sentence context than on a bare
short fragment), then keeps only the spaCy tokens whose characters fall within the
completion span (`Doc.char_span(..., alignment_mode="expand")`, since GPT-2
BPE token boundaries don't necessarily line up with spaCy's word-level
tokenization).

```
uv run analyze_grammar.py
```

Writes `data/grammar_n50.csv`, **completion level** -- one row per
`(template_idx, pair_idx, variant)`, matching `draws_n50.csv`'s grain, not
one row per spaCy token. The number of spaCy tokens per completion varies
(word-level tokenization doesn't line up 1:1 with GPT-2 BPE), so there's no
fixed `pos_1`/`pos_2`/... column scheme the way `draws_n50.csv`'s
`token_1`/`token_2`/... works -- instead each tag type is one space-joined
`*_sequence` string column, in span order:

| column | meaning |
|---|---|
| `template_idx`, `pair_idx`, `variant` | join back to `draws_n50.csv` |
| `completion` | the row's full `N_DRAWS`-token completion, for reference |
| `n_span_tokens` | how many spaCy tokens are in this completion's span |
| `lead_word`, `lead_pos`, `lead_tag`, `lead_dep` | the span's *first* spaCy token -- pulled out as its own scalar columns since it's the specific feature `compare_pairs.py` compares (the word filling the template's predicate-adjective slot) |
| `token_text_sequence` | the span's tokens' text, space-joined, in order |
| `pos_sequence` | coarse POS tags (`token.pos_`), space-joined |
| `tag_sequence` | fine-grained POS tags (`token.tag_`), space-joined |
| `dep_sequence` | dependency relations (`token.dep_`), space-joined |
| `head_text_sequence`, `head_pos_sequence` | each token's syntactic head, space-joined |
| `is_root_sequence` | `"True"`/`"False"` per token, space-joined |

Consumers that want per-token counts (e.g. `compare_pos.py`'s POS-frequency
comparison) split a `*_sequence` column back out with `.str.split()` /
`.explode()` rather than reading a long-format table directly.

Caveat: completions are cut off at exactly `N_DRAWS` tokens, so they
sometimes end mid-clause or mid-word (e.g. "I don" from what was likely "I
don't") — spaCy will occasionally misparse the truncated tail. That's
inherent to the fixed-length design, not a bug in the tagging.
