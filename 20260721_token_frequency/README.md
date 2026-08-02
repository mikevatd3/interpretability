# Token frequency

Moved from `justhousingnotes/exploration/density/density.py`.

Counts every GPT-2 BPE token's occurrence across WikiText-103
(train+validation+test), using `tiktoken`'s `gpt2` encoding directly (no
model load needed -- this is a corpus statistic, not a model probe).
Writes `data/token_counts.csv` (`id, token, count`, most-common first).

Seeded the token list later hand-labeled into `../20260721_city_token_probe/`'s
`draft_labels.csv`, and is the frequency side of the norm-vs-frequency
comparison in `justhousingnotes/exploration/norm_vs_frequency.py`
(not yet migrated).

## Run

```
uv run main.py
```
