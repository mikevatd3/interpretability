# Next-token entropy across a sentence

Moved from `justhousingnotes/exploration/entropies/`.

GPT-2 is causal, so one forward pass over a sentence gives a next-token
distribution at every position for free. `entropy.py` turns each of those
distributions into a Shannon entropy (bits), read two ways:

- `entropy_after_bits[i]` — how open the prediction *after* token i is.
- `entropy_before_bits[i]` — how wide a field of options token i was *chosen
  from* (the same numbers, shifted forward by one position; token 0 has no
  preceding context, so it's `+inf`).

`plot.py` renders either framing as a line chart with the actual tokens as
x-axis labels. `entropy_plot.py` is the runnable entry point (defaults to an
example sentence, prints the per-token dataframe, and opens the plot).

## Run

```
uv run entropy_plot.py
```
