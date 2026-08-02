# City token probe

Moved from `justhousingnotes/exploration/density/cities.py`.

An active-learning loop that trains a linear probe on GPT-2's token
embeddings to separate city-name tokens from everything else:

1. Hand-label a small seed batch (~15) of city tokens in
   `data/draft_labels.csv` (seeded from `../20260721_token_frequency/`'s
   token list).
2. Take the labeled cities plus an equal-size random sample of non-cities,
   and fit a `nn.Linear(768, 1)` probe on their `gpt2-small` embeddings
   (`transformer.W_E`).
3. Score every token with that probe and re-sort the file so likely
   unlabeled cities float to the top.
4. Hand-label more of the top of the file, repeat, until satisfied.
5. Final pass writes `data/final_labels.csv`.

Interactive (`input()` prompts between rounds) -- not a script to run
unattended.

## Run

```
uv run main.py
```
