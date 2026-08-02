# Prompt density

Moved from `justhousingnotes/exploration/density/main.py`.

Several differently-worded prompts that should all complete to the same
answer ("Paris"), run through GPT-2 small with `run_with_cache`. For each
prompt, prints the top-3 next-token predictions at the final position.

The question: how much do the logits shift between phrasings even when the
top prediction is the same? A probe at how many different paths the model
takes to the same output token -- is "Paris" always the same "place" to the
model, or do different phrasings arrive there through different internal
representations?

## Run

```
uv run main.py
```
