# SAE feature ablation walkthrough

This walks through what `sae_ablation.py` + `ablation_experiment.py` do,
step by step. The goal: find which GPT-2 SAE features fire differently on
AAVE vs. SAE guise prompts, then ablate them and see whether that actually
moves the output-level association score (`q(x)`) — a first pass at "are
the identified activated nodes transferable to our problem?" from the
README.

## What it's testing

The original `q(x)` score (in `prompt_level_association_score.py` / now
`main.py`) only looks at the model's *output*: how much more likely a probe
word like " lazy" is after an AAVE-guise prompt vs. an SAE-guise prompt.
It can't tell you *why*. This experiment adds a mechanistic layer underneath
that measurement:

1. Find SAE features (interpretable directions in GPT-2's residual stream)
   that activate differently between the two guises.
2. Zero (or mean-)ablate the most differential ones.
3. Recompute `q(x)` with those features knocked out.
4. If the score moves a lot, those features are plausibly *carrying* the
   bias. If it barely moves, the bias is more diffuse — spread across many
   features rather than concentrated in a few.

## Steps

### 1. Load a model + a pretrained SAE for one layer

```python
from sae_ablation import load_model_and_sae

model, sae = load_model_and_sae("blocks.6.hook_resid_pre")
```

This loads GPT-2 as a TransformerLens `HookedSAETransformer` (needed for
hooking) and a pretrained residual-stream SAE from the `gpt2-small-res-jb`
release, one per layer (0–11). The SAE is spliced into the model with
`use_error_term=True`, which means: as long as nothing touches the SAE's
feature activations, the model behaves *exactly* like plain GPT-2. This is
what makes the later ablation step clean — you're only changing the
specific features you target, nothing else.

### 2. Find the differential features

```python
from sae_ablation import compute_feature_diff

feature_diff = compute_feature_diff(model, sae, pairs, OUTER_PROMPTS, layer=6)
```

For every (AAVE sentence, SAE sentence) pair, wrapped in every outer prompt
template, this runs both guises through the model, reads off the SAE
feature activations at the final token position (the position `q(x)` itself
reads from), and averages `aave_activation - sae_activation` per feature
across the whole dataset. The result is one number per SAE feature (24,576
of them at layer 6): positive means "this feature fires harder on the AAVE
guise," negative means the opposite.

### 3. Pick the top features and ablate them

```python
from sae_ablation import make_ablation_hook, sae_acts_hook_name

top_idx = feature_diff.abs().topk(10).indices
hook = make_ablation_hook(top_idx, mode="zero")
```

`make_ablation_hook` returns a TransformerLens hook function. Passed into
`run_with_hooks` at `sae_acts_hook_name(sae)`, it zeroes out (or, in
`"mean"` mode, replaces with a supplied mean value) just those feature
activations before the SAE decodes back into the residual stream.

### 4. Compare `q(x)` with and without the ablation

```python
from sae_ablation import score_probe_words

baseline = score_probe_words(model, PROBE_WORDS, pairs, OUTER_PROMPTS, fwd_hooks=[])
ablated  = score_probe_words(model, PROBE_WORDS, pairs, OUTER_PROMPTS, fwd_hooks=[(hook_name, hook)])
```

Both calls compute the same `log(P(aave)/P(sae))` score as the original
`Q` class, just run through `model.run_with_hooks` so the ablation can be
switched on or off. `baseline` is the SAE spliced in but untouched (should
match plain GPT-2's `q(x)`); `ablated` is with the top features knocked
out. The delta between them is the effect size.

### 5. Run the whole thing

All of the above is wired together in `ablation_experiment.py`:

```bash
python ablation_experiment.py
```

Knobs at the top of that file:

| Knob | What it controls |
|---|---|
| `LAYER` | Which of GPT-2's 12 layers to probe (0–11) — the README's "try different layers" |
| `TOP_N` | How many differential features to ablate |
| `ABLATION_MODE` | `"zero"` or `"mean"` replacement value |
| `N_PAIRS` | How many of the 2,019 AAVE/SAE pairs to use — `None` for all |

It prints a markdown table (baseline vs. ablated vs. delta per probe word)
and a list of the ablated feature indices with ready-to-click Neuronpedia
links (`https://neuronpedia.org/gpt2-small/{LAYER}-res-jb/{feature}`), so
you can look up what each feature seems to represent.

## Practical notes

- No GPU on this machine — everything runs on CPU. The feature-diff step
  is cached to `cached_feature_diff_layer{LAYER}_n{N_PAIRS}.pt`, so
  re-running with a different `TOP_N` or `ABLATION_MODE` (same layer and
  pair count) skips straight to scoring.
- `N_PAIRS=50` (the default) takes several minutes end-to-end on CPU,
  mostly in the two full scoring passes. The full 2,019-pair dataset will
  take much longer — run it in the background.
- Because `use_error_term=True` makes the un-ablated SAE numerically
  transparent, the `baseline` column here should track the existing
  `cached_q.pt` / `cached_q_multi.pt` results (from `main.py`) reasonably
  closely — if it doesn't, that's a sign something's wired wrong before
  even getting to the ablation.
