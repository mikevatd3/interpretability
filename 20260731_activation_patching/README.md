# Causal tracing: where does the model push toward the biased outcome?

Follow-up to `../20260725_sae_basics/`. That approach (rank SAE features by a
paired t-stat between AAVE/SAE pooled activations) turned out to be pretty
noisy on inspection -- top-ranked "biased" features included things like
"words ending in 'st'", a row where the `aave_text` was actually Czech, and
features that look driven by the human MTurk translation choice to render
AAVE in-group terms as explicit racial-category nouns on the SAE side (see
`../20260725_sae_basics/data/aave_sae_top_pairs_layer10_all_templates.csv`).
None of that tells you *where in the model* the bias-relevant computation
actually happens -- it's a feature-interpretation problem, not a
localization one.

This experiment asks a narrower, more directly answerable question instead:

**For a matched AAVE/SAE pair, which layer's computation is causally
responsible for shifting next-token probability toward the biased trait
word?**

That's activation patching / causal tracing (Meng et al., ROME-style), not
SAE feature interpretation -- it doesn't require naming *what* a component
represents, only *where* the causal effect is concentrated.

## Method sketch

- "corrupted" run = AAVE-guise prompt (produces the biased completion)
- "clean" run = SAE-guise prompt (produces the baseline completion)
- patch one layer's residual stream at a time, **at the final token position
  only** -- AAVE/SAE prompts aren't token-aligned (the two sides of a pair
  are different lengths), so patching every position the way ROME does for
  subject tokens doesn't transfer directly here. The final position is also
  the only thing `q(x)` (in `matched_guise_probing`) ever reads, so this is
  a natural fit rather than a compromise.
- metric: log-prob of a target trait word (or a `katz.txt`-averaged
  `q(x)`-style ratio) under the patched run, relative to the clean/corrupted
  baselines
- sweep across all of `model.cfg.n_layers` (12 for GPT-2 small) -> a
  per-layer localization profile

## Open questions (not yet settled)

- [ ] Patch direction: denoising (clean activation spliced into the
      corrupted run) or noising (corrupted spliced into clean)? These answer
      different questions -- "what's sufficient to restore the biased
      outcome" vs. "what's necessary to cause it."
- [ ] `hook_resid_pre` vs `hook_resid_post` per layer.
- [ ] Single target word vs. averaged over the `katz.txt` list.
- [ ] Fix one `TRAIT_PROMPTS` template for a first pass, or sweep all 9 the
      way `matched_guise_probing`/`sae_basics` do?
- [ ] Once a layer (or narrow band of layers) is implicated, whether to zoom
      into *that* layer's SAE features as a follow-up refinement step,
      instead of using SAE-diffing as the primary localization tool the way
      `sae_basics` did.

## Status

Skeleton only. `main.py` has model/pairs loading working; the actual
patching sweep (`patch_layer_at_last_token`, `run_patching_sweep`) is
stubbed out with the plan in its docstrings, not yet implemented -- waiting
on the open questions above before writing the real logic.
