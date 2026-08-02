"""Causal tracing / activation patching: localize *where* in GPT-2 the
AAVE-vs-SAE guise causes the shift toward biased trait-word completions.

See README.md for the full writeup and the open questions this skeleton is
waiting on. Short version:
  - "corrupted" run = AAVE-guise prompt (produces the biased completion)
  - "clean" run     = SAE-guise prompt (produces the baseline completion)
  - patch each layer's residual stream at the FINAL token position only
    (AAVE/SAE prompts aren't token-aligned, so whole-sequence position-wise
    patching a la ROME doesn't transfer directly here -- and the final
    position is all q(x) ever reads anyway, in matched_guise_probing)
  - metric: log-prob of a target trait word under the patched run, relative
    to the clean/corrupted baselines
  - sweep across all layers -> a per-layer localization profile

load_model/load_pairs/last_token_logits are real and usable now.
patch_layer_at_last_token/run_patching_sweep are NOT implemented yet --
their docstrings capture the plan, but the actual patching logic is waiting
on the open questions in README.md (patch direction, resid_pre vs
resid_post, single word vs katz.txt average).
"""

import csv

import pandas as pd
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer
# from transformer_lens.model_bridge import TransformerBridge


from config import DATA_DIR, DEVICE, LAYER_RANGE, PAIRS_PATH

load_dotenv()


def load_pairs() -> pd.DataFrame:
    """The 2019 intent-equivalent AAVE/SAE tweet pairs -- see
    ../data_library/groenwold_aave_sae/aave_sae_pairs.SOURCE.md
    for where this came from and which papers to cite."""
    return pd.read_csv(
        PAIRS_PATH, sep="\t", header=None, names=["aave", "sae"], quoting=csv.QUOTE_NONE
    )


def load_model() -> HookedTransformer:
    model = HookedTransformer.from_pretrained("gpt2", device=DEVICE)
    model.eval()
    return model


def last_token_logits(model: HookedTransformer, text: str) -> torch.Tensor:
    """Logits at the final token position -- the same read matched_guise_probing's
    Q class takes softmax over to compute q(x)."""
    tokens = model.to_tokens(text)
    with torch.no_grad():
        logits = model(tokens)
    return logits[0, -1, :]


def patch_layer_at_last_token(
    model: HookedTransformer,
    corrupted_text: str,
    clean_text: str,
    layer: int,
) -> torch.Tensor:
    """Run `corrupted_text`, but with layer `layer`'s residual stream at the
    FINAL token position replaced by the value it took during a run of
    `clean_text`. Returns the resulting logits at the final position.

    NOT IMPLEMENTED YET. Sketch once the open questions in README.md are
    settled:
      1. run_with_cache on `clean_text`, grab the activation at
         config.hook_name-equivalent (hook_resid_pre or hook_resid_post,
         TBD) for `layer`, at its own final token position
      2. run `corrupted_text` with a hook on that same point that overwrites
         the final-token-position activation with the value from step 1
         (use model.run_with_hooks + a hook_fn closing over that tensor)
      3. return the resulting logits[0, -1, :]

    (This is the "denoising" direction -- clean spliced into corrupted. The
    reverse "noising" direction swaps which text is run and which is
    patched-from; see README.md's open questions.)
    """
    raise NotImplementedError("patch direction / hook point not yet settled -- see README.md")


def run_patching_sweep(
    model: HookedTransformer,
    pairs: pd.DataFrame,
    target_word: str,
    layers: range = LAYER_RANGE,
) -> pd.DataFrame:
    """For each layer, how much does patching move the metric back toward
    baseline, averaged over `pairs`? This is the actual output of the
    experiment -- a per-layer localization profile, e.g.:

        layer  metric_shift
        0      0.01
        1      0.02
        ...
        7      0.38   <- candidate "where the bias lives"
        8      0.05
        ...

    NOT IMPLEMENTED YET -- depends on patch_layer_at_last_token above, and
    on deciding (see README.md): single target_word logprob vs a
    katz.txt-averaged q(x)-style ratio, and whether to sweep per-template
    (config.TRAIT_PROMPTS) or fix one template for a first pass.
    """
    raise NotImplementedError("depends on patch_layer_at_last_token; see README.md")


def main():
    model = load_model()
    pairs = load_pairs()
    print(f"loaded {model.cfg.n_layers}-layer model on {DEVICE}, {len(pairs)} pairs")
    print(f"data dir: {DATA_DIR}")
    # TODO: run_patching_sweep(...) once the open questions in README.md are settled


if __name__ == "__main__":
    main()
