import time

import numpy as np
import torch
from dotenv import load_dotenv
from sae_lens import SAE, HookedSAETransformer

from config import DEVICE, SAE_RELEASE, acts_hook_name, acts_npz_path, hook_name
from data_prep import prompted_pairs

load_dotenv()

LAYER = 10
VARIANT = "plain"  # "plain" or "outer_prompt" -- see config.VARIANTS


def load_model_and_sae(layer: int) -> tuple[HookedSAETransformer, SAE]:
    """Load the gpt2-small-res-jb SAE for this layer and a matching, correctly-loaded model.

    The model must be loaded with from_pretrained_no_processing and the SAE's own
    model_from_pretrained_kwargs (here: center_writing_weights=True) -- the default
    from_pretrained applies more processing than these SAEs were trained on, and gives
    a much less sparse, effectively wrong, set of feature activations (L0 ~330 instead
    of the ~50 these SAEs actually expect). sae_lens raises a UserWarning about this at
    load time; this is that fix, and it's the one non-obvious part of using this release.

    Reusable outside data collection too -- e.g. a future ablation/patching script
    (see COOLSTUFF.md's causal-test gap-closing step) would import this rather than
    re-loading the model and SAE from scratch.

    Loads onto config.DEVICE -- cuda/mps if this environment has a GPU, cpu otherwise.
    """
    sae = SAE.from_pretrained(SAE_RELEASE, hook_name(layer), device=DEVICE)
    if isinstance(sae, tuple):
        sae = sae[0]

    model = HookedSAETransformer.from_pretrained_no_processing(
        "gpt2", device=DEVICE, **sae.cfg.metadata.model_from_pretrained_kwargs
    )
    model.eval()
    return model, sae


def feature_vectors(
    text: str,
    model: HookedSAETransformer,
    sae: SAE,
    acts_hook: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean-, max-, and last-token-pooled SAE feature activations for one piece of text."""
    tokens = model.to_tokens(text)

    with torch.no_grad():
        _, cache = model.run_with_cache_with_saes(
            tokens, saes=[sae], names_filter=lambda name: name == acts_hook
        )

    acts = cache[acts_hook][0]
    acts = acts[1:]  # drop the prepended BOS position

    # .cpu() is a no-op on CPU tensors and required before .numpy() on cuda/mps ones,
    # so this is safe regardless of which device `model`/`sae` are loaded on.
    return (
        acts.mean(dim=0).cpu().numpy(),
        acts.max(dim=0).values.cpu().numpy(),
        acts[-1].cpu().numpy(),
    )


def collect(layer: int, variant: str) -> None:
    pairs = prompted_pairs(variant)
    model, sae = load_model_and_sae(layer)
    acts_hook = acts_hook_name(layer)

    n = len(pairs)
    d_sae = sae.cfg.d_sae
    aave_mean = np.zeros((n, d_sae), dtype=np.float32)
    aave_max = np.zeros((n, d_sae), dtype=np.float32)
    aave_last = np.zeros((n, d_sae), dtype=np.float32)
    sae_mean = np.zeros((n, d_sae), dtype=np.float32)
    sae_max = np.zeros((n, d_sae), dtype=np.float32)
    sae_last = np.zeros((n, d_sae), dtype=np.float32)

    t0 = time.time()
    for i, row in pairs.iterrows():
        aave_mean[i], aave_max[i], aave_last[i] = feature_vectors(
            row["aave"], model, sae, acts_hook
        )
        sae_mean[i], sae_max[i], sae_last[i] = feature_vectors(
            row["sae"], model, sae, acts_hook
        )

        if (i + 1) % 100 == 0 or i + 1 == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(f"{i + 1}/{n} pairs ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")

    out_path = acts_npz_path(layer, variant)
    np.savez_compressed(
        out_path,
        aave_mean=aave_mean,
        aave_max=aave_max,
        aave_last=aave_last,
        sae_mean=sae_mean,
        sae_max=sae_max,
        sae_last=sae_last,
    )
    print(f"wrote {out_path}")
    print("row i of every array here matches row i of ../../justhousingnotes/data/groenwold_aave_sae/aave_sae_pairs.tsv")


if __name__ == "__main__":
    collect(LAYER, VARIANT)
