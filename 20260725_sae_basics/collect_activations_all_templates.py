import time

import numpy as np
from dotenv import load_dotenv

from collect_activations import feature_vectors, load_model_and_sae
from config import template_acts_npz_path, acts_hook_name
from data_prep import TRAIT_PROMPTS, load_pairs

load_dotenv()

LAYER = 10


def collect_one_template(layer: int, template: str, template_idx: int, model, sae) -> None:
    """Same collection as collect_activations.collect(), for a single template,
    reusing an already-loaded model/sae so the (slow, one-time) SAE/model load
    only happens once across all 9 templates rather than 9 times.

    Deliberately one npz per template rather than one combined array across all
    templates: peak RAM here is ~1.2GB (matching the existing single-variant
    files already on disk), instead of ~10.7GB for all 9 templates' rows held
    at once -- see analyze_diff_all_templates.py for how the per-template
    stats get combined into one result without ever holding all the raw rows
    together.
    """
    pairs = load_pairs()
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
            template.format(t=row["aave"]), model, sae, acts_hook
        )
        sae_mean[i], sae_max[i], sae_last[i] = feature_vectors(
            template.format(t=row["sae"]), model, sae, acts_hook
        )

        if (i + 1) % 200 == 0 or i + 1 == n:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            print(
                f"  template {template_idx}: {i + 1}/{n} pairs "
                f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)"
            )

    out_path = template_acts_npz_path(layer, template_idx)
    np.savez_compressed(
        out_path,
        aave_mean=aave_mean,
        aave_max=aave_max,
        aave_last=aave_last,
        sae_mean=sae_mean,
        sae_max=sae_max,
        sae_last=sae_last,
    )
    print(f"  wrote {out_path}")


def collect_all_templates(layer: int, templates: list[str] = TRAIT_PROMPTS) -> None:
    """Runs collect_one_template for every template, sequentially, reusing one
    loaded model/sae throughout. Runtime: collect_activations.py's single-
    OUTER_PROMPT pass over 2019 pairs takes ~10 minutes (main.py's estimate);
    this is 9 templates x that, so budget roughly 90 minutes total.
    """
    model, sae = load_model_and_sae(layer)

    for template_idx, template in enumerate(templates):
        print(f"[{template_idx + 1}/{len(templates)}] {template!r}")
        collect_one_template(layer, template, template_idx, model, sae)


if __name__ == "__main__":
    collect_all_templates(LAYER)
