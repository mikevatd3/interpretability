# Pair activation patching (swap out a whole layer)

import csv

import pandas as pd
import torch
from dotenv import load_dotenv
from transformer_lens import HookedTransformer
# from transformer_lens.model_bridge import TransformerBridge


from config import DATA_DIR, DEVICE, KATZ_PATH, LAYER_RANGE, PAIRS_PATH

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
    aave_text: str,
    sae_text: str,
    layer: int,
) -> torch.Tensor:
    """Denoising patch: run aave_text, but with layer `layer`'s
    resid_post at the final token position overwritten by the value it took
    on a run of sae_text. Returns logits at the final position."""
    hook_name = f"blocks.{layer}.hook_resid_post"

    sae_tokens = model.to_tokens(sae_text)
    with torch.no_grad():
        _, sae_cache = model.run_with_cache(sae_tokens, names_filter=hook_name)
    sae_last_token_activation = sae_cache[hook_name][0, -1, :]

    def overwrite_last_token_with_sae_activation(activation, hook):
        activation[0, -1, :] = sae_last_token_activation
        return activation

    aave_tokens = model.to_tokens(aave_text)
    with torch.no_grad():
        patched_logits = model.run_with_hooks(
            aave_tokens,
            fwd_hooks=[(hook_name, overwrite_last_token_with_sae_activation)],
        )
    return patched_logits[0, -1, :]


def load_target_words() -> list[str]:
    """The Katz & Braly trait adjectives dialect-prejudice probes the model
    against -- one forward pass reads all of them, so sweeping the whole
    list costs nothing extra over a single word."""
    return [word.strip() for word in KATZ_PATH.read_text().splitlines() if word.strip()]


def target_token_ids(model: HookedTransformer, words: list[str]) -> list[int]:
    """Single-token (with leading space) target words only -- multi-token
    words can't be read off a single last-token logit vector."""
    ids = []
    for word in words:
        token_ids = model.to_tokens(" " + word, prepend_bos=False)[0]
        if len(token_ids) == 1:
            ids.append(token_ids.item())
    return ids


def mean_target_word_logit(logits: torch.Tensor, target_ids: list[int]) -> float:
    return logits[target_ids].mean().item()


def run_patching_sweep(
    model: HookedTransformer,
    pairs: pd.DataFrame,
    target_words: list[str],
    layers: range = LAYER_RANGE,
) -> pd.DataFrame:
    """Per-layer localization profile: how much does patching layer `layer`
    move the AAVE run's target-word logit (averaged over
    `target_words`) back toward the SAE baseline, averaged over
    `pairs`? 0 = no effect, 1 = fully restores the SAE baseline."""
    target_ids = target_token_ids(model, target_words)

    rows = []
    for pair_index, pair in pairs.iterrows():
        aave_text, sae_text = pair["aave"], pair["sae"]

        aave_logit = mean_target_word_logit(
            last_token_logits(model, aave_text), target_ids
        )
        sae_logit = mean_target_word_logit(last_token_logits(model, sae_text), target_ids)
        sae_minus_aave = sae_logit - aave_logit
        if sae_minus_aave == 0:
            continue

        for layer in layers:
            patched_logits = patch_layer_at_last_token(model, aave_text, sae_text, layer)
            patched_logit = mean_target_word_logit(patched_logits, target_ids)
            metric_shift = (patched_logit - aave_logit) / sae_minus_aave
            rows.append({"pair_index": pair_index, "layer": layer, "metric_shift": metric_shift})

    return pd.DataFrame(rows).groupby("layer")["metric_shift"].mean().reset_index()


def main():
    model = load_model()
    pairs = load_pairs()
    print(f"loaded {model.cfg.n_layers}-layer model on {DEVICE}, {len(pairs)} pairs")

    target_words = load_target_words()
    profile = run_patching_sweep(model, pairs.head(20), target_words)
    print(profile)

    DATA_DIR.mkdir(exist_ok=True)
    out_path = DATA_DIR / "layer_patching_profile.csv"
    profile.to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
