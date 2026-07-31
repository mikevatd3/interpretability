from pathlib import Path
from typing import Callable, Literal

import pandas as pd
import torch
from sae_lens import SAE, HookedSAETransformer
from tqdm import tqdm

from prompt_level_association_score import _get_device

AblationMode = Literal["zero", "mean"]


def load_model_and_sae(
    hook_name: str, device: str | None = None
) -> tuple[HookedSAETransformer, SAE]:
    device = device or _get_device()
    sae = SAE.from_pretrained("gpt2-small-res-jb", hook_name, device=device)

    # sae-lens warns that these SAEs were trained against a model loaded
    # with these specific kwargs (e.g. center_writing_weights) -- skipping
    # them would make the SAE's reconstruction subtly wrong.
    model = HookedSAETransformer.from_pretrained_no_processing(
        sae.cfg.metadata.model_name, **sae.cfg.metadata.model_from_pretrained_kwargs
    )
    model.to(device)

    # use_error_term=True splices the SAE in "transparently": as long as
    # nothing touches hook_sae_acts_post, the model's output is unchanged
    # from running without the SAE at all, because the SAE's own
    # reconstruction error gets added back. This is what makes ablation on
    # hook_sae_acts_post a clean, isolated intervention on just the
    # targeted features.
    model.add_sae(sae, use_error_term=True)
    return model, sae


def sae_acts_hook_name(sae: SAE) -> str:
    return f"{sae.cfg.metadata.hook_name}.hook_sae_acts_post"


def make_ablation_hook(
    feature_idx: torch.Tensor,
    mode: AblationMode,
    mean_values: torch.Tensor | None = None,
) -> Callable:
    if mode == "mean":
        if mean_values is None:
            raise ValueError("mean_values is required for mode='mean'")
        replacement = mean_values[feature_idx]
    else:
        replacement = torch.zeros_like(feature_idx, dtype=torch.float32)

    def hook(acts: torch.Tensor, hook) -> torch.Tensor:
        del hook
        acts = acts.clone()
        acts[..., feature_idx] = replacement.to(acts.dtype)
        return acts

    return hook


def _flatten_texts(
    outer_prompts: list[str], pairs: pd.DataFrame, n_pairs: int | None
) -> tuple[list[str], list[str]]:
    rows = pairs if n_pairs is None else pairs.iloc[:n_pairs]
    aave_texts = [
        v.format(t=row["aave"]) for v in outer_prompts for _, row in rows.iterrows()
    ]
    sae_texts = [
        v.format(t=row["sae"]) for v in outer_prompts for _, row in rows.iterrows()
    ]
    return aave_texts, sae_texts


def _last_token_feature_acts(
    model: HookedSAETransformer,
    acts_hook_name: str,
    layer: int,
    texts: list[str],
) -> torch.Tensor:
    _, cache = model.run_with_cache(
        texts,
        padding_side="left",
        stop_at_layer=layer + 1,
        names_filter=acts_hook_name,
    )
    return cache[acts_hook_name][:, -1, :]


def compute_feature_diff(
    model: HookedSAETransformer,
    sae: SAE,
    pairs: pd.DataFrame,
    outer_prompts: list[str],
    layer: int,
    batch_size: int = 16,
    n_pairs: int | None = None,
) -> torch.Tensor:
    """mean(AAVE feature acts) - mean(SAE feature acts) at the final token
    position, per SAE feature, averaged over all (template, pair)
    combinations. Positive => feature fires harder on the AAVE guise.
    """
    acts_hook_name = sae_acts_hook_name(sae)
    aave_texts, sae_texts = _flatten_texts(outer_prompts, pairs, n_pairs)
    n_total = len(aave_texts)

    accumulation = torch.zeros(sae.cfg.d_sae, device=model.cfg.device)

    model.eval()
    with torch.no_grad():
        for start in tqdm(
            range(0, n_total, batch_size),
            desc="Feature diff batches",
            total=(n_total + batch_size - 1) // batch_size,
        ):
            end = start + batch_size
            aave_feats = _last_token_feature_acts(
                model, acts_hook_name, layer, aave_texts[start:end]
            )
            sae_feats = _last_token_feature_acts(
                model, acts_hook_name, layer, sae_texts[start:end]
            )
            accumulation += (aave_feats - sae_feats).sum(dim=0)

    return (accumulation / n_total).cpu()


def score_probe_words(
    model: HookedSAETransformer,
    probe_words: list[str],
    pairs: pd.DataFrame,
    outer_prompts: list[str],
    fwd_hooks: list[tuple[str, Callable]],
    batch_size: int = 16,
    n_pairs: int | None = None,
) -> dict[str, float]:
    """Same log(P(aave)/P(sae)) association score as Q in
    prompt_level_association_score.py, but run through
    model.run_with_hooks so it can be evaluated with or without an
    ablation hook active.
    """
    tokenizer = model.tokenizer
    aave_texts, sae_texts = _flatten_texts(outer_prompts, pairs, n_pairs)
    n_total = len(aave_texts)
    vocab_size = model.cfg.d_vocab

    accumulation = torch.zeros(vocab_size, device=model.cfg.device)

    model.eval()
    with torch.no_grad():
        for start in tqdm(
            range(0, n_total, batch_size),
            desc="Score batches",
            total=(n_total + batch_size - 1) // batch_size,
        ):
            end = start + batch_size

            aave_logits = model.run_with_hooks(
                aave_texts[start:end], padding_side="left", fwd_hooks=fwd_hooks
            )
            sae_logits = model.run_with_hooks(
                sae_texts[start:end], padding_side="left", fwd_hooks=fwd_hooks
            )

            aave_probs = aave_logits[:, -1, :].softmax(dim=-1)
            sae_probs = sae_logits[:, -1, :].softmax(dim=-1)

            accumulation += torch.log(aave_probs / sae_probs).sum(dim=0)

    scores = (accumulation / n_total).cpu()

    # model.tokenizer prepends a BOS token (id 50256, per this SAE's
    # metadata.prepend_bos), so the word's own first subword is the *last*
    # id in the encoding, not the first -- unlike the raw HF tokenizer used
    # in prompt_level_association_score.py's Q, which doesn't add BOS.
    return {
        word: scores[tokenizer.encode(word)[-1]].item() for word in probe_words
    }


def save_feature_diff(feature_diff: torch.Tensor, path: Path) -> None:
    torch.save(feature_diff, path)


def load_feature_diff(path: Path) -> torch.Tensor:
    return torch.load(path)
