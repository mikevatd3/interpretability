import numpy as np
import torch
from sae_lens import SAE, HookedSAETransformer
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_feature

from config import DEVICE, SAE_RELEASE, hook_name


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


def neuronpedia_label(feature: int, layer: int) -> str:
    try:
        info = get_neuronpedia_feature(feature, layer)
        explanations = info.get("explanations") or []
        return explanations[0]["description"] if explanations else "(no label yet)"
    except Exception as e:
        return f"(neuronpedia lookup failed: {e})"


def paired_diff_stats(aave: np.ndarray, sae: np.ndarray) -> dict[str, np.ndarray]:
    """Per-feature paired-comparison stats between two (n_pairs, d_sae) pooled-activation arrays.

    Subtracts row-by-row -- row i of `sae` minus row i of `aave` describe the same
    underlying tweet -- before reducing across pairs. mean(sae - aave) over rows is
    mathematically identical to mean(sae) - mean(aave), so a plain mean-diff never
    actually uses the fact that the rows are matched; the pairing only matters once
    you also compute variance *from the per-pair diff* and rank by a t-statistic
    (mean diff over its own standard error) instead of raw mean diff. That's what
    rewards features that differ *consistently* across pairs, rather than ones with
    a big average gap driven by a handful of outlier pairs.
    """
    diff_matrix = sae - aave  # (n_pairs, d_sae), row-aligned to pairs.tsv
    n_pairs = diff_matrix.shape[0]
    mean_diff = diff_matrix.mean(axis=0)
    std_diff = diff_matrix.std(axis=0, ddof=1)
    t_stat = mean_diff / (std_diff / np.sqrt(n_pairs) + 1e-12)
    sign_consistency = (np.sign(diff_matrix) == np.sign(mean_diff)).mean(axis=0)
    n_active = (diff_matrix != 0).sum(axis=0)
    return {
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "t_stat": t_stat,
        "sign_consistency": sign_consistency,
        "n_active": n_active,
    }


def combine_grouped_stats(
    means: list[np.ndarray],
    variances: list[np.ndarray],
    ns: list[int],
    n_actives: list[np.ndarray],
    sign_consistencies: list[np.ndarray],
):
    """Combine per-template (mean, variance, n) into the exact combined-sample
    mean and variance, without ever holding all templates' raw rows together.

    This is the standard one-way-ANOVA total-sum-of-squares decomposition:
    total SS = within-group SS + between-group SS, i.e.
        sum((n_i - 1) * var_i)  +  sum(n_i * (mean_i - combined_mean)^2)
    divided by (N - 1) gives the exact sample variance of the pooled set, for
    any grouping -- this isn't an approximation of mean_diff/std_diff/t_stat,
    it reproduces exactly what you'd get computing paired_diff_stats on the
    full concatenated (n_templates * n_pairs, d_sae) array directly. Verified
    against that brute-force computation on a tiny slice before this was used
    for real data -- see the smoke test in this module's history.

    n_active is exactly additive (a count of rows where diff != 0, summed
    across disjoint template groups).

    sign_consistency is the one field that is NOT exactly recoverable from
    per-group summaries: the true value needs each row's sign relative to the
    *global* combined mean, not each template's own mean, and that requires
    the raw per-row diffs. What's returned here is the across-templates
    average of each template's own (exact, within-that-template) sign
    consistency -- a reasonable proxy for "does this feature move the same
    direction reliably," but not identical to the single-pass value you'd get
    from the full concatenated array. Flagged in the output column name below.
    """
    means_arr = np.stack(means)  # (n_templates, d_sae)
    vars_arr = np.stack(variances)  # (n_templates, d_sae)
    ns_arr = np.asarray(ns, dtype=np.float64)  # (n_templates,)
    n_actives_arr = np.stack(n_actives)  # (n_templates, d_sae)
    sign_consistencies_arr = np.stack(sign_consistencies)  # (n_templates, d_sae)

    n_total = ns_arr.sum()
    weights = ns_arr[:, None]  # broadcasts against (n_templates, d_sae)

    combined_mean = (weights * means_arr).sum(axis=0) / n_total

    within_ss = ((ns_arr[:, None] - 1) * vars_arr).sum(axis=0)
    between_ss = (weights * (means_arr - combined_mean) ** 2).sum(axis=0)
    combined_var = (within_ss + between_ss) / (n_total - 1)
    combined_std = np.sqrt(combined_var)

    combined_n_active = n_actives_arr.sum(axis=0)
    approx_sign_consistency = sign_consistencies_arr.mean(axis=0)

    return {
        "mean_diff": combined_mean,
        "std_diff": combined_std,
        "n_total": int(n_total),
        "n_active": combined_n_active,
        "sign_consistency_approx": approx_sign_consistency,
    }
