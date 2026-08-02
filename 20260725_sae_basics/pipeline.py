import time

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from config import acts_hook_name, all_templates_diff_csv_path, template_acts_npz_path, top_pairs_csv_path
from core import combine_grouped_stats, feature_vectors, load_model_and_sae, neuronpedia_label, paired_diff_stats
from data_prep import TRAIT_PROMPTS, load_pairs

load_dotenv()

LAYER = 10
POOLING = "mean"  # "mean", "max", or "last" -- which pooled activation to diff
N_LABELED = 200  # how many top rows by |t_stat| get a neuronpedia label
N_TOP_FEATURES = 20  # how many top-|t_stat| features (from the diff table) to drill into
K_PAIRS_PER_FEATURE = 5  # how many (template, pair) examples to surface per feature

# Collection is the ~90-minute step (a full model+SAE pass over 2019 pairs x 9
# templates); analysis and reporting are a lot cheaper but analysis still makes
# up to N_LABELED live Neuronpedia requests. All three stages are skipped if
# their output already exists, so re-running this script is safe and cheap
# once everything's been collected at least once. Flip these to True to force
# a redo.
FORCE_RECOLLECT = False
FORCE_REANALYZE = False


# --- collect ---------------------------------------------------------------


def collect_one_template(layer: int, template: str, template_idx: int, model, sae) -> None:
    """Same collection as collect_all_templates, for a single template,
    reusing an already-loaded model/sae so the (slow, one-time) SAE/model load
    only happens once across all 9 templates rather than 9 times.

    Deliberately one npz per template rather than one combined array across all
    templates: peak RAM here is ~1.2GB (matching the existing single-variant
    files already on disk), instead of ~10.7GB for all 9 templates' rows held
    at once -- see build_diff_table_all_templates below for how the per-template
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
    loaded model/sae throughout. Budget roughly 90 minutes total for all 9
    templates x 2019 pairs.
    """
    model, sae = load_model_and_sae(layer)

    for template_idx, template in enumerate(templates):
        print(f"[{template_idx + 1}/{len(templates)}] {template!r}")
        collect_one_template(layer, template, template_idx, model, sae)


# --- analyze -----------------------------------------------------------------


def build_diff_table_all_templates(
    layer: int, pooling: str, n_labeled: int, templates: list[str] = TRAIT_PROMPTS
) -> pd.DataFrame:
    """Per-feature paired-diff table combining stats across every template in
    `templates` -- n=18171 (9 templates x 2019 pairs) worth of statistical
    power for t_stat, without ever loading more than one template's raw
    activation arrays (~1.2GB) at a time. See collect_all_templates for the
    per-template files this reads, and core.combine_grouped_stats for how
    they're merged.
    """
    means, variances, ns, n_actives, sign_consistencies = [], [], [], [], []
    peak_aave = peak_sae = d_sae = None

    for template_idx in range(len(templates)):
        data = np.load(template_acts_npz_path(layer, template_idx))
        aave = data[f"aave_{pooling}"]
        sae = data[f"sae_{pooling}"]
        if d_sae is None:
            d_sae = aave.shape[1]

        stats = paired_diff_stats(aave, sae)
        means.append(stats["mean_diff"])
        variances.append(stats["std_diff"] ** 2)
        ns.append(aave.shape[0])
        n_actives.append(stats["n_active"])
        sign_consistencies.append(stats["sign_consistency"])

        template_peak_aave = data["aave_max"].max(axis=0)
        template_peak_sae = data["sae_max"].max(axis=0)
        peak_aave = (
            template_peak_aave if peak_aave is None else np.maximum(peak_aave, template_peak_aave)
        )
        peak_sae = (
            template_peak_sae if peak_sae is None else np.maximum(peak_sae, template_peak_sae)
        )

        del data, aave, sae  # this template's arrays are freed before the next npz loads

    assert d_sae is not None, "templates must be non-empty"

    combined = combine_grouped_stats(means, variances, ns, n_actives, sign_consistencies)
    t_stat = combined["mean_diff"] / (combined["std_diff"] / np.sqrt(combined["n_total"]) + 1e-12)

    diff_df = pd.DataFrame(
        {
            "feature": np.arange(d_sae),
            "t_stat": t_stat,
            "mean_diff": combined["mean_diff"],
            "sign_consistency_approx": combined["sign_consistency_approx"],
            "n_active": combined["n_active"],
            "peak_aave": peak_aave,
            "peak_sae": peak_sae,
            "fires_more_on": np.where(combined["mean_diff"] > 0, "sae", "aave"),
        }
    )
    diff_df["abs_t"] = diff_df["t_stat"].abs()
    diff_df = diff_df.sort_values("abs_t", ascending=False).reset_index(drop=True)

    # Labeling all 24576 features would mean 24576 live HTTP requests for a set
    # that's mostly dead/noise anyway -- only the top n_labeled rows by |t_stat|
    # get one. The rest are left blank; pull more later if needed.
    labels = [""] * len(diff_df)
    for rank in range(min(n_labeled, len(diff_df))):
        labels[rank] = neuronpedia_label(int(diff_df.loc[rank, "feature"]), layer)
        if (rank + 1) % 25 == 0:
            print(f"labeled {rank + 1}/{n_labeled}...")
    diff_df["neuronpedia_label"] = labels
    return diff_df


# --- report ------------------------------------------------------------------


def top_pairs_for_features(
    features: list[int], layer: int, pooling: str, k: int = 5, templates: list[str] = TRAIT_PROMPTS
) -> dict[int, pd.DataFrame]:
    """For each feature in `features`, the k (template, pair) combinations with
    the largest |sae_activation - aave_activation| -- the concrete examples
    behind that feature's aggregate mean_diff/t_stat in the diff table, which
    only tells you the feature differs *on average*, not which specific pairs
    or phrasings actually drove it.

    Does one pass over the 9 per-template npz files regardless of how many
    features are requested (rather than one pass per feature), reusing the
    same files collect_all_templates wrote. Peak memory per template stays at
    that template's full pooled-activation arrays (~200MB for aave + sae
    combined at this pooling) plus a handful of small (n_templates * n_pairs,)
    float arrays per requested feature -- trivial even for a couple dozen
    features at once.
    """
    pairs = load_pairs()
    n_pairs = len(pairs)
    n_templates = len(templates)
    n_total = n_templates * n_pairs

    aave_vals = {f: np.empty(n_total, dtype=np.float32) for f in features}
    sae_vals = {f: np.empty(n_total, dtype=np.float32) for f in features}

    for template_idx in range(n_templates):
        data = np.load(template_acts_npz_path(layer, template_idx))
        aave_all = data[f"aave_{pooling}"]
        sae_all = data[f"sae_{pooling}"]
        start = template_idx * n_pairs
        end = start + n_pairs
        for f in features:
            aave_vals[f][start:end] = aave_all[:, f]
            sae_vals[f][start:end] = sae_all[:, f]
        del data, aave_all, sae_all

    results = {}
    for f in features:
        diff = sae_vals[f] - aave_vals[f]
        abs_diff = np.abs(diff)
        top_idx = np.argsort(-abs_diff)[:k]

        rows = []
        for idx in top_idx:
            template_idx, pair_idx = divmod(int(idx), n_pairs)
            rows.append(
                {
                    "feature": f,
                    "template_idx": template_idx,
                    "template": templates[template_idx],
                    "pair_idx": pair_idx,
                    "aave_text": pairs.iloc[pair_idx]["aave"],
                    "sae_text": pairs.iloc[pair_idx]["sae"],
                    "aave_activation": float(aave_vals[f][idx]),
                    "sae_activation": float(sae_vals[f][idx]),
                    "diff": float(diff[idx]),
                }
            )
        results[f] = pd.DataFrame(rows)

    return results


def report(
    layer: int = LAYER,
    pooling: str = POOLING,
    n_top_features: int = N_TOP_FEATURES,
    k: int = K_PAIRS_PER_FEATURE,
) -> pd.DataFrame:
    """Top `n_top_features` by |t_stat| from the diff table, each with its top
    `k` example pairs -- one combined, flat DataFrame (also what gets saved to
    top_pairs_csv_path), plus a print of it as a readable report.
    """
    diff_df = pd.read_csv(all_templates_diff_csv_path(layer, pooling))
    top_features = diff_df.sort_values("abs_t", ascending=False)["feature"].head(n_top_features).tolist()

    per_feature = top_pairs_for_features(top_features, layer, pooling, k=k)

    for feature in top_features:
        feature_row = diff_df[diff_df["feature"] == feature].iloc[0]
        print(
            f"\n=== feature {feature}  t={feature_row['t_stat']:.1f}  "
            f"mean_diff={feature_row['mean_diff']:.3f}  fires_more_on={feature_row['fires_more_on']}  "
            f"{feature_row['neuronpedia_label']!r} ==="
        )
        for _, row in per_feature[feature].iterrows():
            print(f"  diff={row['diff']:+.3f}  (aave={row['aave_activation']:.3f}, sae={row['sae_activation']:.3f})")
            print(f"    template: {row['template']}")
            print(f"    aave: {row['aave_text']!r}")
            print(f"    sae:  {row['sae_text']!r}")

    combined = pd.concat(per_feature.values(), ignore_index=True)

    label_map = diff_df.set_index("feature")["neuronpedia_label"].to_dict()
    combined.insert(1, "neuronpedia_label", combined["feature"].map(label_map))

    return combined


# --- orchestration -----------------------------------------------------------


def ensure_collected(layer: int, templates: list[str] = TRAIT_PROMPTS) -> None:
    paths = [template_acts_npz_path(layer, i) for i in range(len(templates))]
    if all(p.exists() for p in paths) and not FORCE_RECOLLECT:
        print(f"[collect] all {len(paths)} template activation files already exist, skipping (FORCE_RECOLLECT=True to redo)")
        return
    print(f"[collect] running model + SAE over {len(templates)} templates, layer {layer} (~90 min)...")
    collect_all_templates(layer, templates)


def ensure_analyzed(layer: int, pooling: str, n_labeled: int) -> None:
    path = all_templates_diff_csv_path(layer, pooling)
    if path.exists() and not FORCE_REANALYZE:
        print(f"[analyze] {path.name} already exists, skipping (FORCE_REANALYZE=True to redo)")
        return
    print(f"[analyze] building combined paired-diff table, layer {layer}, pooling={pooling}...")
    diff_df = build_diff_table_all_templates(layer, pooling, n_labeled)
    diff_df.to_csv(path, index=False)
    print(f"[analyze] wrote {len(diff_df)} rows to {path.name}")


def main():
    ensure_collected(LAYER)
    ensure_analyzed(LAYER, POOLING, N_LABELED)

    print(f"[report] building top-pairs report for layer {LAYER}, pooling={POOLING}...")
    combined = report()
    out_path = top_pairs_csv_path(LAYER, POOLING)
    combined.to_csv(out_path, index=False)
    print(f"[report] wrote {len(combined)} rows to {out_path}")


if __name__ == "__main__":
    main()
