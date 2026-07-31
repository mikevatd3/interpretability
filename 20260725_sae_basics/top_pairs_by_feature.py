import numpy as np
import pandas as pd
from dotenv import load_dotenv

from collect_activations_all_templates import TRAIT_PROMPTS
from config import all_templates_diff_csv_path, template_acts_npz_path, top_pairs_csv_path
from data_prep import load_pairs

load_dotenv()

LAYER = 10
POOLING = "mean"
N_TOP_FEATURES = 20  # how many top-|t_stat| features (from the diff table) to drill into
K_PAIRS_PER_FEATURE = 5  # how many (template, pair) examples to surface per feature


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
    same files collect_activations_all_templates.py wrote. Peak memory per
    template stays at that template's full pooled-activation arrays (~200MB
    for aave + sae combined at this pooling) plus a handful of small
    (n_templates * n_pairs,) float arrays per requested feature -- trivial
    even for a couple dozen features at once.
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


if __name__ == "__main__":
    combined = report()
    out_path = top_pairs_csv_path(LAYER, POOLING)
    combined.to_csv(out_path, index=False)
    print(f"\nwrote {len(combined)} rows to {out_path}")
