import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sae_lens.analysis.neuronpedia_integration import get_neuronpedia_feature

from config import acts_npz_path, diff_csv_path

load_dotenv()

LAYER = 10
VARIANT = "plain"  # "plain" or "outer_prompt" -- which collected npz to read
POOLING = "mean"  # "mean", "max", or "last" -- which pooled activation to diff
N_LABELED = 200  # how many top rows by |t_stat| get a neuronpedia label


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


def build_diff_table(layer: int, variant: str, pooling: str, n_labeled: int) -> pd.DataFrame:
    """Every feature gets a row, not just a top-N printout, so this can be sorted/
    filtered/joined outside of this script instead of only eyeballing whatever
    n_labeled happened to be."""
    data = np.load(acts_npz_path(layer, variant))
    aave = data[f"aave_{pooling}"]
    sae = data[f"sae_{pooling}"]
    d_sae = aave.shape[1]

    stats = paired_diff_stats(aave, sae)

    diff_df = pd.DataFrame(
        {
            "feature": np.arange(d_sae),
            "t_stat": stats["t_stat"],
            "mean_diff": stats["mean_diff"],
            "sign_consistency": stats["sign_consistency"],
            "n_active": stats["n_active"],
            "peak_aave": data["aave_max"].max(axis=0),
            "peak_sae": data["sae_max"].max(axis=0),
            "fires_more_on": np.where(stats["mean_diff"] > 0, "sae", "aave"),
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


if __name__ == "__main__":
    diff_df = build_diff_table(LAYER, VARIANT, POOLING, N_LABELED)
    out_path = diff_csv_path(LAYER, VARIANT, POOLING)
    diff_df.to_csv(out_path, index=False)
    print(f"wrote {len(diff_df)} rows to {out_path}, sorted by |t_stat| descending")
    print(f"(top {N_LABELED} rows include a neuronpedia label; the rest are blank)")
