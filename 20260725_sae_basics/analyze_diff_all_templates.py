import numpy as np
import pandas as pd
from dotenv import load_dotenv

from analyze_diff import neuronpedia_label, paired_diff_stats
from config import all_templates_diff_csv_path, template_acts_npz_path
from data_prep import TRAIT_PROMPTS

load_dotenv()

LAYER = 10
POOLING = "mean"  # "mean", "max", or "last" -- which pooled activation to diff
N_LABELED = 200  # how many top rows by |t_stat| get a neuronpedia label


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


def build_diff_table_all_templates(
    layer: int, pooling: str, n_labeled: int, templates: list[str] = TRAIT_PROMPTS
) -> pd.DataFrame:
    """Same per-feature paired-diff table as analyze_diff.build_diff_table, but
    combining stats across every template in `templates` -- n=18171 (9
    templates x 2019 pairs) worth of statistical power for t_stat, without
    ever loading more than one template's raw activation arrays (~1.2GB) at a
    time. See collect_activations_all_templates.py for the per-template files
    this reads, and combine_grouped_stats above for how they're merged.
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

    labels = [""] * len(diff_df)
    for rank in range(min(n_labeled, len(diff_df))):
        labels[rank] = neuronpedia_label(int(diff_df.loc[rank, "feature"]), layer)
        if (rank + 1) % 25 == 0:
            print(f"labeled {rank + 1}/{n_labeled}...")
    diff_df["neuronpedia_label"] = labels
    return diff_df


if __name__ == "__main__":
    diff_df = build_diff_table_all_templates(LAYER, POOLING, N_LABELED)
    out_path = all_templates_diff_csv_path(LAYER, POOLING)
    diff_df.to_csv(out_path, index=False)
    print(f"wrote {len(diff_df)} rows to {out_path}, sorted by |t_stat| descending")
    print(f"(top {N_LABELED} rows include a neuronpedia label; the rest are blank)")
