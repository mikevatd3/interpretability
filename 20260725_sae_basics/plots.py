import pandas as pd
import plotly.graph_objects as go

from config import diff_csv_path
from data_prep import load_pairs

LAYER = 10
VARIANT = "plain"  # "plain" or "outer_prompt"
POOLING = "mean"  # "mean", "max", or "last"

BLUE = "#2a78d6"  # categorical slot 1 -- used for "aave" and single-series magnitude
ORANGE = "#eb6834"  # categorical slot 2 -- used for "sae"


def show_plots(layer: int, variant: str, pooling: str) -> None:
    diff_df = pd.read_csv(diff_csv_path(layer, variant, pooling))
    pairs = load_pairs()

    # --- 1. Distribution of the paired t-statistic across all 24576 features ---
    # Most features should sit near zero (no reliable AAVE/SAE difference); log y
    # makes the long tail of features that *do* differ visible against the huge
    # spike at zero.
    fig1 = go.Figure()
    fig1.add_trace(go.Histogram(x=diff_df["t_stat"], nbinsx=200, marker_color=BLUE))
    fig1.update_layout(
        title=f"Paired t-statistic across all {len(diff_df)} features (layer {layer}, {variant}/{pooling})",
        xaxis_title="t-statistic (mean diff / SE of diff, over 2019 pairs)",
        yaxis_title="Feature count",
        yaxis_type="log",
        margin={"l": 60, "r": 40, "t": 60, "b": 60},
    )

    # --- 2. n_active vs |t_stat| -- the sparsity-inflation risk, made visible ---
    # A feature active in only a handful of the 2019 pairs can still produce a
    # large |t| from near-zero variance alone. Plotting these against each other
    # shows whether the top-ranked features are broadly active or suspiciously rare.
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scattergl(
            x=diff_df["n_active"],
            y=diff_df["abs_t"],
            mode="markers",
            marker=dict(color=BLUE, size=5, opacity=0.35),
            text=diff_df["feature"],
            hovertemplate="feature %{text}<br>active in %{x} pairs<br>|t|=%{y:.2f}<extra></extra>",
        )
    )
    fig2.update_layout(
        title="Feature reliability: how many pairs was it active in vs. how big is |t|?",
        xaxis_title="pairs the feature was active in (out of 2019)",
        yaxis_title="|t-statistic|",
        xaxis_type="log",
        margin={"l": 60, "r": 40, "t": 60, "b": 60},
    )

    # --- 3. mean_diff vs t_stat, colored by which side fires more ---
    # Distinguishes "big average gap" from "statistically reliable gap" -- a
    # feature can have a large mean_diff and a small t_stat if it's inconsistent
    # across pairs, or vice versa.
    fig3 = go.Figure()
    for side, color in [("aave", BLUE), ("sae", ORANGE)]:
        subset = diff_df[diff_df["fires_more_on"] == side]
        # neuronpedia_label is only populated for the top N_LABELED=200 rows by
        # |t_stat| (see analyze_diff.py) -- everything else was never fetched,
        # so make that explicit in the tooltip rather than showing a blank line.
        label = subset["neuronpedia_label"].fillna("").replace(
            "", "(unlabeled -- outside top 200 by |t_stat|)"
        )
        custom = pd.DataFrame({"feature": subset["feature"], "label": label}).to_numpy()
        fig3.add_trace(
            go.Scattergl(
                x=subset["mean_diff"],
                y=subset["t_stat"],
                mode="markers",
                name=f"fires more on {side}",
                marker=dict(color=color, size=5, opacity=0.35),
                customdata=custom,
                hovertemplate=(
                    "feature %{customdata[0]}<br>mean_diff=%{x:.3f}<br>t=%{y:.2f}"
                    "<br>%{customdata[1]}<extra></extra>"
                ),
            )
        )
    fig3.update_layout(
        title="Effect size (mean diff) vs. reliability (paired t-statistic)",
        xaxis_title="mean_diff (sae - aave)",
        yaxis_title="t-statistic",
        legend_title_text="",
        margin={"l": 60, "r": 40, "t": 60, "b": 60},
    )

    # --- 4. Text length: aave vs sae, word count ---
    # The "sae" column is a human MTurk translation of the "aave" tweet -- if
    # translators systematically expanded/compressed the text, mean-pooled
    # features would partly reflect length differences rather than dialect
    # content. Overlaid histograms make that visible directly.
    aave_len = pairs["aave"].str.split().str.len()
    sae_len = pairs["sae"].str.split().str.len()

    fig4 = go.Figure()
    fig4.add_trace(go.Histogram(x=aave_len, name="aave", marker_color=BLUE, opacity=0.6, nbinsx=40))
    fig4.add_trace(go.Histogram(x=sae_len, name="sae", marker_color=ORANGE, opacity=0.6, nbinsx=40))
    fig4.update_layout(
        title="Word count per text: aave vs. sae side of each pair",
        xaxis_title="words",
        yaxis_title="count",
        barmode="overlay",
        legend_title_text="",
        margin={"l": 60, "r": 40, "t": 60, "b": 60},
    )
    print(f"mean word count -- aave: {aave_len.mean():.1f}  sae: {sae_len.mean():.1f}")

    fig1.show()
    fig2.show()
    fig3.show()
    fig4.show()


if __name__ == "__main__":
    show_plots(LAYER, VARIANT, POOLING)
