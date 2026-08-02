import numpy as np
import pandas as pd
import plotly.graph_objects as go

BLUE = "#2a78d6"    # entropy_before_bits -- band this token was chosen from
ORANGE = "#eb6834"  # entropy_after_bits  -- how open the next prediction is

COLUMN_LABELS = {
    "entropy_before_bits": "entropy of the distribution this token was chosen from",
    "entropy_after_bits": "entropy of the distribution predicting the next token",
}
COLUMN_COLORS = {
    "entropy_before_bits": BLUE,
    "entropy_after_bits": ORANGE,
}


def _add_unbounded_annotations(fig: go.Figure, df: pd.DataFrame, column: str) -> None:
    """Mark +inf rows (the first token has no preceding context) with a
    callout instead of silently dropping them from the chart."""
    for _, row in df[np.isinf(df[column])].iterrows():
        fig.add_annotation(
            x=row["position"],
            y=1,
            yref="paper",
            yanchor="top",
            text=f"'{row['token_clean']}': ∞<br>(nothing precedes it)",
            showarrow=True,
            arrowhead=2,
            ay=-40,
        )


def plot_entropy(
    df: pd.DataFrame,
    column: str = "entropy_before_bits",
    title: str = "Next-token entropy across a sentence",
) -> go.Figure:
    """Single-series line chart for one of the two entropy framings.

    column="entropy_before_bits" (default): height at token i = how wide a
        field of options the model picked token i out of, i.e. "the model
        chose this word from this tight/wide a band of choices." Token 0's
        value is +inf (nothing precedes it) -- excluded from the line so the
        chart stays on a finite scale, but flagged with an annotation rather
        than just disappearing.
    column="entropy_after_bits": height at token i = how open the
        prediction for the token AFTER i is. Always finite.
    """
    plot_df = df[np.isfinite(df[column])]
    color = COLUMN_COLORS[column]
    label = COLUMN_LABELS[column]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=plot_df["position"],
            y=plot_df[column],
            mode="lines+markers",
            line=dict(color=color, width=2),
            marker=dict(size=8, color=color),
            text=plot_df["token_clean"],
            hovertemplate="%{text}<br>" + label + ": %{y:.2f} bits<extra></extra>",
        )
    )
    _add_unbounded_annotations(fig, df, column)
    fig.update_layout(
        title=title,
        xaxis=dict(
            title="token position (tick label = the token itself)",
            tickmode="array",
            tickvals=df["position"],
            ticktext=df["token_clean"],
        ),
        yaxis_title=f"{label} (bits)",
        margin={"l": 60, "r": 40, "t": 60, "b": 60},
    )
    return fig
