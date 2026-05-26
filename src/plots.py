from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REGIME_COLORS: dict[str, str] = {
    "Calm": "#22c55e",
    "Choppy": "#f59e0b",
    "Stress": "#ef4444",
}


def _rgba(hex_color: str, alpha: float) -> str:
    """Convert '#RRGGBB' + alpha to 'rgba(r,g,b,alpha)' accepted by all plotly versions."""
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

_DARK_BG = "#0e1117"
_PLOT_BG = "#0e1117"
_PAPER_BG = "#0e1117"
_GRID_COLOR = "#2d3748"
_TEXT_COLOR = "#e2e8f0"

_BASE_LAYOUT = dict(
    plot_bgcolor=_PLOT_BG,
    paper_bgcolor=_PAPER_BG,
    font=dict(color=_TEXT_COLOR, size=12),
    xaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
    yaxis=dict(gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=_GRID_COLOR),
    margin=dict(l=50, r=20, t=40, b=40),
)


def _regime_spans(df: pd.DataFrame) -> list[dict]:
    """Return list of {start, end, label} for consecutive regime runs."""
    if df.empty or "state_label" not in df.columns:
        return []
    spans: list[dict] = []
    current_label: str | None = None
    start_date = None

    for _, row in df.iterrows():
        label = row["state_label"]
        if pd.isna(label):
            continue
        if label != current_label:
            if current_label is not None:
                spans.append({"start": start_date, "end": row["date"], "label": current_label})
            current_label = label
            start_date = row["date"]

    if current_label is not None:
        spans.append({"start": start_date, "end": df["date"].iloc[-1], "label": current_label})
    return spans


def _add_oos_line(fig: go.Figure, split_date: str, label: bool = True) -> None:
    """Draw a full-height dashed line at the OOS boundary.

    Uses add_shape/add_annotation rather than add_vline: the latter computes a
    midpoint via sum() over the x endpoints, which raises on string dates in
    recent plotly versions.
    """
    fig.add_shape(
        type="line",
        x0=split_date,
        x1=split_date,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="#94a3b8", dash="dash", width=1),
    )
    if label:
        fig.add_annotation(
            x=split_date,
            y=1,
            yref="paper",
            text="OOS start",
            showarrow=False,
            font=dict(color="#94a3b8", size=11),
            xanchor="left",
            yanchor="bottom",
        )


def plot_spy_with_regimes(
    df: pd.DataFrame,
    split_date: str = "2019-01-01",
) -> go.Figure:
    """SPY log-price line chart with colored regime background bands."""
    fig = go.Figure()

    spans = _regime_spans(df)
    for span in spans:
        color = REGIME_COLORS.get(span["label"], "#888888")
        fig.add_vrect(
            x0=span["start"],
            x1=span["end"],
            fillcolor=color,
            opacity=0.12,
            layer="below",
            line_width=0,
        )

    fig.add_trace(
        go.Scatter(
            x=df["date"],
            y=df["spy_close"],
            mode="lines",
            name="SPY",
            line=dict(color="#60a5fa", width=1.5),
        )
    )

    _add_oos_line(fig, split_date, label=True)

    fig.update_layout(
        **_BASE_LAYOUT,
        title="SPY Price — Regime Background",
        xaxis_title=None,
        yaxis_title="Price (USD)",
        showlegend=False,
        height=400,
    )
    return fig


def plot_regime_probabilities(
    df: pd.DataFrame,
    split_date: str = "2019-01-01",
) -> go.Figure:
    """Stacked area chart of posterior regime probabilities summing to 1."""
    fig = go.Figure()

    for label in ["Calm", "Choppy", "Stress"]:
        col = f"prob_{label.lower()}"
        if col not in df.columns:
            continue
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[col],
                stackgroup="one",
                name=label,
                fillcolor=_rgba(REGIME_COLORS[label], 0.6),
                line=dict(color=REGIME_COLORS[label], width=0.5),
                mode="lines",
            )
        )

    _add_oos_line(fig, split_date, label=False)

    fig.update_layout(
        **_BASE_LAYOUT,
        title="Regime Posterior Probabilities",
        xaxis_title=None,
        yaxis_title="Probability",
        yaxis=dict(range=[0, 1], gridcolor=_GRID_COLOR, zerolinecolor=_GRID_COLOR),
        height=300,
    )
    return fig


def plot_forward_returns(fwd_df: pd.DataFrame) -> go.Figure:
    """Violin plots of 1m / 3m / 6m forward returns grouped by regime."""
    horizons = {"fwd_1m": "1 Month", "fwd_3m": "3 Months", "fwd_6m": "6 Months"}
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=list(horizons.values()),
        shared_yaxes=True,
    )

    for col_idx, (col, title) in enumerate(horizons.items(), start=1):
        for label in ["Calm", "Choppy", "Stress"]:
            subset = fwd_df[fwd_df["state_label"] == label][col].dropna()
            fig.add_trace(
                go.Violin(
                    y=subset * 100,
                    name=label,
                    box_visible=True,
                    meanline_visible=True,
                    fillcolor=_rgba(REGIME_COLORS[label], 0.53),
                    line_color=REGIME_COLORS[label],
                    showlegend=(col_idx == 1),
                ),
                row=1,
                col=col_idx,
            )

    fig.update_layout(
        **_BASE_LAYOUT,
        title="Forward SPY Returns by Regime (in-sample only)",
        yaxis_title="Return (%)",
        height=400,
        violinmode="group",
    )
    return fig


def plot_transition_matrix(trans_matrix: pd.DataFrame) -> go.Figure:
    """Heatmap of regime transition probabilities."""
    labels = trans_matrix.index.tolist()
    z = trans_matrix.values
    text = [[f"{v:.1%}" for v in row] for row in z]

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=labels,
            y=labels,
            text=text,
            texttemplate="%{text}",
            colorscale="Blues",
            showscale=True,
            zmin=0,
            zmax=1,
        )
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        title="Regime Transition Probabilities (day-over-day)",
        xaxis_title="To",
        yaxis_title="From",
        height=380,
    )
    return fig


def plot_regime_comparison(
    in_sample: pd.DataFrame,
    oos: pd.DataFrame,
    metric: str = "annual_return",
    metric_label: str = "Annualised Return",
) -> go.Figure:
    """Grouped bar chart comparing a single metric in-sample vs out-of-sample."""
    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="In-sample (2007–2018)",
            x=in_sample.index.tolist(),
            y=in_sample[metric],
            marker_color=[REGIME_COLORS.get(r, "#888") for r in in_sample.index],
            opacity=0.9,
        )
    )
    fig.add_trace(
        go.Bar(
            name="Out-of-sample (2019–present)",
            x=oos.index.tolist(),
            y=oos[metric],
            marker_color=[REGIME_COLORS.get(r, "#888") for r in oos.index],
            opacity=0.5,
            marker_pattern_shape="/",
        )
    )

    fig.update_layout(
        **_BASE_LAYOUT,
        title=f"{metric_label} — In-Sample vs Out-of-Sample",
        yaxis_title=metric_label,
        barmode="group",
        height=360,
    )
    return fig
