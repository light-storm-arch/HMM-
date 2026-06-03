"""Streamlit HMM Market Regime Detector — entry point."""
from __future__ import annotations

import traceback
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.constants import REGIME_CARD_COLORS, SPLIT_DATE, TRAIN_START

st.set_page_config(
    page_title="HMM Regime Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = Path("models/hmm_fitted.pkl")
CACHE_PATH = Path("data/historical_cache.csv")

FRED_STALE_BUSINESS_DAYS = 5
FRED_DISPLAY_NAMES = {
    "hy_oas": "Baa corporate spread (BAA10Y)",
    "t10y2y": "10Y–2Y yield curve (T10Y2Y)",
    "nfci": "Chicago Fed NFCI",
}


# ── helpers ───────────────────────────────────────────────────────────────────

@st.cache_resource
def _load_model_from_disk():
    from src.hmm_model import load_model
    return load_model(MODEL_PATH)


def _get_model():
    """Return (model, scaler, state_map), preferring any session-state override."""
    if "model_override" in st.session_state:
        return st.session_state["model_override"]
    return _load_model_from_disk()


@st.cache_data(ttl=3600)
def _load_and_build(fred_api_key: str | None) -> tuple[pd.DataFrame, dict]:
    from src.data_loader import load_data
    from src.features import build_features
    raw = load_data(fred_api_key)
    features = build_features(raw)
    fred_last_update = features.attrs.get("fred_last_update", {})
    return features, fred_last_update


def _predict(features_df: pd.DataFrame) -> pd.DataFrame:
    from src.hmm_model import predict_regimes
    model, scaler, state_map = _get_model()
    return predict_regimes(model, scaler, state_map, features_df)


# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(
    regime_min_date: date,
    max_date: date,
    features_df: pd.DataFrame,
    fred_last_update: dict,
) -> tuple[tuple[date, date], str, bool]:
    st.sidebar.title("HMM Regime Detector")
    st.sidebar.markdown("---")

    st.sidebar.subheader("Display window")
    date_range = st.sidebar.slider(
        "Date range",
        min_value=regime_min_date,
        max_value=max_date,
        value=(regime_min_date, max_date),
        format="YYYY-MM-DD",
    )

    st.sidebar.subheader("View")
    period_view = st.sidebar.radio(
        "Show periods",
        ["Both", "In-sample only", "Out-of-sample only"],
        index=0,
    )

    st.sidebar.subheader("Model")
    ack = st.sidebar.checkbox(
        "I understand this invalidates OOS stats",
        key="refit_ack",
        help="Required before the refit button is enabled.",
    )
    if ack:
        st.sidebar.warning(
            "Refitting on full history removes the genuine out-of-sample window. "
            "OOS stats will be meaningless until the page is reloaded."
        )
    refit_clicked = st.sidebar.button(
        "Refit model on full history",
        help="Refits on 2007 → today. Eliminates the true out-of-sample period.",
        disabled=not ack,
    )

    if refit_clicked:
        with st.sidebar.status("Refitting HMM…"):
            from src.hmm_model import fit_hmm
            model, scaler, state_map = fit_hmm(
                features_df, TRAIN_START, str(date.today())
            )
            st.session_state["model_override"] = (model, scaler, state_map)
            st.session_state["refitted"] = True
        st.sidebar.success("Refitted on full history.")
        st.rerun()

    if st.session_state.get("refitted"):
        st.sidebar.info("Using full-history refitted model.")

    st.sidebar.markdown("---")
    _render_fred_freshness(fred_last_update)

    return date_range, period_view, refit_clicked


def _render_fred_freshness(fred_last_update: dict) -> None:
    """Show last-real-update dates for each FRED series with a stale-data warning."""
    st.sidebar.subheader("FRED data freshness")

    if not fred_last_update or all(v is None for v in fred_last_update.values()):
        st.sidebar.warning(
            "FRED data unavailable (no API key configured). The model is running "
            "without macro inputs — set FRED_API_KEY in secrets to enable."
        )
        return

    today = pd.Timestamp(date.today())
    any_stale = False
    lines: list[str] = []
    for col, label in FRED_DISPLAY_NAMES.items():
        ts = fred_last_update.get(col)
        if ts is None or pd.isna(ts):
            lines.append(f"- **{label}**: unavailable")
            any_stale = True
            continue
        business_days = np.busday_count(ts.date(), today.date())
        suffix = f"{business_days} business day{'s' if business_days != 1 else ''} ago"
        lines.append(f"- **{label}**: {ts.date().isoformat()}  ({suffix})")
        if business_days > FRED_STALE_BUSINESS_DAYS:
            any_stale = True

    st.sidebar.markdown("\n".join(lines))
    if any_stale:
        st.sidebar.warning(
            f"At least one FRED series is more than {FRED_STALE_BUSINESS_DAYS} business "
            "days stale. Forward-filled values are being used in its place."
        )
    st.sidebar.caption("Market data updates hourly. FRED series often lag by days.")


# ── tab 1: current regime + history ───────────────────────────────────────────

def render_tab1(regime_df: pd.DataFrame, date_range: tuple[date, date], period_view: str) -> None:
    from src.plots import plot_regime_probabilities, plot_spy_with_regimes

    # Apply period filter to the display slice
    start_d, end_d = date_range
    display_df = regime_df[
        (regime_df["date"] >= pd.Timestamp(start_d))
        & (regime_df["date"] <= pd.Timestamp(end_d))
    ].copy()

    if period_view == "In-sample only":
        display_df = display_df[display_df["date"] < SPLIT_DATE]
    elif period_view == "Out-of-sample only":
        display_df = display_df[display_df["date"] >= SPLIT_DATE]

    # ── current regime cards ──────────────────────────────────────────────────
    st.subheader("Current Regime")

    latest = regime_df.iloc[-1]
    current_label = latest.get("state_label", "—")
    p_calm = latest.get("prob_calm", np.nan)
    p_choppy = latest.get("prob_choppy", np.nan)
    p_stress = latest.get("prob_stress", np.nan)
    latest_date = pd.to_datetime(latest["date"]).strftime("%Y-%m-%d")

    probs = [p for p in (p_calm, p_choppy, p_stress) if not np.isnan(p)]
    confidence_str = f"{max(probs):.0%}" if probs else "—"

    col_label, col_calm, col_choppy, col_stress = st.columns(4)

    bg, accent = REGIME_CARD_COLORS.get(current_label, ("#1e293b", "#94a3b8"))
    col_label.markdown(
        f"""
        <div style="background:{bg};border-left:4px solid {accent};
                    border-radius:8px;padding:16px;text-align:center">
            <div style="font-size:0.75rem;color:#94a3b8">REGIME ({latest_date})</div>
            <div style="font-size:1.8rem;font-weight:700;color:{accent}">{current_label}</div>
            <div style="font-size:0.75rem;color:#94a3b8;margin-top:4px">
                Confidence: <span style="color:#e2e8f0;font-weight:600">{confidence_str}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    for col, label, prob in [
        (col_calm, "Calm", p_calm),
        (col_choppy, "Choppy", p_choppy),
        (col_stress, "Stress", p_stress),
    ]:
        _, accent = REGIME_CARD_COLORS.get(label, ("#1e293b", "#94a3b8"))
        val = f"{prob:.1%}" if not np.isnan(prob) else "—"
        col.markdown(
            f"""
            <div style="background:#1e293b;border-left:4px solid {accent};
                        border-radius:8px;padding:16px;text-align:center">
                <div style="font-size:0.75rem;color:#94a3b8">{label.upper()} PROB</div>
                <div style="font-size:1.8rem;font-weight:700;color:{accent}">{val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # ── charts ────────────────────────────────────────────────────────────────
    st.plotly_chart(
        plot_spy_with_regimes(display_df, SPLIT_DATE),
        use_container_width=True,
    )
    st.plotly_chart(
        plot_regime_probabilities(display_df, SPLIT_DATE),
        use_container_width=True,
    )

    # ── recent days table ─────────────────────────────────────────────────────
    st.subheader("Last 20 Trading Days")
    recent = regime_df.tail(20)[
        ["date", "spy_close", "log_return", "state_label", "prob_stress"]
    ].copy()
    recent["date"] = recent["date"].dt.strftime("%Y-%m-%d")
    recent["spy_close"] = recent["spy_close"].round(2)
    recent["log_return"] = (recent["log_return"] * 100).round(3)
    recent["prob_stress"] = (recent["prob_stress"] * 100).round(1)
    recent.columns = ["Date", "SPY Close", "Daily Return (%)", "Regime", "Stress Prob (%)"]
    st.dataframe(recent.iloc[::-1], use_container_width=True, hide_index=True)

    # ── CSV export ────────────────────────────────────────────────────────────
    export_cols = [
        c
        for c in ["date", "spy_close", "log_return", "state_label",
                  "prob_calm", "prob_choppy", "prob_stress"]
        if c in regime_df.columns
    ]
    export_df = regime_df[export_cols].copy()
    export_df["date"] = pd.to_datetime(export_df["date"]).dt.strftime("%Y-%m-%d")
    max_date_str = pd.to_datetime(regime_df["date"].max()).strftime("%Y%m%d")
    st.download_button(
        "Download regime history (CSV)",
        data=export_df.to_csv(index=False).encode("utf-8"),
        file_name=f"hmm_regimes_{max_date_str}.csv",
        mime="text/csv",
    )


# ── tab 2: backtest stats ─────────────────────────────────────────────────────

def render_tab2(regime_df: pd.DataFrame) -> None:
    from src.backtest import (
        compute_forward_returns,
        compute_regime_stats,
        compute_transition_matrix,
    )
    from src.plots import (
        REGIME_COLORS,
        plot_forward_returns,
        plot_regime_comparison,
        plot_transition_matrix,
    )

    in_stats, oos_stats = compute_regime_stats(regime_df, SPLIT_DATE)

    # ── stats tables ──────────────────────────────────────────────────────────
    st.subheader("Regime-Conditional Statistics")

    def _fmt_stats(df: pd.DataFrame, label: str) -> pd.DataFrame:
        out = df.copy()
        out["annual_return"] = (out["annual_return"] * 100).map("{:.1f}%".format)
        out["annual_vol"] = (out["annual_vol"] * 100).map("{:.1f}%".format)
        out["sharpe"] = out["sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "—")
        out["max_drawdown"] = (out["max_drawdown"] * 100).map("{:.1f}%".format)
        out["pct_days"] = out["pct_days"].map("{:.1f}%".format)
        out["avg_duration_days"] = out["avg_duration_days"].map(
            lambda x: f"{x:.1f}" if pd.notna(x) else "—"
        )
        out["n_days"] = out["n_days"].astype(int)
        out.columns = [
            "Ann. Return", "Ann. Vol", "Sharpe", "Max DD",
            "% of Days", "Avg Duration (d)", "N Days",
        ]
        out.index.name = "Regime"
        return out

    col_is, col_oos = st.columns(2)
    with col_is:
        st.markdown("**In-sample (2007–2018)**")
        st.dataframe(_fmt_stats(in_stats, "in"), use_container_width=True)
    with col_oos:
        st.markdown("**Out-of-sample (2019–present)**")
        st.dataframe(_fmt_stats(oos_stats, "oos"), use_container_width=True)

    if st.session_state.get("refitted"):
        st.warning(
            "Model was refitted on full history — OOS stats above are meaningless "
            "(the model has seen all the data)."
        )

    # ── annualised return comparison chart ───────────────────────────────────
    st.plotly_chart(
        plot_regime_comparison(in_stats, oos_stats, "annual_return", "Annualised Return"),
        use_container_width=True,
    )

    # ── forward return violin plots ───────────────────────────────────────────
    st.subheader("Forward Return Distributions (in-sample only)")
    fwd_df = compute_forward_returns(regime_df, SPLIT_DATE)
    if len(fwd_df) > 10:
        st.plotly_chart(plot_forward_returns(fwd_df), use_container_width=True)
    else:
        st.info("Not enough in-sample data to compute forward return distributions.")

    # ── transition matrix ────────────────────────────────────────────────────
    st.subheader("Regime Transition Matrix")
    col_trans, col_spacer = st.columns([1, 1])
    with col_trans:
        trans = compute_transition_matrix(regime_df)
        st.plotly_chart(plot_transition_matrix(trans), use_container_width=True)

    with col_spacer:
        st.markdown("**Interpretation**")
        st.markdown(
            """
The matrix shows the empirical probability of moving from one regime to another on the
next trading day, computed from Viterbi-decoded states on the full dataset.

High diagonal values → regimes are persistent (strong autocorrelation).
Off-diagonal entries → typical transition paths (e.g. Calm → Choppy before Stress).
            """
        )


# ── tab 3: how it works ───────────────────────────────────────────────────────

def render_tab3() -> None:
    st.subheader("What this tool does")
    st.markdown(
        """
Every day the stock market behaves differently — sometimes calm and grinding higher,
sometimes volatile and directionless, sometimes in outright crisis mode. This tool reads
nine daily market signals, learns to recognise these patterns from history, and tells you
**which of three regimes the market is currently in**, and how confident it is.

It uses a **Hidden Markov Model (HMM)** — a statistical model that assumes markets move
through hidden states you cannot observe directly, but can infer from signals you *can*
observe. The model was trained on data from **2007–2018** only. Everything from 2019 onward
is genuine out-of-sample — the model had never seen those years when it learned the patterns.
        """
    )

    st.divider()

    st.subheader("The three regimes")
    st.markdown(
        """
> **These are volatility-and-stress regimes, not directional calls.**
> The model identifies *how* the market is behaving — not whether prices will go up or down.
        """
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(
            """
<div style="background:#166534;border-left:4px solid #22c55e;border-radius:8px;padding:16px">
<div style="font-size:1.1rem;font-weight:700;color:#22c55e">🟢 Calm</div>
<div style="color:#e2e8f0;margin-top:8px;font-size:0.9rem">
Low volatility. Markets grinding higher or flat. Credit spreads tight.
VIX low and in normal contango. Typical "everything is fine" environment.
</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
<div style="background:#78350f;border-left:4px solid #f59e0b;border-radius:8px;padding:16px">
<div style="font-size:1.1rem;font-weight:700;color:#f59e0b">🟡 Choppy</div>
<div style="color:#e2e8f0;margin-top:8px;font-size:0.9rem">
Elevated volatility. Mixed daily returns — up one day, down the next.
Something is off but not broken. Credit spreads starting to widen.
The "transition zone" between calm and stress.
</div>
</div>
            """,
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            """
<div style="background:#7f1d1d;border-left:4px solid #ef4444;border-radius:8px;padding:16px">
<div style="font-size:1.1rem;font-weight:700;color:#ef4444">🔴 Stress</div>
<div style="color:#e2e8f0;margin-top:8px;font-size:0.9rem">
High volatility. Negative average returns. VIX spiking. Credit spreads
blowing out. Investors fleeing to Treasuries. Crisis behaviour.
</div>
</div>
            """,
            unsafe_allow_html=True,
        )

    st.divider()

    st.subheader("The 9 input signals")
    st.markdown(
        "These are calculated fresh every trading day and fed into the model."
    )

    features_data = {
        "Signal": [
            "SPY daily return",
            "Realized volatility (20-day)",
            "VIX level",
            "VIX term structure",
            "HYG / LQD ratio change",
            "TLT daily return",
            "Baa corporate spread",
            "10Y – 2Y yield curve",
            "NFCI",
        ],
        "What it measures": [
            "Daily log return of SPY — the S&P 500 ETF",
            "How much SPY has been moving over the past month, annualised. The single most important regime signal.",
            "The market's implied fear gauge — derived from S&P 500 options prices",
            "VIX minus 3-month VIX. When near-term fear exceeds 3-month fear (backwardation), it signals acute stress.",
            "5-day change in the ratio of high-yield bonds (HYG) to investment-grade bonds (LQD). Risk-off when HY underperforms.",
            "Daily return of the 20+ year Treasury ETF. Rises in a flight-to-safety, falls when risk appetite is strong.",
            "Moody's Baa corporate bond yield spread over 10-year Treasuries. Widens when credit markets are stressed.",
            "10-year Treasury yield minus 2-year yield. Inversion signals recession risk; steepening signals recovery.",
            "Chicago Fed National Financial Conditions Index — a broad weekly gauge of stress across money markets, debt, and equity.",
        ],
        "Source": [
            "Yahoo Finance", "Calculated from SPY", "Yahoo Finance (^VIX)",
            "Yahoo Finance (^VIX – ^VIX3M)", "Yahoo Finance (HYG, LQD)",
            "Yahoo Finance (TLT)", "FRED (BAA10Y)",
            "FRED (T10Y2Y)", "FRED (NFCI)",
        ],
    }

    st.dataframe(
        features_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Signal": st.column_config.TextColumn(width="medium"),
            "What it measures": st.column_config.TextColumn(width="large"),
            "Source": st.column_config.TextColumn(width="small"),
        },
    )

    st.divider()

    st.subheader("How the model works")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Training (done once, offline)**")
        st.markdown(
            """
1. Collect all 9 signals for every trading day **2007–2018** (~3,000 days)
2. Standardise each signal (zero mean, unit variance) so they're on equal footing
3. Run the HMM fitting algorithm — it finds 3 clusters of days that look statistically
   similar to each other, and learns the probability of moving between clusters day-to-day
4. Label the clusters by volatility rank: lowest vol → **Calm**, middle → **Choppy**,
   highest → **Stress**. This labelling is automatic and consistent across refits.
5. Save the trained model to disk so the app never retrains on load
            """
        )

    with col_b:
        st.markdown("**Live classification (runs daily)**")
        st.markdown(
            """
1. Fetch today's signals from Yahoo Finance and FRED
2. Apply the same standardisation used during training
3. Run **Viterbi decoding** — an algorithm that finds the single most-likely sequence
   of hidden states across the entire price history, all at once
4. Run **forward-backward algorithm** — produces probability estimates for each state
   on each day (the three coloured bars in the metric cards)
5. Display current regime + full history on the dashboard
            """
        )

    st.divider()

    st.subheader("Important caveats")
    st.markdown(
        """
| Caveat | Detail |
|---|---|
| **Not a directional signal** | A Calm regime can occur while markets drift sideways for months. A Stress regime can occur during sharp recoveries. The model identifies *how* the market is behaving, not *where* it is going. |
| **Detection lag** | Viterbi decoding is applied to the full sequence at once — it uses some hindsight. Expect a **5–15 trading-day lag** before a real-time regime shift is detected with confidence. |
| **Out-of-sample boundary** | The model was trained on 2007–2018. Stats for 2019–present are genuine out-of-sample. If you click "Refit on full history", that boundary disappears and OOS stats become meaningless. |
| **Descriptive, not causal** | Regime labels describe a statistical cluster of market behaviour. They do not explain *why* the market shifted. |
| **Single train/test split** | Version 1 uses one fixed train/test split. A rolling walk-forward validation would give more reliable performance estimates. |
        """
    )


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    fred_api_key: str | None = st.secrets.get("FRED_API_KEY", None)

    # ── data loading ──────────────────────────────────────────────────────────
    try:
        features_df, fred_last_update = _load_and_build(fred_api_key)
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python scripts/build_cache.py", language="bash")
        st.stop()

    # ── model loading ─────────────────────────────────────────────────────────
    try:
        _get_model()
    except FileNotFoundError as exc:
        st.error(str(exc))
        st.code("python scripts/fit_model.py", language="bash")
        st.stop()

    # ── regime prediction (before sidebar so we know the true min date) ─────────
    try:
        regime_df = _predict(features_df)
    except Exception:
        st.error("Regime prediction failed. Details:")
        st.code(traceback.format_exc())
        st.stop()

    if regime_df.empty:
        st.warning("No regime data to display — check that features have no unexpected NaNs.")
        st.stop()

    regime_min_date = regime_df["date"].min().date()
    regime_max_date = regime_df["date"].max().date()

    # ── sidebar ───────────────────────────────────────────────────────────────
    date_range, period_view, _ = render_sidebar(
        regime_min_date, regime_max_date, features_df, fred_last_update
    )

    # ── main content ──────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["📊 Regime History", "🔬 Backtest Stats", "📖 How It Works"])

    with tab1:
        render_tab1(regime_df, date_range, period_view)

    with tab2:
        render_tab2(regime_df)

    with tab3:
        render_tab3()


if __name__ == "__main__":
    main()
