"""Streamlit HMM Market Regime Detector — entry point."""
from __future__ import annotations

import traceback
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="HMM Regime Detector",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── constants ─────────────────────────────────────────────────────────────────
SPLIT_DATE = "2019-01-01"
TRAIN_START = "2007-01-01"
MODEL_PATH = Path("models/hmm_fitted.pkl")
CACHE_PATH = Path("data/historical_cache.csv")

REGIME_CARD_COLORS = {
    "Calm": ("#166534", "#22c55e"),   # bg, accent
    "Choppy": ("#78350f", "#f59e0b"),
    "Stress": ("#7f1d1d", "#ef4444"),
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
def _load_and_build(fred_api_key: str | None) -> pd.DataFrame:
    from src.data_loader import load_data
    from src.features import build_features
    raw = load_data(fred_api_key)
    return build_features(raw)


def _predict(features_df: pd.DataFrame) -> pd.DataFrame:
    from src.hmm_model import predict_regimes
    model, scaler, state_map = _get_model()
    return predict_regimes(model, scaler, state_map, features_df)


# ── sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar(features_df: pd.DataFrame) -> tuple[tuple[date, date], str, bool]:
    st.sidebar.title("HMM Regime Detector")
    st.sidebar.markdown("---")

    min_date = features_df["date"].min().date()
    max_date = features_df["date"].max().date()

    st.sidebar.subheader("Display window")
    date_range = st.sidebar.slider(
        "Date range",
        min_value=min_date,
        max_value=max_date,
        value=(date(2015, 1, 1), max_date),
        format="YYYY-MM-DD",
    )

    st.sidebar.subheader("View")
    period_view = st.sidebar.radio(
        "Show periods",
        ["Both", "In-sample only", "Out-of-sample only"],
        index=0,
    )

    st.sidebar.subheader("Model")
    refit_clicked = st.sidebar.button(
        "Refit model on full history",
        help="Refits on 2007 → today. Eliminates the true out-of-sample period.",
    )

    if refit_clicked:
        st.sidebar.warning(
            "Refitting on full history removes the genuine out-of-sample window. "
            "OOS stats will be meaningless after this — reload the page to restore."
        )
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
    st.sidebar.caption("Data updates every hour. FRED series update weekly/monthly.")

    return date_range, period_view, refit_clicked


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

    col_label, col_calm, col_choppy, col_stress = st.columns(4)

    bg, accent = REGIME_CARD_COLORS.get(current_label, ("#1e293b", "#94a3b8"))
    col_label.markdown(
        f"""
        <div style="background:{bg};border-left:4px solid {accent};
                    border-radius:8px;padding:16px;text-align:center">
            <div style="font-size:0.75rem;color:#94a3b8">REGIME ({latest_date})</div>
            <div style="font-size:1.8rem;font-weight:700;color:{accent}">{current_label}</div>
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


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    fred_api_key: str | None = st.secrets.get("FRED_API_KEY", None)

    # ── data loading ──────────────────────────────────────────────────────────
    try:
        features_df = _load_and_build(fred_api_key)
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

    # ── sidebar ───────────────────────────────────────────────────────────────
    date_range, period_view, _ = render_sidebar(features_df)

    # ── regime prediction ─────────────────────────────────────────────────────
    try:
        regime_df = _predict(features_df)
    except Exception:
        st.error("Regime prediction failed. Details:")
        st.code(traceback.format_exc())
        st.stop()

    if regime_df.empty:
        st.warning("No regime data to display — check that features have no unexpected NaNs.")
        st.stop()

    # ── main content ──────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["📊 Regime History", "🔬 Backtest Stats"])

    with tab1:
        render_tab1(regime_df, date_range, period_view)

    with tab2:
        render_tab2(regime_df)


if __name__ == "__main__":
    main()
