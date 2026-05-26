from __future__ import annotations

import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

CACHE_PATH = Path(__file__).parent.parent / "data" / "historical_cache.csv"

HISTORY_START = "2000-01-01"

# yfinance ticker → output column name
YFINANCE_TICKERS: dict[str, str] = {
    "SPY": "spy_close",
    "^VIX": "vix_close",
    "^VIX3M": "vix3m_close",
    "HYG": "hyg_close",
    "LQD": "lqd_close",
    "TLT": "tlt_close",
}

# FRED series ID → output column name
# BAA10Y = Moody's Baa corporate spread over 10-year Treasury (free, history from 1986).
# BAMLH0A0HYM2 (ICE BofA HY OAS) is restricted to recent history on free FRED API keys.
FRED_SERIES: dict[str, str] = {
    "BAA10Y": "hy_oas",
    "T10Y2Y": "t10y2y",
    "NFCI": "nfci",
}

EXPECTED_COLS = [
    "date",
    "spy_close",
    "vix_close",
    "vix3m_close",
    "hyg_close",
    "lqd_close",
    "tlt_close",
    "hy_oas",
    "t10y2y",
    "nfci",
]


def _extract_close(data: pd.DataFrame, ticker: str) -> pd.Series | None:
    """Extract Close from a yfinance DataFrame, handling MultiIndex columns."""
    if data.empty:
        return None
    if isinstance(data.columns, pd.MultiIndex):
        for price_col in ("Close", "Adj Close"):
            if (price_col, ticker) in data.columns:
                return data[(price_col, ticker)]
        return None
    for price_col in ("Close", "Adj Close"):
        if price_col in data.columns:
            return data[price_col]
    return None


def fetch_yfinance(start: str, end: str) -> pd.DataFrame:
    """Download market price data from yfinance, one ticker at a time."""
    series_list: list[pd.Series] = []
    for ticker, col_name in YFINANCE_TICKERS.items():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                raw = yf.download(
                    ticker,
                    start=start,
                    end=end,
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            close = _extract_close(raw, ticker)
            if close is None or close.empty:
                continue
            close.name = col_name
            close.index = pd.to_datetime(close.index).normalize()
            series_list.append(close)
        except Exception:
            pass

    if not series_list:
        return pd.DataFrame(columns=["date"])

    df = pd.concat(series_list, axis=1, sort=True)
    df.index.name = "date"
    df = df.reset_index()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


def fetch_fred(start: str, end: str, api_key: str) -> pd.DataFrame:
    """Download economic series from FRED."""
    try:
        from fredapi import Fred
    except ImportError:
        return pd.DataFrame(columns=["date"])

    fred = Fred(api_key=api_key)
    series_list: list[pd.Series] = []
    for series_id, col_name in FRED_SERIES.items():
        try:
            s = fred.get_series(
                series_id, observation_start=start, observation_end=end
            )
            s.name = col_name
            s.index = pd.to_datetime(s.index).normalize()
            series_list.append(s)
        except Exception:
            pass

    if not series_list:
        return pd.DataFrame(columns=["date"])

    df = pd.concat(series_list, axis=1, sort=True)
    df.index.name = "date"
    df = df.reset_index()
    return df


def _post_process(df: pd.DataFrame) -> pd.DataFrame:
    """Forward-fill prices for market-closed days and carry FRED values."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df.set_index("date")

    market_cols = ["spy_close", "vix_close", "vix3m_close", "hyg_close", "lqd_close", "tlt_close"]
    for col in market_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    fred_cols = ["hy_oas", "t10y2y", "nfci"]
    for col in fred_cols:
        if col in df.columns:
            df[col] = df[col].ffill()

    df = df.reset_index()
    return df


def load_data(fred_api_key: str | None = None) -> pd.DataFrame:
    """Load historical cache, then append any live data through today.

    Raises FileNotFoundError if the cache CSV is missing — run
    scripts/build_cache.py first.
    """
    if not CACHE_PATH.exists():
        raise FileNotFoundError(
            f"Historical cache not found at {CACHE_PATH}.\n"
            "Run  python scripts/build_cache.py  to generate it first."
        )

    cache = pd.read_csv(CACHE_PATH, parse_dates=["date"])
    cache["date"] = pd.to_datetime(cache["date"]).dt.normalize()
    cache = cache.sort_values("date").reset_index(drop=True)

    max_cached = cache["date"].max().date()
    today = date.today()

    if max_cached >= today:
        return _post_process(cache)

    fetch_start = (max_cached + timedelta(days=1)).isoformat()
    # yfinance end is exclusive, so use tomorrow
    fetch_end = (today + timedelta(days=1)).isoformat()

    market_df = fetch_yfinance(fetch_start, fetch_end)

    if fred_api_key and not market_df.empty:
        fred_df = fetch_fred(fetch_start, today.isoformat(), fred_api_key)
        if not fred_df.empty and "date" in fred_df.columns:
            market_df = market_df.merge(fred_df, on="date", how="left")

    if market_df.empty or "date" not in market_df.columns:
        return _post_process(cache)

    for col in EXPECTED_COLS[1:]:
        if col not in market_df.columns:
            market_df[col] = np.nan

    combined = pd.concat(
        [cache[EXPECTED_COLS], market_df[EXPECTED_COLS]],
        ignore_index=True,
    )
    return _post_process(combined)


def load_data_cached(fred_api_key: str | None = None) -> pd.DataFrame:
    """Streamlit-cached version of load_data. Falls back to uncached outside Streamlit."""
    import streamlit as st
    if not hasattr(load_data_cached, "_fn"):
        load_data_cached._fn = st.cache_data(ttl=3600)(load_data)
    return load_data_cached._fn(fred_api_key)
