"""One-time script to build data/historical_cache.csv.

Usage:
    export FRED_API_KEY=your_key_here
    python scripts/build_cache.py

Or pass the key directly:
    python scripts/build_cache.py --fred-key YOUR_KEY

Pulls everything from 2000-01-01 through today and saves to
data/historical_cache.csv. Re-running overwrites the existing file.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

# Allow running from repo root or scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import (
    EXPECTED_COLS,
    FRED_SERIES,
    HISTORY_START,
    YFINANCE_TICKERS,
    _extract_close,
)


def _download_yfinance(start: str, end: str) -> pd.DataFrame:
    import yfinance as yf

    series_list: list[pd.Series] = []
    for ticker, col_name in YFINANCE_TICKERS.items():
        print(f"  Fetching {ticker} ...", end=" ", flush=True)
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
                print("no data")
                continue
            close.name = col_name
            close.index = pd.to_datetime(close.index).normalize()
            series_list.append(close)
            print(f"{len(close)} rows")
        except Exception as exc:
            print(f"ERROR: {exc}")

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)
    df.index.name = "date"
    return df.reset_index()


def _download_fred(start: str, end: str, api_key: str) -> pd.DataFrame:
    from fredapi import Fred

    fred = Fred(api_key=api_key)
    series_list: list[pd.Series] = []
    for series_id, col_name in FRED_SERIES.items():
        print(f"  Fetching FRED:{series_id} ...", end=" ", flush=True)
        try:
            s = fred.get_series(
                series_id, observation_start=start, observation_end=end
            )
            s.name = col_name
            s.index = pd.to_datetime(s.index).normalize()
            series_list.append(s)
            print(f"{len(s)} rows")
        except Exception as exc:
            print(f"ERROR: {exc}")

    if not series_list:
        return pd.DataFrame()

    df = pd.concat(series_list, axis=1)
    df.index.name = "date"
    return df.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build historical cache CSV")
    parser.add_argument("--fred-key", default=None, help="FRED API key (overrides env var)")
    parser.add_argument("--start", default=HISTORY_START, help="Start date (YYYY-MM-DD)")
    args = parser.parse_args()

    fred_key = args.fred_key or os.environ.get("FRED_API_KEY")
    if not fred_key:
        print(
            "ERROR: FRED API key required. Set FRED_API_KEY env var or use --fred-key.",
            file=sys.stderr,
        )
        sys.exit(1)

    end = (date.today() + timedelta(days=1)).isoformat()

    print(f"\nDownloading market data {args.start} → {date.today()} ...")
    market_df = _download_yfinance(args.start, end)

    print(f"\nDownloading FRED data {args.start} → {date.today()} ...")
    fred_df = _download_fred(args.start, date.today().isoformat(), fred_key)

    if market_df.empty:
        print("ERROR: No market data downloaded. Aborting.", file=sys.stderr)
        sys.exit(1)

    if not fred_df.empty:
        combined = market_df.merge(fred_df, on="date", how="left")
    else:
        combined = market_df.copy()
        for col in ("hy_oas", "t10y2y", "nfci"):
            combined[col] = float("nan")

    combined["date"] = pd.to_datetime(combined["date"]).dt.normalize()
    combined = combined.sort_values("date").drop_duplicates(subset=["date"])

    # Ensure all expected columns are present
    for col in EXPECTED_COLS:
        if col not in combined.columns:
            combined[col] = float("nan")
    combined = combined[EXPECTED_COLS]

    out_path = Path(__file__).parent.parent / "data" / "historical_cache.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(out_path, index=False)

    print(f"\nSaved {len(combined):,} rows → {out_path}")
    print(f"Date range: {combined['date'].min().date()} to {combined['date'].max().date()}")
    print("\nNext step: python scripts/fit_model.py")


if __name__ == "__main__":
    main()
