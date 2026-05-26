from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT_DATE = "2019-01-01"
STATE_LABELS = ["Calm", "Choppy", "Stress"]


def _regime_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-regime statistics for a single period slice."""
    rows = []
    for label in STATE_LABELS:
        mask = df["state_label"] == label
        subset = df[mask].copy()
        n_days = len(df)
        n_regime = len(subset)

        if n_regime < 5:
            rows.append(
                {
                    "regime": label,
                    "annual_return": np.nan,
                    "annual_vol": np.nan,
                    "sharpe": np.nan,
                    "max_drawdown": np.nan,
                    "pct_days": n_regime / n_days * 100 if n_days else np.nan,
                    "avg_duration_days": np.nan,
                    "n_days": n_regime,
                }
            )
            continue

        rets = subset["log_return"].fillna(0)
        ann_ret = rets.mean() * 252
        ann_vol = rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

        # Strategy drawdown: invest $1 only while in this regime
        nav = 1.0
        peak = 1.0
        max_dd = 0.0
        for r in rets:
            nav *= 1 + r
            if nav > peak:
                peak = nav
            dd = (nav - peak) / peak
            if dd < max_dd:
                max_dd = dd

        # Average regime duration (consecutive runs of this label in full df)
        df_copy = df[["state_label"]].copy()
        df_copy["run_id"] = (df_copy["state_label"] != df_copy["state_label"].shift()).cumsum()
        durations = (
            df_copy[df_copy["state_label"] == label]
            .groupby("run_id")
            .size()
        )
        avg_dur = durations.mean() if len(durations) > 0 else np.nan

        rows.append(
            {
                "regime": label,
                "annual_return": ann_ret,
                "annual_vol": ann_vol,
                "sharpe": sharpe,
                "max_drawdown": max_dd,
                "pct_days": n_regime / n_days * 100,
                "avg_duration_days": avg_dur,
                "n_days": n_regime,
            }
        )

    return pd.DataFrame(rows).set_index("regime")


def compute_regime_stats(
    regime_df: pd.DataFrame,
    split_date: str = SPLIT_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (in_sample_stats, oos_stats) DataFrames, both indexed by regime label."""
    in_sample = regime_df[regime_df["date"] < split_date].copy()
    oos = regime_df[regime_df["date"] >= split_date].copy()
    return _regime_stats(in_sample), _regime_stats(oos)


def compute_forward_returns(
    regime_df: pd.DataFrame,
    split_date: str = SPLIT_DATE,
) -> pd.DataFrame:
    """Compute 1-month / 3-month / 6-month forward SPY returns by in-sample regime.

    Uses only in-sample data to avoid lookahead, then drops rows where the
    forward window runs past the split boundary.
    """
    in_sample = regime_df[regime_df["date"] < split_date].copy().reset_index(drop=True)

    in_sample["fwd_1m"] = (
        in_sample["spy_close"].shift(-21) / in_sample["spy_close"] - 1
    )
    in_sample["fwd_3m"] = (
        in_sample["spy_close"].shift(-63) / in_sample["spy_close"] - 1
    )
    in_sample["fwd_6m"] = (
        in_sample["spy_close"].shift(-126) / in_sample["spy_close"] - 1
    )

    cols = ["date", "state_label", "fwd_1m", "fwd_3m", "fwd_6m"]
    return in_sample[cols].dropna()


def compute_transition_matrix(regime_df: pd.DataFrame) -> pd.DataFrame:
    """Build an empirical day-over-day regime transition probability matrix."""
    matrix = pd.DataFrame(0.0, index=STATE_LABELS, columns=STATE_LABELS)
    labels_col = regime_df["state_label"].values
    for i in range(len(labels_col) - 1):
        src = labels_col[i]
        dst = labels_col[i + 1]
        if src in STATE_LABELS and dst in STATE_LABELS:
            matrix.loc[src, dst] += 1

    row_sums = matrix.sum(axis=1)
    matrix = matrix.div(row_sums.where(row_sums > 0, other=np.nan), axis=0)
    return matrix
