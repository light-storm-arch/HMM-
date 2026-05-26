from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_COLS = [
    "log_return",
    "realized_vol_20d",
    "vix_level",
    "vix_term",
    "hyg_lqd_ratio_chg",
    "tlt_return",
    "hy_oas",
    "t10y2y",
    "nfci",
]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add all HMM input features to a raw price DataFrame.

    Input DataFrame must have columns: date, spy_close, vix_close, vix3m_close,
    hyg_close, lqd_close, tlt_close, hy_oas, t10y2y, nfci.

    Returns a new DataFrame with original columns plus all FEATURE_COLS.
    Rows at the front will have NaN from rolling windows — callers should
    dropna(subset=FEATURE_COLS) before passing to the HMM.
    """
    out = df.copy()
    out = out.sort_values("date").reset_index(drop=True)

    out["log_return"] = np.log(out["spy_close"] / out["spy_close"].shift(1))

    out["realized_vol_20d"] = (
        out["log_return"].rolling(20).std() * np.sqrt(252)
    )

    out["vix_level"] = out["vix_close"]

    # Positive = backwardation (near-term VIX elevated vs 3-month) = stress signal
    out["vix_term"] = out["vix_close"] - out["vix3m_close"]

    log_ratio = np.log(out["hyg_close"] / out["lqd_close"])
    out["hyg_lqd_ratio_chg"] = log_ratio.diff(5)

    out["tlt_return"] = np.log(out["tlt_close"] / out["tlt_close"].shift(1))

    # FRED series are already in the raw DataFrame; just carry them through
    for col in ("hy_oas", "t10y2y", "nfci"):
        if col not in out.columns:
            out[col] = np.nan

    return out
