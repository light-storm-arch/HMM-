from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

from src.constants import STATE_LABELS, TRAIN_END, TRAIN_START
from src.features import FEATURE_COLS

MODEL_PATH = Path(__file__).parent.parent / "models" / "hmm_fitted.pkl"


def relabel_states(
    model: GaussianHMM,
    features_df: pd.DataFrame,
    scaler: StandardScaler,
) -> dict[int, str]:
    """Map raw hmmlearn state integers to vol-ranked labels.

    Runs Viterbi on features_df, computes per-state mean realized_vol_20d,
    then assigns Calm → lowest vol, Choppy → middle, Stress → highest.
    Returns {raw_int: label_str}.
    """
    df = features_df.dropna(subset=FEATURE_COLS).copy()
    X = scaler.transform(df[FEATURE_COLS].values)
    states = model.predict(X)

    df = df.reset_index(drop=True)
    df["_state"] = states

    state_mean_vol = (
        df.groupby("_state")["realized_vol_20d"].mean().sort_values()
    )
    return {
        int(raw): label
        for raw, label in zip(state_mean_vol.index, STATE_LABELS)
    }


def fit_hmm(
    features_df: pd.DataFrame,
    train_start: str = TRAIN_START,
    train_end: str = TRAIN_END,
) -> tuple[GaussianHMM, StandardScaler, dict[int, str]]:
    """Fit a 3-state GaussianHMM on the specified training window.

    Returns (model, scaler, state_map) where state_map is {raw_int: label}.
    """
    mask = (features_df["date"] >= train_start) & (features_df["date"] <= train_end)
    train_df = features_df[mask].dropna(subset=FEATURE_COLS).copy()

    if len(train_df) < 50:
        raise ValueError(
            f"Only {len(train_df)} usable training rows — check feature "
            "columns for NaNs and confirm the date range."
        )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[FEATURE_COLS].values)

    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=200,
        random_state=42,
    )
    model.fit(X_train)

    state_map = relabel_states(model, train_df, scaler)
    return model, scaler, state_map


def predict_regimes(
    model: GaussianHMM,
    scaler: StandardScaler,
    state_map: dict[int, str],
    features_df: pd.DataFrame,
) -> pd.DataFrame:
    """Decode regimes for the full features_df using fixed model parameters.

    Returns a DataFrame with the original columns plus:
        state_label, viterbi_state, prob_calm, prob_choppy, prob_stress.
    Rows with any NaN in FEATURE_COLS are dropped.
    """
    df = features_df.dropna(subset=FEATURE_COLS).copy().reset_index(drop=True)
    X = scaler.transform(df[FEATURE_COLS].values)

    viterbi_states = model.predict(X)
    posteriors = model.predict_proba(X)  # shape (n, 3), columns = raw state indices

    # Build reverse map: label → raw state index
    reverse_map: dict[str, int] = {v: k for k, v in state_map.items()}

    df["viterbi_state"] = viterbi_states
    df["state_label"] = df["viterbi_state"].map(state_map)

    for label in STATE_LABELS:
        raw_idx = reverse_map.get(label)
        col = f"prob_{label.lower()}"
        df[col] = posteriors[:, raw_idx] if raw_idx is not None else np.nan

    return df


def save_model(
    model: GaussianHMM,
    scaler: StandardScaler,
    state_map: dict[int, str],
    path: Path = MODEL_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "state_map": state_map}, f)


def load_model(
    path: Path = MODEL_PATH,
) -> tuple[GaussianHMM, StandardScaler, dict[int, str]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Fitted model not found at {path}.\n"
            "Run  python scripts/fit_model.py  to generate it."
        )
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["model"], payload["scaler"], payload["state_map"]
