"""Shared constants for the HMM Regime Detector.

Centralizing these prevents drift between modules (previously SPLIT_DATE,
STATE_LABELS, and the regime color palette were duplicated across app.py,
backtest.py, hmm_model.py, and plots.py).
"""
from __future__ import annotations

SPLIT_DATE = "2019-01-01"
TRAIN_START = "2007-01-01"
TRAIN_END = "2018-12-31"

STATE_LABELS = ["Calm", "Choppy", "Stress"]

REGIME_COLORS: dict[str, str] = {
    "Calm": "#22c55e",
    "Choppy": "#f59e0b",
    "Stress": "#ef4444",
}

# (background, accent) pairs for the dashboard regime cards
REGIME_CARD_COLORS: dict[str, tuple[str, str]] = {
    "Calm":   ("#166534", "#22c55e"),
    "Choppy": ("#78350f", "#f59e0b"),
    "Stress": ("#7f1d1d", "#ef4444"),
}
