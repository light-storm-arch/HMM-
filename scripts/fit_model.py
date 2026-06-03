"""Fit the HMM on the 2007–2018 training window and pickle the result.

Usage (from repo root):
    python scripts/fit_model.py

Optionally refit on full history (no true out-of-sample period):
    python scripts/fit_model.py --full-history

The output is models/hmm_fitted.pkl, loaded by app.py on startup.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.constants import TRAIN_END, TRAIN_START
from src.data_loader import load_data
from src.features import FEATURE_COLS, build_features
from src.hmm_model import MODEL_PATH, fit_hmm, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and pickle the HMM")
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Fit on 2007 through today instead of 2007–2018 training window",
    )
    args = parser.parse_args()

    train_end = str(date.today()) if args.full_history else TRAIN_END
    print(f"\nTraining window: {TRAIN_START} → {train_end}")
    if args.full_history:
        print("WARNING: Full-history fit — no true out-of-sample period.")

    print("Loading data ...")
    try:
        raw_df = load_data()
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)

    print("Building features ...")
    features_df = build_features(raw_df)

    usable = features_df.dropna(subset=FEATURE_COLS)
    print(f"Usable rows after feature engineering: {len(usable):,}")

    train_mask = (usable["date"] >= TRAIN_START) & (usable["date"] <= train_end)
    n_train = train_mask.sum()
    print(f"Training rows: {n_train:,}")

    if n_train < 50:
        print("ERROR: Not enough training data. Check historical_cache.csv.", file=sys.stderr)
        sys.exit(1)

    print("Fitting HMM (n_components=3, full covariance, 200 iterations) ...")
    model, scaler, state_map = fit_hmm(features_df, TRAIN_START, train_end)

    print(f"State label map: {state_map}")

    save_model(model, scaler, state_map, MODEL_PATH)
    print(f"\nModel saved → {MODEL_PATH}")
    print("\nNext step: streamlit run app.py")


if __name__ == "__main__":
    main()
