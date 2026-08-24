#!/usr/bin/env python3
"""
scripts/run_train.py
─────────────────────
CLI entrypoint for the model training layer.

Usage:
    python scripts/run_train.py
    python scripts/run_train.py --data data/processed/combined.csv
"""

import argparse
import logging
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.train import train_and_save

from src.cli_utils import configure_logging

configure_logging()


def run_smoke_test() -> None:
    """Optional validation after training."""
    from src.model.predict import load_model, predict_risk
    model = load_model()
    result = predict_risk(model, temp_f=103.0, hour=17, day_of_week=1, month=7)
    print("\n  Smoke test (103°F, 5pm Tuesday July):")
    print(f"    Predicted demand : {result['predicted_demand_mw']:,.0f} MW")
    print(f"    Risk score       : {result['risk_score']:.1f}/100 ({result['risk_level']})")


def main() -> int:
    try:
        parser = argparse.ArgumentParser(description="Train the Heat Risk demand forecasting model.")
        parser.add_argument("--data", default=None, help="Path to combined.csv")
        parser.add_argument("--no-save", action="store_true", help="Don't write model files")
        parser.add_argument("--smoke-test", action="store_true", help="Run a quick validation smoke test after training")
        args = parser.parse_args()

        train_and_save(combined_path=args.data, save=not args.no_save)

        print("\n✅  Training complete:")
        if not args.no_save:
            print("    → models/best_model.pkl")
            print("    → models/model_meta.json")

        if args.smoke_test and not args.no_save:
            run_smoke_test()

        return 0

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
