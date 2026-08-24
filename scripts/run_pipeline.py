#!/usr/bin/env python3
"""
scripts/run_pipeline.py
───────────────────────
CLI entrypoint for the data pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --start 2024-06-01 --end 2024-08-31
    python scripts/run_pipeline.py --start 2024-06-01 --end 2024-08-31 --mode open_meteo
"""

import argparse
import logging
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd
from pathlib import Path

# Allow running from the project root without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.pipeline import run_pipeline

from src.cli_utils import configure_logging
from datetime import datetime

configure_logging()


def valid_date(date_string: str) -> str:
    try:
        return datetime.strptime(date_string, "%Y-%m-%d").strftime("%Y-%m-%d")
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date: {date_string}. Use YYYY-MM-DD")


def main() -> int:
    try:
        parser = argparse.ArgumentParser(description="Run the Heat Risk Agent data pipeline.")
        parser.add_argument(
            "--start",
            type=valid_date,
            default="2024-06-01",
            help="Start date (YYYY-MM-DD). Default: 2024-06-01",
        )
        parser.add_argument(
            "--end",
            type=valid_date,
            default="2024-08-31",
            help="End date (YYYY-MM-DD). Default: 2024-08-31",
        )
        parser.add_argument(
            "--mode",
            default=None,
            choices=["simulate", "open_meteo", "fortyguard"],
            help="Override TEMP_DATA_MODE env var.",
        )
        parser.add_argument(
            "--no-save",
            action="store_true",
            help="Skip writing output files (useful for quick tests).",
        )
        args = parser.parse_args()

        df = run_pipeline(
            start=args.start,
            end=args.end,
            temp_mode=args.mode,
            save=not args.no_save,
        )

        print(f"\nFirst 5 rows of combined dataset:\n{df.head().to_string()}\n")
        print("\n✅  Data pipeline complete:")
        if not args.no_save:
            print("    → data/raw/temp_raw.csv")
            print("    → data/raw/demand_raw.csv")
            print("    → data/processed/combined.csv")
        return 0

    except argparse.ArgumentTypeError as e:
        logging.error(f"Invalid argument: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
