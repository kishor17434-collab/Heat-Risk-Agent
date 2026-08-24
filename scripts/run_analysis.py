#!/usr/bin/env python3
"""
scripts/run_analysis.py
────────────────────────
CLI entrypoint for the correlation analysis layer.

Usage:
    python scripts/run_analysis.py
    python scripts/run_analysis.py --data data/processed/combined.csv
"""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.correlation import run_analysis

from src.cli_utils import configure_logging

configure_logging()


def main() -> int:
    try:
        parser = argparse.ArgumentParser(description="Run correlation analysis on combined dataset.")
        parser.add_argument(
            "--data",
            default=None,
            help="Path to combined.csv (default: data/processed/combined.csv)",
        )
        parser.add_argument(
            "--no-save",
            action="store_true",
            help="Don't write correlation_report.json",
        )
        args = parser.parse_args()

        run_analysis(combined_path=args.data, save=not args.no_save)
        print("\n✅  Analysis complete:")
        if not args.no_save:
            print("    → data/processed/correlation_report.json")
            print("    → data/processed/correlation_plot.png (if matplotlib installed)")
        return 0

    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
