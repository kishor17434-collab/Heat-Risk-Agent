#!/usr/bin/env python3
"""
scripts/run_agent.py
─────────────────────
CLI entrypoint for the autonomous agent loop.

Usage:
    python scripts/run_agent.py                          # simulate mode
    python scripts/run_agent.py --mode simulate
    python scripts/run_agent.py --mode live
    python scripts/run_agent.py --mode simulate --steps 48   # replay 48 hours
    python scripts/run_agent.py --threshold 65               # override threshold
"""

import argparse
import logging
import sys

if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf-8-sig"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent.loop import AgentLoop

from src.cli_utils import configure_logging

configure_logging()


def valid_threshold(value: str) -> float:
    try:
        f_val = float(value)
        if not (0 <= f_val <= 100):
            raise ValueError()
        return f_val
    except ValueError:
        raise argparse.ArgumentTypeError(f"Threshold must be between 0 and 100, got: {value}")


def main() -> int:
    try:
        parser = argparse.ArgumentParser(description="Run the Heat Risk Agent autonomous loop.")
        parser.add_argument(
            "--mode",
            default="simulate",
            choices=["simulate", "live"],
            help="simulate: replay historical data | live: poll real API (default: simulate)",
        )
        parser.add_argument(
            "--steps",
            type=int,
            default=None,
            help="Stop after N steps (default: run full dataset / run forever)",
        )
        parser.add_argument(
            "--threshold",
            type=valid_threshold,
            default=None,
            help="Risk score threshold (0-100) for triggering alerts. Overrides RISK_THRESHOLD env var.",
        )
        args = parser.parse_args()

        agent = AgentLoop(threshold=args.threshold)
        agent.run(mode=args.mode, max_steps=args.steps)

        print("\n✅  Agent loop complete:")
        print("    → logs/agent_decisions.log  (all decisions)")
        print("    → logs/agent_alerts.log     (triggered alerts only)")
        return 0

    except argparse.ArgumentTypeError as e:
        logging.error(f"Invalid argument: {e}")
        return 1
    except Exception as e:
        logging.error(f"Unexpected error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
