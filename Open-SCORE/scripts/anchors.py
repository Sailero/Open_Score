"""Run exactly the approved 2400 rule-anchor episodes, resuming existing CSVs."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_score.eval.anchors import run_anchors
from open_score.eval.report import refresh_report
from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run", default=FORMAL_RUN)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    last_report = time.monotonic()

    def update_report(state):
        nonlocal last_report
        if time.monotonic() - last_report >= 300:
            refresh_report(args.output, run=args.run)
            last_report = time.monotonic()

    try:
        run_anchors(args.output, run=args.run, seed=args.seed, on_progress=update_report)
    finally:
        refresh_report(args.output, run=args.run)


if __name__ == "__main__":
    main()
