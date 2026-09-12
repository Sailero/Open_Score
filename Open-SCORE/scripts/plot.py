"""Refresh the one Markdown report and figures from existing records."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_score.eval.report import refresh_report
from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run", default=FORMAL_RUN)
    args = parser.parse_args()
    print(refresh_report(args.output, run=args.run))


if __name__ == "__main__":
    main()
