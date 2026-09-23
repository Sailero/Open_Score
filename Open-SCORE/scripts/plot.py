"""Refresh a report and figures from existing records."""
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
    parser.add_argument("--paper", action="store_true",
                        help="Render the concise main0921 paper edition and publication figures")
    args = parser.parse_args()
    if args.paper:
        from open_score.eval.paper_report import render_report
        print(render_report(args.output, run=args.run))
        return
    from open_score.eval.experiment import is_profile
    if is_profile(args.output):
        from open_score.utils.resources import configure_workspace, cpu_threads
        configure_workspace()
        cpu_threads()
    print(refresh_report(args.output, run=args.run))


if __name__ == "__main__":
    main()
