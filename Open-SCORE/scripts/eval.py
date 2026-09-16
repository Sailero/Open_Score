"""Evaluate a retained model using the frozen four-configuration protocol."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from open_score.utils.logging import DEFAULT_OUTPUT, FORMAL_RUN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run", default=FORMAL_RUN)
    parser.add_argument("--seed", type=int, default=None, help="Training seed; inferred from checkpoint when omitted")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--depth-sweep", action="store_true",
                        help="Equal-scale 1:1 sweep over cycle depths R=1..6")
    args = parser.parse_args()
    from open_score.eval import evaluate_checkpoint, evaluate_depth_sweep
    fn = evaluate_depth_sweep if args.depth_sweep else evaluate_checkpoint
    fn(args.method, args.checkpoint, output=args.output, run=args.run,
       seed=args.seed, device=args.device)


if __name__ == "__main__":
    main()
