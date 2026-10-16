"""Render the eight shared HAD figures from saved statistics; never run experiments."""
from __future__ import annotations
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
spec = spec_from_file_location("open_score_saved_figures", PROJECT / "open_score/eval/report.py")
report = module_from_spec(spec)
spec.loader.exec_module(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PROJECT/"outputs/main")
    parser.add_argument("--training", action="store_true", help="Refresh the single training curve PNG and its report from saved logs")
    parser.add_argument("--method", default="regir_nomem")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run", default="train")
    args=parser.parse_args()
    paths = report.render_training(args.output, method=args.method, seed=args.seed, run=args.run) if args.training else report.render_figures(args.output)
    for path in paths:
        print(path)


if __name__=="__main__":
    main()
