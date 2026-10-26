"""Render selected HAD figures from saved statistics; never run experiments."""
from __future__ import annotations
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[1]
spec = spec_from_file_location("open_score_saved_figures", PROJECT / "open_score/eval/report.py")
report = module_from_spec(spec)
spec.loader.exec_module(report)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=PROJECT/"outputs/main")
    parser.add_argument("--data-root",type=Path,help="Optional local outputs/main holding saved trajectories; read only")
    args=parser.parse_args()
    for path in report.render_figures(args.output,data_root=args.data_root):
        print(path)


if __name__=="__main__":
    main()
