"""Run the v4 common-executor study without installing the package."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from open_score.research_v4.cli import main


if __name__ == '__main__':
    raise SystemExit(main())
