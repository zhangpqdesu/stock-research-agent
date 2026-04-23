from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocklab.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Redownload raw data from Tushare.")
    args = parser.parse_args()
    run_pipeline(refresh=args.refresh)


if __name__ == "__main__":
    main()
