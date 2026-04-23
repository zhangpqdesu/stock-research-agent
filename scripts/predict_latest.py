from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from stocklab.predict import run_latest_prediction


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="Reload raw data before predicting latest.")
    args = parser.parse_args()
    result = run_latest_prediction(refresh=args.refresh)
    print(result[["trade_date", "ts_code", "name", "industry", "score", "rank"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
