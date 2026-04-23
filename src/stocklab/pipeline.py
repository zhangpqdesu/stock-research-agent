from __future__ import annotations

from .backtest import run_backtest, save_backtest_outputs
from .config import ensure_directories, load_config
from .data import download_all_data
from .features import build_feature_dataset
from .model import train_and_score


def run_pipeline(refresh: bool = False) -> None:
    config = load_config()
    dirs = ensure_directories()
    raw = download_all_data(config, dirs["raw"], refresh=refresh)
    dataset = build_feature_dataset(raw, config)
    dataset.to_csv(dirs["processed"] / "feature_dataset.csv", index=False)
    scored = train_and_score(dataset, config)
    scored.to_csv(dirs["processed"] / "scored_dataset.csv", index=False)
    equity, holdings, metrics = run_backtest(scored, raw["daily"], config)
    save_backtest_outputs(dirs["outputs"], equity, holdings, metrics)
