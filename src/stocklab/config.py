from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config" / "strategy.json"


@dataclass(frozen=True)
class StrategyConfig:
    universe_mode: str
    universe_limit: int | None
    include_fundamentals: bool
    start_date: str
    end_date: str | None
    min_list_days: int
    min_close_price: float
    min_avg_amount_20d: float
    label_horizon: int
    rebalance_weekdays: list[int]
    max_holdings: int
    stop_loss: float
    rank_sell_threshold: int
    buy_commission_rate: float
    sell_commission_rate: float
    sell_stamp_duty_rate: float
    slippage_rate: float
    benchmark_code: str
    universe_index_code: str
    market_ma_window: int
    train_min_rows: int
    rolling_train_days: int


def load_config() -> StrategyConfig:
    load_dotenv(ROOT / ".env")
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return StrategyConfig(**data)


def get_tushare_token() -> str:
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Missing TUSHARE_TOKEN. Put it in .env.")
    return token


def configure_tushare_network() -> None:
    # Tushare often fails behind proxy/VPN. Prefer direct connection for its API host.
    direct_hosts = "api.waditu.com,waditu.com"
    for key in ("NO_PROXY", "no_proxy"):
        current = os.getenv(key, "").strip()
        hosts = [item for item in current.split(",") if item]
        for host in direct_hosts.split(","):
            if host not in hosts:
                hosts.append(host)
        os.environ[key] = ",".join(hosts)


def ensure_directories() -> dict[str, Path]:
    directories = {
        "raw": ROOT / "data" / "raw",
        "processed": ROOT / "data" / "processed",
        "outputs": ROOT / "outputs",
    }
    for path in directories.values():
        path.mkdir(parents=True, exist_ok=True)
    return directories
