from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from time import sleep

import pandas as pd
import tushare as ts
from dotenv import load_dotenv
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "hs300_cache"
RAW_DIR = ROOT / "data" / "raw"


def call_with_retry(fn, retries: int = 5, sleep_seconds: int = 2):
    last_error = None
    for i in range(retries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep(sleep_seconds * (i + 1))
    raise last_error


def _ensure_no_proxy() -> None:
    for key in ("NO_PROXY", "no_proxy"):
        current = os.getenv(key, "").strip()
        parts = [item for item in current.split(",") if item]
        for host in ("api.waditu.com", "waditu.com"):
            if host not in parts:
                parts.append(host)
        os.environ[key] = ",".join(parts)


def _latest_cached_trade_date(cache_subdir: Path) -> str | None:
    if not cache_subdir.exists():
        return None
    latest = None
    for p in cache_subdir.glob("*.csv"):
        if p.name.startswith("_"):
            continue
        try:
            df = pd.read_csv(p, usecols=["trade_date"])
        except Exception:  # noqa: BLE001
            continue
        if df.empty:
            continue
        value = df["trade_date"].astype(str).max()
        latest = max(latest, value) if latest else value
    return latest


def _resolve_latest_hs300_codes(pro, trade_date: str) -> list[str]:
    local_index_weight = RAW_DIR / "index_weight.csv"
    if local_index_weight.exists():
        weights = pd.read_csv(local_index_weight)
    else:
        weights = call_with_retry(lambda: pro.index_weight(index_code="000300.SH", trade_date=trade_date))
        if weights is None or weights.empty:
            start_date = (pd.Timestamp(trade_date) - pd.Timedelta(days=10)).strftime("%Y%m%d")
            weights = call_with_retry(
                lambda: pro.index_weight(index_code="000300.SH", start_date=start_date, end_date=trade_date)
            )
    if weights is None or weights.empty:
        raise RuntimeError(f"Unable to resolve HS300 constituents up to {trade_date}.")
    weights["trade_date"] = pd.to_datetime(weights["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    latest_trade_date = weights["trade_date"].max()
    latest_weights = weights.loc[weights["trade_date"].eq(latest_trade_date)].copy()
    return sorted(latest_weights["con_code"].dropna().unique().tolist())


def _append_frame(existing_path: Path, part: pd.DataFrame) -> None:
    if part is None or part.empty:
        return
    if existing_path.exists():
        existing = pd.read_csv(existing_path)
        merged = pd.concat([existing, part], ignore_index=True).drop_duplicates()
    else:
        merged = part
    merged.to_csv(existing_path, index=False)


def update_history_incrementally(pro) -> str:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "daily").mkdir(parents=True, exist_ok=True)
    (CACHE_DIR / "daily_basic").mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    latest_daily = _latest_cached_trade_date(CACHE_DIR / "daily")
    latest_basic = _latest_cached_trade_date(CACHE_DIR / "daily_basic")
    latest_cached = max(filter(None, [latest_daily, latest_basic])) if latest_daily or latest_basic else None

    if latest_cached is None:
        raise RuntimeError("No local HS300 cache found. Run scripts/download_hs300_history.py first.")

    start_date = (pd.Timestamp(latest_cached) + pd.Timedelta(days=1)).strftime("%Y%m%d")
    end_date = datetime.now().strftime("%Y%m%d")
    if start_date > end_date:
        return latest_cached

    codes = _resolve_latest_hs300_codes(pro, end_date)
    for ts_code in tqdm(codes, desc="update_daily", unit="stock"):
        daily_part = call_with_retry(
            lambda c=ts_code: pro.daily(ts_code=c, start_date=start_date, end_date=end_date)
        )
        basic_part = call_with_retry(
            lambda c=ts_code: pro.daily_basic(
                ts_code=c,
                start_date=start_date,
                end_date=end_date,
                fields=(
                    "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
                    "pe,pb,ps_ttm,dv_ttm,total_mv,circ_mv"
                ),
            )
        )
        _append_frame(CACHE_DIR / "daily" / f"{ts_code.replace('.', '_')}.csv", daily_part)
        _append_frame(CACHE_DIR / "daily_basic" / f"{ts_code.replace('.', '_')}.csv", basic_part)

    benchmark_part = call_with_retry(lambda: pro.index_daily(ts_code="000300.SH", start_date=start_date, end_date=end_date))
    _append_frame(RAW_DIR / "benchmark.csv", benchmark_part)

    newest = _latest_cached_trade_date(CACHE_DIR / "daily") or latest_cached
    return newest


def main() -> None:
    load_dotenv(ROOT / ".env")
    token = os.getenv("TUSHARE_TOKEN")
    if not token:
        raise RuntimeError("Missing TUSHARE_TOKEN. Put it in .env or export it.")

    _ensure_no_proxy()
    ts.set_token(token)
    pro = ts.pro_api(token)
    newest = update_history_incrementally(pro)
    print(f"HS300 cache updated through {newest}")


if __name__ == "__main__":
    main()
