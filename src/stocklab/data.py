from __future__ import annotations

from pathlib import Path
from time import sleep

import pandas as pd
import tushare as ts

from .config import StrategyConfig, configure_tushare_network, get_tushare_token


class TushareClient:
    def __init__(self) -> None:
        configure_tushare_network()
        self.pro = ts.pro_api(get_tushare_token())

    def stock_basic(self) -> pd.DataFrame:
        return self.pro.stock_basic(
            exchange="",
            list_status="L",
            fields="ts_code,symbol,name,area,industry,market,list_date"
        )

    def daily(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.pro.daily(ts_code=ts_code, trade_date=trade_date, start_date=start_date, end_date=end_date)

    def daily_basic(
        self,
        ts_code: str | None = None,
        trade_date: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.pro.daily_basic(
            ts_code=ts_code,
            trade_date=trade_date,
            start_date=start_date,
            end_date=end_date,
            fields=(
                "ts_code,trade_date,close,turnover_rate,turnover_rate_f,volume_ratio,"
                "pe,pb,ps_ttm,dv_ttm,total_mv,circ_mv"
            ),
        )

    def fina_indicator(
        self,
        ts_code: str | None = None,
        period: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.pro.fina_indicator(
            ts_code=ts_code,
            period=period,
            start_date=start_date,
            end_date=end_date,
            fields=(
                "ts_code,end_date,ann_date,roe,roe_dt,roa,grossprofit_margin,"
                "netprofit_margin,assets_turn,assets_to_eqt,ocfps,eps"
            ),
        )

    def balancesheet(
        self,
        ts_code: str | None = None,
        period: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.pro.balancesheet(
            ts_code=ts_code,
            period=period,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,ann_date,end_date,total_assets,total_liab,money_cap,accounts_receiv,inventories"
        )

    def cashflow(
        self,
        ts_code: str | None = None,
        period: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame:
        return self.pro.cashflow(
            ts_code=ts_code,
            period=period,
            start_date=start_date,
            end_date=end_date,
            fields="ts_code,ann_date,end_date,n_cashflow_act"
        )

    def index_daily(self, ts_code: str, start_date: str, end_date: str | None) -> pd.DataFrame:
        return self.pro.index_daily(ts_code=ts_code, start_date=start_date, end_date=end_date)

    def index_weight(self, index_code: str, start_date: str, end_date: str | None) -> pd.DataFrame:
        return self.pro.index_weight(index_code=index_code, start_date=start_date, end_date=end_date)


def cached_csv(path: Path, loader, refresh: bool = False) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_csv(path)
    frame = _with_retry(loader)
    frame.to_csv(path, index=False)
    return frame


def _date_ranges(start_date: str, end_date: str | None, days_per_chunk: int = 90) -> list[tuple[str, str]]:
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date) if end_date else pd.Timestamp.today().normalize()
    ranges = []
    current = start
    while current <= end:
        chunk_end = min(current + pd.Timedelta(days=days_per_chunk - 1), end)
        ranges.append((current.strftime("%Y%m%d"), chunk_end.strftime("%Y%m%d")))
        current = chunk_end + pd.Timedelta(days=1)
    return ranges


def _with_retry(loader, retries: int = 3, sleep_seconds: int = 2) -> pd.DataFrame:
    last_error = None
    for attempt in range(retries):
        try:
            return loader()
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            if attempt < retries - 1:
                sleep(sleep_seconds * (attempt + 1))
    raise last_error


def _fetch_by_ranges(path: Path, ranges: list[tuple[str, str]], loader, refresh: bool = False) -> pd.DataFrame:
    if path.exists() and not refresh:
        return pd.read_csv(path)
    parts = []
    for start_date, end_date in ranges:
        print(f"Fetching {path.name}: {start_date}-{end_date}", flush=True)
        part = _with_retry(lambda s=start_date, e=end_date: loader(s, e))
        if part is not None and not part.empty:
            parts.append(part)
    if not parts:
        frame = pd.DataFrame()
    else:
        frame = pd.concat(parts, ignore_index=True).drop_duplicates()
    frame.to_csv(path, index=False)
    return frame


def _fetch_by_ranges_resume(
    path: Path,
    ranges: list[tuple[str, str]],
    loader,
    refresh: bool = False,
) -> pd.DataFrame:
    chunk_dir = path.parent / "chunks" / path.stem
    if refresh and chunk_dir.exists():
        for item in chunk_dir.glob("*.csv"):
            item.unlink()
    chunk_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    for start_date, end_date in ranges:
        chunk_path = chunk_dir / f"{start_date}_{end_date}.csv"
        if chunk_path.exists():
            print(f"Using cached chunk {chunk_path.name}", flush=True)
            part = pd.read_csv(chunk_path)
        else:
            print(f"Fetching {path.name}: {start_date}-{end_date}", flush=True)
            part = _with_retry(lambda s=start_date, e=end_date: loader(s, e))
            if part is None:
                part = pd.DataFrame()
            part.to_csv(chunk_path, index=False)
        if not part.empty:
            parts.append(part)

    frame = pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame()
    frame.to_csv(path, index=False)
    return frame


def _fetch_by_codes(
    path: Path,
    codes: list[str],
    loader,
    refresh: bool = False,
) -> pd.DataFrame:
    chunk_dir = path.parent / "chunks" / path.stem
    if refresh and chunk_dir.exists():
        for item in chunk_dir.glob("*.csv"):
            item.unlink()
    chunk_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    failed_codes = []
    for code in codes:
        chunk_path = chunk_dir / f"{code.replace('.', '_')}.csv"
        if chunk_path.exists():
            print(f"Using cached chunk {chunk_path.name}", flush=True)
            part = pd.read_csv(chunk_path)
        else:
            print(f"Fetching {path.name}: {code}", flush=True)
            try:
                part = _with_retry(lambda c=code: loader(c))
                if part is None:
                    part = pd.DataFrame()
                part.to_csv(chunk_path, index=False)
            except Exception as exc:  # noqa: BLE001
                print(f"Skipping {code} for {path.name}: {exc}", flush=True)
                failed_codes.append(code)
                continue
        if not part.empty:
            parts.append(part)

    if failed_codes:
        failed_path = chunk_dir / "_failed_codes.txt"
        failed_path.write_text("\n".join(failed_codes), encoding="utf-8")
    frame = pd.concat(parts, ignore_index=True).drop_duplicates() if parts else pd.DataFrame()
    frame.to_csv(path, index=False)
    return frame


def _load_local_cache(cache_dir: Path) -> pd.DataFrame:
    files = sorted([path for path in cache_dir.glob("*.csv") if not path.name.startswith("_")])
    if not files:
        return pd.DataFrame()
    parts = [pd.read_csv(path) for path in files]
    return pd.concat(parts, ignore_index=True).drop_duplicates()


def _resolve_universe_codes(client: TushareClient, config: StrategyConfig, raw_dir: Path, refresh: bool) -> list[str]:
    if config.universe_mode != "hs300":
        return []

    weight_ranges = _date_ranges(config.start_date, config.end_date, days_per_chunk=180)
    weights = _fetch_by_ranges(
        raw_dir / "index_weight.csv",
        weight_ranges,
        lambda start_date, end_date: client.index_weight(config.universe_index_code, start_date, end_date),
        refresh=refresh,
    )
    if weights.empty:
        raise RuntimeError("HS300 universe requested, but index_weight returned no rows.")
    weights["trade_date"] = pd.to_datetime(weights["trade_date"])
    latest_trade_date = weights["trade_date"].max()
    latest_weights = weights.loc[weights["trade_date"].eq(latest_trade_date)].copy()
    codes = sorted(latest_weights["con_code"].dropna().unique().tolist())
    if config.universe_limit:
        codes = codes[: config.universe_limit]
    return codes


def download_all_data(config: StrategyConfig, raw_dir: Path, refresh: bool = False) -> dict[str, pd.DataFrame]:
    client = TushareClient()
    finance_ranges = _date_ranges(config.start_date, config.end_date, days_per_chunk=120)
    stock_basic = cached_csv(raw_dir / "stock_basic.csv", client.stock_basic, refresh=refresh)
    local_cache_root = raw_dir.parent.parent / "hs300_cache"

    if config.universe_mode == "hs300":
        universe_codes = _resolve_universe_codes(client, config, raw_dir, refresh)
        local_daily_dir = local_cache_root / "daily"
        local_basic_dir = local_cache_root / "daily_basic"
        if local_daily_dir.exists() and local_basic_dir.exists() and not refresh:
            daily = _load_local_cache(local_daily_dir)
            daily_basic = _load_local_cache(local_basic_dir)
            daily.to_csv(raw_dir / "daily.csv", index=False)
            daily_basic.to_csv(raw_dir / "daily_basic.csv", index=False)
        else:
            daily = _fetch_by_codes(
                raw_dir / "daily.csv",
                universe_codes,
                lambda ts_code: client.daily(ts_code=ts_code, start_date=config.start_date, end_date=config.end_date),
                refresh=refresh,
            )
            daily_basic = _fetch_by_codes(
                raw_dir / "daily_basic.csv",
                universe_codes,
                lambda ts_code: client.daily_basic(ts_code=ts_code, start_date=config.start_date, end_date=config.end_date),
                refresh=refresh,
            )
        if config.include_fundamentals:
            fina_indicator = _fetch_by_codes(
                raw_dir / "fina_indicator.csv",
                universe_codes,
                lambda ts_code: client.fina_indicator(ts_code=ts_code, start_date=config.start_date, end_date=config.end_date),
                refresh=refresh,
            )
            balancesheet = _fetch_by_codes(
                raw_dir / "balancesheet.csv",
                universe_codes,
                lambda ts_code: client.balancesheet(ts_code=ts_code, start_date=config.start_date, end_date=config.end_date),
                refresh=refresh,
            )
            cashflow = _fetch_by_codes(
                raw_dir / "cashflow.csv",
                universe_codes,
                lambda ts_code: client.cashflow(ts_code=ts_code, start_date=config.start_date, end_date=config.end_date),
                refresh=refresh,
            )
        else:
            fina_indicator = pd.DataFrame()
            balancesheet = pd.DataFrame()
            cashflow = pd.DataFrame()
    else:
        daily_ranges = _date_ranges(config.start_date, config.end_date, days_per_chunk=20)
        basic_ranges = _date_ranges(config.start_date, config.end_date, days_per_chunk=20)
        daily = _fetch_by_ranges_resume(
            raw_dir / "daily.csv",
            daily_ranges,
            lambda start_date, end_date: client.daily(start_date=start_date, end_date=end_date),
            refresh=refresh,
        )
        daily_basic = _fetch_by_ranges_resume(
            raw_dir / "daily_basic.csv",
            basic_ranges,
            lambda start_date, end_date: client.daily_basic(start_date=start_date, end_date=end_date),
            refresh=refresh,
        )
        if config.include_fundamentals:
            fina_indicator = _fetch_by_ranges(
                raw_dir / "fina_indicator.csv",
                finance_ranges,
                lambda start_date, end_date: client.fina_indicator(start_date=start_date, end_date=end_date),
                refresh=refresh,
            )
            balancesheet = _fetch_by_ranges(
                raw_dir / "balancesheet.csv",
                finance_ranges,
                lambda start_date, end_date: client.balancesheet(start_date=start_date, end_date=end_date),
                refresh=refresh,
            )
            cashflow = _fetch_by_ranges(
                raw_dir / "cashflow.csv",
                finance_ranges,
                lambda start_date, end_date: client.cashflow(start_date=start_date, end_date=end_date),
                refresh=refresh,
            )
        else:
            fina_indicator = pd.DataFrame()
            balancesheet = pd.DataFrame()
            cashflow = pd.DataFrame()
    benchmark = cached_csv(
        raw_dir / "benchmark.csv",
        lambda: client.index_daily(config.benchmark_code, config.start_date, config.end_date),
        refresh=refresh,
    )
    return {
        "daily": daily,
        "daily_basic": daily_basic,
        "stock_basic": stock_basic,
        "fina_indicator": fina_indicator,
        "balancesheet": balancesheet,
        "cashflow": cashflow,
        "benchmark": benchmark,
    }
