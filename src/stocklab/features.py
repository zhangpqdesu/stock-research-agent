from __future__ import annotations

import numpy as np
import pandas as pd

from .config import StrategyConfig


FEATURE_COLUMNS = [
    "ret_5",
    "ret_10",
    "ret_20",
    "ret_60",
    "vol_10",
    "vol_20",
    "max_drawdown_20",
    "price_ma_10",
    "price_ma_20",
    "price_ma_60",
    "turnover_rate",
    "turnover_rate_f",
    "volume_ratio",
    "avg_amount_20",
    "pb",
    "pe",
    "ps_ttm",
    "dv_ttm",
    "total_mv",
    "circ_mv",
]

FUNDAMENTAL_FEATURE_COLUMNS = [
    "roe",
    "roe_dt",
    "roa",
    "grossprofit_margin",
    "netprofit_margin",
    "assets_turn",
    "assets_to_eqt",
    "ocfps",
    "eps",
    "debt_to_assets",
    "cash_to_assets",
    "inventory_to_assets",
    "receivables_to_assets",
]


def _prepare_main_board(stock_basic: pd.DataFrame) -> pd.DataFrame:
    frame = stock_basic.copy()
    frame["list_date"] = pd.to_datetime(frame["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    frame = frame[frame["market"].eq("主板")]
    frame = frame[~frame["name"].str.contains("ST", na=False)]
    return frame


def _prepare_financials(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if any(key not in data or data[key].empty for key in ("fina_indicator", "balancesheet", "cashflow")):
        return pd.DataFrame()
    indicator = data["fina_indicator"].copy()
    balance = data["balancesheet"].copy()
    cashflow = data["cashflow"].copy()

    for frame in (indicator, balance, cashflow):
        frame["ann_date"] = pd.to_datetime(frame["ann_date"].astype(str), format="%Y%m%d", errors="coerce")
        frame["end_date"] = pd.to_datetime(frame["end_date"].astype(str), format="%Y%m%d", errors="coerce")

    merged = indicator.merge(
        balance,
        on=["ts_code", "ann_date", "end_date"],
        how="outer",
    ).merge(
        cashflow,
        on=["ts_code", "ann_date", "end_date"],
        how="outer",
    )

    merged["debt_to_assets"] = merged["total_liab"] / merged["total_assets"]
    merged["cash_to_assets"] = merged["money_cap"] / merged["total_assets"]
    merged["inventory_to_assets"] = merged["inventories"] / merged["total_assets"]
    merged["receivables_to_assets"] = merged["accounts_receiv"] / merged["total_assets"]
    merged = merged.sort_values(["ts_code", "ann_date"])
    return merged


def build_feature_dataset(
    data: dict[str, pd.DataFrame],
    config: StrategyConfig,
    require_labels: bool = True,
) -> pd.DataFrame:
    stock_basic = _prepare_main_board(data["stock_basic"])
    daily = data["daily"].copy()
    daily_basic = data["daily_basic"].copy()
    benchmark = data["benchmark"].copy()
    financials = _prepare_financials(data)

    for frame in (daily, daily_basic, benchmark):
        frame["trade_date"] = pd.to_datetime(frame["trade_date"].astype(str), format="%Y%m%d", errors="coerce")

    daily = daily.merge(
        daily_basic.drop(columns=["close"], errors="ignore"),
        on=["ts_code", "trade_date"],
        how="left",
    )
    daily = daily.merge(stock_basic[["ts_code", "name", "industry", "list_date"]], on="ts_code", how="inner")
    daily = daily.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    daily["amount"] = daily["amount"].astype(float)
    daily["pct_chg"] = daily["pct_chg"].astype(float) / 100.0
    daily["close"] = daily["close"].astype(float)
    daily["list_days"] = (daily["trade_date"] - daily["list_date"]).dt.days

    daily["ret_5"] = daily.groupby("ts_code")["close"].pct_change(5)
    daily["ret_10"] = daily.groupby("ts_code")["close"].pct_change(10)
    daily["ret_20"] = daily.groupby("ts_code")["close"].pct_change(20)
    daily["ret_60"] = daily.groupby("ts_code")["close"].pct_change(60)
    daily["vol_10"] = daily.groupby("ts_code")["pct_chg"].transform(lambda s: s.rolling(10).std())
    daily["vol_20"] = daily.groupby("ts_code")["pct_chg"].transform(lambda s: s.rolling(20).std())
    daily["avg_amount_20"] = daily.groupby("ts_code")["amount"].transform(lambda s: s.rolling(20).mean())
    daily["ma_10"] = daily.groupby("ts_code")["close"].transform(lambda s: s.rolling(10).mean())
    daily["ma_20"] = daily.groupby("ts_code")["close"].transform(lambda s: s.rolling(20).mean())
    daily["ma_60"] = daily.groupby("ts_code")["close"].transform(lambda s: s.rolling(60).mean())
    daily["price_ma_10"] = daily["close"] / daily["ma_10"] - 1.0
    daily["price_ma_20"] = daily["close"] / daily["ma_20"] - 1.0
    daily["price_ma_60"] = daily["close"] / daily["ma_60"] - 1.0
    daily["rolling_max_20"] = daily.groupby("ts_code")["close"].transform(lambda s: s.rolling(20).max())
    daily["max_drawdown_20"] = daily["close"] / daily["rolling_max_20"] - 1.0

    benchmark = benchmark.sort_values("trade_date").copy()
    benchmark["benchmark_ret"] = benchmark["close"].pct_change()
    benchmark["benchmark_future"] = benchmark["close"].shift(-config.label_horizon) / benchmark["close"] - 1.0
    benchmark["benchmark_ma"] = benchmark["close"].rolling(config.market_ma_window).mean()
    benchmark["market_trend_on"] = benchmark["close"] > benchmark["benchmark_ma"]
    daily = daily.merge(
        benchmark[["trade_date", "benchmark_ret", "benchmark_future", "close", "market_trend_on"]].rename(
            columns={"close": "benchmark_close"}
        ),
        on="trade_date",
        how="left",
    )
    daily["future_return"] = daily.groupby("ts_code")["close"].shift(-config.label_horizon) / daily["close"] - 1.0
    daily["target"] = daily["future_return"] - daily["benchmark_future"]

    feature_columns = FEATURE_COLUMNS.copy()
    if config.include_fundamentals and not financials.empty:
        financials = financials.sort_values(["ts_code", "ann_date"])
        daily = daily.sort_values(["ts_code", "trade_date"])
        daily = pd.merge_asof(
            daily,
            financials,
            by="ts_code",
            left_on="trade_date",
            right_on="ann_date",
            direction="backward",
        )
        feature_columns.extend(FUNDAMENTAL_FEATURE_COLUMNS)

    mask = (
        (daily["list_days"] >= config.min_list_days)
        & (daily["close"] >= config.min_close_price)
        & (daily["avg_amount_20"] >= config.min_avg_amount_20d)
        & daily["market_trend_on"].fillna(False)
    )
    dataset = daily.loc[mask].copy()
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    required_columns = feature_columns + (["target", "future_return"] if require_labels else [])
    dataset = dataset.dropna(subset=required_columns)
    dataset["trade_date"] = pd.to_datetime(dataset["trade_date"])
    dataset["rebalance_flag"] = dataset["trade_date"].dt.weekday.isin(config.rebalance_weekdays)
    dataset["feature_columns"] = ",".join(feature_columns)
    return dataset.reset_index(drop=True)
