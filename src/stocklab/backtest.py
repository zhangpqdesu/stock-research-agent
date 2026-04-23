from __future__ import annotations

import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .config import StrategyConfig


@dataclass
class BacktestMetrics:
    annual_return: float
    annual_volatility: float
    sharpe: float
    max_drawdown: float
    total_return: float
    win_rate: float
    average_turnover: float


def _calc_metrics(equity: pd.DataFrame, turnover: list[float]) -> BacktestMetrics:
    daily_ret = equity["strategy_return"].fillna(0.0)
    nav = (1.0 + daily_ret).cumprod()
    running_max = nav.cummax()
    drawdown = nav / running_max - 1.0
    annual_return = nav.iloc[-1] ** (252 / max(len(nav), 1)) - 1.0
    annual_volatility = daily_ret.std(ddof=0) * np.sqrt(252)
    sharpe = annual_return / annual_volatility if annual_volatility > 0 else 0.0
    return BacktestMetrics(
        annual_return=float(annual_return),
        annual_volatility=float(annual_volatility),
        sharpe=float(sharpe),
        max_drawdown=float(drawdown.min()),
        total_return=float(nav.iloc[-1] - 1.0),
        win_rate=float((daily_ret > 0).mean()),
        average_turnover=float(np.mean(turnover) if turnover else 0.0),
    )


def run_backtest(scored: pd.DataFrame, raw_daily: pd.DataFrame, config: StrategyConfig) -> tuple[pd.DataFrame, pd.DataFrame, BacktestMetrics]:
    price = raw_daily[["ts_code", "trade_date", "open", "close", "pct_chg"]].copy()
    price["trade_date"] = pd.to_datetime(price["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    price["open"] = price["open"].astype(float)
    price["close"] = price["close"].astype(float)
    price = price.sort_values(["ts_code", "trade_date"]).copy()
    price["next_open"] = price.groupby("ts_code")["open"].shift(-1)
    price["hold_ret"] = price["next_open"] / price["open"] - 1.0

    ranked = scored.sort_values(["trade_date", "score"], ascending=[True, False]).copy()
    ranked["rank"] = ranked.groupby("trade_date")["score"].rank(method="first", ascending=False)

    holdings: dict[str, dict[str, float]] = {}
    signal_dates = sorted(ranked["trade_date"].unique())
    execution_schedule = {}
    all_dates = sorted(price["trade_date"].unique())
    for signal_date in signal_dates:
        later_dates = [d for d in all_dates if d > signal_date]
        if later_dates:
            execution_schedule[later_dates[0]] = signal_date

    records: list[dict] = []
    turnover: list[float] = []

    ranked_by_date = {d: g.copy() for d, g in ranked.groupby("trade_date")}
    price_by_date = {d: g.set_index("ts_code") for d, g in price.groupby("trade_date")}
    executed_positions: list[dict] = []

    for trade_date in all_dates:
        day_prices = price_by_date.get(trade_date)
        if day_prices is None:
            continue
        old_holdings = {code: payload.copy() for code, payload in holdings.items()}
        target_holdings = {code: payload.copy() for code, payload in holdings.items()}

        sell_triggered = set()
        for code, payload in list(target_holdings.items()):
            if code not in day_prices.index:
                continue
            open_price = float(day_prices.loc[code, "open"])
            drawdown = open_price / payload["entry_price"] - 1.0
            if drawdown <= -config.stop_loss:
                sell_triggered.add(code)
                del target_holdings[code]

        if trade_date in execution_schedule:
            signal_date = execution_schedule[trade_date]
            signal_ranking = ranked_by_date[signal_date].sort_values("rank").copy()
            candidates = signal_ranking[signal_ranking["rank"] <= config.rank_sell_threshold]
            retained = {}
            for code, payload in target_holdings.items():
                if code in candidates["ts_code"].tolist():
                    retained[code] = payload

            selected_rows: list[pd.Series] = []
            if retained:
                retained_rows = signal_ranking[signal_ranking["ts_code"].isin(retained.keys())].sort_values("rank")
                selected_rows.extend([row for _, row in retained_rows.iterrows()])

            additions = signal_ranking[~signal_ranking["ts_code"].isin({row["ts_code"] for row in selected_rows})]
            additions = _select_top_distinct_industries(additions, config.max_holdings - len(selected_rows))
            selected_rows.extend([row for _, row in additions.iterrows()])
            selected_rows = selected_rows[: config.max_holdings]

            new_holdings: dict[str, dict[str, float]] = {}
            selected_codes = [row["ts_code"] for row in selected_rows]
            for row in selected_rows:
                code = row["ts_code"]
                if code not in day_prices.index:
                    continue
                new_holdings[code] = {
                    "weight": 1.0 / len(selected_codes) if selected_codes else 0.0,
                    "entry_price": retained.get(code, {}).get("entry_price", day_prices.loc[code, "open"]),
                }
                executed_positions.append(
                    {
                        "trade_date": trade_date,
                        "signal_date": signal_date,
                        "ts_code": code,
                        "name": row["name"],
                        "industry": row["industry"],
                        "score": row["score"],
                        "rank": row["rank"],
                        "future_return": row.get("future_return"),
                        "target": row.get("target"),
                    }
                )
            target_holdings = new_holdings

        old_weights = {code: float(payload["weight"]) for code, payload in old_holdings.items()}
        new_weights = {code: float(payload["weight"]) for code, payload in target_holdings.items()}
        buy_turnover = sum(max(new_weights.get(code, 0.0) - old_weights.get(code, 0.0), 0.0) for code in set(old_weights) | set(new_weights))
        sell_turnover = sum(max(old_weights.get(code, 0.0) - new_weights.get(code, 0.0), 0.0) for code in set(old_weights) | set(new_weights))
        turnover_today = buy_turnover + sell_turnover

        transaction_cost = (
            buy_turnover * (config.buy_commission_rate + config.slippage_rate)
            + sell_turnover * (config.sell_commission_rate + config.sell_stamp_duty_rate + config.slippage_rate)
        )

        daily_return = 0.0
        for code, payload in target_holdings.items():
            if code not in day_prices.index:
                continue
            hold_ret = day_prices.loc[code, "hold_ret"]
            if pd.isna(hold_ret):
                continue
            daily_return += float(payload["weight"]) * float(hold_ret)
        daily_return -= transaction_cost
        holdings = target_holdings

        records.append(
            {
                "trade_date": trade_date,
                "strategy_return": daily_return,
                "holdings": ",".join(sorted(holdings)),
                "turnover": turnover_today,
            }
        )
        turnover.append(turnover_today)

    equity = pd.DataFrame(records).sort_values("trade_date")
    equity["nav"] = (1.0 + equity["strategy_return"]).cumprod()
    holdings_df = pd.DataFrame(executed_positions).sort_values(["trade_date", "rank"]) if executed_positions else pd.DataFrame(
        columns=["trade_date", "signal_date", "ts_code", "name", "industry", "score", "rank", "future_return", "target"]
    )
    metrics = _calc_metrics(equity, turnover)
    return equity, holdings_df, metrics


def _select_top_distinct_industries(candidates: pd.DataFrame, max_holdings: int) -> pd.DataFrame:
    if max_holdings <= 0 or candidates.empty:
        return pd.DataFrame(columns=candidates.columns)
    selected_rows = []
    used_industries: set[str] = set()
    for _, row in candidates.sort_values("rank").iterrows():
        industry = row.get("industry")
        if industry in used_industries and len(selected_rows) < max_holdings - 1:
            continue
        selected_rows.append(row)
        if pd.notna(industry):
            used_industries.add(industry)
        if len(selected_rows) >= max_holdings:
            break
    if len(selected_rows) < max_holdings:
        chosen_codes = {row["ts_code"] for row in selected_rows}
        for _, row in candidates.sort_values("rank").iterrows():
            if row["ts_code"] in chosen_codes:
                continue
            selected_rows.append(row)
            if len(selected_rows) >= max_holdings:
                break
    return pd.DataFrame(selected_rows)


def save_backtest_outputs(output_dir, equity: pd.DataFrame, holdings: pd.DataFrame, metrics: BacktestMetrics) -> None:
    equity.to_csv(output_dir / "equity_curve.csv", index=False)
    holdings.to_csv(output_dir / "holdings.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, ensure_ascii=False, indent=2)
