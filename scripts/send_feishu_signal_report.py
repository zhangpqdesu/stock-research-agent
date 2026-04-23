from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TARGET = "ou_7e7ea3aabc37536de9cadfa9ff8e2d0d"


def run_cmd(args: list[str]) -> None:
    result = subprocess.run(args, cwd=ROOT, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def latest_market_status() -> dict[str, object]:
    benchmark = pd.read_csv(ROOT / "data" / "raw" / "benchmark.csv")
    benchmark["trade_date"] = pd.to_datetime(benchmark["trade_date"].astype(str), format="%Y%m%d", errors="coerce")
    benchmark = benchmark.sort_values("trade_date").copy()
    benchmark["ma60"] = benchmark["close"].rolling(60).mean()
    benchmark["trend_on"] = benchmark["close"] > benchmark["ma60"]
    row = benchmark.iloc[-1]
    return {
        "trade_date": row["trade_date"].strftime("%Y-%m-%d"),
        "close": float(row["close"]),
        "ma60": float(row["ma60"]) if pd.notna(row["ma60"]) else None,
        "trend_on": bool(row["trend_on"]) if pd.notna(row["trend_on"]) else False,
    }


def latest_data_status() -> dict[str, str]:
    daily = pd.read_csv(ROOT / "data" / "raw" / "daily.csv")
    daily_basic = pd.read_csv(ROOT / "data" / "raw" / "daily_basic.csv")
    benchmark = pd.read_csv(ROOT / "data" / "raw" / "benchmark.csv")
    return {
        "daily": str(daily["trade_date"].astype(str).max()),
        "daily_basic": str(daily_basic["trade_date"].astype(str).max()),
        "benchmark": str(benchmark["trade_date"].astype(str).max()),
    }


def compose_message() -> str:
    metrics = json.loads((ROOT / "outputs" / "metrics.json").read_text(encoding="utf-8"))
    latest_prediction_summary = json.loads(
        (ROOT / "outputs" / "latest_prediction_summary.json").read_text(encoding="utf-8")
    )
    latest_prediction = pd.read_csv(ROOT / "outputs" / "latest_prediction.csv")
    market = latest_market_status()
    data_status = latest_data_status()

    top2 = latest_prediction.head(2)[["ts_code", "name", "industry", "score"]]
    top2_lines = [
        f"{row.ts_code} {row.name} / {row.industry} / score={row.score:.4f}"
        for row in top2.itertuples(index=False)
    ]

    can_open = market["trend_on"]
    if can_open:
        decision = "可以开仓"
        reason = f"沪深300最新收盘 {market['close']:.2f} 高于60日均线 {market['ma60']:.2f}。"
    else:
        decision = "不建议开仓"
        reason = f"沪深300最新收盘 {market['close']:.2f} 低于60日均线 {market['ma60']:.2f}，市场过滤未通过。"

    lines = [
        "HS300 日线模型日报",
        f"开仓结论：{decision}",
        f"原因：{reason}",
        f"最新有效预测日：{latest_prediction_summary['prediction_trade_date']}",
        "前2候选：",
        *[f"- {line}" for line in top2_lines],
        (
            "最新数据日期："
            f"daily={data_status['daily']} / daily_basic={data_status['daily_basic']} / benchmark={data_status['benchmark']}"
        ),
        (
            "回测摘要："
            f"年化={metrics['annual_return']:.2%} / 夏普={metrics['sharpe']:.2f} / 最大回撤={metrics['max_drawdown']:.2%}"
        ),
    ]
    return "\n".join(lines)


def send_message(target: str, message: str, dry_run: bool) -> None:
    if dry_run:
        print(message)
        return
    cmd = [
        "openclaw",
        "message",
        "send",
        "--channel",
        "feishu",
        "--target",
        target,
        "--message",
        message,
    ]
    run_cmd(cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model + prediction and send a Feishu summary via OpenClaw.")
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Feishu target id")
    parser.add_argument("--skip-run", action="store_true", help="Only compose/send from existing outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print message instead of sending")
    args = parser.parse_args()

    if not args.skip_run:
        # Tushare 日线一般在 17:00 后更新；先尝试拉取最近交易日增量数据，再跑全流程。
        run_cmd([sys.executable, "scripts/update_latest_data.py"])
        run_cmd([sys.executable, "scripts/run_pipeline.py"])
        run_cmd([sys.executable, "scripts/predict_latest.py"])

    message = compose_message()
    send_message(args.target, message, args.dry_run)
    if args.dry_run:
        return
    print("Feishu report sent.")


if __name__ == "__main__":
    main()
