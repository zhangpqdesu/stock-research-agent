from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


st.set_page_config(page_title="主板双股票模型", layout="wide")
st.title("Tushare 主板双股票日线模型")

output_dir = ROOT / "outputs"
equity_path = output_dir / "equity_curve.csv"
holdings_path = output_dir / "holdings.csv"
metrics_path = output_dir / "metrics.json"

if not equity_path.exists():
    st.warning("尚未生成回测结果，请先运行 python3 scripts/run_pipeline.py")
    st.stop()

equity = pd.read_csv(equity_path, parse_dates=["trade_date"])
holdings = pd.read_csv(holdings_path, parse_dates=["trade_date"])
metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

col1, col2, col3, col4 = st.columns(4)
col1.metric("总收益", f"{metrics['total_return']:.2%}")
col2.metric("年化收益", f"{metrics['annual_return']:.2%}")
col3.metric("夏普", f"{metrics['sharpe']:.2f}")
col4.metric("最大回撤", f"{metrics['max_drawdown']:.2%}")

fig = px.line(equity, x="trade_date", y="nav", title="策略净值曲线")
st.plotly_chart(fig, use_container_width=True)

st.subheader("最近持仓候选")
latest_date = holdings["trade_date"].max()
st.dataframe(
    holdings[holdings["trade_date"].eq(latest_date)]
    .sort_values("rank")
    .reset_index(drop=True),
    use_container_width=True,
)

st.subheader("回测日收益")
st.plotly_chart(
    px.bar(equity.tail(120), x="trade_date", y="strategy_return", title="最近120个交易日收益"),
    use_container_width=True,
)

st.subheader("历史持仓记录")
st.dataframe(holdings.sort_values(["trade_date", "rank"], ascending=[False, True]), use_container_width=True)
