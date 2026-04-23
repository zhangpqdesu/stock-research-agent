# Tushare 主板双股票日线模型

这个项目提供一个适合个人开发者的 A 股主板日线研究框架，包含：

- Tushare 数据下载与本地缓存
- 主板股票池过滤
- 轻量特征工程
- LightGBM 横截面回归模型
- 最多持有 2 只股票的周频回测
- Streamlit Dashboard 展示回测结果

## 策略设定

- 市场：A 股主板
- 调仓频率：每周
- 持仓数：最多 2 只
- 权重：等权
- 标签：未来 10 日超额收益
- 过滤：ST、次新、停牌、低流动性、低价股
- 风控：市场趋势过滤、个股止损、排名退出

## 快速开始

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/run_pipeline.py
streamlit run dashboard/app.py
```

如果使用当前环境，也可以直接运行：

```bash
python3 scripts/run_pipeline.py
streamlit run dashboard/app.py
```

## 输出目录

- `data/raw/`：原始数据缓存
- `data/processed/`：特征与训练数据
- `outputs/`：回测结果、持仓、指标和图表数据

## 说明

- `.env` 中存放 `TUSHARE_TOKEN`
- 首次运行会联网下载数据，耗时取决于 Tushare 接口速度
- 当前实现优先保证研究闭环，便于后续替换因子、模型和交易规则
- 如果你平时挂 VPN，代码会尝试对 `api.waditu.com` 设置 `NO_PROXY` 直连；如果你的 VPN 是系统级全局隧道，仍需要手动分流或临时断开 VPN
