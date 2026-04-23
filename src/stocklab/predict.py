from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from lightgbm import LGBMRegressor

from .config import ensure_directories, load_config
from .data import download_all_data
from .features import build_feature_dataset


def run_latest_prediction(refresh: bool = False) -> pd.DataFrame:
    config = load_config()
    dirs = ensure_directories()
    raw = download_all_data(config, dirs["raw"], refresh=refresh)
    dataset = build_feature_dataset(raw, config, require_labels=False)
    feature_columns = dataset["feature_columns"].iloc[0].split(",")

    latest_trade_date = dataset["trade_date"].max()
    latest_slice = dataset.loc[dataset["trade_date"].eq(latest_trade_date)].copy()

    train_end = latest_trade_date - pd.Timedelta(days=config.label_horizon)
    train_start = latest_trade_date - pd.Timedelta(days=config.rolling_train_days)
    train = dataset.loc[
        (dataset["trade_date"] >= train_start)
        & (dataset["trade_date"] <= train_end)
    ].dropna(subset=feature_columns + ["target"])

    if len(train) < config.train_min_rows:
        raise RuntimeError(f"Not enough training rows for latest prediction: {len(train)}")

    latest_slice = latest_slice.dropna(subset=feature_columns)
    latest_slice = latest_slice.loc[latest_slice["rebalance_flag"]]
    if latest_slice.empty:
        latest_rebalance = dataset.loc[dataset["rebalance_flag"], "trade_date"].max()
        latest_slice = dataset.loc[dataset["trade_date"].eq(latest_rebalance)].dropna(subset=feature_columns).copy()

    model = LGBMRegressor(
        objective="regression",
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )
    model.fit(train[feature_columns], train["target"])
    latest_slice["score"] = model.predict(latest_slice[feature_columns])
    latest_slice = latest_slice.sort_values("score", ascending=False).reset_index(drop=True)
    latest_slice["rank"] = latest_slice.index + 1

    prediction_path = dirs["outputs"] / "latest_prediction.csv"
    latest_slice.to_csv(prediction_path, index=False)

    summary = {
        "prediction_trade_date": latest_slice["trade_date"].iloc[0].strftime("%Y-%m-%d"),
        "top_pick": latest_slice.loc[0, "ts_code"] if not latest_slice.empty else None,
        "top_2": latest_slice.head(config.max_holdings)["ts_code"].tolist(),
        "rows_scored": int(len(latest_slice)),
    }
    with (dirs["outputs"] / "latest_prediction_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return latest_slice
