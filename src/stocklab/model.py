from __future__ import annotations

import pandas as pd
from lightgbm import LGBMRegressor

from .config import StrategyConfig


def train_and_score(dataset: pd.DataFrame, config: StrategyConfig) -> pd.DataFrame:
    frame = dataset.sort_values(["trade_date", "ts_code"]).copy()
    frame["score"] = pd.NA
    feature_columns = frame["feature_columns"].iloc[0].split(",")
    trade_dates = sorted(frame.loc[frame["rebalance_flag"], "trade_date"].unique())

    for trade_date in trade_dates:
        train_start = trade_date - pd.Timedelta(days=config.rolling_train_days)
        train_mask = (frame["trade_date"] < trade_date) & (frame["trade_date"] >= train_start)
        train = frame.loc[train_mask].dropna(subset=feature_columns + ["target"])
        test = frame.loc[frame["trade_date"].eq(trade_date)].dropna(subset=feature_columns)

        if len(train) < config.train_min_rows or test.empty:
            continue

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
        frame.loc[test.index, "score"] = model.predict(test[feature_columns])

    return frame.dropna(subset=["score"]).reset_index(drop=True)
