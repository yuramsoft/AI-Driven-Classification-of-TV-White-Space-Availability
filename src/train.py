from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline

from features import FEATURES, TARGET, validate_training_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Feature table CSV")
    parser.add_argument("--model-out", default="models/tvws_power.joblib")
    parser.add_argument("--metrics-out", default="models/metrics.json")
    args = parser.parse_args()

    raw = pd.read_csv(args.csv)
    if "group_id" not in raw.columns:
        raise ValueError("CSV must include group_id for spatially grouped evaluation")
    groups = raw.loc[validate_training_frame(raw).index, "group_id"]
    data = validate_training_frame(raw)
    groups = groups.loc[data.index]

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=17)
    train_idx, test_idx = next(splitter.split(data[FEATURES], data[TARGET], groups))
    train, test = data.iloc[train_idx], data.iloc[test_idx]

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("forest", RandomForestRegressor(
            n_estimators=500,
            min_samples_leaf=3,
            max_features=0.9,
            random_state=17,
            n_jobs=-1,
        )),
    ])
    model.fit(train[FEATURES], train[TARGET])
    pred = model.predict(test[FEATURES])
    metrics = {
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mae_db": float(mean_absolute_error(test[TARGET], pred)),
        "rmse_db": float(mean_squared_error(test[TARGET], pred) ** 0.5),
        "r2": float(r2_score(test[TARGET], pred)),
        "split": "GroupShuffleSplit by group_id; replace with spatial blocks/campaign holdout",
    }

    model_path = Path(args.model_out)
    metrics_path = Path(args.metrics_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "features": FEATURES}, model_path)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
