from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

NUMERIC = [
    "latitude", "longitude", "channel", "frequency_mhz", "rx_power_dbm",
    "tx_distance_km", "tx_height_m", "rx_height_m", "terrain_loss_db",
    "landcover_urban_fraction", "landcover_forest_fraction",
]
REQUIRED = [
    "frequency_mhz", "rx_power_dbm", "tx_distance_km", "tx_height_m",
    "rx_height_m", "terrain_loss_db", "landcover_urban_fraction",
    "landcover_forest_fraction", "group_id",
]


def clean_frame(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = raw.copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_")
                  for c in df.columns]
    if df.columns.duplicated().any():
        raise ValueError("Column normalization created duplicate names")
    missing = sorted(set(REQUIRED) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in NUMERIC:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "measurement_time" in df:
        df["measurement_time"] = pd.to_datetime(
            df["measurement_time"], errors="coerce", utc=True
        )

    before = len(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    valid = df[REQUIRED].notna().all(axis=1)
    if {"latitude", "longitude"}.issubset(df.columns):
        valid &= df["latitude"].between(-90, 90) & df["longitude"].between(-180, 180)
    valid &= df["frequency_mhz"].gt(0) & df["tx_distance_km"].ge(0)
    valid &= df["tx_height_m"].gt(0) & df["rx_height_m"].gt(0)
    for col in ("landcover_urban_fraction", "landcover_forest_fraction"):
        valid &= df[col].between(0, 1)

    cleaned = df.loc[valid].drop_duplicates().copy()
    if "measurement_time" in cleaned:
        cleaned["measurement_time"] = cleaned["measurement_time"].dt.strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        )
    audit = {
        "input_rows": int(before),
        "output_rows": int(len(cleaned)),
        "rejected_or_duplicate_rows": int(before - len(cleaned)),
    }
    return cleaned, audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean TVWS measurement CSV")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default="data/processed/features.csv")
    parser.add_argument("--report", default="data/processed/cleaning_report.json")
    args = parser.parse_args()
    clean, audit = clean_frame(pd.read_csv(args.input))
    if clean.empty:
        raise SystemExit("No valid rows remain; inspect source data and cleaning rules")
    output, report = Path(args.output), Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(output, index=False)
    report.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
