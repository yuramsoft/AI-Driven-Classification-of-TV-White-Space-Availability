from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate_demo_data(rows: int = 5000, seed: int = 17) -> pd.DataFrame:
    if rows < 100:
        raise ValueError("Use at least 100 rows to create a useful demo dataset")
    rng = np.random.default_rng(seed)
    # Example study-area box only; replace with the project's real study area.
    latitude = rng.uniform(34.0, 35.0, rows)
    longitude = rng.uniform(-119.0, -118.0, rows)
    frequency = rng.choice([503.0, 509.0, 587.0, 593.0], rows)
    distance = rng.uniform(1.0, 80.0, rows)
    tx_height = rng.uniform(30.0, 250.0, rows)
    rx_height = rng.uniform(2.0, 30.0, rows)
    urban = rng.beta(2.0, 3.0, rows)
    forest = rng.beta(2.0, 4.0, rows) * (1.0 - urban)
    terrain_loss = rng.gamma(2.0, 4.0, rows)
    # Deliberately simplified synthetic response; not a standards-compliant model.
    rx_power = (
        -38.0 - 20.0 * np.log10(distance) - 0.035 * terrain_loss
        - 7.0 * urban - 3.0 * forest
        + 0.012 * (tx_height - 100.0) + rng.normal(0.0, 5.0, rows)
    )
    # A coarse grid identifier helps demonstrate grouped splitting.
    group_id = (np.floor((latitude - 34.0) * 10).astype(int) * 10
                + np.floor((longitude + 119.0) * 10).astype(int)).astype(str)
    return pd.DataFrame({
        "latitude": latitude, "longitude": longitude,
        "channel": rng.choice([17, 18, 32, 33], rows),
        "frequency_mhz": frequency, "rx_power_dbm": rx_power,
        "tx_distance_km": distance, "tx_height_m": tx_height,
        "rx_height_m": rx_height, "terrain_loss_db": terrain_loss,
        "landcover_urban_fraction": urban,
        "landcover_forest_fraction": forest,
        "measurement_time": pd.Timestamp("2025-01-01", tz="UTC").isoformat(),
        "group_id": group_id,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic TVWS demo rows")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out", default="data/raw/synthetic_demo.csv")
    args = parser.parse_args()
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    generate_demo_data(args.rows, args.seed).to_csv(output, index=False)
    print(f"Wrote {args.rows} synthetic rows to {output}")


if __name__ == "__main__":
    main()
