from __future__ import annotations

import numpy as np
import pandas as pd

FEATURES = [
    "frequency_mhz",
    "tx_distance_km",
    "tx_height_m",
    "rx_height_m",
    "terrain_loss_db",
    "landcover_urban_fraction",
    "landcover_forest_fraction",
]
TARGET = "rx_power_dbm"
REQUIRED = FEATURES + [TARGET]


def validate_training_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy; fail loudly rather than silently changing units."""
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    clean = frame[REQUIRED].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if clean.empty:
        raise ValueError("No complete finite training rows remain")
    if (clean["frequency_mhz"] <= 0).any():
        raise ValueError("frequency_mhz must be positive")
    if (clean["tx_distance_km"] < 0).any():
        raise ValueError("tx_distance_km cannot be negative")
    for col in ("landcover_urban_fraction", "landcover_forest_fraction"):
        if not clean[col].between(0, 1).all():
            raise ValueError(f"{col} must be a fraction in [0, 1]")
    return clean
