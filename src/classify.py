from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionPolicy:
    # Replace these example values with the applicable, reviewed policy.
    protected_power_threshold_dbm: float = -90.0
    uncertainty_margin_db: float = 6.0
    min_required_margin_db: float = 0.0


def classify_channel(
    predicted_power_dbm: float,
    uncertainty_db: float,
    policy: DecisionPolicy,
    regulator_allows: bool,
) -> dict:
    """Conservative research classification; regulator gate always wins."""
    if not regulator_allows:
        return {"class": "unavailable", "reason": "regulatory_constraint"}
    if uncertainty_db < 0:
        raise ValueError("uncertainty_db must be non-negative")

    conservative_power = predicted_power_dbm + uncertainty_db + policy.uncertainty_margin_db
    margin = policy.protected_power_threshold_dbm - conservative_power
    if margin >= policy.min_required_margin_db:
        label = "available"
        reason = "predicted_protected_signal_below_threshold_with_margin"
    elif predicted_power_dbm - uncertainty_db > policy.protected_power_threshold_dbm:
        label = "unavailable"
        reason = "predicted_protected_signal_above_threshold"
    else:
        label = "uncertain"
        reason = "prediction_interval_overlaps_threshold"

    return {
        "class": label,
        "reason": reason,
        "predicted_power_dbm": round(float(predicted_power_dbm), 2),
        "uncertainty_db": round(float(uncertainty_db), 2),
        "conservative_power_dbm": round(float(conservative_power), 2),
        "available_margin_db": round(float(margin), 2),
    }