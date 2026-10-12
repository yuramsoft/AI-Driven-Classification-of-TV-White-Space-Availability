# AI-Driven Classification of TV White-Space Availability

## 1. Project overview
The project aims to develop an artificial intelligence-based system for identifying and classifying Television White Spaces (TVWS) within the UHF television broadcasting spectrum. The system will process measured spectrum data to determine whether individual frequency channels or spectrum segments is **available**, **unavailable**, or **uncertain** at a specified location and time. The system combines terrain, land-cover, transmitter, and measurement data; predicts received power and uncertainty; and applies regulatory protection rules before declaring a channel available.

**Important:** This is a research/prototyping design, not an authorization to transmit. Use the applicable national regulator's rules and certified databases/devices for operational decisions. A model prediction alone must never override regulatory constraints.

### Goals
- Produce channel-level availability maps for a selected study area.
- Estimate protected TV signal strength, with calibrated uncertainty.
- Explain influential features and validate performance on geographically separated locations.
- Provide an API and a small map-oriented prediction workflow.

### Scope assumptions
- The jurisdiction, frequency plan, and regulator are configurable; do not hard-code rules from one country.
- Inputs include geospatial observations with channel/frequency and coordinates, plus licensed-transmitter metadata and environmental raster/vector features.
- Output is a planning/research classification. A compliance layer supplied by the regulator or domain expert is authoritative.

## 2. End-to-end architecture
1. **Data ingestion:** measurement campaigns, transmitter registry, elevation/terrain, land cover, clutter/buildings where licensed, and jurisdictional rules.
2. **Feature engineering:** geodesic distance, bearing, terrain profile/obstruction, land-cover fractions, frequency, antenna heights/gains, and time/context fields.
3. **Model:** predict received TV power (dBm) and prediction uncertainty; use the prediction plus protection margin and regulatory constraints to classify.
4. **Evaluation:** spatially blocked holdout, per-channel metrics, calibration, and baseline comparison.
5. **Serving:** FastAPI endpoint accepts a location, channel, and features; returns class, estimated power, confidence interval, and reason codes.
6. **Monitoring:** track data drift, calibration, missing inputs, and field-measurement error; retrain only after review.

## 3. Repository layout

```text
tvws-classifier/
├── data/                 # Raw and processed data; never commit restricted data
├── config/               # Jurisdiction and feature configuration
├── models/               # Trained model artifacts (versioned separately)
├── src/
│   ├── generate_data.py  # Reproducible synthetic demonstration data
│   ├── clean_data.py         # CSV validation, cleaning, and audit report
│   ├── features.py           # Model feature validation
│   ├── train.py          # Spatially separated model training
│   ├── classify.py       # Regulatory-aware decision function
│   └── api.py            # FastAPI service
├── tests/
├── requirements.txt
└── README.md
```

## 4. Training Data

Training data should contain one row per receiver measurement and candidate channel. Recommended columns:

| Column | Meaning |
|---|---|
| `latitude`, `longitude` | Receiver position in decimal degrees |
| `channel`, `frequency_mhz` | Candidate TV channel and center frequency |
| `rx_power_dbm` | Measured protected-signal power; document receiver/antenna calibration |
| `tx_distance_km` | Distance to relevant licensed transmitter |
| `tx_height_m`, `rx_height_m` | Antenna heights above ground |
| `terrain_loss_db` | Profile-derived terrain attenuation estimate |
| `landcover_*` | Fractions or encoded category from a documented land-cover source |
| `measurement_time` | UTC timestamp or derived temporal fields |
| `group_id` | Spatial block / campaign identifier for leakage-resistant split |


### Training notes
- Confirm the `group_id` is a real spatial grouping (for example, grid cell or campaign), not a row identifier.
- Add a simple propagation-model baseline and compare its errors against the ML model.
- Tune hyperparameters using grouped cross-validation on training groups only.
- Report MAE/RMSE in dB, residual plots, per-channel errors, and uncertainty interval coverage.
- Fit preprocessing only on training data, as the pipeline above does.

## 5. Data generation — `src/generate_data.py`

This creates a **synthetic demonstration dataset only** so the pipeline can be tested before real measurements are available. Its simulated received power is not propagation truth, evidence of channel availability, or suitable for transmission decisions. Keep synthetic and measured data clearly separated.

## 6. Data cleaning — `src/clean_data.py`

This script normalizes column names, parses numeric fields, rejects impossible values, drops rows missing required model inputs/target, and writes an audit summary. It does not invent missing measurements or silently clip outliers. Review the rejected-row report and adapt valid ranges to the sensor and jurisdiction before using real data.

## 7. Classification policy — `src/classify.py`

The threshold below is an explicit configurable placeholder, not a universal TVWS rule. A real deployment must load the relevant regulator's protection thresholds, adjacent-channel rules, guard bands, geographic exclusions, antenna assumptions, and required margins from an approved source.

The example uncertainty argument is a placeholder interface. Do not substitute an arbitrary constant in production: estimate uncertainty from held-out residuals, ensembles, conformal intervals, or a validated probabilistic model and document its coverage.

## 8. API — `src/api.py`

**API limitation:** `regulator_allows` must be produced by a trusted regulatory rules service, not supplied unchecked by a public client. In a real system, authenticate callers and compute that field server-side.


## 9. Validation and safety checklist

- [ ] Verify measurement units, calibration, frequency/channel mapping, and coordinate datum.
- [ ] Separate train/test by spatial blocks and preferably by measurement campaign/time.
- [ ] Check performance by channel, terrain class, urban/rural class, and distance band.
- [ ] Calibrate prediction intervals and measure empirical coverage on held-out locations.
- [ ] Compare with an accepted propagation baseline and investigate systematic residuals.
- [ ] Apply regulator-mandated protection criteria and uncertainty margins outside the ML model.
- [ ] Fail closed for missing, stale, out-of-area, or conflicting regulatory inputs.
- [ ] Log model/data versions and decision reason codes; protect location data.
- [ ] Treat results as advisory until validated and approved for the jurisdiction.


## 10. Run locally
## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt

# Generate clearly labeled synthetic demo data (not real measurements)
python src/generate_data.py --rows 5000 --seed 17

# Clean/validate the input and write an audit report
python src/clean_data.py --input data/raw/synthetic_demo.csv

# Train and evaluate using grouped holdout
python src/train.py --csv data/processed/features.csv

# Run the research API
uvicorn src.api:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API documentation. Synthetic rows are for software testing only and must not be reported as measured radio evidence.

Example request body (use feature values from your dataset and a validated uncertainty estimate):

```json
{
  "frequency_mhz": 587.0,
  "tx_distance_km": 24.5,
  "tx_height_m": 125.0,
  "rx_height_m": 10.0,
  "terrain_loss_db": 18.0,
  "landcover_urban_fraction": 0.35,
  "landcover_forest_fraction": 0.2,
  "uncertainty_db": 4.0,
  "regulator_allows": true
}
```
## 11. Recommended next implementation milestones

1. Choose jurisdiction, study area, channels, and authoritative regulatory source.
2. Obtain licensed transmitter and measurement data with documented rights and calibration.
3. Build terrain/land-cover feature extraction and generate spatial blocks.
4. Establish propagation baseline; train and validate model with grouped cross-validation.
5. Add calibrated uncertainty and policy review; test conservative edge cases.
6. Deploy behind authentication with monitoring, audit logs, and human review.

-

## Author

**Ibrahim Mustapha, PhD**

Department of Electrical & Electronic Engineering
University of Maiduguri
Communications Engineering / Wireless Communications
Python & Machine Learning

---

## License

This project is released under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This project is primarily an educational and research-oriented implementation. Predictions produced by the initial model should not be treated as a substitute for detailed radio-frequency planning, field measurements, or validated commercial propagation models.
