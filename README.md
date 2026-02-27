# EcoExchange AI - ML Backend

Production-ready ML backend for carbon emission forecasting, sustainability optimization, and circular economy waste matching.

## Features

1.  **Carbon Forecaster (v2.0)**: Uses Meta's Prophet model with anti-hallucination guardrails (hard caps at 2 STDs). Predicts Scope 1, 2, and 3 emissions with fixed confidence intervals (±12%).
2.  **Sustainability Suggester**: XGBoost multi-output classifier recommending interventions with ROI/payback tracking and dynamic logic explanations.
3.  **Waste Matcher**: Geospatial matching with automatic radius expansion, match quality labels, transport cost estimation, and CO2 savings calculation.
4.  **Industry Benchmarking**: Real-time comparison against industry averages and best-in-class performance.

## Project Structure

```text
EcoExchange-API/
├── data/
│   ├── generate_data.py             (8000 row high-fidelity data generator)
│   ├── emissions_training.csv       (High continuity training data)
│   ├── suggestions_training.csv     (Budget-aware recommendation data)
│   └── waste_listings.csv           (3000 row Indian geographic clusters)
├── models/
│   ├── train_forecaster.py          (Prophet Training with monthly seasonality)
│   ├── train_suggester.py           (XGBoost Training)
│   └── saved/                       (Serialized Models, Benchmarks.json)
├── api/
│   ├── main.py                      (FastAPI Entry Point)
│   ├── predict_emissions.py         (Forecasting + Anomaly Detection + Benchmarking)
│   ├── suggest.py                   (Budget-aware Suggestions)
│   ├── match.py                     (Radius-expanding Waste Matcher)
│   └── benchmark.py                 (Industry Benchmark Endpoint)
├── requirements.txt
└── README.md
```

## Setup & Running

1.  **Generate Training Data**: (8000 rows, consistent inputs)
    ```bash
    python data/generate_data.py
    ```

2.  **Train Forecasting Models**: (Changepoint prior scale 0.01)
    ```bash
    python models/train_forecaster.py
    ```

3.  **Train Suggestion Model**:
    ```bash
    python models/train_suggester.py
    ```

4.  **Launch API**:
    ```bash
    uvicorn api.main:app --reload --port 8000
    ```

## API Documentation

### 1. Predict Emissions (`POST /predict/emissions`)
**Enhanced Response Fields**:
- `trend_percent`: Monthly percentage change.
- `benchmark_comparison`: Industry average vs your average + percentile ranking.
- `data_quality_score`: Reliability score (0.40 - 0.95) based on historical data volume.
- `anomalies`: Z-score based detection (threshold > 1.0).

### 2. Get Suggestions (`POST /suggestions`)
**Enhanced Response Fields**:
- `why_recommended`: Dynamic explanation based on input data (e.g. "Your power factor is low...").
- `roi_percent`: Annualized return on investment.
- `payback_note`: Flags long-term vs standard investments.
- `confidence`: Probability score capped at 0.98 for realism.

### 3. Match Waste (`POST /match`)
**Enhanced Features**:
- `radius_expanded`: Boolean indicating if search was expanded to 500km to find matches.
- `quality_label`: "Excellent", "Good", "Fair", or "Possible".
- `estimated_transport_cost_inr`: Calculated at ₹45/km/ton (Indian average).
- `co2_saved_kg`: Based on material recycling factors.

### 4. Industry Benchmark (`GET /benchmark/{industry}`)
Returns performance targets, top emission sources, and common optimizations for a given industry.

## Anti-Hallucination Guardrails
The system uses a strict **±2 Standard Deviation Hard Cap** on all Prophet forecasts. This prevents the model from predicting impossible jumps (e.g., historical max is 100k, model cannot predict 150k if it's outside 2 STDs).
- Confidence intervals are fixed at **±12%** for practical business planning.
- Suggestions confidence is capped at **0.98** to maintain credibility.
