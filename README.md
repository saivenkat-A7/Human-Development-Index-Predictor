# Human Development Index (HDI) Predictor

A production-ready Flask web app that predicts a country's HDI tier
(**Very High / High / Medium / Low**) from life expectancy, education,
and income indicators, using a Random Forest model trained against the
official UNDP HDI methodology.

## Features

- Predicts HDI category **and** continuous HDI score
- Every prediction shown alongside the official UNDP formula result for transparency
- Interactive Bootstrap UI with scenario quick-fill buttons (Very High / Medium / Low)
- REST API endpoint (`/api/predict`) for programmatic access
- Confusion matrix & feature importance plots generated during training
- Dockerized for easy deployment

## Tech Stack

Python · Flask · scikit-learn · Pandas · NumPy · Matplotlib · Seaborn · Bootstrap 5

## Folder Structure

```
hdi-predictor/
├── app.py                       # Flask application
├── requirements.txt
├── Dockerfile
├── .gitignore
├── README.md
├── data/
│   ├── generate_dataset.py      # Synthetic dataset generator (UNDP formula based)
│   └── hdi_dataset.csv          # Generated on first run
├── model/
│   ├── train_model.py           # Trains classifier + regressor
│   ├── hdi_classifier.pkl       # Generated after training
│   ├── hdi_regressor.pkl        # Generated after training
│   ├── scaler.pkl               # Generated after training
│   └── label_encoder.pkl        # Generated after training
├── utils/
│   └── hdi_calculator.py        # Official UNDP HDI formula implementation
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── result.html
│   └── about.html
└── static/
    ├── css/style.css
    ├── js/script.js
    └── images/                  # confusion_matrix.png, feature_importance.png (generated)
```

## Setup (Local / Anaconda)

### 1. Create environment

```bash
conda create -n hdi-predictor python=3.11 -y
conda activate hdi-predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate dataset

```bash
python data/generate_dataset.py
```

### 4. Train the model

```bash
python model/train_model.py
```

This prints classifier accuracy / regressor R² and saves model artifacts
into `model/`, plus a confusion matrix and feature-importance chart into
`static/images/`.

### 5. Run the app

```bash
python app.py
```

Visit **http://127.0.0.1:5000**

## Running with Docker

```bash
docker build -t hdi-predictor .
docker run -p 5000:5000 hdi-predictor
```

## API Usage

```bash
curl "http://127.0.0.1:5000/api/predict?life_expectancy=78.5&mean_years_schooling=11&expected_years_schooling=15&gni_per_capita=32000"
```

Response:

```json
{
  "inputs": {...},
  "ml_prediction": {
    "category": "High",
    "hdi_score": 0.7912,
    "probabilities": {"Very High": 0.31, "High": 0.62, "Medium": 0.07, "Low": 0.0}
  },
  "official_formula_result": {
    "hdi_score": 0.798,
    "category": "High",
    "health_index": 0.887,
    "education_index": 0.7,
    "income_index": 0.79
  }
}
```

## HDI Methodology Reference

- Life expectancy goalposts: 20–85 years
- Mean years of schooling goalposts: 0–15 years
- Expected years of schooling goalposts: 0–18 years
- GNI per capita (PPP $) goalposts: 100–75,000 (log scale)
- HDI = geometric mean of the three normalized dimension indices
- Tier cutoffs: Very High ≥ 0.800, High ≥ 0.700, Medium ≥ 0.550, Low < 0.550

## System Requirements

**Hardware:** Intel Core i3+, 4GB RAM min, 10GB free storage, internet for package installs.
**Software:** Windows/Linux/macOS, Python 3.x, Anaconda Navigator, Jupyter/Spyder (optional), Flask.


