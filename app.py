"""
app.py
------
Flask web application for the HDI (Human Development Index) Predictor.

Routes:
  GET  /            -> input form
  POST /predict      -> runs ML prediction + official formula, renders result
  GET  /api/predict  -> JSON API version (query params or POST JSON)
  GET  /about        -> project info page
"""

import os
import io
import base64
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for

from utils.hdi_calculator import full_breakdown, LE_MIN, LE_MAX, MYS_MIN, MYS_MAX, EYS_MIN, EYS_MAX, GNI_MIN, GNI_MAX

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "model")

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "hdi-predictor-dev-secret-key")  # set SECRET_KEY env var in production

FEATURES = [
    "life_expectancy",
    "mean_years_schooling",
    "expected_years_schooling",
    "gni_per_capita",
]

CATEGORY_COLORS = {
    "Very High": "#2e7d32",
    "High": "#66bb6a",
    "Medium": "#fbc02d",
    "Low": "#e53935",
}


def load_artifacts():
    """Loads model artifacts if present. Returns None values if not yet trained."""
    paths = {
        "clf": os.path.join(MODEL_DIR, "hdi_classifier.pkl"),
        "reg": os.path.join(MODEL_DIR, "hdi_regressor.pkl"),
        "scaler": os.path.join(MODEL_DIR, "scaler.pkl"),
        "encoder": os.path.join(MODEL_DIR, "label_encoder.pkl"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        return None, None, None, None
    clf = joblib.load(paths["clf"])
    reg = joblib.load(paths["reg"])
    scaler = joblib.load(paths["scaler"])
    encoder = joblib.load(paths["encoder"])
    return clf, reg, scaler, encoder


clf, reg, scaler, encoder = load_artifacts()


def make_probability_chart(class_names, probabilities):
    """Builds a base64-encoded bar chart of class probabilities for inline HTML embedding."""
    colors = [CATEGORY_COLORS.get(c, "#42a5f5") for c in class_names]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    bars = ax.bar(class_names, probabilities, color=colors)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")
    ax.set_title("Model Confidence by HDI Category")
    for bar, prob in zip(bars, probabilities):
        ax.text(bar.get_x() + bar.get_width() / 2, prob + 0.02, f"{prob*100:.1f}%",
                 ha="center", va="bottom", fontsize=9)
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def validate_inputs(form):
    errors = []
    try:
        le = float(form.get("life_expectancy"))
        mys = float(form.get("mean_years_schooling"))
        eys = float(form.get("expected_years_schooling"))
        gni = float(form.get("gni_per_capita"))
    except (TypeError, ValueError):
        return None, ["All fields are required and must be numeric."]

    if not (0 <= le <= 100):
        errors.append(f"Life expectancy should be between 0 and 100 years.")
    if not (0 <= mys <= 20):
        errors.append("Mean years of schooling should be between 0 and 20.")
    if not (0 <= eys <= 25):
        errors.append("Expected years of schooling should be between 0 and 25.")
    if not (0 < gni <= 200000):
        errors.append("GNI per capita should be a positive value (max 200,000).")

    if errors:
        return None, errors
    return {"life_expectancy": le, "mean_years_schooling": mys,
            "expected_years_schooling": eys, "gni_per_capita": gni}, []


@app.route("/")
def home():
    model_ready = clf is not None
    return render_template("index.html", model_ready=model_ready)


@app.route("/predict", methods=["POST"])
def predict():
    if clf is None:
        flash("Model is not trained yet. Run 'python model/train_model.py' first.")
        return redirect(url_for("home"))

    data, errors = validate_inputs(request.form)
    if errors:
        for e in errors:
            flash(e)
        return redirect(url_for("home"))

    X = np.array([[data[f] for f in FEATURES]])
    X_scaled = scaler.transform(X)

    pred_encoded = clf.predict(X_scaled)[0]
    pred_category = encoder.inverse_transform([pred_encoded])[0]
    probabilities = clf.predict_proba(X_scaled)[0]
    class_names = list(encoder.classes_)

    predicted_score = float(reg.predict(X_scaled)[0])

    # Official formula-based result, for transparency / comparison
    official = full_breakdown(**data)

    chart_b64 = make_probability_chart(class_names, probabilities)

    return render_template(
        "result.html",
        inputs=data,
        ml_category=pred_category,
        ml_score=round(predicted_score, 4),
        official=official,
        chart_b64=chart_b64,
        category_color=CATEGORY_COLORS.get(pred_category, "#42a5f5"),
    )


@app.route("/api/predict", methods=["GET", "POST"])
def api_predict():
    if clf is None:
        return jsonify({"error": "Model not trained. Run model/train_model.py first."}), 503

    payload = request.get_json(silent=True) or request.args
    data, errors = validate_inputs(payload)
    if errors:
        return jsonify({"errors": errors}), 400

    X = np.array([[data[f] for f in FEATURES]])
    X_scaled = scaler.transform(X)

    pred_encoded = clf.predict(X_scaled)[0]
    pred_category = encoder.inverse_transform([pred_encoded])[0]
    probabilities = clf.predict_proba(X_scaled)[0]
    class_names = list(encoder.classes_)
    predicted_score = float(reg.predict(X_scaled)[0])

    official = full_breakdown(**data)

    return jsonify({
        "inputs": data,
        "ml_prediction": {
            "category": pred_category,
            "hdi_score": round(predicted_score, 4),
            "probabilities": {c: round(float(p), 4) for c, p in zip(class_names, probabilities)},
        },
        "official_formula_result": official,
    })


@app.route("/about")
def about():
    return render_template("about.html")


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
