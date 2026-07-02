"""
train_model.py
---------------
Trains two models on the HDI dataset:
  1. RandomForestClassifier -> predicts HDI category (Very High/High/Medium/Low)
  2. RandomForestRegressor  -> predicts continuous HDI score (0-1)

Saves:
  model/hdi_classifier.pkl
  model/hdi_regressor.pkl
  model/scaler.pkl
  model/label_encoder.pkl
  static/images/confusion_matrix.png
  static/images/feature_importance.png

Run:
    python model/train_model.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    mean_absolute_error, r2_score
)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(BASE_DIR, "data", "hdi_dataset.csv")
MODEL_DIR = os.path.dirname(__file__)
IMG_DIR = os.path.join(BASE_DIR, "static", "images")

FEATURES = [
    "life_expectancy",
    "mean_years_schooling",
    "expected_years_schooling",
    "gni_per_capita",
]

os.makedirs(IMG_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        print("Dataset not found. Generating it first...")
        sys.path.append(BASE_DIR)
        from data.generate_dataset import main as generate_main
        generate_main()
    return pd.read_csv(DATA_PATH)


def train_classifier(X_train, X_test, y_train, y_test, class_names):
    clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n[Classifier] Accuracy: {acc:.4f}")
    print(classification_report(y_test, preds, target_names=class_names))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("HDI Category - Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=120)
    plt.close()

    # Feature importance plot
    importances = clf.feature_importances_
    plt.figure(figsize=(6, 4))
    sns.barplot(x=importances, y=FEATURES, hue=FEATURES, palette="viridis", legend=False)
    plt.title("Feature Importance (Classifier)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "feature_importance.png"), dpi=120)
    plt.close()

    return clf, acc


def train_regressor(X_train, X_test, y_train, y_test):
    reg = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_split=4,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\n[Regressor] MAE: {mae:.4f} | R2: {r2:.4f}")
    return reg, mae, r2


def main():
    df = load_data()

    X = df[FEATURES].values
    y_cat_raw = df["hdi_category"].values
    y_score = df["hdi_score"].values

    encoder = LabelEncoder()
    y_cat = encoder.fit_transform(y_cat_raw)
    class_names = list(encoder.classes_)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test, yreg_train, yreg_test = train_test_split(
        X_scaled, y_cat, y_score, test_size=0.2, random_state=42, stratify=y_cat
    )

    clf, acc = train_classifier(X_train, X_test, y_train, y_test, class_names)
    reg, mae, r2 = train_regressor(X_train, X_test, yreg_train, yreg_test)

    joblib.dump(clf, os.path.join(MODEL_DIR, "hdi_classifier.pkl"))
    joblib.dump(reg, os.path.join(MODEL_DIR, "hdi_regressor.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
    joblib.dump(encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

    print("\nSaved model artifacts to:", MODEL_DIR)
    print(f"Final Classifier Accuracy: {acc:.4f} | Regressor R2: {r2:.4f}")


if __name__ == "__main__":
    main()
