"""
generate_dataset.py
--------------------
Generates a realistic synthetic dataset of "countries" spanning the full
range of the official UNDP HDI indicator scales, computes the true HDI
category using the official formula, and adds small measurement noise so
the downstream ML classifier has to genuinely learn the relationship
rather than just re-deriving the closed-form formula.

Run:
    python data/generate_dataset.py
Produces:
    data/hdi_dataset.csv
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.hdi_calculator import compute_hdi, classify_hdi

RANDOM_SEED = 42
N_SAMPLES = 4000

TIER_ARCHETYPES = {
    # tier: (LE range, MYS range, EYS range, GNI range, weight)
    "Very High": {"le": (75, 85), "mys": (10, 15), "eys": (14, 18), "gni": (30000, 75000)},
    "High":      {"le": (68, 76), "mys": (7, 11),  "eys": (11, 15), "gni": (9000, 32000)},
    "Medium":    {"le": (58, 70), "mys": (4, 8),   "eys": (8, 12),  "gni": (2500, 11000)},
    "Low":       {"le": (45, 62), "mys": (1, 5),   "eys": (4, 9),   "gni": (300, 3500)},
}


def generate_row(rng: np.random.Generator, tier: str) -> dict:
    a = TIER_ARCHETYPES[tier]
    le = rng.uniform(*a["le"])
    mys = rng.uniform(*a["mys"])
    eys = rng.uniform(*a["eys"])
    gni = rng.uniform(*a["gni"])

    # Add realistic measurement noise (simulating imperfect real-world data)
    le += rng.normal(0, 1.5)
    mys += rng.normal(0, 0.6)
    eys += rng.normal(0, 0.6)
    gni *= (1 + rng.normal(0, 0.05))

    # Clip to plausible real-world bounds
    le = float(np.clip(le, 40, 90))
    mys = float(np.clip(mys, 0, 15))
    eys = float(np.clip(eys, 0, 20))
    gni = float(np.clip(gni, 200, 120000))

    hdi_score = compute_hdi(le, mys, eys, gni)
    category = classify_hdi(hdi_score)

    return {
        "life_expectancy": round(le, 2),
        "mean_years_schooling": round(mys, 2),
        "expected_years_schooling": round(eys, 2),
        "gni_per_capita": round(gni, 2),
        "hdi_score": hdi_score,
        "hdi_category": category,
    }


def main():
    rng = np.random.default_rng(RANDOM_SEED)
    tiers = list(TIER_ARCHETYPES.keys())
    rows = []

    for _ in range(N_SAMPLES):
        tier = rng.choice(tiers)  # uniform across tiers -> balanced classes
        rows.append(generate_row(rng, tier))

    df = pd.DataFrame(rows)

    out_path = os.path.join(os.path.dirname(__file__), "hdi_dataset.csv")
    df.to_csv(out_path, index=False)

    print(f"Generated {len(df)} rows -> {out_path}")
    print("\nClass distribution (post real-formula classification):")
    print(df["hdi_category"].value_counts())


if __name__ == "__main__":
    main()
