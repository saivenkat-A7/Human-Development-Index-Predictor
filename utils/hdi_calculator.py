"""
hdi_calculator.py
------------------
Implements the official UNDP Human Development Index (HDI) methodology.

HDI = (I_health * I_education * I_income) ** (1/3)

Where:
  I_health    = (LE - LE_min) / (LE_max - LE_min)
  I_education = (MYS_index + EYS_index) / 2
  I_income    = (ln(GNI) - ln(GNI_min)) / (ln(GNI_max) - ln(GNI_min))

Official goalposts (UNDP Human Development Report Technical Notes):
  Life expectancy (LE):            20  - 85   years
  Mean years of schooling (MYS):    0  - 15   years
  Expected years of schooling (EYS): 0 - 18   years
  GNI per capita (PPP $):          100 - 75000

HDI classification tiers (official UNDP cutoffs):
  Very High : HDI >= 0.800
  High      : 0.700 <= HDI < 0.800
  Medium    : 0.550 <= HDI < 0.700
  Low       : HDI < 0.550
"""

import math

# Official UNDP goalposts
LE_MIN, LE_MAX = 20, 85
MYS_MIN, MYS_MAX = 0, 15
EYS_MIN, EYS_MAX = 0, 18
GNI_MIN, GNI_MAX = 100, 75000


def life_expectancy_index(le: float) -> float:
    le = max(min(le, LE_MAX), LE_MIN)
    return (le - LE_MIN) / (LE_MAX - LE_MIN)


def education_index(mys: float, eys: float) -> float:
    mys = max(min(mys, MYS_MAX), MYS_MIN)
    eys = max(min(eys, EYS_MAX), EYS_MIN)
    mys_index = mys / MYS_MAX
    eys_index = eys / EYS_MAX
    return (mys_index + eys_index) / 2


def income_index(gni: float) -> float:
    gni = max(min(gni, GNI_MAX), GNI_MIN)
    return (math.log(gni) - math.log(GNI_MIN)) / (math.log(GNI_MAX) - math.log(GNI_MIN))


def compute_hdi(life_expectancy: float, mean_years_schooling: float,
                 expected_years_schooling: float, gni_per_capita: float) -> float:
    """Returns the official HDI score (0-1) using the UNDP geometric mean formula."""
    i_health = life_expectancy_index(life_expectancy)
    i_edu = education_index(mean_years_schooling, expected_years_schooling)
    i_income = income_index(gni_per_capita)

    # Guard against zero (geometric mean fails on 0)
    i_health = max(i_health, 1e-6)
    i_edu = max(i_edu, 1e-6)
    i_income = max(i_income, 1e-6)

    hdi = (i_health * i_edu * i_income) ** (1 / 3)
    return round(hdi, 4)


def classify_hdi(hdi_score: float) -> str:
    """Official UNDP tier cutoffs."""
    if hdi_score >= 0.800:
        return "Very High"
    elif hdi_score >= 0.700:
        return "High"
    elif hdi_score >= 0.550:
        return "Medium"
    else:
        return "Low"


def full_breakdown(life_expectancy: float, mean_years_schooling: float,
                    expected_years_schooling: float, gni_per_capita: float) -> dict:
    """Returns the HDI score, tier, and each dimension sub-index (for UI display)."""
    i_health = round(life_expectancy_index(life_expectancy), 4)
    i_edu = round(education_index(mean_years_schooling, expected_years_schooling), 4)
    i_income = round(income_index(gni_per_capita), 4)
    hdi = compute_hdi(life_expectancy, mean_years_schooling, expected_years_schooling, gni_per_capita)
    return {
        "hdi_score": hdi,
        "category": classify_hdi(hdi),
        "health_index": i_health,
        "education_index": i_edu,
        "income_index": i_income,
    }
