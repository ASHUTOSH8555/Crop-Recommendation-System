"""
advisor.py
Crop rotation logic, climate risk scoring,
fertilizer gap analysis, 12-month crop calendar
"""

import numpy as np
from models import CROP_NUTRIENT_NEEDS, CROP_PROFILES

# ── Crop rotation rules ───────────────────────────────────────────────────────
# Maps previous crop → list of good follow-up crops (nutrient balance logic)

ROTATION_RULES = {
    "Rice":       ["Wheat","Mustard","Lentil","Chickpea","Potato"],
    "Wheat":      ["Rice","Soybean","Chickpea","Lentil","Sunflower"],
    "Maize":      ["Soybean","Groundnut","Chickpea","Wheat","Potato"],
    "Cotton":     ["Wheat","Chickpea","Soybean","Mustard","Lentil"],
    "Sugarcane":  ["Wheat","Mustard","Onion","Garlic","Potato"],
    "Soybean":    ["Wheat","Maize","Cotton","Sunflower","Rice"],
    "Groundnut":  ["Wheat","Maize","Cotton","Sorghum","Rice"],
    "Chickpea":   ["Wheat","Maize","Cotton","Rice","Sunflower"],
    "Lentil":     ["Wheat","Maize","Rice","Cotton","Sunflower"],
    "Potato":     ["Wheat","Maize","Soybean","Mustard","Onion"],
    "Tomato":     ["Onion","Garlic","Maize","Wheat","Mustard"],
    "Onion":      ["Tomato","Maize","Wheat","Soybean","Rice"],
    "Garlic":     ["Tomato","Maize","Wheat","Soybean","Rice"],
    "Mustard":    ["Rice","Maize","Soybean","Chickpea","Wheat"],
    "Sunflower":  ["Wheat","Maize","Soybean","Chickpea","Rice"],
    "Banana":     ["Sugarcane","Rice","Maize","Groundnut","Soybean"],
    "Coffee":     ["Banana","Maize","Soybean","Rice","Tea"],
    "Tea":        ["Coffee","Maize","Soybean","Rice","Banana"],
    "Jute":       ["Rice","Wheat","Mustard","Lentil","Chickpea"],
    "Mango":      ["Groundnut","Soybean","Maize","Wheat","Rice"],
    "Grapes":     ["Wheat","Mustard","Onion","Garlic","Chickpea"],
    "Watermelon": ["Wheat","Maize","Onion","Garlic","Mustard"],
}

def rotation_advice(previous_crop: str, recommendations: list) -> dict:
    """
    Given previous crop and current recommendations,
    flag which are good rotations and which to avoid.
    """
    if not previous_crop or previous_crop not in ROTATION_RULES:
        return {"good": [], "avoid": [], "note": "No rotation history provided."}

    good_rotations = ROTATION_RULES.get(previous_crop, [])
    result = {"good": [], "avoid": [], "note": f"Based on previous crop: {previous_crop}"}

    for crop, conf in recommendations:
        if crop in good_rotations:
            result["good"].append(crop)
        elif crop == previous_crop:
            result["avoid"].append(crop)  # monoculture warning

    return result


# ── Climate risk scoring ──────────────────────────────────────────────────────

def climate_risk(crop: str, weather: dict) -> dict:
    """
    Score water stress, frost risk, heat stress for a crop
    given current weather conditions.
    Returns risk levels: low / medium / high
    """
    p = CROP_PROFILES.get(crop)
    if not p:
        return {}

    temp     = weather["temperature"]
    temp_std = weather.get("temp_std", 3.0)
    rain     = weather["rainfall"]
    rain_std = weather.get("rain_std", 20.0)

    t_min, t_max = p["t"]
    r_min, r_max = p["r"]

    # Water stress
    if rain < r_min * 0.6:
        water = "high"
    elif rain < r_min:
        water = "medium"
    else:
        water = "low"

    # Frost risk (temp drops below crop minimum)
    frost_temp = temp - 2 * temp_std
    if frost_temp < t_min - 5:
        frost = "high"
    elif frost_temp < t_min:
        frost = "medium"
    else:
        frost = "low"

    # Heat stress
    heat_temp = temp + 2 * temp_std
    if heat_temp > t_max + 5:
        heat = "high"
    elif heat_temp > t_max:
        heat = "medium"
    else:
        heat = "low"

    # Overall risk
    levels = {"low": 0, "medium": 1, "high": 2}
    overall_score = max(levels[water], levels[frost], levels[heat])
    overall = ["low", "medium", "high"][overall_score]

    return {
        "water_stress": water,
        "frost_risk":   frost,
        "heat_stress":  heat,
        "overall":      overall,
    }


# ── Fertilizer gap analysis ───────────────────────────────────────────────────

def fertilizer_gap(crop: str, current_N: float, current_P: float,
                   current_K: float, current_ph: float) -> dict:
    """
    Compare current soil nutrients vs what the crop ideally needs.
    Returns how much to add (positive = add, negative = excess).
    """
    needs = CROP_NUTRIENT_NEEDS.get(crop)
    if not needs:
        return {}

    gap_N  = round(needs["N"]  - current_N,  1)
    gap_P  = round(needs["P"]  - current_P,  1)
    gap_K  = round(needs["K"]  - current_K,  1)
    gap_ph = round(needs["ph"] - current_ph, 2)

    def label(val):
        if val > 10:   return "add"
        if val < -10:  return "excess"
        return "optimal"

    ph_action = "add lime" if gap_ph > 0.3 else ("add sulfur" if gap_ph < -0.3 else "optimal")

    return {
        "N":  {"gap": gap_N,  "action": label(gap_N),  "unit": "kg/ha"},
        "P":  {"gap": gap_P,  "action": label(gap_P),  "unit": "kg/ha"},
        "K":  {"gap": gap_K,  "action": label(gap_K),  "unit": "kg/ha"},
        "ph": {"gap": gap_ph, "action": ph_action},
    }


# ── 12-month crop calendar ────────────────────────────────────────────────────

from data_pipeline import fetch_weather, get_soil_params, MONTH_NAMES

def crop_calendar(lat: float, lon: float, address: str,
                  model, scaler, le) -> dict:
    """
    Generate best crop recommendation for every month of the year.
    Returns {month_name: (crop, confidence%)}
    """
    from models import recommend
    N, P, K, ph = get_soil_params(address)
    calendar = {}
    for m in range(1, 13):
        weather = fetch_weather(lat, lon, m)
        recs    = recommend(model, scaler, le, N, P, K,
                            weather["temperature"], weather["humidity"],
                            ph, weather["rainfall"], top_n=1)
        calendar[MONTH_NAMES[m]] = recs[0] if recs else ("Unknown", 0.0)
    return calendar


# ── Yield prediction ──────────────────────────────────────────────────────────

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler as _SS
from sklearn.model_selection import train_test_split as _tts

# Baseline yield (ton/ha) per crop under ideal conditions
BASELINE_YIELD = {
    "Rice": 4.5,   "Wheat": 3.5,   "Maize": 5.5,   "Chickpea": 1.2,
    "Lentil": 1.0, "Cotton": 1.8,  "Sugarcane": 65, "Soybean": 2.5,
    "Groundnut": 2.0, "Sunflower": 1.8, "Banana": 35, "Mango": 8.0,
    "Coffee": 1.5, "Tea": 2.2,     "Jute": 2.5,    "Mustard": 1.5,
    "Potato": 22,  "Tomato": 25,   "Onion": 18,    "Garlic": 8.0,
    "Watermelon": 30, "Grapes": 12,
}

def _build_yield_model():
    """Train a Random Forest Regressor to predict yield from soil+weather."""
    rng  = np.random.default_rng(0)
    rows = []
    for crop, base in BASELINE_YIELD.items():
        p = CROP_PROFILES.get(crop, {})
        if not p:
            continue
        for _ in range(150):
            N    = rng.uniform(*p["N"])
            P    = rng.uniform(*p["P"])
            K    = rng.uniform(*p["K"])
            temp = rng.uniform(*p["t"])
            hum  = rng.uniform(*p["h"])
            ph   = rng.uniform(*p["ph"])
            rain = rng.uniform(*p["r"])
            # Yield varies ±30% around baseline based on how central conditions are
            t_mid = (p["t"][0]+p["t"][1])/2
            r_mid = (p["r"][0]+p["r"][1])/2
            t_score = 1 - abs(temp-t_mid)/(p["t"][1]-p["t"][0]+1)
            r_score = 1 - abs(rain-r_mid)/(p["r"][1]-p["r"][0]+1)
            noise   = rng.uniform(0.85, 1.15)
            yield_  = base * ((t_score+r_score)/2) * noise
            rows.append([N,P,K,temp,hum,ph,rain,yield_])

    df  = pd.DataFrame(rows, columns=["N","P","K","temp","hum","ph","rain","yield"])
    X   = df.drop("yield", axis=1).values
    y   = df["yield"].values
    sc  = _SS()
    Xs  = sc.fit_transform(X)
    mdl = RandomForestRegressor(n_estimators=100, random_state=42)
    mdl.fit(Xs, y)
    return mdl, sc

_yield_model, _yield_scaler = _build_yield_model()

def predict_yield(crop: str, N, P, K, temp, humidity, ph, rainfall) -> dict:
    """Predict expected yield in ton/ha for the given crop and conditions."""
    x  = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    xs = _yield_scaler.transform(x)
    y  = float(_yield_model.predict(xs)[0])
    base = BASELINE_YIELD.get(crop, 3.0)
    pct  = round((y / base) * 100, 1)
    return {
        "predicted_ton_ha": round(y, 2),
        "baseline_ton_ha":  base,
        "yield_pct":        pct,          # % of ideal baseline
        "rating":           "excellent" if pct >= 90 else
                            "good"      if pct >= 70 else
                            "moderate"  if pct >= 50 else "poor",
    }


# ── Water requirement calculator ─────────────────────────────────────────────

# Crop evapotranspiration (ET) in mm/month under standard conditions
CROP_ET_MM_MONTH = {
    "Rice": 200,  "Wheat": 120,  "Maize": 150,  "Chickpea": 90,
    "Lentil": 80, "Cotton": 160, "Sugarcane": 180, "Soybean": 130,
    "Groundnut": 120, "Sunflower": 140, "Banana": 170, "Mango": 100,
    "Coffee": 130, "Tea": 150,   "Jute": 160,   "Mustard": 100,
    "Potato": 130, "Tomato": 140, "Onion": 110, "Garlic": 100,
    "Watermelon": 150, "Grapes": 110,
}

def water_requirement(crop: str, rainfall_mm: float) -> dict:
    """
    Calculate irrigation water needed given crop ET and actual rainfall.
    Positive deficit = irrigation needed. Negative = surplus.
    """
    et   = CROP_ET_MM_MONTH.get(crop, 130)
    deficit = round(et - rainfall_mm, 1)
    if deficit > 50:
        status = "high irrigation needed"
    elif deficit > 0:
        status = "moderate irrigation needed"
    elif deficit > -30:
        status = "rainfall sufficient"
    else:
        status = "drainage may be needed"

    return {
        "crop_et_mm":       et,
        "rainfall_mm":      round(rainfall_mm, 1),
        "deficit_mm":       deficit,
        "status":           status,
        "irrigation_needed": deficit > 0,
    }


# ── Pest and disease risk ─────────────────────────────────────────────────────

# Rules: (temp_min, temp_max, humidity_min) → pest/disease name + severity
PEST_RULES = {
    "Rice": [
        (25, 35, 80, "Brown Planthopper",      "high"),
        (20, 30, 75, "Blast (fungal)",          "high"),
        (15, 28, 70, "Sheath Blight",           "medium"),
    ],
    "Wheat": [
        (10, 20, 70, "Yellow Rust",             "high"),
        (15, 25, 65, "Aphids",                  "medium"),
        (5,  18, 75, "Powdery Mildew",          "medium"),
    ],
    "Maize": [
        (25, 35, 60, "Fall Armyworm",           "high"),
        (20, 30, 70, "Northern Leaf Blight",    "medium"),
        (18, 28, 55, "Stem Borer",              "medium"),
    ],
    "Cotton": [
        (25, 38, 60, "Bollworm",                "high"),
        (20, 35, 55, "Whitefly",                "high"),
        (22, 32, 65, "Aphids",                  "medium"),
    ],
    "Potato": [
        (10, 22, 80, "Late Blight (Phytophthora)","high"),
        (15, 25, 70, "Early Blight",            "medium"),
        (12, 20, 65, "Aphids / Virus vectors",  "medium"),
    ],
    "Tomato": [
        (20, 30, 75, "Early Blight",            "high"),
        (22, 32, 70, "Whitefly / Leaf curl virus","high"),
        (18, 28, 65, "Fusarium Wilt",           "medium"),
    ],
    "Banana": [
        (25, 35, 80, "Panama Wilt (Fusarium)",  "high"),
        (22, 32, 75, "Black Sigatoka",          "high"),
        (20, 30, 70, "Banana Weevil",           "medium"),
    ],
    "Sugarcane": [
        (25, 35, 70, "Red Rot",                 "high"),
        (20, 32, 65, "Pyrilla (leafhopper)",    "medium"),
        (22, 30, 75, "Smut",                    "medium"),
    ],
}

DEFAULT_PESTS = [
    (20, 35, 70, "General fungal risk",         "medium"),
    (25, 38, 55, "Insect pest activity",        "medium"),
]

def pest_disease_risk(crop: str, temp: float, humidity: float) -> list:
    """
    Return list of likely pests/diseases given crop, temperature, humidity.
    Each entry: {name, severity, reason}
    """
    rules   = PEST_RULES.get(crop, DEFAULT_PESTS)
    threats = []
    for t_min, t_max, h_min, name, severity in rules:
        if t_min <= temp <= t_max and humidity >= h_min:
            threats.append({
                "name":     name,
                "severity": severity,
                "reason":   f"temp {temp}°C in [{t_min}-{t_max}], humidity {humidity}% ≥ {h_min}%",
            })
    if not threats:
        threats.append({
            "name":     "No major threats detected",
            "severity": "low",
            "reason":   "Current conditions outside known risk ranges",
        })
    return threats
