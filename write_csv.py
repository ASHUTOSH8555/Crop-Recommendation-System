"""
write_csv.py
Persist recommendation results to CSV and summarize past runs.
"""

import csv
import os
from collections import Counter

RESULTS_CSV = os.path.join(os.path.dirname(__file__), "recommendation_results.csv")

FIELDNAMES = [
    "location", "month_name", "season", "lat", "lon",
    "N", "P", "K", "ph",
    "temperature", "humidity", "rainfall", "weather_source",
    "top_crop", "top_confidence",
    "rec_2", "rec_3", "rec_4", "rec_5",
    "yield_ton_ha", "yield_rating", "yield_pct",
    "water_deficit_mm", "water_status",
    "overall_risk", "water_stress", "frost_risk", "heat_stress",
    "net_profit_usd", "profitable",
    "rotation_good", "rotation_avoid",
    "fert_N_gap", "fert_P_gap", "fert_K_gap", "fert_ph_gap",
    "previous_crop",
]


def write_result(
    location, month_name, season, lat, lon,
    N, P, K, ph,
    weather, recs, yield_info, water_info,
    risk, profit, rotation, fert_gap,
    previous_crop=None,
):
    file_exists = os.path.isfile(RESULTS_CSV)

    def _rec_str(idx):
        if idx < len(recs):
            return "{} ({:.1f}%)".format(recs[idx][0], recs[idx][1])
        return ""

    row = {
        "location":         location,
        "month_name":       month_name,
        "season":           season,
        "lat":              round(lat, 4),
        "lon":              round(lon, 4),
        "N": N, "P": P, "K": K, "ph": ph,
        "temperature":      weather.get("temperature"),
        "humidity":         weather.get("humidity"),
        "rainfall":         weather.get("rainfall"),
        "weather_source":   weather.get("source", ""),
        "top_crop":         recs[0][0] if recs else "",
        "top_confidence":   round(recs[0][1], 1) if recs else "",
        "rec_2":            _rec_str(1),
        "rec_3":            _rec_str(2),
        "rec_4":            _rec_str(3),
        "rec_5":            _rec_str(4),
        "yield_ton_ha":     yield_info.get("predicted_ton_ha", ""),
        "yield_rating":     yield_info.get("rating", ""),
        "yield_pct":        yield_info.get("yield_pct", ""),
        "water_deficit_mm": round(water_info.get("deficit_mm", 0), 1),
        "water_status":     water_info.get("status", ""),
        "overall_risk":     risk.get("overall", ""),
        "water_stress":     risk.get("water_stress", ""),
        "frost_risk":       risk.get("frost_risk", ""),
        "heat_stress":      risk.get("heat_stress", ""),
        "net_profit_usd":   profit.get("net_profit", ""),
        "profitable":       profit.get("profitable", ""),
        "rotation_good":    "|".join(rotation.get("good", [])),
        "rotation_avoid":   "|".join(rotation.get("avoid", [])),
        "fert_N_gap":       fert_gap.get("N", {}).get("gap", ""),
        "fert_P_gap":       fert_gap.get("P", {}).get("gap", ""),
        "fert_K_gap":       fert_gap.get("K", {}).get("gap", ""),
        "fert_ph_gap":      fert_gap.get("ph", {}).get("gap", ""),
        "previous_crop":    previous_crop or "",
    }

    with open(RESULTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return RESULTS_CSV


def load_results():
    if not os.path.isfile(RESULTS_CSV):
        return []
    with open(RESULTS_CSV, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summarize_results():
    rows = load_results()
    if not rows:
        return {"total": 0, "top_crops": [], "avg_profit_usd": None}

    top_crops = Counter(r["top_crop"] for r in rows if r.get("top_crop")).most_common(5)

    profits = []
    for r in rows:
        try:
            profits.append(float(r["net_profit_usd"]))
        except (ValueError, TypeError, KeyError):
            pass

    avg_profit = round(sum(profits) / len(profits), 2) if profits else None

    return {
        "total":          len(rows),
        "top_crops":      top_crops,
        "avg_profit_usd": avg_profit,
    }
