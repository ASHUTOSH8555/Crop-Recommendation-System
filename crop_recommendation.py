# -*- coding: utf-8 -*-
"""
Crop Recommendation System
User inputs: location + month/season
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from data_pipeline import parse_month, geocode, fetch_weather, get_soil_params, MONTH_NAMES, get_season
from models import build_training_data, train_models, recommend, explain_shap, FEATURES
from advisor import rotation_advice, climate_risk, fertilizer_gap, crop_calendar, predict_yield, water_requirement, pest_disease_risk
from market import profit_estimate

RISK_COLOR = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}
RISK_ICON  = {"low": "[OK]", "medium": "[!!]", "high": "[XX]"}

def show_full_report(recs, location, month_name, weather, model_accuracies,
                     shap_vals, risk, fert_gap, rotation, cal,
                     yield_info, water_info, pests, profit):
    fig = plt.figure(figsize=(24, 18), facecolor="#0f0f1a")
    fig.suptitle(f"Crop Report  |  {location}  |  {month_name}",
                 color="white", fontsize=15, fontweight="bold", y=0.99)
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.6, wspace=0.4)
    top = recs[0][0] if recs else ""
    fig.add_subplot(gs[0, 0]); _panel_recommendations(plt.gca(), recs, location, month_name)
    fig.add_subplot(gs[0, 1]); _panel_model_accuracy(plt.gca(), model_accuracies)
    fig.add_subplot(gs[0, 2]); _panel_shap(plt.gca(), shap_vals, top)
    fig.add_subplot(gs[1, 0]); _panel_risk(plt.gca(), risk, top)
    fig.add_subplot(gs[1, 1]); _panel_fertilizer(plt.gca(), fert_gap, top)
    fig.add_subplot(gs[1, 2]); _panel_rotation(plt.gca(), rotation)
    fig.add_subplot(gs[2, 0]); _panel_yield(plt.gca(), yield_info, top)
    fig.add_subplot(gs[2, 1]); _panel_water(plt.gca(), water_info, top)
    fig.add_subplot(gs[2, 2]); _panel_profit(plt.gca(), profit, top)
    fig.add_subplot(gs[3, 0]); _panel_pests(plt.gca(), pests, top)
    fig.add_subplot(gs[3, 1:]); _panel_calendar(plt.gca(), cal)
    fname = f"report_{location.replace(' ','_')}_{month_name}.png"
    plt.savefig(fname, dpi=120, facecolor=fig.get_facecolor(), bbox_inches="tight")
    print(f"\n  Report saved: {fname}")
    plt.show()

def _style(ax, title):
    ax.set_facecolor("#16213e")
    ax.tick_params(colors="white", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)

def _panel_recommendations(ax, recs, location, month_name):
    _style(ax, "Top Crop Recommendations")
    crops  = [r[0] for r in recs][::-1]
    scores = [r[1] for r in recs][::-1]
    colors = ["#2ecc71","#3498db","#e67e22","#9b59b6","#e74c3c"][::-1]
    bars   = ax.barh(crops, scores, color=colors, edgecolor="#222", height=0.55)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{s}%", va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xlim(0, max(scores)+18)
    ax.set_xlabel("Confidence (%)", color="#aaa", fontsize=8)
    ax.xaxis.label.set_color("#aaa")

def _panel_model_accuracy(ax, accuracies):
    _style(ax, "ML Model Accuracy")
    names  = list(accuracies.keys())
    vals   = [accuracies[n]*100 for n in names]
    best   = max(vals)
    colors = ["#2ecc71" if v == best else "#3498db" for v in vals]
    bars   = ax.bar(range(len(names)), vals, color=colors, edgecolor="#222", width=0.5)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{v:.1f}%", ha="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ","\n") for n in names], color="white", fontsize=7)
    ax.set_ylim(0, 115)
    ax.set_ylabel("Accuracy (%)", color="#aaa", fontsize=8)
    ax.legend(handles=[mpatches.Patch(color="#2ecc71", label="Best"),
                        mpatches.Patch(color="#3498db", label="Others")],
              facecolor="#0f0f1a", labelcolor="white", fontsize=7, loc="lower right")

def _panel_shap(ax, shap_vals, crop_name):
    _style(ax, f"Why '{crop_name}'? (SHAP)")
    if not shap_vals or "error" in shap_vals:
        ax.text(0.5, 0.5, "SHAP unavailable", ha="center", va="center",
                color="#aaa", fontsize=8, transform=ax.transAxes)
        return
    items  = sorted(shap_vals.items(), key=lambda x: abs(x[1]))
    feats  = [i[0] for i in items]
    vals   = [i[1] for i in items]
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in vals]
    ax.barh(feats, vals, color=colors, edgecolor="#222", height=0.55)
    ax.axvline(0, color="#aaa", linewidth=0.8)
    ax.set_xlabel("SHAP value", color="#aaa", fontsize=7)
    ax.legend(handles=[mpatches.Patch(color="#2ecc71", label="Positive"),
                        mpatches.Patch(color="#e74c3c", label="Negative")],
              facecolor="#0f0f1a", labelcolor="white", fontsize=7)

def _panel_risk(ax, risk, crop_name):
    _style(ax, f"Climate Risk: {crop_name}")
    ax.axis("off")
    if not risk:
        ax.text(0.5, 0.5, "No risk data", ha="center", va="center",
                color="#aaa", transform=ax.transAxes)
        return
    items = [("Water Stress", risk.get("water_stress","?")),
             ("Frost Risk",   risk.get("frost_risk","?")),
             ("Heat Stress",  risk.get("heat_stress","?")),
             ("Overall Risk", risk.get("overall","?"))]
    for i, (label, level) in enumerate(items):
        color = RISK_COLOR.get(level, "#aaa")
        y = 0.82 - i * 0.22
        ax.text(0.05, y, label, color="white", fontsize=9, transform=ax.transAxes)
        ax.text(0.65, y, level.upper(), color=color, fontsize=9,
                fontweight="bold", transform=ax.transAxes)

def _panel_fertilizer(ax, fert_gap, crop_name):
    _style(ax, f"Fertilizer Gap: {crop_name}")
    ax.axis("off")
    if not fert_gap:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                color="#aaa", transform=ax.transAxes)
        return
    rows = []
    for key in ["N","P","K","ph"]:
        info = fert_gap.get(key, {})
        rows.append([key, f"{info.get('gap',0):+.1f} {info.get('unit','')}",
                     info.get("action","?")])
    col_colors = [["#1a1a2e"]*3]*len(rows)
    for i, row in enumerate(rows):
        if row[2] == "add":     col_colors[i][2] = "#1a4a1a"
        elif row[2] == "excess":col_colors[i][2] = "#4a1a1a"
        else:                   col_colors[i][2] = "#1a3a1a"
    tbl = ax.table(cellText=rows, colLabels=["Nutrient","Gap","Action"],
                   cellLoc="center", loc="center",
                   cellColours=col_colors, colColours=["#0f0f2a"]*3)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_text_props(color="white")
        cell.set_edgecolor("#333")

def _panel_rotation(ax, rotation):
    _style(ax, "Crop Rotation Advice")
    ax.axis("off")
    ax.text(0.05, 0.92, rotation.get("note",""), color="#aaa",
            fontsize=8, transform=ax.transAxes)
    good  = rotation.get("good", [])
    avoid = rotation.get("avoid", [])
    if good:
        ax.text(0.05, 0.75, "Good rotations:", color="#2ecc71",
                fontsize=8, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.60, ", ".join(good), color="white",
                fontsize=8, transform=ax.transAxes)
    if avoid:
        ax.text(0.05, 0.42, "Avoid (monoculture):", color="#e74c3c",
                fontsize=8, fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.27, ", ".join(avoid), color="white",
                fontsize=8, transform=ax.transAxes)
    if not good and not avoid:
        ax.text(0.05, 0.55, "All crops suitable for rotation.",
                color="#2ecc71", fontsize=8, transform=ax.transAxes)

def _panel_yield(ax, yield_info, crop_name):
    _style(ax, f"Yield Prediction: {crop_name}")
    ax.axis("off")
    if not yield_info:
        ax.text(0.5,0.5,"No data",ha="center",va="center",color="#aaa",transform=ax.transAxes)
        return
    rating_color = {"excellent":"#2ecc71","good":"#3498db","moderate":"#f39c12","poor":"#e74c3c"}
    color = rating_color.get(yield_info["rating"], "#aaa")
    ax.text(0.5, 0.80, f"{yield_info['predicted_ton_ha']} ton/ha",
            ha="center", color="white", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.60, f"Rating: {yield_info['rating'].upper()}",
            ha="center", color=color, fontsize=11, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.42, f"Baseline: {yield_info['baseline_ton_ha']} ton/ha",
            ha="center", color="#aaa", fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.26, f"Achieved: {yield_info['yield_pct']}% of ideal",
            ha="center", color="#aaa", fontsize=9, transform=ax.transAxes)

def _panel_water(ax, water_info, crop_name):
    _style(ax, f"Water Requirement: {crop_name}")
    ax.axis("off")
    if not water_info:
        ax.text(0.5,0.5,"No data",ha="center",va="center",color="#aaa",transform=ax.transAxes)
        return
    status_color = {
        "high irrigation needed":    "#e74c3c",
        "moderate irrigation needed":"#f39c12",
        "rainfall sufficient":       "#2ecc71",
        "drainage may be needed":    "#3498db",
    }
    color = status_color.get(water_info["status"], "#aaa")
    for i, (label, val) in enumerate([
        ("Crop ET demand", f"{water_info['crop_et_mm']} mm/month"),
        ("Rainfall",       f"{water_info['rainfall_mm']} mm/month"),
        ("Deficit",        f"{water_info['deficit_mm']:+.1f} mm"),
    ]):
        y = 0.80 - i * 0.20
        ax.text(0.05, y, label, color="#aaa",  fontsize=9, transform=ax.transAxes)
        ax.text(0.65, y, val,   color="white", fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.18, water_info["status"].upper(),
            ha="center", color=color, fontsize=9, fontweight="bold", transform=ax.transAxes)

def _panel_profit(ax, profit, crop_name):
    _style(ax, f"Profit Estimate: {crop_name}")
    ax.axis("off")
    if not profit:
        ax.text(0.5,0.5,"No data",ha="center",va="center",color="#aaa",transform=ax.transAxes)
        return
    net_color = "#2ecc71" if profit["profitable"] else "#e74c3c"
    for i, (label, val) in enumerate([
        ("Market price",    f"${profit['price_usd_ton']}/ton"),
        ("Expected yield",  f"{profit['yield_ton_ha']} ton/ha"),
        ("Gross revenue",   f"${profit['gross_revenue']}"),
        ("Production cost", f"${profit['production_cost']}"),
    ]):
        y = 0.88 - i * 0.18
        ax.text(0.05, y, label, color="#aaa",  fontsize=8, transform=ax.transAxes)
        ax.text(0.65, y, val,   color="white", fontsize=8, fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.10, "Net Profit:", color="#aaa", fontsize=9, transform=ax.transAxes)
    ax.text(0.50, 0.10, f"${profit['net_profit']}",
            color=net_color, fontsize=12, fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, -0.02, f"Source: {profit['price_source']}",
            color="#555", fontsize=6, transform=ax.transAxes)

def _panel_pests(ax, pests, crop_name):
    _style(ax, f"Pest & Disease Risk: {crop_name}")
    ax.axis("off")
    if not pests:
        ax.text(0.5,0.5,"No data",ha="center",va="center",color="#aaa",transform=ax.transAxes)
        return
    sev_color = {"high":"#e74c3c","medium":"#f39c12","low":"#2ecc71"}
    for i, threat in enumerate(pests[:4]):
        y     = 0.85 - i * 0.22
        color = sev_color.get(threat["severity"], "#aaa")
        ax.text(0.03, y,      f"[{threat['severity'].upper()}] {threat['name']}",
                color=color, fontsize=8, fontweight="bold", transform=ax.transAxes)
        ax.text(0.03, y-0.09, threat["reason"],
                color="#888", fontsize=6.5, transform=ax.transAxes)

def _panel_calendar(ax, cal):
    _style(ax, "12-Month Crop Calendar")
    months = list(cal.keys())
    crops  = [cal[m][0] for m in months]
    confs  = [cal[m][1] for m in months]
    unique_crops = list(dict.fromkeys(crops))
    palette    = plt.cm.get_cmap("tab20", max(len(unique_crops), 1))
    crop_color = {c: palette(i) for i, c in enumerate(unique_crops)}
    bars = ax.bar(months, confs, color=[crop_color[c] for c in crops],
                  edgecolor="#222", width=0.7)
    for bar, crop, conf in zip(bars, crops, confs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                crop, ha="center", va="bottom", color="white", fontsize=7, rotation=30)
    ax.set_ylim(0, 120)
    ax.set_ylabel("Confidence (%)", color="#aaa", fontsize=8)
    ax.set_xticklabels(months, color="white", fontsize=8, rotation=20)
    handles = [mpatches.Patch(color=crop_color[c], label=c) for c in unique_crops]
    ax.legend(handles=handles, facecolor="#0f0f1a", labelcolor="white",
              fontsize=7, loc="upper right", ncol=4)

def print_box(lines, width=58):
    print("  +" + "-"*width + "+")
    for line in lines:
        print(f"  | {line:<{width-1}}|")
    print("  +" + "-"*width + "+")

def main():
    print("\n  ADVANCED CROP RECOMMENDATION SYSTEM")
    print("  Location + Month -> Weather + Soil + ML + SHAP + Risk\n")

    print("  Loading and training ML models (RF, SVM, LR, Neural Net)...")
    df = build_training_data()
    trained_models, best_name, scaler, le, model_accuracies = train_models(df)
    best_model = trained_models[best_name]
    rf_model   = trained_models["Random Forest"]
    print("  Models ready.\n")

    previous_crop = None

    while True:
        print("-" * 62)
        location_input = input("\n  Enter Location  (city / state / country) : ").strip()
        if location_input.lower() in ("quit","exit","q",""):
            print("\n  Goodbye!\n"); break

        month_input = input("  Enter Month or Season (e.g. March / Kharif / Winter) : ").strip()
        if not month_input:
            print("  Please enter a month or season."); continue

        prev_input = input("  Previous crop grown here? (press Enter to skip) : ").strip()
        if prev_input:
            previous_crop = prev_input.strip().title()

        try:
            month      = parse_month(month_input)
            month_name = MONTH_NAMES[month]
            season     = get_season(month)

            print(f"\n  Locating '{location_input}'...")
            lat, lon, address = geocode(location_input)
            print(f"  Found    : {address[:72]}")
            print(f"  Lat/Lon  : {lat:.4f}, {lon:.4f}  |  Season: {season}")

            print(f"  Fetching historical weather for {month_name}...")
            weather = fetch_weather(lat, lon, month)
            print(f"  Temp: {weather['temperature']}C  Humidity: {weather['humidity']}%  "
                  f"Rainfall: {weather['rainfall']}mm  [{weather['source']}]")

            N, P, K, ph = get_soil_params(address)
            print(f"  Soil  N:{N}  P:{P}  K:{K}  pH:{ph}")

            recs = recommend(best_model, scaler, le, N, P, K,
                             weather["temperature"], weather["humidity"],
                             ph, weather["rainfall"], top_n=5)

            print("  Computing SHAP explanation...")
            shap_vals  = explain_shap(rf_model, scaler, le, N, P, K,
                                      weather["temperature"], weather["humidity"],
                                      ph, weather["rainfall"], recs[0][0])
            risk       = climate_risk(recs[0][0], weather)
            fert       = fertilizer_gap(recs[0][0], N, P, K, ph)
            rotation   = rotation_advice(previous_crop, recs)
            yield_info = predict_yield(recs[0][0], N, P, K,
                                       weather["temperature"], weather["humidity"],
                                       ph, weather["rainfall"])
            water_info = water_requirement(recs[0][0], weather["rainfall"])
            pests      = pest_disease_risk(recs[0][0],
                                           weather["temperature"], weather["humidity"])
            profit     = profit_estimate(recs[0][0], yield_info["predicted_ton_ha"])

            gen_cal = input("\n  Generate 12-month crop calendar? (y/n) : ").strip().lower()
            if gen_cal == "y":
                print("  Building calendar...")
                cal = crop_calendar(lat, lon, address, best_model, scaler, le)
            else:
                cal = {m: ("--", 0) for m in MONTH_NAMES[1:]}

            # Terminal output
            print()
            print_box(
                [f"TOP RECOMMENDATIONS  |  {location_input}  |  {month_name}",
                 "-"*56,
                 f"{'Rank':<5} {'Crop':<18} {'Confidence':>12}",
                 "-"*56] +
                [f"{i+1:<5} {crop:<18} {conf:>10.1f}%" for i,(crop,conf) in enumerate(recs)]
            )

            if shap_vals and "error" not in shap_vals:
                print("\n  WHY THIS CROP? (SHAP)")
                for feat, val in sorted(shap_vals.items(), key=lambda x: abs(x[1]), reverse=True):
                    print(f"    {feat:<14} {val:+.4f}")

            if risk:
                print(f"\n  CLIMATE RISK for {recs[0][0]}:")
                for k, v in risk.items():
                    print(f"    {k:<16} {v.upper()}")

            if fert:
                print(f"\n  FERTILIZER GAP for {recs[0][0]}:")
                for nutrient, info in fert.items():
                    print(f"    {nutrient:<6} gap: {info.get('gap',0):+.1f} {info.get('unit','')}  -> {info.get('action','?')}")

            if rotation.get("good") or rotation.get("avoid"):
                print(f"\n  ROTATION ({rotation['note']}):")
                if rotation["good"]:  print(f"    Good : {', '.join(rotation['good'])}")
                if rotation["avoid"]: print(f"    Avoid: {', '.join(rotation['avoid'])}")

            print(f"\n  YIELD: {yield_info['predicted_ton_ha']} ton/ha  "
                  f"({yield_info['rating'].upper()}, {yield_info['yield_pct']}% of ideal)")
            print(f"  WATER: deficit {water_info['deficit_mm']:+.1f}mm  -> {water_info['status']}")
            print(f"  PROFIT: ${profit['net_profit']}  ({'PROFITABLE' if profit['profitable'] else 'LOSS'})")

            print(f"\n  PESTS for {recs[0][0]}:")
            for t in pests:
                print(f"    [{t['severity'].upper():<6}] {t['name']}")

            if gen_cal == "y":
                print("\n  12-MONTH CALENDAR:")
                for mon, (crop, conf) in cal.items():
                    print(f"    {mon:<12} {crop:<18} {conf:.1f}%")

            show_full_report(recs, location_input, month_name, weather,
                             model_accuracies, shap_vals, risk, fert, rotation, cal,
                             yield_info, water_info, pests, profit)

        except ValueError as e:
            print(f"\n  [Error] {e}")
        except Exception as e:
            print(f"\n  [Error] {e}")
            import traceback; traceback.print_exc()

        if input("\n  Try another location? (y/n) : ").strip().lower() != "y":
            print("\n  Goodbye!\n"); break

if __name__ == "__main__":
    main()
