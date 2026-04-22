"""
app.py  —  Flask web server for the Crop Recommendation System
Run:  python app.py
Open: http://localhost:5000
"""

import io
import base64
import traceback
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — must be before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec

from flask import Flask, render_template, request, jsonify

from data_pipeline import parse_month, geocode, fetch_weather, get_soil_params, MONTH_NAMES, get_season
from models       import build_training_data, train_models, recommend, explain_shap, FEATURES
from advisor      import (rotation_advice, climate_risk, fertilizer_gap,
                          crop_calendar, predict_yield, water_requirement, pest_disease_risk)
from market       import profit_estimate
from write_csv    import write_result, summarize_results

app = Flask(__name__)

# ── Boot: train / load models once at startup ─────────────────────────────────
print("Loading models…")
_df                                          = build_training_data()
_trained, _best_name, _scaler, _le, _accs   = train_models(_df)
_best_model                                  = _trained[_best_name]
_rf_model                                    = _trained["Random Forest"]
print(f"Models ready  (best: {_best_name})")

RISK_COLOR = {"low": "#2ecc71", "medium": "#f39c12", "high": "#e74c3c"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110,
                facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return b64


def _build_report_figure(recs, location, month_name, weather,
                          shap_vals, risk, fert, rotation,
                          yield_info, water_info, pests, profit, cal):
    fig = plt.figure(figsize=(22, 16), facecolor="#0f0f1a")
    fig.suptitle(f"Crop Report  |  {location}  |  {month_name}",
                 color="white", fontsize=14, fontweight="bold", y=0.99)
    gs  = gridspec.GridSpec(4, 3, figure=fig, hspace=0.65, wspace=0.42)
    top = recs[0][0] if recs else ""

    def _style(ax, title):
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="white", labelsize=8)
        for sp in ax.spines.values(): sp.set_color("#333")
        ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=6)

    # Panel 1 — recommendations
    ax = fig.add_subplot(gs[0, 0]); _style(ax, "Top Crop Recommendations")
    crops  = [r[0] for r in recs][::-1]; scores = [r[1] for r in recs][::-1]
    colors = ["#2ecc71","#3498db","#e67e22","#9b59b6","#e74c3c"][::-1]
    bars   = ax.barh(crops, scores, color=colors, edgecolor="#222", height=0.55)
    for bar, s in zip(bars, scores):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{s}%", va="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xlim(0, max(scores)+18); ax.set_xlabel("Confidence (%)", color="#aaa", fontsize=8)

    # Panel 2 — model accuracy
    ax = fig.add_subplot(gs[0, 1]); _style(ax, "ML Model Accuracy")
    names = list(_accs.keys()); vals = [_accs[n]*100 for n in names]
    best  = max(vals)
    clrs  = ["#2ecc71" if v == best else "#3498db" for v in vals]
    bars2 = ax.bar(range(len(names)), vals, color=clrs, edgecolor="#222", width=0.5)
    for bar, v in zip(bars2, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.3,
                f"{v:.1f}%", ha="center", color="white", fontsize=8, fontweight="bold")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([n.replace(" ","\n") for n in names], color="white", fontsize=7)
    ax.set_ylim(0, 115); ax.set_ylabel("Accuracy (%)", color="#aaa", fontsize=8)

    # Panel 3 — SHAP
    ax = fig.add_subplot(gs[0, 2]); _style(ax, f"Why '{top}'? (SHAP)")
    if shap_vals and "error" not in shap_vals:
        items = sorted(shap_vals.items(), key=lambda x: abs(x[1]))
        feats = [i[0] for i in items]; svals = [i[1] for i in items]
        sc    = ["#2ecc71" if v > 0 else "#e74c3c" for v in svals]
        ax.barh(feats, svals, color=sc, edgecolor="#222", height=0.55)
        ax.axvline(0, color="#aaa", linewidth=0.8)
        ax.set_xlabel("SHAP value", color="#aaa", fontsize=7)
    else:
        ax.text(0.5, 0.5, "SHAP unavailable", ha="center", va="center",
                color="#aaa", fontsize=8, transform=ax.transAxes)

    # Panel 4 — climate risk
    ax = fig.add_subplot(gs[1, 0]); _style(ax, f"Climate Risk: {top}"); ax.axis("off")
    items4 = [("Water Stress", risk.get("water_stress","?")),
              ("Frost Risk",   risk.get("frost_risk","?")),
              ("Heat Stress",  risk.get("heat_stress","?")),
              ("Overall Risk", risk.get("overall","?"))]
    for i, (lbl, lvl) in enumerate(items4):
        y = 0.82 - i*0.22
        ax.text(0.05, y, lbl,        color="white",                    fontsize=9, transform=ax.transAxes)
        ax.text(0.65, y, lvl.upper(),color=RISK_COLOR.get(lvl,"#aaa"), fontsize=9,
                fontweight="bold", transform=ax.transAxes)

    # Panel 5 — fertilizer
    ax = fig.add_subplot(gs[1, 1]); _style(ax, f"Fertilizer Gap: {top}"); ax.axis("off")
    if fert:
        rows = [[k, f"{fert[k].get('gap',0):+.1f} {fert[k].get('unit','')}",
                 fert[k].get("action","?")] for k in ["N","P","K","ph"]]
        cc   = [["#1a1a2e"]*3]*4
        for i, row in enumerate(rows):
            cc[i][2] = "#1a4a1a" if row[2]=="add" else ("#4a1a1a" if row[2]=="excess" else "#1a3a1a")
        tbl = ax.table(cellText=rows, colLabels=["Nutrient","Gap","Action"],
                       cellLoc="center", loc="center", cellColours=cc, colColours=["#0f0f2a"]*3)
        tbl.auto_set_font_size(False); tbl.set_fontsize(8); tbl.scale(1, 1.6)
        for (r,c), cell in tbl.get_celld().items():
            cell.set_text_props(color="white"); cell.set_edgecolor("#333")

    # Panel 6 — rotation
    ax = fig.add_subplot(gs[1, 2]); _style(ax, "Crop Rotation Advice"); ax.axis("off")
    ax.text(0.05, 0.92, rotation.get("note",""), color="#aaa", fontsize=8, transform=ax.transAxes)
    good = rotation.get("good",[]); avoid = rotation.get("avoid",[])
    if good:
        ax.text(0.05, 0.75, "Good rotations:", color="#2ecc71", fontsize=8,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.60, ", ".join(good), color="white", fontsize=8, transform=ax.transAxes)
    if avoid:
        ax.text(0.05, 0.42, "Avoid (monoculture):", color="#e74c3c", fontsize=8,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.05, 0.27, ", ".join(avoid), color="white", fontsize=8, transform=ax.transAxes)
    if not good and not avoid:
        ax.text(0.05, 0.55, "All crops suitable for rotation.",
                color="#2ecc71", fontsize=8, transform=ax.transAxes)

    # Panel 7 — yield
    ax = fig.add_subplot(gs[2, 0]); _style(ax, f"Yield Prediction: {top}"); ax.axis("off")
    rc = {"excellent":"#2ecc71","good":"#3498db","moderate":"#f39c12","poor":"#e74c3c"}
    ax.text(0.5, 0.80, f"{yield_info['predicted_ton_ha']} ton/ha",
            ha="center", color="white", fontsize=18, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.60, f"Rating: {yield_info['rating'].upper()}",
            ha="center", color=rc.get(yield_info['rating'],"#aaa"), fontsize=11,
            fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.42, f"Baseline: {yield_info['baseline_ton_ha']} ton/ha",
            ha="center", color="#aaa", fontsize=9, transform=ax.transAxes)
    ax.text(0.5, 0.26, f"Achieved: {yield_info['yield_pct']}% of ideal",
            ha="center", color="#aaa", fontsize=9, transform=ax.transAxes)

    # Panel 8 — water
    ax = fig.add_subplot(gs[2, 1]); _style(ax, f"Water Requirement: {top}"); ax.axis("off")
    sc2 = {"high irrigation needed":"#e74c3c","moderate irrigation needed":"#f39c12",
           "rainfall sufficient":"#2ecc71","drainage may be needed":"#3498db"}
    for i, (lbl, val) in enumerate([
        ("Crop ET demand", f"{water_info['crop_et_mm']} mm/month"),
        ("Rainfall",       f"{water_info['rainfall_mm']} mm/month"),
        ("Deficit",        f"{water_info['deficit_mm']:+.1f} mm"),
    ]):
        y = 0.80 - i*0.20
        ax.text(0.05, y, lbl, color="#aaa",  fontsize=9, transform=ax.transAxes)
        ax.text(0.65, y, val, color="white", fontsize=9, fontweight="bold", transform=ax.transAxes)
    ax.text(0.5, 0.18, water_info["status"].upper(),
            ha="center", color=sc2.get(water_info["status"],"#aaa"),
            fontsize=9, fontweight="bold", transform=ax.transAxes)

    # Panel 9 — profit
    ax = fig.add_subplot(gs[2, 2]); _style(ax, f"Profit Estimate: {top}"); ax.axis("off")
    nc = "#2ecc71" if profit["profitable"] else "#e74c3c"
    for i, (lbl, val) in enumerate([
        ("Market price",    f"${profit['price_usd_ton']}/ton"),
        ("Expected yield",  f"{profit['yield_ton_ha']} ton/ha"),
        ("Gross revenue",   f"${profit['gross_revenue']}"),
        ("Production cost", f"${profit['production_cost']}"),
    ]):
        y = 0.88 - i*0.18
        ax.text(0.05, y, lbl, color="#aaa",  fontsize=8, transform=ax.transAxes)
        ax.text(0.65, y, val, color="white", fontsize=8, fontweight="bold", transform=ax.transAxes)
    ax.text(0.05, 0.10, "Net Profit:", color="#aaa", fontsize=9, transform=ax.transAxes)
    ax.text(0.50, 0.10, f"${profit['net_profit']}",
            color=nc, fontsize=12, fontweight="bold", transform=ax.transAxes)

    # Panel 10 — pests
    ax = fig.add_subplot(gs[3, 0]); _style(ax, f"Pest & Disease Risk: {top}"); ax.axis("off")
    pc = {"high":"#e74c3c","medium":"#f39c12","low":"#2ecc71"}
    for i, t in enumerate(pests[:4]):
        y = 0.85 - i*0.22
        ax.text(0.03, y,      f"[{t['severity'].upper()}] {t['name']}",
                color=pc.get(t['severity'],"#aaa"), fontsize=8,
                fontweight="bold", transform=ax.transAxes)
        ax.text(0.03, y-0.09, t["reason"],
                color="#888", fontsize=6.5, transform=ax.transAxes)

    # Panel 11-12 — calendar
    ax = fig.add_subplot(gs[3, 1:]); _style(ax, "12-Month Crop Calendar")
    months = list(cal.keys()); ccrops = [cal[m][0] for m in months]
    cconfs = [cal[m][1] for m in months]
    unique = list(dict.fromkeys(ccrops))
    pal    = plt.cm.get_cmap("tab20", max(len(unique),1))
    cmap   = {c: pal(i) for i, c in enumerate(unique)}
    cbars  = ax.bar(months, cconfs, color=[cmap[c] for c in ccrops], edgecolor="#222", width=0.7)
    for bar, crop, conf in zip(cbars, ccrops, cconfs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                crop, ha="center", va="bottom", color="white", fontsize=7, rotation=30)
    ax.set_ylim(0, 125); ax.set_ylabel("Confidence (%)", color="#aaa", fontsize=8)
    ax.set_xticklabels(months, color="white", fontsize=8, rotation=20)
    handles = [mpatches.Patch(color=cmap[c], label=c) for c in unique]
    ax.legend(handles=handles, facecolor="#0f0f1a", labelcolor="white",
              fontsize=7, loc="upper right", ncol=4)

    return fig


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/recommend", methods=["POST"])
def api_recommend():
    data          = request.get_json()
    location_input = data.get("location", "").strip()
    month_input    = data.get("month", "").strip()
    prev_crop      = data.get("previous_crop", "").strip().title() or None
    gen_cal        = data.get("calendar", False)

    if not location_input or not month_input:
        return jsonify({"error": "Location and month are required."}), 400

    try:
        month      = parse_month(month_input)
        month_name = MONTH_NAMES[month]
        season     = get_season(month)

        lat, lon, address = geocode(location_input)
        weather           = fetch_weather(lat, lon, month)
        N, P, K, ph       = get_soil_params(address)

        recs       = recommend(_best_model, _scaler, _le, N, P, K,
                               weather["temperature"], weather["humidity"],
                               ph, weather["rainfall"], top_n=5)
        shap_vals  = explain_shap(_rf_model, _scaler, _le, N, P, K,
                                  weather["temperature"], weather["humidity"],
                                  ph, weather["rainfall"], recs[0][0])
        risk       = climate_risk(recs[0][0], weather)
        fert       = fertilizer_gap(recs[0][0], N, P, K, ph)
        rotation   = rotation_advice(prev_crop, recs)
        yield_info = predict_yield(recs[0][0], N, P, K,
                                   weather["temperature"], weather["humidity"],
                                   ph, weather["rainfall"])
        water_info = water_requirement(recs[0][0], weather["rainfall"])
        pests      = pest_disease_risk(recs[0][0],
                                       weather["temperature"], weather["humidity"])
        profit     = profit_estimate(recs[0][0], yield_info["predicted_ton_ha"])

        if gen_cal:
            cal = crop_calendar(lat, lon, address, _best_model, _scaler, _le)
        else:
            cal = {m: ("--", 0) for m in MONTH_NAMES[1:]}

        # Build report image
        fig    = _build_report_figure(recs, location_input, month_name, weather,
                                      shap_vals, risk, fert, rotation,
                                      yield_info, water_info, pests, profit, cal)
        img_b64 = _fig_to_b64(fig)

        # Save to CSV
        write_result(
            location=location_input, month_name=month_name, season=season,
            lat=lat, lon=lon, N=N, P=P, K=K, ph=ph,
            weather=weather, recs=recs, yield_info=yield_info,
            water_info=water_info, risk=risk, profit=profit,
            rotation=rotation, fert_gap=fert, previous_crop=prev_crop,
        )

        return jsonify({
            "location":   address,
            "month":      month_name,
            "season":     season,
            "lat":        round(lat, 4),
            "lon":        round(lon, 4),
            "weather":    weather,
            "soil":       {"N": N, "P": P, "K": K, "ph": ph},
            "best_model": _best_name,
            "recs":       [{"crop": c, "confidence": s} for c, s in recs],
            "shap":       shap_vals,
            "risk":       risk,
            "fertilizer": fert,
            "rotation":   rotation,
            "yield":      yield_info,
            "water":      water_info,
            "pests":      pests,
            "profit":     profit,
            "calendar":   {m: {"crop": v[0], "confidence": v[1]} for m, v in cal.items()},
            "report_img": img_b64,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {e}"}), 500


@app.route("/summary")
def api_summary():
    s = summarize_results()
    return jsonify({
        "total_runs":      s.get("total", 0),
        "top_crop":        s["top_crops"][0][0] if s.get("top_crops") else None,
        "avg_net_profit":  s.get("avg_profit_usd"),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
