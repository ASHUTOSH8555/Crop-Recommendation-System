"""
market.py
Market price fetching + profit estimation.
Uses free public APIs with fallback to curated price table.
"""

import requests

# ── Fallback price table (USD per ton, approximate global averages) ───────────
FALLBACK_PRICES_USD_TON = {
    # CSV crops (22)
    "Rice": 420,       "Maize": 200,       "Chickpea": 800,
    "Kidneybeans": 900,"Pigeonpeas": 750,  "Mothbeans": 700,
    "Mungbean": 850,   "Blackgram": 800,   "Lentil": 700,
    "Pomegranate": 900,"Banana": 300,      "Mango": 600,
    "Grapes": 900,     "Watermelon": 180,  "Muskmelon": 250,
    "Apple": 700,      "Orange": 400,      "Papaya": 350,
    "Coconut": 280,    "Cotton": 1600,     "Jute": 350,
    "Coffee": 2800,
    # Extra crops
    "Wheat": 220,      "Sugarcane": 35,    "Soybean": 480,
    "Groundnut": 900,  "Sunflower": 450,   "Tea": 2500,
    "Mustard": 550,    "Potato": 200,      "Tomato": 350,
    "Onion": 250,      "Garlic": 1200,
}

def fetch_market_price(crop: str) -> dict:
    """
    Try to fetch live price from World Bank Commodity API.
    Falls back to curated table if unavailable.
    Returns price in USD/ton and source label.
    """
    # World Bank commodity price API (free, no key)
    WB_CODES = {
        "Rice":      "PRICENPQ",   # Rice, Thai 5%
        "Wheat":     "PWHEAMT",    # Wheat, US HRW
        "Maize":     "PMAIZMMT",   # Maize
        "Soybean":   "PSOYB",      # Soybeans
        "Cotton":    "PCOTTIND",   # Cotton
        "Sugarcane": "PSUGAUSA",   # Sugar (proxy)
        "Coffee":    "PCOFFOTM",   # Coffee, other mild
        "Tea":       "PTEA",       # Tea
    }
    code = WB_CODES.get(crop)
    if code:
        try:
            url = f"https://api.worldbank.org/v2/en/indicator/{code}?format=json&mrv=1"
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and len(data) > 1:
                entries = data[1]
                if entries:
                    val = entries[0].get("value")
                    if val:
                        # World Bank returns in USD/mt (metric ton)
                        return {
                            "price_usd_ton": round(float(val), 2),
                            "source": "World Bank Commodity API (live)",
                        }
        except Exception:
            pass

    # Fallback
    price = FALLBACK_PRICES_USD_TON.get(crop, 300)
    return {
        "price_usd_ton": price,
        "source": "curated average (2024)",
    }


def profit_estimate(crop: str, yield_ton_ha: float, area_ha: float = 1.0) -> dict:
    """
    Estimate gross profit for a given crop, yield, and area.
    Returns price, gross revenue, estimated cost, and net profit.
    """
    price_info = fetch_market_price(crop)
    price      = price_info["price_usd_ton"]

    gross_revenue = round(yield_ton_ha * area_ha * price, 2)

    # Rough production cost estimates (USD/ha) — agronomic averages
    COST_PER_HA = {
        # CSV crops
        "Rice": 400,       "Maize": 350,       "Chickpea": 250,
        "Kidneybeans": 300,"Pigeonpeas": 250,  "Mothbeans": 200,
        "Mungbean": 250,   "Blackgram": 250,   "Lentil": 220,
        "Pomegranate": 800,"Banana": 900,      "Mango": 600,
        "Grapes": 1200,    "Watermelon": 400,  "Muskmelon": 380,
        "Apple": 1000,     "Orange": 700,      "Papaya": 500,
        "Coconut": 400,    "Cotton": 600,      "Jute": 300,
        "Coffee": 1200,
        # Extra crops
        "Wheat": 300,      "Sugarcane": 500,   "Soybean": 280,
        "Groundnut": 350,  "Sunflower": 300,   "Tea": 1000,
        "Mustard": 250,    "Potato": 800,      "Tomato": 700,
        "Onion": 500,      "Garlic": 600,
    }
    cost       = COST_PER_HA.get(crop, 400) * area_ha
    net_profit = round(gross_revenue - cost, 2)

    return {
        "price_usd_ton":   price,
        "price_source":    price_info["source"],
        "yield_ton_ha":    round(yield_ton_ha, 2),
        "area_ha":         area_ha,
        "gross_revenue":   gross_revenue,
        "production_cost": round(cost, 2),
        "net_profit":      net_profit,
        "profitable":      net_profit > 0,
    }
