"""
market.py
Market price fetching + profit estimation.
Uses free public APIs with fallback to curated price table.
"""

import requests

# ── Fallback price table (USD per ton, approximate global averages) ───────────
FALLBACK_PRICES_USD_TON = {
    "Rice":       420,  "Wheat":      220,  "Maize":      200,
    "Chickpea":   800,  "Lentil":     700,  "Cotton":    1600,
    "Sugarcane":   35,  "Soybean":    480,  "Groundnut":  900,
    "Sunflower":  450,  "Banana":     300,  "Mango":      600,
    "Coffee":    2800,  "Tea":        2500, "Jute":       350,
    "Mustard":    550,  "Potato":     200,  "Tomato":     350,
    "Onion":      250,  "Garlic":    1200,  "Watermelon": 180,
    "Grapes":     900,
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
        "Rice": 400, "Wheat": 300, "Maize": 350, "Cotton": 600,
        "Sugarcane": 500, "Soybean": 280, "Groundnut": 350,
        "Potato": 800, "Tomato": 700, "Onion": 500, "Garlic": 600,
        "Banana": 900, "Coffee": 1200, "Tea": 1000,
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
