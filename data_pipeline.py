"""
data_pipeline.py
Handles: geocoding, real weather fetch, soil nutrient mapping
"""

import requests
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from geopy.geocoders import Nominatim

# ── Month / Season helpers ────────────────────────────────────────────────────

MONTH_NAMES = ["","January","February","March","April","May","June",
               "July","August","September","October","November","December"]

MONTH_MAP = {
    "january":1,"february":2,"march":3,"april":4,"may":5,"june":6,
    "july":7,"august":8,"september":9,"october":10,"november":11,"december":12,
    "jan":1,"feb":2,"mar":3,"apr":4,"jun":6,"jul":7,"aug":8,
    "sep":9,"oct":10,"nov":11,"dec":12,
}

SEASON_MONTHS = {
    "kharif":  [6,7,8,9],
    "rabi":    [10,11,12,1],
    "zaid":    [2,3,4,5],
    "spring":  [3,4,5],
    "summer":  [5,6,7,8],
    "monsoon": [6,7,8,9],
    "autumn":  [9,10,11],
    "winter":  [11,12,1,2],
}

def parse_month(text: str) -> int:
    t = text.strip().lower()
    if t in MONTH_MAP:
        return MONTH_MAP[t]
    if t in SEASON_MONTHS:
        months = SEASON_MONTHS[t]
        return months[len(months)//2]
    try:
        m = int(t)
        if 1 <= m <= 12:
            return m
    except ValueError:
        pass
    raise ValueError(f"Cannot parse '{text}' as a month or season.")

def get_season(month: int) -> str:
    for season, months in SEASON_MONTHS.items():
        if month in months:
            return season.capitalize()
    return "Unknown"

# ── Geocoding ─────────────────────────────────────────────────────────────────

# Built-in location DB — works fully offline
LOCATION_DB = {
    "bahraich":       (27.5742, 81.5957, "Bahraich, Uttar Pradesh, India"),
    "lucknow":        (26.8467, 80.9462, "Lucknow, Uttar Pradesh, India"),
    "delhi":          (28.6139, 77.2090, "Delhi, India"),
    "new delhi":      (28.6139, 77.2090, "Delhi, India"),
    "mumbai":         (19.0760, 72.8777, "Mumbai, Maharashtra, India"),
    "kolkata":        (22.5726, 88.3639, "Kolkata, West Bengal, India"),
    "chennai":        (13.0827, 80.2707, "Chennai, Tamil Nadu, India"),
    "bangalore":      (12.9716, 77.5946, "Bangalore, Karnataka, India"),
    "bengaluru":      (12.9716, 77.5946, "Bangalore, Karnataka, India"),
    "hyderabad":      (17.3850, 78.4867, "Hyderabad, Telangana, India"),
    "pune":           (18.5204, 73.8567, "Pune, Maharashtra, India"),
    "ahmedabad":      (23.0225, 72.5714, "Ahmedabad, Gujarat, India"),
    "jaipur":         (26.9124, 75.7873, "Jaipur, Rajasthan, India"),
    "varanasi":       (25.3176, 82.9739, "Varanasi, Uttar Pradesh, India"),
    "agra":           (27.1767, 78.0081, "Agra, Uttar Pradesh, India"),
    "kanpur":         (26.4499, 80.3319, "Kanpur, Uttar Pradesh, India"),
    "prayagraj":      (25.4358, 81.8463, "Prayagraj, Uttar Pradesh, India"),
    "allahabad":      (25.4358, 81.8463, "Prayagraj, Uttar Pradesh, India"),
    "gorakhpur":      (26.7606, 83.3732, "Gorakhpur, Uttar Pradesh, India"),
    "meerut":         (28.9845, 77.7064, "Meerut, Uttar Pradesh, India"),
    "ghaziabad":      (28.6692, 77.4538, "Ghaziabad, Uttar Pradesh, India"),
    "noida":          (28.5355, 77.3910, "Noida, Uttar Pradesh, India"),
    "mathura":        (27.4924, 77.6737, "Mathura, Uttar Pradesh, India"),
    "ayodhya":        (26.7922, 82.1998, "Ayodhya, Uttar Pradesh, India"),
    "faizabad":       (26.7922, 82.1998, "Ayodhya, Uttar Pradesh, India"),
    "sitapur":        (27.5630, 80.6830, "Sitapur, Uttar Pradesh, India"),
    "lakhimpur":      (27.9500, 80.7833, "Lakhimpur Kheri, Uttar Pradesh, India"),
    "gonda":          (27.1333, 81.9667, "Gonda, Uttar Pradesh, India"),
    "basti":          (26.8000, 82.7333, "Basti, Uttar Pradesh, India"),
    "patna":          (25.5941, 85.1376, "Patna, Bihar, India"),
    "bhopal":         (23.2599, 77.4126, "Bhopal, Madhya Pradesh, India"),
    "indore":         (22.7196, 75.8577, "Indore, Madhya Pradesh, India"),
    "nagpur":         (21.1458, 79.0882, "Nagpur, Maharashtra, India"),
    "surat":          (21.1702, 72.8311, "Surat, Gujarat, India"),
    "amritsar":       (31.6340, 74.8723, "Amritsar, Punjab, India"),
    "ludhiana":       (30.9010, 75.8573, "Ludhiana, Punjab, India"),
    "chandigarh":     (30.7333, 76.7794, "Chandigarh, India"),
    "dehradun":       (30.3165, 78.0322, "Dehradun, Uttarakhand, India"),
    "shimla":         (31.1048, 77.1734, "Shimla, Himachal Pradesh, India"),
    "guwahati":       (26.1445, 91.7362, "Guwahati, Assam, India"),
    "bhubaneswar":    (20.2961, 85.8245, "Bhubaneswar, Odisha, India"),
    "visakhapatnam":  (17.6868, 83.2185, "Visakhapatnam, Andhra Pradesh, India"),
    "coimbatore":     (11.0168, 76.9558, "Coimbatore, Tamil Nadu, India"),
    "kochi":          (9.9312,  76.2673, "Kochi, Kerala, India"),
    "mysore":         (12.2958, 76.6394, "Mysore, Karnataka, India"),
    "punjab":         (31.1471, 75.3412, "Punjab, India"),
    "haryana":        (29.0588, 76.0856, "Haryana, India"),
    "uttar pradesh":  (26.8467, 80.9462, "Uttar Pradesh, India"),
    "bihar":          (25.5941, 85.1376, "Bihar, India"),
    "rajasthan":      (27.0238, 74.2179, "Rajasthan, India"),
    "gujarat":        (22.2587, 71.1924, "Gujarat, India"),
    "maharashtra":    (19.7515, 75.7139, "Maharashtra, India"),
    "karnataka":      (15.3173, 75.7139, "Karnataka, India"),
    "tamil nadu":     (11.1271, 78.6569, "Tamil Nadu, India"),
    "kerala":         (10.8505, 76.2711, "Kerala, India"),
    "west bengal":    (22.9868, 87.8550, "West Bengal, India"),
    "odisha":         (20.9517, 85.0985, "Odisha, India"),
    "assam":          (26.2006, 92.9376, "Assam, India"),
    "madhya pradesh": (23.4734, 77.9470, "Madhya Pradesh, India"),
    "andhra pradesh": (15.9129, 79.7400, "Andhra Pradesh, India"),
    "telangana":      (18.1124, 79.0193, "Telangana, India"),
    "jharkhand":      (23.6102, 85.2799, "Jharkhand, India"),
    "chhattisgarh":   (21.2787, 81.8661, "Chhattisgarh, India"),
    "new york":       (40.7128, -74.0060, "New York, USA"),
    "london":         (51.5074,  -0.1278, "London, UK"),
    "paris":          (48.8566,   2.3522, "Paris, France"),
    "beijing":        (39.9042, 116.4074, "Beijing, China"),
    "tokyo":          (35.6762, 139.6503, "Tokyo, Japan"),
    "sydney":         (-33.8688,151.2093, "Sydney, Australia"),
    "cairo":          (30.0444,  31.2357, "Cairo, Egypt"),
    "nairobi":        (-1.2921,  36.8219, "Nairobi, Kenya"),
    "karachi":        (24.8607,  67.0011, "Karachi, Pakistan"),
    "lahore":         (31.5204,  74.3587, "Lahore, Pakistan"),
    "islamabad":      (33.6844,  73.0479, "Islamabad, Pakistan"),
    "dhaka":          (23.8103,  90.4125, "Dhaka, Bangladesh"),
    "bangkok":        (13.7563, 100.5018, "Bangkok, Thailand"),
    "jakarta":        (-6.2088, 106.8456, "Jakarta, Indonesia"),
    "manila":         (14.5995, 120.9842, "Manila, Philippines"),
    "hanoi":          (21.0285, 105.8542, "Hanoi, Vietnam"),
    "yangon":         (16.8661,  96.1951, "Yangon, Myanmar"),
    "california":     (36.7783,-119.4179, "California, USA"),
    "texas":          (31.9686, -99.9018, "Texas, USA"),
    "iowa":           (41.8780, -93.0977, "Iowa, USA"),
    "brazil":         (-14.235, -51.9253, "Brazil"),
    "argentina":      (-38.416, -63.6167, "Argentina"),
    "nigeria":        (9.0820,   8.6753,  "Nigeria"),
    "ethiopia":       (9.1450,  40.4897,  "Ethiopia"),
    "ukraine":        (48.3794,  31.1656, "Ukraine"),
    "russia":         (61.5240, 105.3188, "Russia"),
    "canada":         (56.1304,-106.3468, "Canada"),
    "australia":      (-25.274, 133.7751, "Australia"),
    "france":         (46.2276,   2.2137, "France"),
    "germany":        (51.1657,  10.4515, "Germany"),
    "china":          (35.8617, 104.1954, "China"),
    "indonesia":      (-0.7893, 113.9213, "Indonesia"),
    "pakistan":       (30.3753,  69.3451, "Pakistan"),
    "bangladesh":     (23.6850,  90.3563, "Bangladesh"),
    "myanmar":        (21.9162,  95.9560, "Myanmar"),
    "thailand":       (15.8700, 100.9925, "Thailand"),
    "vietnam":        (14.0583, 108.2772, "Vietnam"),
    "philippines":    (12.8797, 121.7740, "Philippines"),
}

def geocode(location: str) -> tuple:
    """
    Geocode a location. Uses built-in DB first (works offline),
    then falls back to live Nominatim if available.
    """
    key = location.strip().lower()

    # Exact match
    if key in LOCATION_DB:
        lat, lon, address = LOCATION_DB[key]
        print(f"  [offline DB match]")
        return lat, lon, address

    # Partial match
    for db_key, (lat, lon, address) in LOCATION_DB.items():
        if db_key in key or key in db_key:
            print(f"  [offline DB partial match: {db_key}]")
            return lat, lon, address

    # Live geocoding
    try:
        geolocator = Nominatim(user_agent="crop_recommender_v2")
        result = geolocator.geocode(location, timeout=10)
        if result:
            return result.latitude, result.longitude, result.address
    except Exception:
        pass

    raise ValueError(
        f"Location '{location}' not found.\n"
        f"  No internet or location not in offline DB.\n"
        f"  Try nearby cities: Lucknow, Delhi, Patna, Jaipur, Mumbai, etc."
    )

# ── Weather ───────────────────────────────────────────────────────────────────

def fetch_weather(lat: float, lon: float, month: int) -> dict:
    """Fetch 10-year historical climate normals from Open-Meteo archive."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude":   lat,
        "longitude":  lon,
        "start_date": "2013-01-01",
        "end_date":   "2023-12-31",
        "daily":      "temperature_2m_mean,precipitation_sum,relative_humidity_2m_mean",
        "timezone":   "auto",
    }
    try:
        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        daily = resp.json().get("daily", {})
        times  = daily.get("time", [])
        temps  = daily.get("temperature_2m_mean", [])
        rains  = daily.get("precipitation_sum", [])
        humids = daily.get("relative_humidity_2m_mean", [])

        t_vals, r_vals, h_vals = [], [], []
        for i, t in enumerate(times):
            if int(t.split("-")[1]) == month:
                if i < len(temps)  and temps[i]  is not None: t_vals.append(temps[i])
                if i < len(rains)  and rains[i]  is not None: r_vals.append(rains[i])
                if i < len(humids) and humids[i] is not None: h_vals.append(humids[i])

        if t_vals:
            # Also compute variance for risk scoring
            return {
                "temperature":      round(float(np.mean(t_vals)), 2),
                "temp_std":         round(float(np.std(t_vals)), 2),
                "rainfall":         round(float(np.sum(r_vals) / 10), 2),
                "rain_std":         round(float(np.std(r_vals)), 2),
                "humidity":         round(float(np.mean(h_vals)), 2),
                "source":           "Open-Meteo archive (2013-2023)",
            }
    except Exception as e:
        print(f"  [weather] {e} — using seasonal estimate.")
    return _fallback_weather(month)

def _fallback_weather(month: int) -> dict:
    if month in [12,1,2]:
        return {"temperature":15.0,"temp_std":3.0,"rainfall":30.0,"rain_std":10.0,"humidity":55.0,"source":"fallback"}
    elif month in [3,4,5]:
        return {"temperature":28.0,"temp_std":4.0,"rainfall":50.0,"rain_std":15.0,"humidity":45.0,"source":"fallback"}
    elif month in [6,7,8,9]:
        return {"temperature":30.0,"temp_std":2.0,"rainfall":200.0,"rain_std":40.0,"humidity":85.0,"source":"fallback"}
    else:
        return {"temperature":22.0,"temp_std":3.0,"rainfall":60.0,"rain_std":20.0,"humidity":65.0,"source":"fallback"}

# ── Soil profiles ─────────────────────────────────────────────────────────────
# (N, P, K, ph)  — regional averages

SOIL_PROFILES = {
    "punjab":        (80,50,50,7.2), "haryana":       (75,48,48,7.5),
    "uttar pradesh": (70,45,45,7.0), "bihar":         (65,40,40,6.8),
    "west bengal":   (60,38,38,6.5), "odisha":        (55,35,35,6.2),
    "andhra pradesh":(50,30,30,6.0), "telangana":     (50,30,30,6.0),
    "karnataka":     (45,28,28,6.2), "tamil nadu":    (45,28,28,6.0),
    "kerala":        (40,25,25,5.8), "maharashtra":   (55,35,35,6.5),
    "gujarat":       (60,38,38,7.0), "rajasthan":     (35,20,20,7.8),
    "madhya pradesh":(65,42,42,6.8), "assam":         (50,30,30,5.5),
    "jharkhand":     (45,28,28,5.8), "chhattisgarh":  (50,32,32,6.0),
    "himachal":      (55,35,35,6.5), "uttarakhand":   (60,38,38,6.5),
    "california":    (70,45,45,6.5), "texas":         (60,38,38,7.0),
    "iowa":          (90,55,55,6.8), "florida":       (50,30,30,6.2),
    "brazil":        (55,35,35,5.8), "argentina":     (75,48,48,6.5),
    "china":         (65,42,42,6.5), "indonesia":     (45,28,28,5.5),
    "nigeria":       (40,25,25,6.0), "ethiopia":      (50,32,32,6.2),
    "france":        (70,45,45,6.8), "germany":       (75,48,48,6.5),
    "ukraine":       (80,50,50,6.8), "russia":        (65,42,42,6.5),
    "australia":     (55,35,35,6.5), "canada":        (70,45,45,6.8),
    "pakistan":      (70,45,45,7.5), "bangladesh":    (55,35,35,6.5),
    "myanmar":       (50,32,32,6.0), "thailand":      (50,30,30,5.8),
    "vietnam":       (45,28,28,5.5), "philippines":   (45,28,28,5.8),
}

DEFAULT_SOIL = (60, 38, 38, 6.5)

def get_soil_params(address: str) -> tuple:
    addr_lower = address.lower()
    for region, params in SOIL_PROFILES.items():
        if region in addr_lower:
            return params
    for region, params in SOIL_PROFILES.items():
        if any(region in w for w in addr_lower.split()):
            return params
    return DEFAULT_SOIL
