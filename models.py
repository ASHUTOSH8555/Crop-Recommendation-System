"""
models.py
ML models: Random Forest, SVM, Logistic Regression, Neural Network
+ SHAP explainability

Training data: real Crop Recommendation dataset (Crop_recommendation.csv)
augmented with synthetic samples for crops not in the CSV.
"""

import os
import pickle
import hashlib
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

_MODEL_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".model_cache.pkl")
_DATA_HASH_FILE   = os.path.join(os.path.dirname(__file__), ".data_hash.txt")


def _data_hash(df: pd.DataFrame) -> str:
    """Quick hash of training data shape + first/last rows to detect changes."""
    sig = f"{df.shape}{list(df.columns)}{df.iloc[0].tolist()}{df.iloc[-1].tolist()}"
    return hashlib.md5(sig.encode()).hexdigest()


def _save_model_cache(trained, best_name, scaler, le, accuracies, data_hash):
    payload = dict(trained=trained, best_name=best_name,
                   scaler=scaler, le=le, accuracies=accuracies)
    with open(_MODEL_CACHE_FILE, "wb") as f:
        pickle.dump(payload, f)
    with open(_DATA_HASH_FILE, "w") as f:
        f.write(data_hash)


def _load_model_cache(data_hash: str):
    """Return cached models if data hasn't changed, else None."""
    if not os.path.isfile(_MODEL_CACHE_FILE) or not os.path.isfile(_DATA_HASH_FILE):
        return None
    with open(_DATA_HASH_FILE) as f:
        cached_hash = f.read().strip()
    if cached_hash != data_hash:
        return None
    try:
        with open(_MODEL_CACHE_FILE, "rb") as f:
            p = pickle.load(f)
        return p["trained"], p["best_name"], p["scaler"], p["le"], p["accuracies"]
    except Exception:
        return None

# ── Crop profiles (N, P, K, temp, humidity, ph, rainfall) ranges ─────────────
# Used for: fertilizer gap, climate risk, yield model, pest rules
# Values derived from the real dataset statistics + agronomic references

CROP_PROFILES = {
    # --- CSV crops (22) ---
    "Rice":        dict(N=(60,100), P=(35,60),  K=(35,45),  t=(20,27),h=(80,85),ph=(5.0,8.0),r=(182,299)),
    "Maize":       dict(N=(60,100), P=(35,60),  K=(15,25),  t=(18,27),h=(55,75),ph=(5.5,7.5),r=(60,110)),
    "Chickpea":    dict(N=(20,60),  P=(55,80),  K=(75,85),  t=(17,21),h=(14,20),ph=(5.9,9.0),r=(65,95)),
    "Kidneybeans": dict(N=(0,40),   P=(55,80),  K=(15,25),  t=(15,24),h=(18,25),ph=(5.5,6.0),r=(60,150)),
    "Pigeonpeas":  dict(N=(0,40),   P=(55,80),  K=(15,25),  t=(18,37),h=(30,68),ph=(4.5,7.2),r=(90,200)),
    "Mothbeans":   dict(N=(0,40),   P=(35,60),  K=(15,25),  t=(24,32),h=(40,65),ph=(3.5,10.0),r=(30,75)),
    "Mungbean":    dict(N=(0,40),   P=(35,60),  K=(15,25),  t=(27,30),h=(80,90),ph=(6.2,7.2),r=(36,60)),
    "Blackgram":   dict(N=(20,60),  P=(55,80),  K=(15,25),  t=(25,35),h=(60,70),ph=(6.5,7.8),r=(60,75)),
    "Lentil":      dict(N=(0,40),   P=(55,80),  K=(15,25),  t=(18,30),h=(60,70),ph=(5.9,7.8),r=(35,55)),
    "Pomegranate": dict(N=(0,40),   P=(5,30),   K=(35,45),  t=(18,25),h=(85,95),ph=(5.5,7.2),r=(100,113)),
    "Banana":      dict(N=(80,120), P=(70,95),  K=(45,55),  t=(25,30),h=(75,85),ph=(5.5,6.5),r=(90,120)),
    "Mango":       dict(N=(0,40),   P=(15,40),  K=(25,35),  t=(27,36),h=(45,55),ph=(4.5,7.0),r=(89,101)),
    "Grapes":      dict(N=(0,40),   P=(120,145),K=(195,205),t=(8,42), h=(80,85),ph=(5.5,6.5),r=(65,75)),
    "Watermelon":  dict(N=(80,120), P=(5,30),   K=(45,55),  t=(24,27),h=(80,90),ph=(6.0,7.0),r=(40,60)),
    "Muskmelon":   dict(N=(80,120), P=(5,30),   K=(45,55),  t=(27,30),h=(90,95),ph=(6.0,6.8),r=(20,30)),
    "Apple":       dict(N=(0,40),   P=(120,145),K=(195,205),t=(21,24),h=(90,95),ph=(5.5,6.5),r=(100,125)),
    "Orange":      dict(N=(0,40),   P=(5,30),   K=(5,15),   t=(10,35),h=(90,95),ph=(6.0,8.0),r=(100,120)),
    "Papaya":      dict(N=(30,70),  P=(45,70),  K=(45,55),  t=(23,43),h=(90,95),ph=(6.5,7.0),r=(40,250)),
    "Coconut":     dict(N=(0,40),   P=(5,30),   K=(25,35),  t=(25,30),h=(90,100),ph=(5.5,6.5),r=(130,225)),
    "Cotton":      dict(N=(100,140),P=(35,60),  K=(15,25),  t=(22,26),h=(75,85),ph=(5.8,8.0),r=(60,100)),
    "Jute":        dict(N=(60,100), P=(35,60),  K=(35,45),  t=(23,27),h=(70,90),ph=(6.0,7.5),r=(150,200)),
    "Coffee":      dict(N=(80,120), P=(15,40),  K=(25,35),  t=(23,28),h=(50,70),ph=(6.0,7.5),r=(115,200)),
    # --- Extra crops (synthetic, kept for backward compat) ---
    "Wheat":       dict(N=(80,100), P=(40,60),  K=(40,60),  t=(10,22),h=(50,70),ph=(6.0,7.5),r=(40,100)),
    "Sugarcane":   dict(N=(100,120),P=(50,70),  K=(50,70),  t=(25,35),h=(70,85),ph=(6.0,7.5),r=(100,200)),
    "Soybean":     dict(N=(20,40),  P=(60,80),  K=(40,60),  t=(20,30),h=(60,80),ph=(6.0,7.0),r=(60,100)),
    "Groundnut":   dict(N=(15,25),  P=(55,75),  K=(25,45),  t=(25,35),h=(50,70),ph=(5.5,7.0),r=(50,100)),
    "Sunflower":   dict(N=(60,80),  P=(50,70),  K=(40,60),  t=(20,30),h=(40,60),ph=(6.0,7.5),r=(40,80)),
    "Tea":         dict(N=(80,100), P=(20,40),  K=(20,40),  t=(18,28),h=(70,90),ph=(4.5,6.0),r=(150,250)),
    "Mustard":     dict(N=(60,80),  P=(40,60),  K=(30,50),  t=(10,20),h=(50,70),ph=(6.0,7.5),r=(30,60)),
    "Potato":      dict(N=(80,100), P=(60,80),  K=(80,100), t=(15,22),h=(70,85),ph=(5.0,6.5),r=(50,100)),
    "Tomato":      dict(N=(70,90),  P=(60,80),  K=(60,80),  t=(20,28),h=(60,80),ph=(6.0,7.0),r=(40,80)),
    "Onion":       dict(N=(60,80),  P=(50,70),  K=(50,70),  t=(15,25),h=(50,70),ph=(6.0,7.5),r=(30,60)),
    "Garlic":      dict(N=(50,70),  P=(40,60),  K=(40,60),  t=(12,22),h=(50,70),ph=(6.0,7.5),r=(30,60)),
}

FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

# Ideal nutrient requirements per crop (for fertilizer gap analysis)
CROP_NUTRIENT_NEEDS = {
    crop: {
        "N":  (p["N"][0]+p["N"][1])/2,
        "P":  (p["P"][0]+p["P"][1])/2,
        "K":  (p["K"][0]+p["K"][1])/2,
        "ph": (p["ph"][0]+p["ph"][1])/2,
    }
    for crop, p in CROP_PROFILES.items()
}

# ── CSV label → display name mapping ─────────────────────────────────────────
# The CSV uses lowercase labels; we normalise to Title Case for display.
CSV_LABEL_MAP = {
    "rice": "Rice", "maize": "Maize", "chickpea": "Chickpea",
    "kidneybeans": "Kidneybeans", "pigeonpeas": "Pigeonpeas",
    "mothbeans": "Mothbeans", "mungbean": "Mungbean",
    "blackgram": "Blackgram", "lentil": "Lentil",
    "pomegranate": "Pomegranate", "banana": "Banana", "mango": "Mango",
    "grapes": "Grapes", "watermelon": "Watermelon", "muskmelon": "Muskmelon",
    "apple": "Apple", "orange": "Orange", "papaya": "Papaya",
    "coconut": "Coconut", "cotton": "Cotton", "jute": "Jute",
    "coffee": "Coffee",
}

# ── Training data ─────────────────────────────────────────────────────────────

def _load_csv_data() -> pd.DataFrame:
    """Load the real Crop_recommendation.csv dataset."""
    csv_path = os.path.join(os.path.dirname(__file__), "Crop_recommendation.csv")
    if not os.path.exists(csv_path):
        return pd.DataFrame()
    df = pd.read_csv(csv_path)
    # Normalise column names
    df.columns = [c.strip() for c in df.columns]
    # The CSV has a 'label' column; rename to 'crop' and title-case it
    if "label" in df.columns:
        df = df.rename(columns={"label": "crop"})
    df["crop"] = df["crop"].str.strip().str.lower().map(CSV_LABEL_MAP).fillna(df["crop"].str.title())
    # Keep only the 7 feature columns + crop
    needed = FEATURES + ["crop"]
    df = df[[c for c in needed if c in df.columns]].dropna()
    return df


def _synthetic_augment(n_per_crop: int = 50) -> pd.DataFrame:
    """
    Generate a small synthetic supplement for crops not in the CSV
    (Wheat, Sugarcane, Soybean, etc.) so the system still handles them.
    """
    rng = np.random.default_rng(42)
    # Only augment crops absent from the CSV
    csv_crops = set(CSV_LABEL_MAP.values())
    rows = []
    for crop, p in CROP_PROFILES.items():
        if crop in csv_crops:
            continue  # already covered by real data
        for _ in range(n_per_crop):
            rows.append({
                "N":           rng.uniform(*p["N"]),
                "P":           rng.uniform(*p["P"]),
                "K":           rng.uniform(*p["K"]),
                "temperature": rng.uniform(*p["t"]),
                "humidity":    rng.uniform(*p["h"]),
                "ph":          rng.uniform(*p["ph"]),
                "rainfall":    rng.uniform(*p["r"]),
                "crop":        crop,
            })
    return pd.DataFrame(rows)


def build_training_data() -> pd.DataFrame:
    """
    Primary training data = real CSV (2200 rows, 22 crops).
    Supplemented with synthetic rows for any extra crops defined in CROP_PROFILES.
    """
    real = _load_csv_data()
    synth = _synthetic_augment()

    if real.empty:
        print("  [Warning] Crop_recommendation.csv not found — using synthetic data only.")
        df = synth
    else:
        n_real = len(real)
        n_synth = len(synth)
        source_note = f"real CSV ({n_real} rows)"
        if n_synth > 0:
            source_note += f" + synthetic ({n_synth} rows for {len(synth['crop'].unique())} extra crops)"
        print(f"  [Data] {source_note}")
        df = pd.concat([real, synth], ignore_index=True)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)

# ── Neural Network (PyTorch) ──────────────────────────────────────────────────

def _build_nn(input_dim: int, num_classes: int):
    try:
        import torch
        import torch.nn as nn

        class CropNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(input_dim, 64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 128),       nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, 64),       nn.ReLU(),
                    nn.Linear(64, num_classes)
                )
            def forward(self, x):
                return self.net(x)

        return CropNet, torch, nn
    except ImportError:
        return None, None, None


class NeuralNetWrapper:
    """Sklearn-compatible wrapper around a PyTorch crop classifier."""
    def __init__(self, input_dim, num_classes, epochs=40, lr=1e-3):
        self.input_dim   = input_dim
        self.num_classes = num_classes
        self.epochs      = epochs
        self.lr          = lr
        self.model       = None
        self.available   = False

    def fit(self, X, y):
        CropNet, torch, nn = _build_nn(self.input_dim, self.num_classes)
        if CropNet is None:
            return self
        self.torch = torch
        self.model = CropNet()
        optimizer  = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion  = nn.CrossEntropyLoss()
        Xt = torch.tensor(X, dtype=torch.float32)
        yt = torch.tensor(y, dtype=torch.long)
        self.model.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss = criterion(self.model(Xt), yt)
            loss.backward()
            optimizer.step()
        self.available = True
        return self

    def predict(self, X):
        if not self.available:
            return np.zeros(len(X), dtype=int)
        import torch
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32))
            return out.argmax(dim=1).numpy()

    def predict_proba(self, X):
        if not self.available:
            return np.ones((len(X), self.num_classes)) / self.num_classes
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            out = self.model(torch.tensor(X, dtype=torch.float32))
            return F.softmax(out, dim=1).numpy()


# ── Train all models ──────────────────────────────────────────────────────────

def train_models(df: pd.DataFrame):
    dhash = _data_hash(df)
    cached = _load_model_cache(dhash)
    if cached:
        trained, best_name, scaler, le, accuracies = cached
        print(f"  [Cache] Loaded models from disk (best: {best_name} "
              f"{accuracies[best_name]:.4f})")
        return trained, best_name, scaler, le, accuracies

    X = df[FEATURES].values
    le = LabelEncoder()
    y  = le.fit_transform(df["crop"])
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        Xs, y, test_size=0.2, random_state=42, stratify=y
    )

    nn_model = NeuralNetWrapper(input_dim=len(FEATURES), num_classes=len(le.classes_))

    candidates = {
        "Random Forest":       RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM":                 SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Neural Network":      nn_model,
    }

    print("\n  Training models...")
    accuracies = {}
    trained    = {}
    for name, m in candidates.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        acc   = accuracy_score(y_test, preds)
        accuracies[name] = acc
        trained[name]    = m
        tag = " <-- best" if acc == max(accuracies.values()) else ""
        print(f"    {name:<22} accuracy: {acc:.4f}{tag}")

    best_name = max(accuracies, key=accuracies.get)
    print(f"\n  Best model: {best_name} ({accuracies[best_name]:.4f})")

    try:
        _save_model_cache(trained, best_name, scaler, le, accuracies, dhash)
        print("  [Cache] Models saved to disk for next run.")
    except Exception as e:
        print(f"  [Cache] Could not save: {e}")

    return trained, best_name, scaler, le, accuracies


# ── Predict ───────────────────────────────────────────────────────────────────

def recommend(model, scaler, le, N, P, K, temp, humidity, ph, rainfall, top_n=5):
    x  = np.array([[N, P, K, temp, humidity, ph, rainfall]])
    xs = scaler.transform(x)
    if hasattr(model, "predict_proba"):
        probs   = model.predict_proba(xs)[0]
        top_idx = np.argsort(probs)[::-1][:top_n]
        return [(le.classes_[i], round(float(probs[i])*100, 1)) for i in top_idx]
    pred = model.predict(xs)[0]
    return [(le.classes_[pred], 100.0)]


# ── SHAP explainability ───────────────────────────────────────────────────────

def explain_shap(rf_model, scaler, le, N, P, K, temp, humidity, ph, rainfall, top_crop: str):
    """
    Return a dict of {feature: shap_value} explaining why top_crop was recommended.
    Uses TreeExplainer on the Random Forest.
    """
    try:
        import shap
        x  = np.array([[N, P, K, temp, humidity, ph, rainfall]])
        xs = scaler.transform(x)

        explainer  = shap.TreeExplainer(rf_model)
        shap_vals  = explainer.shap_values(xs)   # shape: [n_samples, n_features, n_classes] or [n_classes, n_samples, n_features]

        crop_idx = list(le.classes_).index(top_crop)

        # Handle both shap output shapes
        sv = np.array(shap_vals)
        if sv.ndim == 3 and sv.shape[0] == xs.shape[0]:
            # shape: [n_samples, n_features, n_classes]
            vals = sv[0, :, crop_idx]
        elif sv.ndim == 3:
            # shape: [n_classes, n_samples, n_features]
            vals = sv[crop_idx][0]
        else:
            vals = sv[0]

        return {feat: round(float(v), 4) for feat, v in zip(FEATURES, vals)}
    except Exception as e:
        return {"error": str(e)}
