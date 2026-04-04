"""
models.py
ML models: Random Forest, SVM, Logistic Regression, Neural Network
+ SHAP explainability
"""

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

# ── Crop profiles for synthetic training data ─────────────────────────────────
# (N, P, K, temp, humidity, ph, rainfall) ranges

CROP_PROFILES = {
    "Rice":       dict(N=(60,80),  P=(40,60),  K=(40,60),  t=(20,30),h=(80,90),ph=(5.5,7.0),r=(150,300)),
    "Wheat":      dict(N=(80,100), P=(40,60),  K=(40,60),  t=(10,22),h=(50,70),ph=(6.0,7.5),r=(40,100)),
    "Maize":      dict(N=(70,90),  P=(45,65),  K=(45,65),  t=(18,27),h=(55,75),ph=(5.5,7.5),r=(60,110)),
    "Chickpea":   dict(N=(30,50),  P=(60,80),  K=(70,90),  t=(15,25),h=(14,25),ph=(5.5,7.0),r=(60,100)),
    "Lentil":     dict(N=(15,25),  P=(55,75),  K=(15,25),  t=(15,25),h=(60,70),ph=(6.0,7.5),r=(40,60)),
    "Cotton":     dict(N=(110,130),P=(45,55),  K=(15,25),  t=(23,33),h=(75,85),ph=(6.0,8.0),r=(60,80)),
    "Sugarcane":  dict(N=(100,120),P=(50,70),  K=(50,70),  t=(25,35),h=(70,85),ph=(6.0,7.5),r=(100,200)),
    "Soybean":    dict(N=(20,40),  P=(60,80),  K=(40,60),  t=(20,30),h=(60,80),ph=(6.0,7.0),r=(60,100)),
    "Groundnut":  dict(N=(15,25),  P=(55,75),  K=(25,45),  t=(25,35),h=(50,70),ph=(5.5,7.0),r=(50,100)),
    "Sunflower":  dict(N=(60,80),  P=(50,70),  K=(40,60),  t=(20,30),h=(40,60),ph=(6.0,7.5),r=(40,80)),
    "Banana":     dict(N=(90,110), P=(70,90),  K=(45,65),  t=(25,35),h=(75,85),ph=(5.5,7.0),r=(100,150)),
    "Mango":      dict(N=(15,25),  P=(15,25),  K=(25,35),  t=(25,35),h=(45,55),ph=(4.5,7.0),r=(90,110)),
    "Coffee":     dict(N=(90,110), P=(25,45),  K=(25,45),  t=(23,28),h=(55,65),ph=(6.0,7.0),r=(150,200)),
    "Tea":        dict(N=(80,100), P=(20,40),  K=(20,40),  t=(18,28),h=(70,90),ph=(4.5,6.0),r=(150,250)),
    "Jute":       dict(N=(70,90),  P=(40,60),  K=(35,55),  t=(23,33),h=(75,90),ph=(6.0,7.0),r=(150,200)),
    "Mustard":    dict(N=(60,80),  P=(40,60),  K=(30,50),  t=(10,20),h=(50,70),ph=(6.0,7.5),r=(30,60)),
    "Potato":     dict(N=(80,100), P=(60,80),  K=(80,100), t=(15,22),h=(70,85),ph=(5.0,6.5),r=(50,100)),
    "Tomato":     dict(N=(70,90),  P=(60,80),  K=(60,80),  t=(20,28),h=(60,80),ph=(6.0,7.0),r=(40,80)),
    "Onion":      dict(N=(60,80),  P=(50,70),  K=(50,70),  t=(15,25),h=(50,70),ph=(6.0,7.5),r=(30,60)),
    "Garlic":     dict(N=(50,70),  P=(40,60),  K=(40,60),  t=(12,22),h=(50,70),ph=(6.0,7.5),r=(30,60)),
    "Watermelon": dict(N=(90,110), P=(15,25),  K=(45,65),  t=(25,35),h=(80,90),ph=(5.5,7.0),r=(40,60)),
    "Grapes":     dict(N=(15,25),  P=(15,25),  K=(25,35),  t=(8,18), h=(80,90),ph=(5.5,7.0),r=(60,80)),
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

# ── Training data ─────────────────────────────────────────────────────────────

def build_training_data(n_per_crop: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    rows = []
    for crop, p in CROP_PROFILES.items():
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
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)

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
