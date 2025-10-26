import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import os


def train_top_models(X, y):
    # ✅ Encode disease labels into numeric form for XGBoost/LightGBM
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Save the label encoder for decoding predictions later
    os.makedirs("src/models", exist_ok=True)
    joblib.dump(le, "src/models/label_encoder.pkl")

    # Define models
    models = {
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42),
        "LightGBM": LGBMClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
    }

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    results = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        joblib.dump(model, f"src/models/{name}.pkl")
        print(f"✅ {name} trained — Accuracy: {acc:.4f}")
        results.append((name, acc))

    # Return top 3 models by accuracy
    return sorted(results, key=lambda x: x[1], reverse=True)[:3]


def load_top_models():
    model_paths = [
        "src/models/RandomForest.pkl",
        "src/models/XGBoost.pkl",
        "src/models/LightGBM.pkl",
        "src/models/LogisticRegression.pkl"
    ]
    models = [joblib.load(p) for p in model_paths if os.path.exists(p)]
    return models


def ensemble_predict(models, input_data, all_diseases, return_all=False):
    # Load the label encoder to decode predictions
    le = joblib.load("src/models/label_encoder.pkl")

    # Average probabilities from all models
    probs = np.mean([m.predict_proba(input_data) for m in models], axis=0)
    decoded_diseases = le.inverse_transform(np.arange(len(le.classes_)))

    if return_all:
        preds = [decoded_diseases[np.argmax(p)] for p in probs]
        return preds
    else:
        pred_disease = decoded_diseases[np.argmax(probs[0])]
        return probs[0], pred_disease
