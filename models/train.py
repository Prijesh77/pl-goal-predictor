"""
Premier League Goal Scorer Probability Model
Uses Random Forest with cross-validation, feature importance, and model persistence.
"""

import pandas as pd
import numpy as np
import joblib
import os
import json
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, roc_auc_score, confusion_matrix,
    accuracy_score, f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

# ── Config ──────────────────────────────────────────────────────────────────
FEATURES = [
    "position_encoded", "is_home", "minutes_played",
    "shots", "shots_on_target", "xg",
    "key_passes", "dribbles_completed", "touches_in_box",
    "form_last_5_goals", "opponent_strength", "player_skill",
]
TARGET = "scored"
MODEL_DIR = "models"


def load_data(path: str = "data/pl_player_match_data.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"📊 Loaded {len(df)} records | Goal rate: {df[TARGET].mean():.2%}")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features for better model performance."""
    df = df.copy()
    df["shot_accuracy"] = df["shots_on_target"] / (df["shots"] + 1)
    df["xg_per_shot"] = df["xg"] / (df["shots"] + 1)
    df["involvement"] = df["key_passes"] + df["dribbles_completed"] + df["touches_in_box"]
    df["strength_diff"] = df["player_skill"] - df["opponent_strength"]
    return df


def train_and_evaluate():
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load & prepare data
    df = load_data()
    df = engineer_features(df)

    all_features = FEATURES + ["shot_accuracy", "xg_per_shot", "involvement", "strength_diff"]
    X = df[all_features]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\n📐 Train: {len(X_train)} | Test: {len(X_test)}")

    # ── Models to compare ──────────────────────────────────────────────────
    models = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_split=10,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=4, learning_rate=0.05,
            subsample=0.8, random_state=42
        ),
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
        ]),
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    print("\n🔬 Cross-validation results:")
    print("-" * 50)

    for name, model in models.items():
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

        results[name] = {
            "cv_auc_mean": cv_scores.mean(),
            "cv_auc_std": cv_scores.std(),
            "test_auc": roc_auc_score(y_test, y_proba),
            "accuracy": accuracy_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
        }

        print(f"\n{name}:")
        print(f"  CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"  Test AUC: {results[name]['test_auc']:.4f}")
        print(f"  Accuracy: {results[name]['accuracy']:.4f} | F1: {results[name]['f1']:.4f}")

    # ── Best model ─────────────────────────────────────────────────────────
    best_name = max(results, key=lambda k: results[k]["test_auc"])
    best_model = models[best_name]
    print(f"\n🏆 Best model: {best_name} (AUC: {results[best_name]['test_auc']:.4f})")

    # Calibrate probabilities
    calibrated = CalibratedClassifierCV(best_model, cv=5, method="sigmoid")
    calibrated.fit(X_train, y_train)

    # Feature importance
    if hasattr(best_model, "feature_importances_"):
        importances = best_model.feature_importances_
    elif hasattr(best_model, "named_steps"):
        importances = best_model.named_steps["clf"].coef_[0]
    else:
        importances = np.zeros(len(all_features))

    feature_importance = dict(zip(all_features, importances.tolist()))
    sorted_fi = dict(sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True))

    print("\n📊 Top features:")
    for feat, imp in list(sorted_fi.items())[:5]:
        print(f"  {feat}: {imp:.4f}")

    # ── Save artifacts ─────────────────────────────────────────────────────
    joblib.dump(calibrated, f"{MODEL_DIR}/goal_predictor.pkl")
    joblib.dump(all_features, f"{MODEL_DIR}/feature_names.pkl")

    metadata = {
        "best_model": best_name,
        "features": all_features,
        "metrics": results[best_name],
        "feature_importance": sorted_fi,
        "all_results": results,
    }

    with open(f"{MODEL_DIR}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ Model saved to {MODEL_DIR}/goal_predictor.pkl")
    print(f"✅ Metadata saved to {MODEL_DIR}/model_metadata.json")

    return calibrated, all_features, metadata


if __name__ == "__main__":
    train_and_evaluate()
