"""
FastAPI Backend — Premier League Goal Scorer Probability API
Endpoints for predictions, player stats, and model metadata.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI(
    title="PL Goal Scorer Predictor API",
    description="Premier League goal scoring probability prediction using Random Forest ML model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model artifacts ────────────────────────────────────────────────────
MODEL_PATH = "models/goal_predictor.pkl"
META_PATH = "models/model_metadata.json"

try:
    model = joblib.load(MODEL_PATH)
    with open(META_PATH) as f:
        metadata = json.load(f)
    feature_names = metadata["features"]
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️  Model not found. Run models/train.py first. Error: {e}")
    model = None
    metadata = {}
    feature_names = []

OPPONENT_STRENGTH = {
    "Man City": 0.92, "Liverpool": 0.88, "Arsenal": 0.85, "Chelsea": 0.80,
    "Man Utd": 0.76, "Spurs": 0.75, "Newcastle": 0.74, "Aston Villa": 0.73,
    "Brighton": 0.70, "West Ham": 0.68, "Wolves": 0.65, "Brentford": 0.64,
    "Fulham": 0.63, "Everton": 0.62, "Bournemouth": 0.60, "Crystal Palace": 0.59,
    "Nott'm Forest": 0.61, "Burnley": 0.55, "Luton": 0.52, "Sheffield Utd": 0.50,
}

PLAYER_SKILL = {
    "Erling Haaland": 0.92, "Mohamed Salah": 0.88, "Harry Kane": 0.87,
    "Phil Foden": 0.85, "Cole Palmer": 0.83, "Bukayo Saka": 0.82,
    "Son Heung-min": 0.80, "Marcus Rashford": 0.78, "Ollie Watkins": 0.77,
    "Gabriel Martinelli": 0.76, "Darwin Nunez": 0.75, "Jarrod Bowen": 0.74,
    "Nicolas Jackson": 0.72, "Chris Wood": 0.70, "Dominic Solanke": 0.68,
}

POSITION_ENCODED = {"ST": 3, "RW": 2, "LW": 2, "AM": 1, "CM": 0}


# ── Schemas ─────────────────────────────────────────────────────────────────
class PredictionRequest(BaseModel):
    player_name: str = Field(..., example="Erling Haaland")
    opponent: str = Field(..., example="Arsenal")
    position: str = Field(..., example="ST")
    is_home: bool = Field(..., example=True)
    minutes_played: int = Field(..., ge=1, le=90, example=90)
    shots: int = Field(..., ge=0, le=20, example=5)
    shots_on_target: int = Field(..., ge=0, le=15, example=3)
    xg: float = Field(..., ge=0.0, le=5.0, example=0.75)
    key_passes: int = Field(..., ge=0, le=15, example=2)
    dribbles_completed: int = Field(..., ge=0, le=15, example=1)
    touches_in_box: int = Field(..., ge=0, le=20, example=6)
    form_last_5_goals: float = Field(..., ge=0.0, le=5.0, example=3.0)


class PredictionResponse(BaseModel):
    player_name: str
    opponent: str
    goal_probability: float
    predicted_scored: bool
    confidence: str
    model_version: str


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "PL Goal Scorer Predictor API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": ["/predict", "/players", "/opponents", "/model/info", "/health"]
    }


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run train.py first.")

    opp_strength = OPPONENT_STRENGTH.get(req.opponent, 0.65)
    player_skill = PLAYER_SKILL.get(req.player_name, 0.70)
    pos_enc = POSITION_ENCODED.get(req.position, 0)

    # Engineered features
    shot_accuracy = req.shots_on_target / (req.shots + 1)
    xg_per_shot = req.xg / (req.shots + 1)
    involvement = req.key_passes + req.dribbles_completed + req.touches_in_box
    strength_diff = player_skill - opp_strength

    features = np.array([[
        pos_enc, int(req.is_home), req.minutes_played,
        req.shots, req.shots_on_target, req.xg,
        req.key_passes, req.dribbles_completed, req.touches_in_box,
        req.form_last_5_goals, opp_strength, player_skill,
        shot_accuracy, xg_per_shot, involvement, strength_diff,
    ]])

    probability = float(model.predict_proba(features)[0][1])
    predicted = probability >= 0.45

    if probability >= 0.70:
        confidence = "High"
    elif probability >= 0.45:
        confidence = "Medium"
    else:
        confidence = "Low"

    # Log to DB (optional — won't crash if DB unavailable)
    try:
        from database.db import log_prediction
        log_prediction(
            req.player_name, req.opponent, req.is_home,
            req.shots, req.shots_on_target, req.xg,
            probability, predicted
        )
    except Exception:
        pass

    return PredictionResponse(
        player_name=req.player_name,
        opponent=req.opponent,
        goal_probability=round(probability, 4),
        predicted_scored=predicted,
        confidence=confidence,
        model_version=metadata.get("best_model", "Random Forest"),
    )


@app.get("/players")
def get_players():
    return {"players": list(PLAYER_SKILL.keys())}


@app.get("/opponents")
def get_opponents():
    return {"opponents": list(OPPONENT_STRENGTH.keys())}


@app.get("/model/info")
def model_info():
    if not metadata:
        raise HTTPException(status_code=503, detail="Model metadata not available.")
    return {
        "model_type": metadata.get("best_model"),
        "features": metadata.get("features"),
        "metrics": metadata.get("metrics"),
        "top_features": dict(list(metadata.get("feature_importance", {}).items())[:6]),
    }


@app.get("/predictions/recent")
def recent_predictions():
    try:
        from database.db import get_recent_predictions
        rows = get_recent_predictions(20)
        return {"predictions": [
            {
                "player": r[0], "opponent": r[1], "is_home": r[2],
                "shots": r[3], "xg": r[4], "probability": round(r[5], 3),
                "predicted_scored": r[6], "timestamp": str(r[7])
            } for r in rows
        ]}
    except Exception as e:
        return {"predictions": [], "note": "Database unavailable"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
