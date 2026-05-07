# ⚽ Premier League Goal Scorer Predictor

> A full-stack machine learning system that predicts the probability of a Premier League player scoring in a match — powered by Random Forest, FastAPI, PostgreSQL, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green?logo=fastapi)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.5-orange?logo=scikit-learn)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16-blue?logo=postgresql)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35-red?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)

---

## 📌 Overview

This project is a production-style ML pipeline that:

- **Generates** realistic Premier League match statistics for 15 top players
- **Trains** and compares 3 ML models (Random Forest, Gradient Boosting, Logistic Regression)
- **Serves** predictions via a REST API (FastAPI)
- **Stores** all predictions in PostgreSQL for historical analysis
- **Visualises** results on an interactive Streamlit dashboard

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────┐
│                   Streamlit Dashboard                │
│            (localhost:8501 / port 8501)              │
└────────────────────┬────────────────────────────────┘
                     │ HTTP requests
┌────────────────────▼────────────────────────────────┐
│                   FastAPI Backend                    │
│               (localhost:8000 / port 8000)           │
│  /predict  /players  /model/info  /predictions      │
└────────────┬───────────────────┬────────────────────┘
             │                   │
┌────────────▼──────┐  ┌────────▼────────────────────┐
│   ML Model        │  │   PostgreSQL Database        │
│   Random Forest   │  │   (predictions log,          │
│   (calibrated)    │  │    player data, history)     │
└───────────────────┘  └─────────────────────────────┘
```

---

## 🤖 ML Pipeline

### Features Used
| Feature | Description |
|---|---|
| `xg` | Expected goals from match |
| `shots_on_target` | Shots on target |
| `player_skill` | Player skill rating (0–1) |
| `touches_in_box` | Touches inside penalty box |
| `opponent_strength` | Opponent defensive rating |
| `shot_accuracy` | shots_on_target / shots |
| `form_last_5_goals` | Goals in last 5 matches |
| `strength_diff` | player_skill − opponent_strength |
| `is_home` | Home advantage flag |

### Model Comparison
| Model | CV AUC | Test AUC | F1 |
|---|---|---|---|
| **Random Forest** ✅ | ~0.88 | ~0.87 | ~0.74 |
| Gradient Boosting | ~0.86 | ~0.85 | ~0.71 |
| Logistic Regression | ~0.79 | ~0.78 | ~0.65 |

---

## 🚀 Quick Start

### Option 1 — Docker (Recommended)

```bash
git clone https://github.com/YOUR_USERNAME/pl-goal-predictor.git
cd pl-goal-predictor

cp .env.example .env
docker compose up -d
```

Then open:
- **Dashboard** → http://localhost:8501
- **API Docs** → http://localhost:8000/docs

### Option 2 — Local Setup

```bash
# 1. Clone
git clone https://github.com/YOUR_USERNAME/pl-goal-predictor.git
cd pl-goal-predictor

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate data & train model
python data/generate_data.py
python models/train.py

# 4. Setup database (optional)
cp .env.example .env
python database/db.py

# 5. Start API
uvicorn api.main:app --reload

# 6. Start dashboard (new terminal)
streamlit run dashboard/app.py
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Predict goal probability |
| `GET` | `/players` | List all players |
| `GET` | `/opponents` | List all opponents |
| `GET` | `/model/info` | Model metrics & feature importance |
| `GET` | `/predictions/recent` | Recent predictions from DB |

### Example Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "player_name": "Erling Haaland",
    "opponent": "Arsenal",
    "position": "ST",
    "is_home": true,
    "minutes_played": 90,
    "shots": 5,
    "shots_on_target": 3,
    "xg": 0.85,
    "key_passes": 1,
    "dribbles_completed": 1,
    "touches_in_box": 7,
    "form_last_5_goals": 4.0
  }'
```

### Example Response

```json
{
  "player_name": "Erling Haaland",
  "opponent": "Arsenal",
  "goal_probability": 0.7423,
  "predicted_scored": true,
  "confidence": "High",
  "model_version": "Random Forest"
}
```

---

## 📁 Project Structure

```
pl-goal-predictor/
├── data/
│   └── generate_data.py      # Synthetic dataset generator
├── models/
│   └── train.py              # ML training pipeline
├── api/
│   ├── main.py               # FastAPI application
│   └── Dockerfile
├── dashboard/
│   ├── app.py                # Streamlit dashboard
│   ├── Dockerfile
│   └── requirements-dashboard.txt
├── database/
│   └── db.py                 # PostgreSQL schema & helpers
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## 🛠️ Tech Stack

- **ML**: scikit-learn (Random Forest, Gradient Boosting, Logistic Regression)
- **API**: FastAPI + Uvicorn
- **Database**: PostgreSQL + psycopg2
- **Dashboard**: Streamlit
- **Containerisation**: Docker + Docker Compose
- **Data**: pandas + numpy

---

## 👤 Author

Built by Prijesh Shrestha

---

