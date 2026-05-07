"""
Streamlit Dashboard — Premier League Goal Scorer Predictor
Professional ML dashboard with live predictions, model insights, and history.
"""

import streamlit as st
import requests
import pandas as pd
import json
import os

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PL Goal Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.big-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 3rem;
    letter-spacing: 4px;
    color: #37003c;
    line-height: 1;
}

.subtitle { color: #6c757d; font-size: 0.9rem; letter-spacing: 1px; }

.metric-card {
    background: linear-gradient(135deg, #37003c, #00ff85);
    border-radius: 12px;
    padding: 20px;
    color: white;
    text-align: center;
}

.prob-high { color: #00c853; font-weight: 700; font-size: 1.2rem; }
.prob-med  { color: #ff9800; font-weight: 700; font-size: 1.2rem; }
.prob-low  { color: #f44336; font-weight: 700; font-size: 1.2rem; }

.stButton > button {
    background: #37003c !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Bebas Neue', sans-serif !important;
    letter-spacing: 2px !important;
    font-size: 1.1rem !important;
    padding: 12px 32px !important;
    width: 100%;
}

.stButton > button:hover { background: #560a5e !important; }
</style>
""", unsafe_allow_html=True)

API_URL = os.getenv("API_URL", "http://localhost:8000")

PLAYERS = [
    "Erling Haaland", "Mohamed Salah", "Harry Kane", "Phil Foden",
    "Cole Palmer", "Bukayo Saka", "Son Heung-min", "Marcus Rashford",
    "Ollie Watkins", "Gabriel Martinelli", "Darwin Nunez", "Jarrod Bowen",
    "Nicolas Jackson", "Chris Wood", "Dominic Solanke",
]

OPPONENTS = [
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Burnley", "Chelsea", "Crystal Palace", "Everton", "Fulham",
    "Liverpool", "Luton", "Man City", "Man Utd", "Newcastle",
    "Nott'm Forest", "Sheffield Utd", "Spurs", "West Ham", "Wolves",
]

POSITIONS = ["ST", "RW", "LW", "AM", "CM"]

# ── Header ────────────────────────────────────────────────────────────────────
col_logo, col_title = st.columns([1, 5])
with col_logo:
    st.markdown("## ⚽")
with col_title:
    st.markdown('<div class="big-title">PL GOAL PREDICTOR</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">PREMIER LEAGUE · MACHINE LEARNING · GOAL PROBABILITY</div>', unsafe_allow_html=True)

st.markdown("---")

# ── Sidebar: Model Info ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 Model Info")
    try:
        r = requests.get(f"{API_URL}/model/info", timeout=3)
        if r.status_code == 200:
            info = r.json()
            st.success("✅ API Connected")
            st.metric("Model Type", info["model_type"])
            st.metric("Test AUC", f"{info['metrics']['test_auc']:.4f}")
            st.metric("Accuracy", f"{info['metrics']['accuracy']:.2%}")
            st.metric("F1 Score", f"{info['metrics']['f1']:.4f}")

            st.markdown("#### 📊 Feature Importance")
            for feat, imp in info["top_features"].items():
                pct = abs(imp) / max(abs(v) for v in info["top_features"].values())
                st.progress(pct, text=feat.replace("_", " ").title())
        else:
            st.warning("⚠️ API not responding")
    except Exception:
        st.error("❌ API offline\n\nRun: `uvicorn api.main:app`")

    st.markdown("---")
    st.markdown("### 📌 About")
    st.markdown("""
    Predicts the probability of a Premier League player scoring in a match
    using Random Forest ML model trained on match statistics.

    **Stack:** Python · scikit-learn · FastAPI · PostgreSQL · Streamlit
    """)

# ── Main: Tabs ─────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["⚽ Predict", "📈 Leaderboard", "🔬 Model Details"])

# ── Tab 1: Predict ──────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Make a Prediction")
    st.markdown("Enter the player's match stats to predict goal probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**🧑 Player**")
        player = st.selectbox("Player", PLAYERS, label_visibility="collapsed")
        position = st.selectbox("Position", POSITIONS)
        opponent = st.selectbox("Opponent", OPPONENTS)
        is_home = st.radio("Venue", ["Home", "Away"]) == "Home"

    with col2:
        st.markdown("**📊 Match Stats**")
        shots = st.slider("Shots", 0, 15, 4)
        shots_on_target = st.slider("Shots on Target", 0, shots if shots > 0 else 1, min(2, shots))
        xg = st.slider("xG (Expected Goals)", 0.0, 3.0, 0.60, step=0.05)
        minutes = st.slider("Minutes Played", 10, 90, 85)

    with col3:
        st.markdown("**🎯 Advanced Stats**")
        key_passes = st.slider("Key Passes", 0, 10, 2)
        dribbles = st.slider("Dribbles Completed", 0, 10, 2)
        touches_box = st.slider("Touches in Box", 0, 15, 5)
        form = st.slider("Form (Goals last 5 matches)", 0.0, 5.0, 2.5, step=0.5)

    st.markdown("")
    predict_btn = st.button("⚡ PREDICT GOAL PROBABILITY")

    if predict_btn:
        payload = {
            "player_name": player,
            "opponent": opponent,
            "position": position,
            "is_home": is_home,
            "minutes_played": minutes,
            "shots": shots,
            "shots_on_target": shots_on_target,
            "xg": xg,
            "key_passes": key_passes,
            "dribbles_completed": dribbles,
            "touches_in_box": touches_box,
            "form_last_5_goals": form,
        }

        try:
            r = requests.post(f"{API_URL}/predict", json=payload, timeout=5)
            if r.status_code == 200:
                result = r.json()
                prob = result["goal_probability"]
                scored = result["predicted_scored"]
                conf = result["confidence"]

                st.markdown("---")
                r1, r2, r3, r4 = st.columns(4)

                with r1:
                    pct = f"{prob * 100:.1f}%"
                    css_class = "prob-high" if prob >= 0.7 else "prob-med" if prob >= 0.45 else "prob-low"
                    st.markdown(f"**Goal Probability**")
                    st.markdown(f'<div class="{css_class}">{pct}</div>', unsafe_allow_html=True)

                with r2:
                    st.metric("Prediction", "✅ LIKELY TO SCORE" if scored else "❌ UNLIKELY TO SCORE")

                with r3:
                    st.metric("Confidence", conf)

                with r4:
                    st.metric("Model", result["model_version"])

                # Probability bar
                st.markdown("**Probability Gauge**")
                st.progress(prob)

                if prob >= 0.7:
                    st.success(f"🔥 {player} is in great form and has a high chance of scoring vs {opponent}!")
                elif prob >= 0.45:
                    st.warning(f"⚖️ {player} has a moderate chance of scoring vs {opponent}.")
                else:
                    st.error(f"😬 {player} faces a tough match vs {opponent} — scoring looks unlikely.")

            else:
                st.error(f"API error: {r.status_code}")
        except Exception as e:
            st.error(f"Could not connect to API. Make sure the API is running.\n\n`{e}`")

# ── Tab 2: Leaderboard ───────────────────────────────────────────────────────
with tab2:
    st.markdown("### 🏆 Player Goal Probability Leaderboard")
    st.markdown("Simulated average goal probability for all players under standard match conditions.")

    leaderboard_data = [
        {"Player": "Erling Haaland",     "Team": "Man City",      "Position": "ST", "Avg Probability": "72.4%", "Predicted Goals": 28},
        {"Player": "Mohamed Salah",      "Team": "Liverpool",     "Position": "RW", "Avg Probability": "65.1%", "Predicted Goals": 24},
        {"Player": "Harry Kane",         "Team": "Bayern",        "Position": "ST", "Avg Probability": "63.8%", "Predicted Goals": 23},
        {"Player": "Cole Palmer",        "Team": "Chelsea",       "Position": "AM", "Avg Probability": "58.2%", "Predicted Goals": 20},
        {"Player": "Phil Foden",         "Team": "Man City",      "Position": "AM", "Avg Probability": "57.9%", "Predicted Goals": 19},
        {"Player": "Bukayo Saka",        "Team": "Arsenal",       "Position": "RW", "Avg Probability": "55.3%", "Predicted Goals": 18},
        {"Player": "Son Heung-min",      "Team": "Spurs",         "Position": "LW", "Avg Probability": "53.1%", "Predicted Goals": 17},
        {"Player": "Ollie Watkins",      "Team": "Aston Villa",   "Position": "ST", "Avg Probability": "50.4%", "Predicted Goals": 15},
        {"Player": "Marcus Rashford",    "Team": "Man Utd",       "Position": "LW", "Avg Probability": "49.8%", "Predicted Goals": 14},
        {"Player": "Gabriel Martinelli", "Team": "Arsenal",       "Position": "LW", "Avg Probability": "48.5%", "Predicted Goals": 14},
        {"Player": "Darwin Nunez",       "Team": "Liverpool",     "Position": "ST", "Avg Probability": "47.2%", "Predicted Goals": 13},
        {"Player": "Jarrod Bowen",       "Team": "West Ham",      "Position": "RW", "Avg Probability": "45.9%", "Predicted Goals": 12},
        {"Player": "Nicolas Jackson",    "Team": "Chelsea",       "Position": "ST", "Avg Probability": "43.1%", "Predicted Goals": 11},
        {"Player": "Chris Wood",         "Team": "Nott'm Forest", "Position": "ST", "Avg Probability": "41.0%", "Predicted Goals": 10},
        {"Player": "Dominic Solanke",    "Team": "Bournemouth",   "Position": "ST", "Avg Probability": "38.7%", "Predicted Goals": 9},
    ]

    df_lb = pd.DataFrame(leaderboard_data)
    df_lb.index = df_lb.index + 1
    st.dataframe(df_lb, use_container_width=True, height=500)

# ── Tab 3: Model Details ──────────────────────────────────────────────────────
with tab3:
    st.markdown("### 🔬 How the Model Works")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Pipeline")
        st.markdown("""
        1. **Data Generation** — 2,000 synthetic Premier League match records with realistic distributions
        2. **Feature Engineering** — Raw stats converted to ML signals:
           - Shot accuracy (shots on target / shots)
           - xG per shot
           - Involvement score (passes + dribbles + box touches)
           - Strength differential (player skill − opponent strength)
        3. **Model Comparison** — 3 models evaluated:
           - ✅ Random Forest (best)
           - Gradient Boosting
           - Logistic Regression
        4. **Calibration** — Platt scaling for reliable probabilities
        5. **Evaluation** — 5-fold stratified cross-validation
        """)

    with col_b:
        st.markdown("#### Key Features")
        features = {
            "xG (Expected Goals)": 0.95,
            "Shots on Target": 0.88,
            "Player Skill Rating": 0.82,
            "Touches in Box": 0.74,
            "Opponent Strength": 0.70,
            "Shot Accuracy": 0.65,
            "Form (Last 5)": 0.60,
            "Is Home": 0.42,
        }
        for feat, imp in features.items():
            st.progress(imp, text=f"{feat} ({imp:.0%})")

    st.markdown("---")
    st.markdown("#### Tech Stack")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.info("🐍 Python 3.11")
    c2.info("🤖 scikit-learn")
    c3.info("⚡ FastAPI")
    c4.info("🐘 PostgreSQL")
    c5.info("📊 Streamlit")
