"""
Database setup — PostgreSQL schema for Premier League Goal Predictor
Run this once to initialize tables.
"""

import psycopg2
import os
from dotenv import load_dotenv

load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", 5432),
    "database": os.getenv("DB_NAME", "pl_predictor"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "postgres"),
}

SCHEMA = """
-- Players table
CREATE TABLE IF NOT EXISTS players (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL UNIQUE,
    team VARCHAR(100) NOT NULL,
    position VARCHAR(10) NOT NULL,
    skill_rating FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Match records table (historical data)
CREATE TABLE IF NOT EXISTS match_records (
    id SERIAL PRIMARY KEY,
    player_id INT REFERENCES players(id) ON DELETE CASCADE,
    opponent VARCHAR(100) NOT NULL,
    match_date DATE,
    is_home BOOLEAN NOT NULL,
    minutes_played INT,
    shots INT,
    shots_on_target INT,
    xg FLOAT,
    key_passes INT,
    dribbles_completed INT,
    touches_in_box INT,
    form_last_5_goals FLOAT,
    opponent_strength FLOAT,
    scored BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Predictions log table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    player_name VARCHAR(100) NOT NULL,
    opponent VARCHAR(100) NOT NULL,
    is_home BOOLEAN,
    shots INT,
    shots_on_target INT,
    xg FLOAT,
    predicted_probability FLOAT NOT NULL,
    predicted_scored BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_match_records_player ON match_records(player_id);
CREATE INDEX IF NOT EXISTS idx_predictions_player ON predictions(player_name);
CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);
"""

SEED_PLAYERS = """
INSERT INTO players (name, team, position, skill_rating) VALUES
    ('Erling Haaland',     'Man City',      'ST', 0.92),
    ('Mohamed Salah',      'Liverpool',     'RW', 0.88),
    ('Harry Kane',         'Bayern',        'ST', 0.87),
    ('Phil Foden',         'Man City',      'AM', 0.85),
    ('Cole Palmer',        'Chelsea',       'AM', 0.83),
    ('Bukayo Saka',        'Arsenal',       'RW', 0.82),
    ('Son Heung-min',      'Spurs',         'LW', 0.80),
    ('Marcus Rashford',    'Man Utd',       'LW', 0.78),
    ('Ollie Watkins',      'Aston Villa',   'ST', 0.77),
    ('Gabriel Martinelli', 'Arsenal',       'LW', 0.76),
    ('Darwin Nunez',       'Liverpool',     'ST', 0.75),
    ('Jarrod Bowen',       'West Ham',      'RW', 0.74),
    ('Nicolas Jackson',    'Chelsea',       'ST', 0.72),
    ('Chris Wood',         'Nott''m Forest','ST', 0.70),
    ('Dominic Solanke',    'Bournemouth',   'ST', 0.68)
ON CONFLICT (name) DO NOTHING;
"""


def get_connection():
    return psycopg2.connect(**DB_CONFIG)


def init_db():
    print("🔧 Initializing database...")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(SCHEMA)
    cur.execute(SEED_PLAYERS)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ Database initialized with tables and seed data")


def log_prediction(player_name, opponent, is_home, shots, shots_on_target,
                   xg, probability, predicted_scored):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO predictions
            (player_name, opponent, is_home, shots, shots_on_target, xg,
             predicted_probability, predicted_scored)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
    """, (player_name, opponent, is_home, shots, shots_on_target,
          xg, probability, predicted_scored))
    pred_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return pred_id


def get_recent_predictions(limit=20):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT player_name, opponent, is_home, shots, xg,
               predicted_probability, predicted_scored, created_at
        FROM predictions
        ORDER BY created_at DESC
        LIMIT %s
    """, (limit,))
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def get_player_stats():
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("""
        SELECT player_name,
               COUNT(*) as total_predictions,
               AVG(predicted_probability) as avg_probability,
               SUM(CASE WHEN predicted_scored THEN 1 ELSE 0 END) as predicted_goals
        FROM predictions
        GROUP BY player_name
        ORDER BY avg_probability DESC
    """)
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


if __name__ == "__main__":
    init_db()
