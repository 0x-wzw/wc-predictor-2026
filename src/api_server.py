#!/usr/bin/env python3
"""
WC Predictor API Server
=======================
FastAPI server for match winner predictions.

Setup:
    export WC_DATA_DIR=./data
    pip install fastapi uvicorn
    uvicorn src.api_server:app --reload

Public endpoints:
    GET /health
    GET /predict/{team1}/{team2}
    GET /tournament/winner
"""

import os
import json
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime

try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
except ImportError:
    print("Run: pip install fastapi uvicorn")
    raise

# Load predictor
from .predictor import MatchupPredictor, PredictionResult

app = FastAPI(
    title="WC 2026 Predictor API",
    description="Predict World Cup 2026 match outcomes using historical performance data",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize predictor
_DATA_DIR = os.environ.get('WC_DATA_DIR', './data')
predictor = MatchupPredictor(data_dir=_DATA_DIR)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/teams")
def list_teams():
    """List all available teams for prediction"""
    return {"teams": predictor.get_available_teams()}


@app.get("/predict/{team1}/{team2}")
def predict_match(team1: str, team2: str, neutral: bool = True):
    """
    Predict outcome of match between two teams.
    
    Args:
        team1: Home team name
        team2: Away team name
        neutral: If true, no home advantage applied
    """
    try:
        result = predictor.predict(team1, team2, neutral_venue=neutral)
        return {
            "home_team": team1,
            "away_team": team2,
            "neutral_venue": neutral,
            "prediction": result,
            "generated_at": datetime.utcnow().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/tournament/winner")
def tournament_winner_probabilities():
    """Get winner probabilities for all teams"""
    return predictor.get_tournament_probabilities()


@app.get("/team/{team_name}")
def team_stats(team_name: str):
    """Get historical stats for a team"""
    stats = predictor.get_team_stats(team_name)
    if not stats:
        raise HTTPException(status_code=404, detail=f"Team {team_name} not found")
    return stats


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
