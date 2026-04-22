#!/usr/bin/env python3
"""
WC Predictor API Server
=======================
FastAPI server for match winner predictions and tournament simulation.

Setup:
    export WC_DATA_DIR=./data
    pip install fastapi uvicorn
    uvicorn src.api_server:app --reload

Public endpoints:
    GET /health
    GET /teams
    GET /predict/{team1}/{team2}
    GET /team/{team_name}
    
    # Tournament endpoints
    GET /tournament/simulate?runs=1000
    GET /tournament/groups
    GET /tournament/group/{group_id}
    GET /tournament/bracket
    POST /tournament/matchup
"""

import os
import json
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict
from datetime import datetime
from pydantic import BaseModel

try:
    from fastapi import FastAPI, HTTPException, Query
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
except ImportError:
    print("Run: pip install fastapi uvicorn")
    raise

# Load modules
from .predictor import MatchupPredictor
from .tournament_model import TournamentModel, TournamentSimulation
from .group_stage import WC_2026_GROUPS, GroupResult
from .knockout import KnockoutResult

app = FastAPI(
    title="WC 2026 Predictor API",
    description="Predict World Cup 2026 match outcomes and run full tournament simulations",
    version="1.1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Initialize models
_DATA_DIR = os.environ.get('WC_DATA_DIR', './data')
predictor = MatchupPredictor(data_dir=_DATA_DIR)
tournament_model = TournamentModel(predictor)


# ============================================================================
# Health & Info
# ============================================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "1.1.0",
        "components": {
            "predictor": "loaded",
            "tournament_model": "loaded"
        },
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/teams")
def list_teams():
    """List all available teams for prediction"""
    return {
        "count": len(predictor.get_available_teams()),
        "teams": predictor.get_available_teams()
    }


# ============================================================================
# Match Predictions
# ============================================================================

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
            "prediction": result.__dict__,
            "generated_at": datetime.utcnow().isoformat()
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.get("/team/{team_name}")
def team_stats(team_name: str):
    """Get historical stats for a team"""
    stats = predictor.get_team_stats(team_name)
    if not stats:
        raise HTTPException(status_code=404, detail=f"Team {team_name} not found")
    return {
        "team": team_name,
        "stats": stats,
        "generated_at": datetime.utcnow().isoformat()
    }


# ============================================================================
# Tournament Simulation Endpoints
# ============================================================================

class SimulationRequest(BaseModel):
    runs: int = 1000
    include_bracket: bool = False


@app.get("/tournament/simulate")
def simulate_tournament(
    runs: int = Query(1000, ge=100, le=50000, description="Number of Monte Carlo simulations"),
    include_bracket: bool = Query(False, description="Include sample bracket in response")
):
    """
    Run full tournament Monte Carlo simulation.
    
    Returns winner probabilities, advancement odds, and most likely finals.
    """
    try:
        simulation = tournament_model.run_monte_carlo(runs=runs)
        
        # Sort winner probabilities
        sorted_winners = sorted(
            simulation.winner_probs.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        result = {
            "simulations": simulation.runs,
            "generated_at": simulation.timestamp,
            "winner_probabilities": [
                {"team": team, "probability": prob, "odds": 1/prob if prob > 0 else 999}
                for team, prob in sorted_winners[:20]
            ],
            "advancement_probabilities": {
                stage: sorted(probs.items(), key=lambda x: x[1], reverse=True)[:10]
                for stage, probs in simulation.advancement_probs.items()
            },
            "most_likely_finals": [
                {"matchup": list(matchup), "probability": count/simulation.runs}
                for matchup, count in simulation.common_finals[:5]
            ]
        }
        
        if include_bracket:
            # Run single simulation for sample bracket
            _, ko_result = tournament_model.run_single_tournament()
            result["sample_bracket"] = ko_result.bracket
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/tournament/groups")
def list_groups():
    """List all WC 2026 groups and their teams"""
    return {
        "groups": {
            name: teams
            for name, teams in WC_2026_GROUPS.items()
        },
        "format": "8 groups of 4 teams",
        "advancement": "Top 2 from each group advance to R16"
    }


@app.get("/tournament/group/{group_id}")
def predict_group(group_id: str):
    """
    Predict group standings with uncertainty.
    
    Args:
        group_id: Group letter (A, B, C, D, E, F, G, H)
    """
    group_id = group_id.upper()
    
    if group_id not in WC_2026_GROUPS:
        raise HTTPException(
            status_code=404, 
            detail=f"Group {group_id} not found. Valid: A-H"
        )
    
    try:
        prediction = tournament_model.predict_group_standings(group_id)
        return prediction
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class BracketRequest(BaseModel):
    scenario: str = "random"  # "random", "favorites", or specific team list


@app.get("/tournament/bracket")
def get_bracket(scenario: str = Query("random", description="Scenario: random, favorites, or champs")):
    """
    Generate a sample tournament bracket.
    
    Returns complete bracket with matchups through finals.
    """
    try:
        # Run single simulation
        group_results, ko_result = tournament_model.run_single_tournament()
        
        # Format bracket nicely
        bracket = {
            "groups": {
                name: {
                    "winner": result.get_qualified(1)[0],
                    "runner_up": result.get_qualified(2)[1] if len(result.get_qualified(2)) > 1 else None,
                    "standings": [
                        {
                            "position": i+1,
                            "team": s.team,
                            "points": s.points,
                            "gd": s.goal_diff
                        }
                        for i, s in enumerate(result.standings)
                    ]
                }
                for name, result in group_results.items()
            },
            "knockout": ko_result.bracket,
            "final_result": {
                "winner": ko_result.winner,
                "runner_up": ko_result.runner_up,
                "third_place": ko_result.third_place
            }
        }
        
        return {
            "scenario": scenario,
            "generated_at": datetime.utcnow().isoformat(),
            "bracket": bracket
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class MatchupRequest(BaseModel):
    team1: str
    team2: str
    round: Optional[str] = None  # "group", "r16", "qf", "sf", "final"


@app.post("/tournament/matchup")
def predict_knockout_matchup(request: MatchupRequest):
    """
    Predict specific knockout matchup.
    
    Returns probabilities considering knockout pressure (extra time possible).
    """
    try:
        base_pred = predictor.predict(
            request.team1, 
            request.team2, 
            neutral_venue=True
        )
        
        # Adjust for knockout if specified
        if request.round and request.round != "group":
            # Knockout: reduce draw probability (settled by ET)
            draw_adjustment = base_pred.draw_prob * 0.3  # 70% of draws go to ET
            
            # Redistribute to wins
            base_pred.home_win_prob += draw_adjustment * 0.5
            base_pred.away_win_prob += draw_adjustment * 0.5
            base_pred.draw_prob -= draw_adjustment
        
        return {
            "team1": request.team1,
            "team2": request.team2,
            "round": request.round or "neutral",
            "probabilities": {
                "team1_win": round(base_pred.home_win_prob, 3),
                "draw": round(base_pred.draw_prob, 3),
                "team2_win": round(base_pred.away_win_prob, 3)
            },
            "factors": base_pred.key_factors,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================================
# Live Odds Comparison
# ============================================================================

@app.get("/tournament/odds")
def get_live_odds():
    """
    Get current winner odds from model vs market.
    
    Returns model probabilities compared to implied market odds.
    """
    try:
        # Quick Monte Carlo (1k runs)
        simulation = tournament_model.run_monte_carlo(runs=1000)
        
        # Format as odds
        odds = []
        for team, prob in sorted(simulation.winner_probs.items(), key=lambda x: x[1], reverse=True)[:15]:
            decimal_odds = 1 / prob if prob > 0 else 999
            american_odds = int((decimal_odds - 1) * 100) if decimal_odds > 2 else int(-100 / (decimal_odds - 1))
            
            odds.append({
                "team": team,
                "implied_probability": round(prob, 4),
                "decimal_odds": round(decimal_odds, 2),
                "american_odds": american_odds
            })
        
        return {
            "source": "wc-predictor-model",
            "simulations": 1000,
            "generated_at": datetime.utcnow().isoformat(),
            "odds": odds
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Legacy endpoint (for backward compatibility)
# ============================================================================

@app.get("/tournament/winner")
def tournament_winner_probabilities_legacy():
    """Get winner probabilities for all teams (legacy)"""
    return {
        "note": "use /tournament/simulate for full results",
        "teams": predictor.get_tournament_probabilities()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
