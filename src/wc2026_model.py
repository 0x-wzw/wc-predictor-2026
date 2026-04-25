#!/usr/bin/env python3
"""
WC2026 Match Prediction Model - Production Version
===================================================
Deployable model with training pipeline, persistence, and API.

Usage:
  python wc2026_model.py train --data path/to/historical.csv
  python wc2026_model.py predict --match "France,Brazil"
  python wc2026_model.py serve --port 8000
"""

import json
import random
import math
import os
import pickle
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime

# Model paths
MODEL_DIR = os.path.expanduser("~/.hermes/models/wc2026")
MODEL_PATH = os.path.join(MODEL_DIR, "match_predictor_v1.pkl")
DATA_PATH = os.path.expanduser("~/.hermes/data/wc2026_signals/model_predictions.json")

# WC2026 Groups
WC2026_GROUPS = {
    "A": ["Mexico", "South Africa", "Korea Republic", "Czechia"],
    "B": ["Canada", "Bosnia-Herzegovina", "Qatar", "Switzerland"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkiye"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Belgium", "Egypt", "Iran", "New Zealand"],
    "H": ["Spain", "Cape Verde", "Saudi Arabia", "Uruguay"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Argentina", "Algeria", "Austria", "Jordan"],
    "K": ["Portugal", "Congo Dr", "Uzbekistan", "Colombia"],
    "L": ["England", "Croatia", "Ghana", "Panama"],
}

NAME_MAP = {
    "Korea Republic": "South Korea",
    "Iran": "IR Iran",
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Côte d'Ivoire",
    "Congo Dr": "Congo DR",
    "Turkiye": "Turkey",
    "Curaçao": "Curacao",
    "Usa": "USA",
}

def normalize_name(name: str) -> str:
    """Normalize team names between sources."""
    name = name.strip()
    # Check direct mapping
    if name in NAME_MAP:
        return NAME_MAP[name]
    # Case variations
    name_lower = name.lower()
    for k, v in NAME_MAP.items():
        if k.lower() == name_lower:
            return v
    return name

@dataclass
class MatchFeatures:
    """Features used for match prediction."""
    team1: str
    team2: str
    team1_rating: float
    team2_rating: float
    rating_diff: float
    neutral_venue: bool = True
    tournament_knockout: bool = False
    
@dataclass
class MatchPrediction:
    """Complete match prediction output."""
    match_id: str
    timestamp: str
    team1: str
    team2: str
    
    # 1X2
    team1_win_prob: float
    draw_prob: float
    team2_win_prob: float
    
    # Expected goals
    team1_xg: float
    team2_xg: float
    
    # Asian Handicap
    hdp_lines: Dict[str, Dict]
    
    # Over/Under
    ou_lines: Dict[str, Dict]
    
    # BTTS
    btts_yes_prob: float
    
    # Correct score (top 5)
    correct_score: List[Tuple[str, float]]
    
    # Model confidence
    confidence: float  # 0-1 based on data quality
    
    def to_dict(self) -> dict:
        return {
            "match_id": self.match_id,
            "timestamp": self.timestamp,
            "teams": {"home": self.team1, "away": self.team2},
            "1x2": {
                "home_win": {"prob": round(self.team1_win_prob, 4), "decimal": round(1/self.team1_win_prob, 2) if self.team1_win_prob > 0 else None},
                "draw": {"prob": round(self.draw_prob, 4), "decimal": round(1/self.draw_prob, 2) if self.draw_prob > 0 else None},
                "away_win": {"prob": round(self.team2_win_prob, 4), "decimal": round(1/self.team2_win_prob, 2) if self.team2_win_prob > 0 else None},
            },
            "expected_goals": {"home": round(self.team1_xg, 2), "away": round(self.team2_xg, 2)},
            "asian_handicap": {k: {"home": round(v.get("home", 0), 3), "away": round(v.get("away", 0), 3)} 
                              for k, v in self.hdp_lines.items()},
            "over_under": {k.replace("O/U ", "ou_"): {"over": round(v["over_prob"], 3), "under": round(v["under_prob"], 3)} 
                         for k, v in self.ou_lines.items()},
            "btts": {"yes": round(self.btts_yes_prob, 3), "no": round(1-self.btts_yes_prob, 3)},
            "correct_score": [{"score": s, "prob": round(p, 4)} for s, p in self.correct_score[:5]],
            "confidence": round(self.confidence, 3),
        }

class WC2026MatchModel:
    """
    Production match prediction model for WC2026.
    
    Architecture:
    - Team strength ratings derived from tournament win probabilities
    - Poisson-based goal distribution
    - Monte Carlo simulation for market probabilities
    """
    
    VERSION = "1.0.0"
    
    def __init__(self):
        self.ratings: Dict[str, float] = {}
        self.raw_probs: Dict[str, float] = {}
        self.avg_goals = 2.6
        self.home_advantage = 0.15
        self.fitted = False
        self.last_updated = None
        
    def fit(self, tournament_probs: Dict[str, float]) -> 'WC2026MatchModel':
        """
        Fit model from tournament win probabilities.
        
        Args:
            tournament_probs: Dict[team_name, probability_of_winning_tournament]
        
        Returns:
            self for chaining
        """
        self.raw_probs = tournament_probs
        self.ratings = self._compute_ratings(tournament_probs)
        self.fitted = True
        self.last_updated = datetime.utcnow().isoformat()
        return self
    
    def _compute_ratings(self, win_probs: Dict[str, float]) -> Dict[str, float]:
        """Convert tournament win probabilities to team strength ratings."""
        max_prob = max(win_probs.values())
        ratings = {}
        
        for team, prob in win_probs.items():
            if prob > 0:
                # Log-scale strength relative to best team
                strength = math.log(prob / max_prob + 0.01)
                # Map to Elo-like scale: 2000 = best, ~1300 = weakest
                rating = 2000 + 200 * strength
                ratings[normalize_name(team)] = max(1300, min(2100, rating))
            else:
                ratings[normalize_name(team)] = 1300
        
        return ratings
    
    def get_rating(self, team: str) -> float:
        """Get team rating, with fallback for unknown teams."""
        normalized = normalize_name(team)
        if normalized in self.ratings:
            return self.ratings[normalized]
        # Fallback: interpolate
        return 1500
    
    def predict(self, team1: str, team2: str, 
                neutral: bool = True,
                n_sims: int = 10000) -> MatchPrediction:
        """
        Predict a match between two teams.
        
        Args:
            team1: Home team name
            team2: Away team name
            neutral: Whether match is at neutral venue
            n_sims: Number of Monte Carlo simulations
        
        Returns:
            MatchPrediction with all markets
        """
        if not self.fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")
        
        # Get ratings
        r1 = self.get_rating(team1)
        r2 = self.get_rating(team2)
        
        # Calculate expected goals
        base_xg = 1.3
        rating_diff = (r1 - r2) / 400  # Elo difference
        
        team1_xg = base_xg * (1 + rating_diff * 0.3)
        if not neutral:
            team1_xg *= (1 + self.home_advantage)
        team2_xg = base_xg * (1 - rating_diff * 0.3)
        
        # Monte Carlo simulation
        results = []
        for _ in range(n_sims):
            g1 = self._sample_poisson(team1_xg)
            g2 = self._sample_poisson(team2_xg)
            results.append((g1, g2))
        
        # Calculate 1X2
        t1_wins = sum(1 for g1, g2 in results if g1 > g2)
        draws = sum(1 for g1, g2 in results if g1 == g2)
        t2_wins = sum(1 for g1, g2 in results if g1 < g2)
        
        t1_prob = t1_wins / n_sims
        draw_prob = draws / n_sims
        t2_prob = t2_wins / n_sims
        
        # Calculate markets
        hdp_lines = self._calc_hdp(results)
        ou_lines = self._calc_ou(results)
        btts = sum(1 for g1, g2 in results if g1 > 0 and g2 > 0) / n_sims
        
        # Correct score distribution
        scores = defaultdict(int)
        for g1, g2 in results:
            scores[f"{g1}-{g2}"] += 1
        correct_score = sorted([(s, c/n_sims) for s, c in scores.items()],
                                key=lambda x: x[1], reverse=True)[:10]
        
        # Confidence based on rating difference (more confident when teams are different)
        confidence = min(0.95, 0.5 + abs(r1 - r2) / 1000)
        
        return MatchPrediction(
            match_id=f"{normalize_name(team1)}_vs_{normalize_name(team2)}_{datetime.utcnow().strftime('%Y%m%d')}",
            timestamp=datetime.utcnow().isoformat(),
            team1=team1,
            team2=team2,
            team1_win_prob=t1_prob,
            draw_prob=draw_prob,
            team2_win_prob=t2_prob,
            team1_xg=team1_xg,
            team2_xg=team2_xg,
            hdp_lines=hdp_lines,
            ou_lines=ou_lines,
            btts_yes_prob=btts,
            correct_score=correct_score,
            confidence=confidence
        )
    
    def _sample_poisson(self, lam: float) -> int:
        """Sample from Poisson distribution."""
        # Knuth's algorithm for small lambda
        if lam < 30:
            L = math.exp(-lam)
            k = 0
            p = 1.0
            while p > L:
                k += 1
                p *= random.random()
            return max(0, k - 1)
        else:
            # Normal approximation for large lambda
            z = random.gauss(0, 1)
            return max(0, int(lam + z * math.sqrt(lam)))
    
    def _calc_hdp(self, results: List[Tuple[int, int]]) -> Dict[str, Dict]:
        """Calculate Asian Handicap probabilities."""
        lines = [-1.5, -1.0, -0.5, 0, 0.5, 1.0, 1.5]
        hdp = {}
        
        for h in lines:
            home_wins = 0
            for g1, g2 in results:
                adj = g1 + h - g2
                if adj > 0:
                    home_wins += 1
            hdp[f"HDP_{h:+.1f}"] = {
                "home": home_wins / len(results),
                "away": 1 - (home_wins / len(results))
            }
        return hdp
    
    def _calc_ou(self, results: List[Tuple[int, int]]) -> Dict[str, Dict]:
        """Calculate Over/Under probabilities."""
        lines = [1.5, 2.0, 2.5, 3.0, 3.5]
        ou = {}
        
        for line in lines:
            overs = sum(1 for g1, g2 in results if g1 + g2 > line)
            unders = sum(1 for g1, g2 in results if g1 + g2 < line)
            ou[f"O/U {line}"] = {
                "over_prob": overs / len(results),
                "under_prob": unders / len(results)
            }
        return ou
    
    def save(self, path: str = MODEL_PATH) -> None:
        """Save model to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump({
                'ratings': self.ratings,
                'raw_probs': self.raw_probs,
                'version': self.VERSION,
                'last_updated': self.last_updated,
                'fitted': self.fitted
            }, f)
        print(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str = MODEL_PATH) -> 'WC2026MatchModel':
        """Load model from disk."""
        with open(path, 'rb') as f:
            data = pickle.load(f)
        
        model = cls()
        model.ratings = data['ratings']
        model.raw_probs = data['raw_probs']
        model.fitted = data.get('fitted', True)
        model.last_updated = data.get('last_updated')
        return model
    
    @classmethod
    def load_from_source(cls, source_path: str = DATA_PATH) -> 'WC2026MatchModel':
        """Load and fit from v3 model source data."""
        with open(source_path) as f:
            probs = json.load(f)
        return cls().fit(probs)

# API Server for deployment
class PredictionAPI:
    """Simple HTTP API for predictions."""
    
    def __init__(self, model: WC2026MatchModel):
        self.model = model
    
    def predict(self, team1: str, team2: str) -> dict:
        """Generate prediction for API response."""
        pred = self.model.predict(team1, team2)
        return pred.to_dict()
    
    def predict_group(self, group: str) -> List[dict]:
        """Predict all matches in a group."""
        teams = WC2026_GROUPS.get(group.upper(), [])
        predictions = []
        
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                pred = self.model.predict(t1, t2)
                predictions.append(pred.to_dict())
        
        return predictions

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="WC2026 Match Prediction Model")
    subparsers = parser.add_subparsers(dest='command')
    
    # Fit command
    fit_parser = subparsers.add_parser('fit', help='Fit model from tournament probs')
    fit_parser.add_argument('--source', default=DATA_PATH, help='Path to tournament probs JSON')
    fit_parser.add_argument('--save', action='store_true', help='Save fitted model')
    
    # Predict command
    pred_parser = subparsers.add_parser('predict', help='Predict match')
    pred_parser.add_argument('--match', required=True, help='Team1,Team2')
    pred_parser.add_argument('--model', default=MODEL_PATH, help='Path to model file')
    
    # Group command
    group_parser = subparsers.add_parser('group', help='Predict all matches in group')
    group_parser.add_argument('--group', required=True, help='Group letter (A-L)')
    group_parser.add_argument('--format', choices=['json', 'table'], default='table')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--port', type=int, default=8000)
    
    # Batch command
    batch_parser = subparsers.add_parser('batch', help='Batch predict all groups')
    batch_parser.add_argument('--output', '-o', help='Output JSON file')
    
    args = parser.parse_args()
    
    if args.command == 'fit':
        print(f"Fitting model from {args.source}...")
        model = WC2026MatchModel.load_from_source(args.source)
        print(f"Model fitted with {len(model.ratings)} teams")
        print(f"Top ratings: {sorted(model.ratings.items(), key=lambda x: x[1], reverse=True)[:5]}")
        
        if args.save:
            model.save()
    
    elif args.command == 'predict':
        t1, t2 = args.match.split(',')
        
        if os.path.exists(args.model):
            model = WC2026MatchModel.load(args.model)
        else:
            print("Model not found, loading from source...")
            model = WC2026MatchModel.load_from_source()
        
        pred = model.predict(t1, t2)
        print(json.dumps(pred.to_dict(), indent=2))
    
    elif args.command == 'group':
        model = WC2026MatchModel.load_from_source()
        api = PredictionAPI(model)
        preds = api.predict_group(args.group)
        
        if args.format == 'json':
            print(json.dumps(preds, indent=2))
        else:
            print(f"\nGroup {args.group.upper()} Predictions")
            print("=" * 60)
            for p in preds:
                print(f"\n{p['teams']['home']} vs {p['teams']['away']}")
                print(f"  1X2: {p['1x2']['home_win']['prob']*100:.0f}% / "
                      f"{p['1x2']['draw']['prob']*100:.0f}% / "
                      f"{p['1x2']['away_win']['prob']*100:.0f}%")
                print(f"  xG: {p['expected_goals']['home']:.1f} - {p['expected_goals']['away']:.1f}")
                print(f"  BTTS: {p['btts']['yes']*100:.0f}%")
    
    elif args.command == 'batch':
        model = WC2026MatchModel.load_from_source()
        api = PredictionAPI(model)
        
        all_predictions = {}
        for group in WC2026_GROUPS.keys():
            all_predictions[group] = api.predict_group(group)
        
        output = {
            "timestamp": datetime.utcnow().isoformat(),
            "model_version": WC2026MatchModel.VERSION,
            "groups": all_predictions
        }
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"Saved {len(WC2026_GROUPS)} groups to {args.output}")
        else:
            print(json.dumps(output, indent=2))
    
    elif args.command == 'serve':
        # Simple Flask-like server
        try:
            from http.server import HTTPServer, BaseHTTPRequestHandler
            import urllib.parse
            
            model = WC2026MatchModel.load_from_source()
            api = PredictionAPI(model)
            
            class Handler(BaseHTTPRequestHandler):
                def do_GET(self):
                    parsed = urllib.parse.urlparse(self.path)
                    
                    if parsed.path == '/predict':
                        params = urllib.parse.parse_qs(parsed.query)
                        t1 = params.get('team1', [''])[0]
                        t2 = params.get('team2', [''])[0]
                        
                        if t1 and t2:
                            result = api.predict(t1, t2)
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps(result).encode())
                        else:
                            self.send_error(400, "Missing team1 or team2")
                    
                    elif parsed.path == '/group':
                        params = urllib.parse.parse_qs(parsed.query)
                        group = params.get('group', [''])[0]
                        
                        if group:
                            result = api.predict_group(group)
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps(result).encode())
                        else:
                            self.send_error(400, "Missing group")
                    
                    elif parsed.path == '/health':
                        self.send_response(200)
                        self.send_header('Content-type', 'application/json')
                        self.end_headers()
                        self.wfile.write(json.dumps({
                            "status": "ok",
                            "model_version": WC2026MatchModel.VERSION,
                            "teams_loaded": len(model.ratings)
                        }).encode())
                    
                    else:
                        self.send_error(404)
                
                def log_message(self, format, *args):
                    pass  # Suppress logs
            
            server = HTTPServer(('0.0.0.0', args.port), Handler)
            print(f"API server running on http://0.0.0.0:{args.port}")
            print(f"Endpoints:")
            print(f"  GET /health")
            print(f"  GET /predict?team1=France&team2=Brazil")
            print(f"  GET /group?group=H")
            server.serve_forever()
            
        except KeyboardInterrupt:
            print("\nServer stopped")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
