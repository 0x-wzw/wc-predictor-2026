#!/usr/bin/env python3
"""
WC2026 Match Prediction Model - Enhanced with Asian Handicap
=============================================================
Monte Carlo-based tournament simulator with:
- Match-level W/D/L predictions
- Asian Handicap lines (HDP -0.5, -1.0, -1.5, +0.5, etc.)
- Over/Under lines (2.5, 3.0, 3.5)
- Both Teams to Score (BTTS)
- Corner predictions
- Correct score distribution
"""

import json
import random
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

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

NAME_MAP = {"Korea Republic": "South Korea", "Iran": "IR Iran", "Cape Verde": "Cabo Verde",
            "Ivory Coast": "Côte d'Ivoire", "Congo Dr": "Congo DR", "Turkiye": "Turkey"}

def normalize_name(name: str) -> str:
    return NAME_MAP.get(name, name)

@dataclass
class MatchResult:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    winner: Optional[str] = None
    
    @property
    def is_draw(self): return self.home_goals == self.away_goals
    
    @property
    def goal_diff(self): return self.home_goals - self.away_goals
    
    @property
    def total_goals(self): return self.home_goals + self.away_goals
    
    @property
    def btts(self): return self.home_goals > 0 and self.away_goals > 0

@dataclass 
class AsianHandicapResult:
    """Asian Handicap bet result"""
    handicap: float  # e.g., -0.5, -1.0, -1.25, +0.5, +1.0
    line_name: str   # e.g., "HDP -0.5", "HDP -1.0"
    home_win: bool   # Did home team win on this handicap?
    push: bool       # Was it a push (for whole numbers)?
    
@dataclass
class MatchPrediction:
    """Full prediction for a match"""
    home_team: str
    away_team: str
    
    # 1X2 Odds
    home_win_prob: float
    draw_prob: float
    away_win_prob: float
    
    # Expected goals
    home_xg: float
    away_xg: float
    
    # Asian Handicap probabilities
    hdp_lines: Dict[str, Dict[str, float]]  # line -> {home_prob, push_prob, away_prob}
    
    # Over/Under
    ou_lines: Dict[str, Dict[str, float]]   # line -> {over_prob, under_prob, push_prob}
    
    # BTTS
    btts_yes_prob: float
    btts_no_prob: float
    
    # Correct score distribution (top 10)
    correct_score: List[Tuple[str, float]]
    
    # First half predictions
    fh_home_prob: float
    fh_draw_prob: float
    fh_away_prob: float
    
    def to_dict(self) -> dict:
        return {
            "match": f"{self.home_team} vs {self.away_team}",
            "1x2": {
                "home": {"prob": round(self.home_win_prob, 4), "implied": round(1/self.home_win_prob, 2) if self.home_win_prob > 0 else None},
                "draw": {"prob": round(self.draw_prob, 4), "implied": round(1/self.draw_prob, 2) if self.draw_prob > 0 else None},
                "away": {"prob": round(self.away_win_prob, 4), "implied": round(1/self.away_prob, 2) if self.away_win_prob > 0 else None},
            },
            "asian_handicap": self.hdp_lines,
            "over_under": self.ou_lines,
            "btts": {"yes": round(self.btts_yes_prob, 4), "no": round(self.btts_no_prob, 4)},
            "expected_goals": {"home": round(self.home_xg, 2), "away": round(self.away_xg, 2)},
            "correct_score": self.correct_score[:5],
        }

class EnhancedMatchPredictor:
    """Enhanced predictor with Asian Handicap, O/U, BTTS capabilities"""
    
    def __init__(self, team_ratings: Dict[str, float]):
        self.raw_probs = team_ratings
        self.ratings = self._compute_ratings(team_ratings)
        self.avg_goals = 2.6
        self.home_advantage = 0.15
        
    def _compute_ratings(self, win_probs: Dict) -> Dict:
        max_prob = max(win_probs.values())
        ratings = {}
        for team, prob in win_probs.items():
            if prob > 0:
                strength = math.log(prob / max_prob + 0.01)
                ratings[team] = max(1300, min(2100, 2000 + 200 * strength))
            else:
                ratings[team] = 1300
        return ratings
    
    def get_rating(self, team: str) -> float:
        return self.ratings.get(normalize_name(team), 1500)
    
    def predict_match(self, team1: str, team2: str, neutral: bool = True) -> MatchPrediction:
        """Generate full match prediction with all markets"""
        r1, r2 = self.get_rating(team1), self.get_rating(team2)
        
        # Expected goals calculation
        base_xg = 1.3  # base per team
        rating_diff = (r1 - r2) / 400  # in Elo units
        
        # Home team expected goals
        home_xg = base_xg * (1 + rating_diff * 0.3)
        if not neutral:
            home_xg *= (1 + self.home_advantage)
        
        # Away team expected goals
        away_xg = base_xg * (1 - rating_diff * 0.3)
        
        # 1X2 probabilities (Dixon-Coles inspired)
        home_win, draw, away_win = self._calculate_1x2(home_xg, away_xg)
        
        # Simulate many runs for market probabilities
        n_sims = 10000
        results = []
        for _ in range(n_sims):
            hg = self._sample_poisson(home_xg)
            ag = self._sample_poisson(away_xg)
            results.append((hg, ag))
        
        # Calculate Asian Handicap probabilities
        hdp_lines = self._calculate_hdp_probabilities(results)
        
        # Calculate O/U probabilities
        ou_lines = self._calculate_ou_probabilities(results)
        
        # BTTS
        btts_yes = sum(1 for hg, ag in results if hg > 0 and ag > 0) / n_sims
        
        # Correct score distribution
        score_dist = defaultdict(int)
        for hg, ag in results:
            score_dist[f"{hg}-{ag}"] += 1
        correct_score = sorted([(s, c/n_sims) for s, c in score_dist.items()], 
                               key=lambda x: x[1], reverse=True)[:10]
        
        # First half (assume 45% of goals happen in first half on average)
        fh_home_xg = home_xg * 0.45
        fh_away_xg = away_xg * 0.45
        fh_home, fh_draw, fh_away = self._calculate_1x2(fh_home_xg, fh_away_xg)
        
        return MatchPrediction(
            home_team=team1,
            away_team=team2,
            home_win_prob=home_win,
            draw_prob=draw,
            away_win_prob=away_win,
            home_xg=home_xg,
            away_xg=away_xg,
            hdp_lines=hdp_lines,
            ou_lines=ou_lines,
            btts_yes_prob=btts_yes,
            btts_no_prob=1-btts_yes,
            correct_score=correct_score,
            fh_home_prob=fh_home,
            fh_draw_prob=fh_draw,
            fh_away_prob=fh_away
        )
    
    def _calculate_1x2(self, home_xg: float, away_xg: float) -> Tuple[float, float, float]:
        """Calculate 1X2 probabilities from expected goals"""
        # Poisson-based calculation with draw adjustment
        max_goals = 10
        
        home_win = 0
        draw = 0
        away_win = 0
        
        for hg in range(max_goals):
            for ag in range(max_goals):
                p = (self._poisson_pmf(hg, home_xg) * 
                     self._poisson_pmf(ag, away_xg))
                if hg > ag:
                    home_win += p
                elif hg == ag:
                    draw += p
                else:
                    away_win += p
        
        # Dixon-Coles draw adjustment (low-scoring draws are more likely)
        if home_xg + away_xg < 2.5:
            draw *= 1.1
            home_win *= 0.95
            away_win *= 0.95
        
        total = home_win + draw + away_win
        return home_win/total, draw/total, away_win/total
    
    def _calculate_hdp_probabilities(self, results: List[Tuple[int, int]]) -> Dict:
        """Calculate Asian Handicap win probabilities"""
        hdp_lines = {}
        
        # Common Asian Handicap lines
        lines = [-1.5, -1.0, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0, 1.5]
        
        for hdp in lines:
            home_wins = 0
            pushes = 0
            away_wins = 0
            
            for hg, ag in results:
                # Apply handicap to home team
                adj_diff = (hg + hdp) - ag
                
                if hdp == -1.0:  # Special: half loss on -1.0 if win by exactly 1
                    if hg - ag > 1:
                        home_wins += 1
                    elif hg == ag + 1:
                        pushes += 1  # Actually half loss, simplified here
                    else:
                        away_wins += 1
                elif hdp == 1.0:
                    if hg + 1 > ag:
                        home_wins += 1
                    elif hg + 1 == ag:
                        pushes += 1
                    else:
                        away_wins += 1
                else:
                    if adj_diff > 0:
                        home_wins += 1
                    elif adj_diff == 0:
                        pushes += 1
                    else:
                        away_wins += 1
            
            n = len(results)
            line_name = f"HDP {hdp:+.2f}" if hdp != 0 else "HDP 0.0"
            hdp_lines[line_name] = {
                "home_win_prob": round(home_wins/n, 4),
                "push_prob": round(pushes/n, 4),
                "away_win_prob": round(away_wins/n, 4),
                "home_implied": round(n/home_wins, 2) if home_wins > 0 else None,
            }
        
        return hdp_lines
    
    def _calculate_ou_probabilities(self, results: List[Tuple[int, int]]) -> Dict:
        """Calculate Over/Under probabilities"""
        ou_lines = {}
        
        for line in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]:
            overs = sum(1 for hg, ag in results if hg + ag > line)
            unders = sum(1 for hg, ag in results if hg + ag < line)
            pushes = sum(1 for hg, ag in results if hg + ag == line)
            
            n = len(results)
            ou_lines[f"O/U {line}"] = {
                "over_prob": round(overs/n, 4),
                "under_prob": round(unders/n, 4),
                "push_prob": round(pushes/n, 4),
                "over_implied": round(n/overs, 2) if overs > 0 else None,
                "under_implied": round(n/unders, 2) if unders > 0 else None,
            }
        
        return ou_lines
    
    def _poisson_pmf(self, k: int, lam: float) -> float:
        return (lam ** k) * math.exp(-lam) / math.factorial(k)
    
    def _sample_poisson(self, lam: float) -> int:
        u = random.random()
        cumsum = 0
        k = 0
        while k < 15:
            p = self._poisson_pmf(k, lam)
            cumsum += p
            if u < cumsum:
                return k
            k += 1
        return k

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--match", nargs=2, metavar=("T1", "T2"))
    parser.add_argument("--group", help="Predict all matches in group")
    parser.add_argument("--asian-lines", action="store_true", help="Show Asian Handicap lines")
    parser.add_argument("--ou", action="store_true", help="Show Over/Under lines")
    parser.add_argument("--btts", action="store_true", help="Show BTTS")
    parser.add_argument("--all", action="store_true", help="Show all markets")
    args = parser.parse_args()
    
    # Load model
    model_path = os.path.expanduser("~/.hermes/data/wc2026_signals/model_predictions.json")
    with open(model_path) as f:
        win_probs = json.load(f)
    
    predictor = EnhancedMatchPredictor(win_probs)
    
    show_all = args.all
    show_hdp = show_all or args.asian_lines
    show_ou = show_all or args.ou
    show_btts = show_all or args.btts
    
    if args.match:
        t1, t2 = args.match
        pred = predictor.predict_match(t1, t2)
        
        print(f"\n{'='*60}")
        print(f"{t1} vs {t2}")
        print(f"{'='*60}")
        
        print(f"\n📊 1X2 Odds:")
        print(f"  {t1} win: {pred.home_win_prob*100:.1f}% (implied: {1/pred.home_win_prob:.2f}x)")
        print(f"  Draw:      {pred.draw_prob*100:.1f}% (implied: {1/pred.draw_prob:.2f}x)")
        print(f"  {t2} win: {pred.away_win_prob*100:.1f}% (implied: {1/pred.away_win_prob:.2f}x)")
        print(f"\nExpected Goals: {t1} {pred.home_xg:.2f} - {pred.away_xg:.2f} {t2}")
        
        if show_hdp or show_all:
            print(f"\n🎯 Asian Handicap Lines:")
            for line, probs in pred.hdp_lines.items():
                print(f"  {line}: Home {probs['home_win_prob']*100:.1f}% | Push {probs['push_prob']*100:.1f}% | Away {probs['away_win_prob']*100:.1f}%")
        
        if show_ou or show_all:
            print(f"\n📈 Over/Under Lines:")
            for line, probs in pred.ou_lines.items():
                print(f"  {line}: Over {probs['over_prob']*100:.1f}% | Under {probs['under_prob']*100:.1f}%")
        
        if show_btts or show_all:
            print(f"\n🥅 Both Teams to Score:")
            print(f"  Yes: {pred.btts_yes_prob*100:.1f}%")
            print(f"  No:  {pred.btts_no_prob*100:.1f}%")
        
        if show_all:
            print(f"\n📝 Top Correct Scores:")
            for score, prob in pred.correct_score:
                print(f"  {score}: {prob*100:.2f}%")
    
    elif args.group:
        group = args.group.upper()
        teams = WC2026_GROUPS[group]
        
        print(f"\n{'='*60}")
        print(f"Group {group} - All Match Predictions")
        print(f"{'='*60}")
        
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                pred = predictor.predict_match(t1, t2)
                print(f"\n🎯 {t1} vs {t2}")
                print(f"   1X2: H{pred.home_win_prob*100:.0f}% D{pred.draw_prob*100:.0f}% A{pred.away_win_prob*100:.0f}%")
                print(f"   xG: {pred.home_xg:.1f}-{pred.away_xg:.1f} | BTTS: {pred.btts_yes_prob*100:.0f}%")
                if show_hdp:
                    hdp_05 = pred.hdp_lines.get("HDP -0.50", {})
                    print(f"   HDP -0.5: {hdp_05.get('home_win_prob', 0)*100:.0f}%")

if __name__ == "__main__":
    main()
