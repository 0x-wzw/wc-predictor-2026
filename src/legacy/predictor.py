#!/usr/bin/env python3
"""
Matchup Predictor Core
======================
Predict World Cup match outcomes using historical data.
"""

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

DATA_DIR = os.environ.get('WC_DATA_DIR', './data')
OUTPUT_DIR = os.environ.get('WC_OUTPUT_DIR', './results')


@dataclass
class PredictionResult:
    """Prediction result for a matchup"""
    home_team: str
    away_team: str
    home_win_prob: float
    away_win_prob: float
    draw_prob: float
    confidence: float
    key_factors: Dict


class MatchupPredictor:
    """Predict WC match outcomes using historical performance"""
    
    def __init__(self, data_dir: str = None):
        self.data_dir = data_dir or DATA_DIR
        self.team_profiles = self._load_profiles()
        self.h2h_cache = {}
    
    def _load_profiles(self) -> Dict:
        """Load team performance profiles"""
        profile_path = os.path.join(self.data_dir, 'team_historical_metrics.json')
        if not os.path.exists(profile_path):
            # Return empty if no data
            return {}
        with open(profile_path, 'r') as f:
            return json.load(f)
    
    def get_available_teams(self) -> List[str]:
        """List teams available for prediction"""
        return list(self.team_profiles.keys())
    
    def get_team_stats(self, team: str) -> Optional[Dict]:
        """Get historical stats for a team"""
        return self.team_profiles.get(team)
    
    def _calculate_form_factor(self, team: str) -> float:
        """Calculate recent form factor (0.0-1.0)"""
        profile = self.team_profiles.get(team, {})
        
        # Weight recent tournaments more
        wc_appearances = profile.get('world_cup_appearances', 0)
        recent_appearances = profile.get('recent_appearances', 0)
        avg_stage = profile.get('avg_stage_reached', 50)
        
        # Normalize
        appearance_score = min(recent_appearances / 5, 1.0)
        stage_score = max(0, 1.0 - (avg_stage / 32))  # Closer to 1 = better
        
        return (appearance_score * 0.5 + stage_score * 0.5)
    
    def _calculate_experience_gap(self, team1: str, team2: str) -> float:
        """Calculate experience gap between teams (-1.0 to 1.0)"""
        p1 = self.team_profiles.get(team1, {})
        p2 = self.team_profiles.get(team2, {})
        
        form1 = self._calculate_form_factor(team1)
        form2 = self._calculate_form_factor(team2)
        
        return form1 - form2
    
    def _calculate_h2h_advantage(self, team1: str, team2: str) -> float:
        """Calculate head-to-head advantage from historical data"""
        cache_key = tuple(sorted([team1, team2]))
        
        if cache_key in self.h2h_cache:
            return self.h2h_cache[cache_key]
        
        # Load historical matches for h2h
        matches_path = os.path.join(self.data_dir, 'world_cup_matches.json')
        if not os.path.exists(matches_path):
            return 0.0
        
        try:
            with open(matches_path, 'r') as f:
                data = json.load(f)
        except:
            return 0.0
        
        team1_wins = 0
        team2_wins = 0
        total = 0
        
        for match in data.get('matches', []):
            home = match.get('home_team')
            away = match.get('away_team')
            
            if {home, away} != {team1, team2}:
                continue
            
            total += 1
            home_goals = match.get('home_goals')
            away_goals = match.get('away_goals')
            
            if not isinstance(home_goals, int) or not isinstance(away_goals, int):
                continue
            
            if home_goals > away_goals:
                if home == team1:
                    team1_wins += 1
                else:
                    team2_wins += 1
            elif away_goals > home_goals:
                if away == team1:
                    team1_wins += 1
                else:
                    team2_wins += 1
        
        if total < 1:
            return 0.0
        
        # Advantage for team1
        team1_win_rate = team1_wins / total
        advantage = (team1_win_rate - 0.5) * 2  # Normalize to -1 to 1
        self.h2h_cache[cache_key] = advantage
        
        return advantage
    
    def predict(self, home_team: str, away_team: str, 
                neutral_venue: bool = True) -> PredictionResult:
        """
        Predict match outcome.
        
        Args:
            home_team: Home team name
            away_team: Away team name
            neutral_venue: If true, no home advantage
            
        Returns:
            PredictionResult with probabilities
        """
        if home_team not in self.team_profiles:
            raise ValueError(f"Team not found: {home_team}")
        if away_team not in self.team_profiles:
            raise ValueError(f"Team not found: {away_team}")
        
        # Calculate factors
        experience_gap = self._calculate_experience_gap(home_team, away_team)
        h2h_advantage = self._calculate_h2h_advantage(home_team, away_team)
        
        # Base probabilities (50/50)
        base_home = 0.33
        base_draw = 0.33
        base_away = 0.33
        
        # Adjust for experience
        exp_factor = experience_gap * 0.15
        base_home += exp_factor
        base_away -= exp_factor
        
        # Adjust for h2h
        h2h_factor = h2h_advantage * 0.1
        base_home += h2h_factor
        base_away -= h2h_factor
        
        # Normalize
        total = base_home + base_draw + base_away
        home_win_prob = base_home / total
        draw_prob = base_draw / total
        away_win_prob = base_away / total
        
        # Confidence based on data quality
        profile1 = self.team_profiles[home_team]
        profile2 = self.team_profiles[away_team]
        matches_count = min(
            profile1.get('total_matches', 10),
            profile2.get('total_matches', 10)
        )
        confidence = min(matches_count / 50, 1.0)
        
        return PredictionResult(
            home_team=home_team,
            away_team=away_team,
            home_win_prob=round(max(0.05, min(0.9, home_win_prob)), 3),
            away_win_prob=round(max(0.05, min(0.9, away_win_prob)), 3),
            draw_prob=round(max(0.1, min(0.4, draw_prob)), 3),
            confidence=round(confidence, 2),
            key_factors={
                'experience_gap': round(experience_gap, 2),
                'h2h_advantage': round(h2h_advantage, 2),
                'home_advantage': 0.0 if neutral_venue else 0.05
            }
        )
    
    def get_tournament_probabilities(self) -> Dict:
        """Get tournament winner probabilities from simulation results"""
        sim_path = os.path.join(OUTPUT_DIR, 'probabilities.json')
        if os.path.exists(sim_path):
            with open(sim_path, 'r') as f:
                return json.load(f)
        return {}


# Simple CLI interface
if __name__ == "__main__":
    import sys
    
    predictor = MatchupPredictor()
    
    if len(sys.argv) > 2:
        team1, team2 = sys.argv[1], sys.argv[2]
        result = predictor.predict(team1, team2)
        
        print(f"\nPrediction: {team1} vs {team2}")
        print(f"  {team1} win: {result.home_win_prob:.1%}")
        print(f"  Draw: {result.draw_prob:.1%}")
        print(f"  {team2} win: {result.away_win_prob:.1%}")
        print(f"  Confidence: {result.confidence:.1%}")
    else:
        print("Usage: python predictor.py "<Team1>" "<Team2>"")
        print(f"\nAvailable teams: {len(predictor.get_available_teams())}")
