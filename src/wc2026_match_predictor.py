#!/usr/bin/env python3
"""
WC2026 Match Prediction Model
Monte Carlo-based tournament simulator with group stage and knockout bracket.
Derives match-level W/D/L from team strength ratings.
"""

import json
import random
import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import os

# WC2026 Groups (from FIFA)
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

# Name normalization
NAME_MAP = {
    "Korea Republic": "South Korea",
    "Iran": "IR Iran",
    "Cape Verde": "Cabo Verde",
    "Ivory Coast": "Côte d'Ivoire",
    "Congo Dr": "Congo DR",
    "Turkiye": "Turkey",
}

def normalize_name(name: str) -> str:
    return NAME_MAP.get(name, name)

def reverse_normalize(name: str) -> str:
    rev = {v: k for k, v in NAME_MAP.items()}
    return rev.get(name, name)

@dataclass
class MatchResult:
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    winner: Optional[str] = None  # None for draw
    
    @property
    def is_draw(self) -> bool:
        return self.home_goals == self.away_goals
    
    @property
    def goal_diff(self) -> int:
        return self.home_goals - self.away_goals

@dataclass 
class GroupStanding:
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    
    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against
    
    def add_match(self, result: MatchResult, is_home: bool):
        self.played += 1
        if is_home:
            self.goals_for += result.home_goals
            self.goals_against += result.away_goals
            if result.winner == self.team:
                self.wins += 1
                self.points += 3
            elif result.is_draw:
                self.draws += 1
                self.points += 1
            else:
                self.losses += 1
        else:
            self.goals_for += result.away_goals
            self.goals_against += result.home_goals
            if result.winner == self.team:
                self.wins += 1
                self.points += 3
            elif result.is_draw:
                self.draws += 1
                self.points += 1
            else:
                self.losses += 1

class MatchPredictor:
    """Predicts match outcomes from team strength ratings."""
    
    def __init__(self, team_ratings: Dict[str, float]):
        """
        team_ratings: Dict[team_name, tournament_win_probability]
        """
        self.raw_ratings = team_ratings
        # Convert win probabilities to Elo-like ratings
        self.ratings = self._compute_ratings(team_ratings)
        # Goal scoring parameters (estimated from historical data)
        self.avg_goals_per_match = 2.6
        self.home_advantage = 0.15  # 15% boost for home-like conditions
        
    def _compute_ratings(self, win_probs: Dict[str, float]) -> Dict[str, float]:
        """Convert tournament win probabilities to team strength ratings.
        
        Tournament win prob = strength * difficulty_of_path
        We extract strength by normalizing to the max probability.
        """
        # Find max probability (strongest team by model)
        max_prob = max(win_probs.values())
        
        ratings = {}
        for team, prob in win_probs.items():
            if prob > 0:
                # Team strength = relative to best team
                # Use log scale to spread out differences
                strength = math.log(prob / max_prob + 0.01)  # +0.01 to avoid -inf
                # Scale to Elo-like range: 2000 = best team, ~1500 = weakest
                rating = 2000 + 200 * strength
                ratings[team] = max(1300, min(2100, rating))
            else:
                ratings[team] = 1300
        return ratings
    
    def get_rating(self, team: str) -> float:
        team_norm = normalize_name(team)
        if team_norm in self.ratings:
            return self.ratings[team_norm]
        # Fallback: interpolate from similar teams
        return 1500
    
    def predict_match(self, team1: str, team2: str, neutral: bool = True) -> MatchResult:
        """
        Predict a single match with W/D/L probabilities and simulated score.
        Returns MatchResult with goals and winner.
        """
        r1, r2 = self.get_rating(team1), self.get_rating(team2)
        
        # Expected score probability (Elo-based)
        expected = 1 / (1 + 10 ** ((r2 - r1) / 400))
        
        # Add home advantage if not neutral
        if not neutral:
            expected += self.home_advantage
            expected = min(0.95, max(0.05, expected))
        
        # Determine outcome by sampling
        u = random.random()
        
        # Draw probability peaks when teams are even
        draw_prob = 0.25 - 0.15 * abs(expected - 0.5)  # 10-25% draw rate
        draw_prob = max(0.08, min(0.35, draw_prob))
        
        # Adjust win probs to account for draws
        t1_win = expected * (1 - draw_prob)
        t2_win = (1 - expected) * (1 - draw_prob)
        
        # Normalize
        total = t1_win + t2_win + draw_prob
        t1_win /= total
        t2_win /= total
        draw_prob /= total
        
        # Simulate goals (Poisson-like distribution)
        if u < t1_win:
            # Team 1 wins
            t1_goals = max(1, self._sample_goals(expected * 1.5))
            t2_goals = self._sample_goals((1 - expected) * 0.8)
            t2_goals = min(t2_goals, t1_goals - 1) if t2_goals >= t1_goals else t2_goals
            return MatchResult(team1, team2, t1_goals, t2_goals, team1)
        elif u < t1_win + draw_prob:
            # Draw
            goals = self._sample_goals(self.avg_goals_per_match / 2)
            return MatchResult(team1, team2, goals, goals, None)
        else:
            # Team 2 wins
            t2_goals = max(1, self._sample_goals((1 - expected) * 1.5))
            t1_goals = self._sample_goals(expected * 0.8)
            t1_goals = min(t1_goals, t2_goals - 1) if t1_goals >= t2_goals else t1_goals
            return MatchResult(team1, team2, t1_goals, t2_goals, team2)
    
    def _sample_goals(self, expected: float) -> int:
        """Sample goals from Poisson-like distribution."""
        # Simple approximation
        u = random.random()
        goals = 0
        cum_prob = 0.0
        while goals < 10:
            prob = (expected ** goals) * math.exp(-expected) / math.factorial(goals)
            cum_prob += prob
            if u < cum_prob:
                return goals
            goals += 1
        return goals
    
    def get_match_odds(self, team1: str, team2: str, neutral: bool = True) -> Dict:
        """Return W/D/L probabilities for a match."""
        r1, r2 = self.get_rating(team1), self.get_rating(team2)
        expected = 1 / (1 + 10 ** ((r2 - r1) / 400))
        
        if not neutral:
            expected += self.home_advantage
            expected = min(0.95, max(0.05, expected))
        
        draw_prob = 0.25 - 0.15 * abs(expected - 0.5)
        draw_prob = max(0.08, min(0.35, draw_prob))
        
        t1_win = expected * (1 - draw_prob)
        t2_win = (1 - expected) * (1 - draw_prob)
        
        total = t1_win + t2_win + draw_prob
        
        return {
            "team1": team1,
            "team2": team2,
            "team1_win": round(t1_win / total, 4),
            "draw": round(draw_prob / total, 4),
            "team2_win": round(t2_win / total, 4),
            "team1_implied": round(1 / (t1_win / total), 2) if t1_win > 0 else None,
            "draw_implied": round(1 / (draw_prob / total), 2) if draw_prob > 0 else None,
            "team2_implied": round(1 / (t2_win / total), 2) if t2_win > 0 else None,
        }

class GroupStageSimulator:
    """Simulates WC2026 group stage (4 teams, round-robin)."""
    
    def simulate_group(self, name: str, teams: List[str], 
                       predictor: MatchPredictor) -> Dict:
        """Simulate all 6 matches in a group and return standings."""
        standings = {t: GroupStanding(t) for t in teams}
        matches = []
        
        # Round-robin: each team plays every other once
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                result = predictor.predict_match(t1, t2, neutral=True)
                matches.append(result)
                standings[t1].add_match(result, True)
                standings[t2].add_match(result, False)
        
        # Sort by FIFA rules: points, GD, GF, head-to-head
        sorted_teams = sorted(
            standings.values(),
            key=lambda s: (s.points, s.goal_diff, s.goals_for),
            reverse=True
        )
        
        # Handle ties (simplified: random for now)
        # Full implementation would check head-to-head
        
        return {
            "group": name,
            "teams": [
                {
                    "rank": i + 1,
                    "team": s.team,
                    "played": s.played,
                    "wins": s.wins,
                    "draws": s.draws,
                    "losses": s.losses,
                    "gf": s.goals_for,
                    "ga": s.goals_against,
                    "gd": s.goal_diff,
                    "points": s.points,
                    "qualified": i < 2  # Top 2 advance
                }
                for i, s in enumerate(sorted_teams)
            ],
            "matches": [
                {
                    "home": m.home_team,
                    "away": m.away_team,
                    "score": f"{m.home_goals}-{m.away_goals}",
                    "winner": m.winner
                }
                for m in matches
            ]
        }

class KnockoutSimulator:
    """Simulates WC2026 knockout bracket."""
    
    # WC2026 R16 mapping: Group winner vs Runner-up from paired group
    R16_MAPPING = [
        ("1A", "2B"), ("1C", "2D"), ("1E", "2F"), ("1G", "2H"),
        ("1B", "2A"), ("1D", "2C"), ("1F", "2E"), ("1H", "2G"),
        ("1I", "2J"), ("1K", "2L"), ("1J", "2I"), ("1L", "2K"),
    ]
    
    def build_bracket(self, group_results: Dict[str, Dict]) -> List[Tuple[str, str]]:
        """Build R16 matchups from group results."""
        bracket = []
        for pos1, pos2 in self.R16_MAPPING:
            g1, rank1 = pos1[1], int(pos1[0])
            g2, rank2 = pos2[1], int(pos2[0])
            
            team1 = group_results[g1]["teams"][rank1 - 1]["team"]
            team2 = group_results[g2]["teams"][rank2 - 1]["team"]
            bracket.append((team1, team2))
        
        return bracket
    
    def simulate_knockout_match(self, team1: str, team2: str, 
                                 predictor: MatchPredictor) -> str:
        """Simulate single knockout match (ET/penalties if draw)."""
        result = predictor.predict_match(team1, team2, neutral=True)
        
        if result.winner:
            return result.winner
        
        # Draw - go to extra time/penalties
        # Simplified: 50/50 with slight favorite bias
        r1, r2 = predictor.get_rating(team1), predictor.get_rating(team2)
        fav_prob = 0.5 + 0.1 * ((r1 - r2) / 400)
        fav_prob = max(0.3, min(0.7, fav_prob))
        
        return team1 if random.random() < fav_prob else team2
    
    def simulate_bracket(self, bracket: List[Tuple[str, str]], 
                         predictor: MatchPredictor) -> Dict:
        """Simulate full knockout bracket."""
        # R16
        r16_winners = []
        r16_results = []
        for t1, t2 in bracket:
            winner = self.simulate_knockout_match(t1, t2, predictor)
            r16_winners.append(winner)
            r16_results.append({"match": f"{t1} vs {t2}", "winner": winner})
        
        # Quarter-finals (QF)
        qf_pairs = [(r16_winners[i], r16_winners[i+1]) 
                    for i in range(0, len(r16_winners), 2)]
        qf_winners = []
        qf_results = []
        for t1, t2 in qf_pairs:
            winner = self.simulate_knockout_match(t1, t2, predictor)
            qf_winners.append(winner)
            qf_results.append({"match": f"{t1} vs {t2}", "winner": winner})
        
        # Semi-finals (SF)
        sf_pairs = [(qf_winners[i], qf_winners[i+1]) 
                    for i in range(0, len(qf_winners), 2)]
        sf_winners = []
        sf_results = []
        for t1, t2 in sf_pairs:
            winner = self.simulate_knockout_match(t1, t2, predictor)
            sf_winners.append(winner)
            sf_results.append({"match": f"{t1} vs {t2}", "winner": winner})
        
        # Final + 3rd place
        finalist1, finalist2 = sf_winners[0], sf_winners[1]
        third_place_teams = [sf_results[0]["match"].split(" vs ")[0],
                            sf_results[0]["match"].split(" vs ")[1],
                            sf_results[1]["match"].split(" vs ")[0],
                            sf_results[1]["match"].split(" vs ")[1]]
        third_place_teams = [t for t in third_place_teams if t not in sf_winners]
        
        champion = self.simulate_knockout_match(finalist1, finalist2, predictor)
        third_place = self.simulate_knockout_match(third_place_teams[0], 
                                                    third_place_teams[1], predictor)
        
        return {
            "r16": r16_results,
            "qf": qf_results,
            "sf": sf_results,
            "final": {"match": f"{finalist1} vs {finalist2}", "winner": champion},
            "third_place": {"match": f"{third_place_teams[0]} vs {third_place_teams[1]}", 
                          "winner": third_place},
            "champion": champion,
            "runner_up": finalist2 if champion == finalist1 else finalist1,
            "third_place_winner": third_place
        }

class WC2026Predictor:
    """Full WC2026 prediction system."""
    
    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = os.path.expanduser("~/.hermes/data/wc2026_signals/model_predictions.json")
        
        with open(model_path) as f:
            self.win_probs = json.load(f)
        
        self.predictor = MatchPredictor(self.win_probs)
        self.group_sim = GroupStageSimulator()
        self.knockout_sim = KnockoutSimulator()
    
    def simulate_tournament(self, runs: int = 1000) -> Dict:
        """Run Monte Carlo simulation of full tournament."""
        winners = defaultdict(int)
        finals = defaultdict(int)
        semis = defaultdict(int)
        quarters = defaultdict(int)
        r16 = defaultdict(int)
        group_winners = defaultdict(int)
        
        all_simulations = []
        
        for _ in range(runs):
            # Group stage
            group_results = {}
            for name, teams in WC2026_GROUPS.items():
                result = self.group_sim.simulate_group(name, teams, self.predictor)
                group_results[name] = result
                for t in result["teams"][:2]:  # Top 2
                    group_winners[t["team"]] += 1
            
            # Knockout
            bracket = self.knockout_sim.build_bracket(group_results)
            for m in bracket:
                r16[m[0]] += 1
                r16[m[1]] += 1
            
            knockout = self.knockout_sim.simulate_bracket(bracket, self.predictor)
            
            for match in knockout["qf"]:
                quarters[match["winner"]] += 1
            for match in knockout["sf"]:
                semis[match["winner"]] += 1
            
            finals[knockout["champion"]] += 1
            finals[knockout["runner_up"]] += 1
            winners[knockout["champion"]] += 1
            
            if len(all_simulations) < 3:  # Save sample runs
                all_simulations.append({
                    "groups": group_results,
                    "knockout": knockout
                })
        
        # Compute probabilities
        def to_prob(counts):
            return {k: round(v / runs, 4) for k, v in sorted(counts.items(), 
                                                            key=lambda x: x[1], reverse=True)}
        
        return {
            "simulations": runs,
            "winner_probabilities": to_prob(winners),
            "final_probabilities": to_prob(finals),
            "semi_probabilities": to_prob(semis),
            "quarter_probabilities": to_prob(quarters),
            "r16_probabilities": {k: round(v / runs, 4) for k, v in sorted(r16.items(),
                                                                          key=lambda x: x[1], reverse=True)},
            "sample_runs": all_simulations
        }
    
    def predict_match(self, team1: str, team2: str) -> Dict:
        """Get W/D/L odds for a specific match."""
        return self.predictor.get_match_odds(team1, team2)
    
    def predict_group_matches(self, group_name: str) -> List[Dict]:
        """Predict all matches in a group with odds."""
        teams = WC2026_GROUPS[group_name]
        predictions = []
        
        for i, t1 in enumerate(teams):
            for t2 in teams[i+1:]:
                odds = self.predictor.get_match_odds(t1, t2)
                predictions.append({
                    "match": f"{t1} vs {t2}",
                    "odds": odds,
                    "expected_winner": t1 if odds["team1_win"] > odds["team2_win"] else t2
                })
        
        return predictions

def main():
    """Run prediction and output results."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--match", nargs=2, metavar=("TEAM1", "TEAM2"),
                      help="Predict single match odds")
    parser.add_argument("--group", help="Predict all matches in group (A-L)")
    parser.add_argument("--output", "-o", help="Output JSON file")
    args = parser.parse_args()
    
    predictor = WC2026Predictor()
    
    if args.match:
        team1, team2 = args.match
        odds = predictor.predict_match(team1, team2)
        print(f"\n{team1} vs {team2}")
        print(f"  {team1} win: {odds['team1_win']*100:.1f}% (implied: {odds['team1_implied']:.2f}x)")
        print(f"  Draw:       {odds['draw']*100:.1f}% (implied: {odds['draw_implied']:.2f}x)")
        print(f"  {team2} win: {odds['team2_win']*100:.1f}% (implied: {odds['team2_implied']:.2f}x)")
    
    elif args.group:
        preds = predictor.predict_group_matches(args.group.upper())
        print(f"\nGroup {args.group.upper()} Match Predictions")
        print("=" * 50)
        for p in preds:
            odds = p["odds"]
            print(f"\n{p['match']}:")
            print(f"  {odds['team1']} win: {odds['team1_win']*100:.1f}%")
            print(f"  Draw:         {odds['draw']*100:.1f}%")
            print(f"  {odds['team2']} win: {odds['team2_win']*100:.1f}%")
    
    else:
        print(f"Running {args.runs} Monte Carlo simulations...")
        results = predictor.simulate_tournament(runs=args.runs)
        
        print("\n" + "=" * 60)
        print("WC2026 TOURNAMENT SIMULATION RESULTS")
        print("=" * 60)
        
        print("\n🏆 WINNER PROBABILITIES (Top 10):")
        for team, prob in list(results["winner_probabilities"].items())[:10]:
            print(f"  {team:20s}: {prob*100:5.2f}%  (implied: {1/prob:.1f}x)")
        
        print("\n🥈 FINAL APPEARANCE (Top 10):")
        for team, prob in list(results["final_probabilities"].items())[:10]:
            print(f"  {team:20s}: {prob*100:5.2f}%")
        
        print("\n⚽ SEMI-FINAL APPEARANCE (Top 10):")
        for team, prob in list(results["semi_probabilities"].items())[:10]:
            print(f"  {team:20s}: {prob*100:5.2f}%")
        
        # Sample run
        print("\n📋 SAMPLE TOURNAMENT RUN:")
        sample = results["sample_runs"][0]
        print("\nGroup Stage:")
        for gname, gdata in sample["groups"].items():
            qualified = [t["team"] for t in gdata["teams"][:2]]
            print(f"  Group {gname}: {qualified[0]}, {qualified[1]} advance")
        
        ko = sample["knockout"]
        print(f"\nChampion: {ko['champion']}")
        print(f"Runner-up: {ko['runner_up']}")
        print(f"3rd Place: {ko['third_place_winner']}")
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nFull results saved to {args.output}")
    
    return predictor

if __name__ == "__main__":
    main()
