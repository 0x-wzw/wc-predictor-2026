#!/usr/bin/env python3
"""
Tournament Model - Full WC 2026 Simulation
==========================================
Orchestrates group stage → knockout → finals with Monte Carlo simulation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from collections import Counter
import random
import json
from datetime import datetime

from group_stage import GroupStageSimulator, WC_2026_GROUPS, GroupResult
from knockout import KnockoutSimulator, KnockoutResult
from predictor import MatchupPredictor


@dataclass
class TournamentSimulation:
    """Complete tournament Monte Carlo results"""
    runs: int
    timestamp: str
    
    # Winner probabilities
    winner_probs: Dict[str, float] = field(default_factory=dict)
    runner_up_probs: Dict[str, float] = field(default_factory=dict)
    
    # Advancement probabilities by stage
    advancement_probs: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # Structure: {"group": p, "r16": p, "qf": p, "sf": p, "final": p, "champion": p}
    
    # Most common matchups at each stage
    common_finals: List[Tuple] = field(default_factory=list)
    common_semis: List[Tuple] = field(default_factory=list)
    
    def get_team_odds(self, team: str) -> Dict:
        """Get all odds for a team"""
        return {
            "win": self.winner_probs.get(team, 0.0),
            "reach_final": self.advancement_probs.get("final", {}).get(team, 0.0),
            "reach_semis": self.advancement_probs.get("sf", {}).get(team, 0.0),
            "reach_quarters": self.advancement_probs.get("qf", {}).get(team, 0.0),
            "reach_r16": self.advancement_probs.get("r16", {}).get(team, 0.0),
        }


class TournamentModel:
    """
    Full tournament simulation with Monte Carlo analysis.
    """
    
    def __init__(self, predictor: MatchupPredictor = None, 
                 groups: Dict = None):
        self.predictor = predictor or MatchupPredictor()
        self.group_sim = GroupStageSimulator(self.predictor)
        self.ko_sim = KnockoutSimulator(self.predictor)
        self.groups = groups or WC_2026_GROUPS
    
    def run_single_tournament(self) -> Tuple[Dict[str, GroupResult], KnockoutResult]:
        """
        Run single tournament simulation.
        
        Returns:
            (group_results, knockout_result)
        """
        # Simulate groups
        group_results = self.group_sim.simulate_all_groups(self.groups)
        
        # Simulate knockout
        knockout_result = self.ko_sim.simulate_tournament(group_results)
        
        return group_results, knockout_result
    
    def run_monte_carlo(self, runs: int = 10000) -> TournamentSimulation:
        """
        Run Monte Carlo simulation to get probability distributions.
        
        Args:
            runs: Number of tournament simulations
            
        Returns:
            TournamentSimulation with aggregated results
        """
        print(f"Running {runs} tournament simulations...")
        
        # Counters for statistics
        winner_counter = Counter()
        runner_up_counter = Counter()
        third_place_counter = Counter()
        
        # Advancement tracking
        advancement = {
            "group": Counter(),      # Qualified from group
            "r16": Counter(),        # Reached R16
            "qf": Counter(),         # Reached QF
            "sf": Counter(),         # Reached SF
            "final": Counter(),      # Reached Final
            "champion": Counter()    # Won tournament
        }
        
        # Matchup tracking
        finals_matchups = Counter()
        semis_matchups = []
        
        for i in range(runs):
            if (i + 1) % 1000 == 0:
                print(f"  Completed {i + 1}/{runs} runs...")
            
            # Run one tournament
            group_results, ko_result = self.run_single_tournament()
            
            # Track winners
            winner_counter[ko_result.winner] += 1
            runner_up_counter[ko_result.runner_up] += 1
            third_place_counter[ko_result.third_place] += 1
            advancement["champion"][ko_result.winner] += 1
            
            # Track finalists
            advancement["final"][ko_result.winner] += 1
            advancement["final"][ko_result.runner_up] += 1
            finals_matchups[tuple(sorted([ko_result.winner, ko_result.runner_up]))] += 1
            
            # Track semifinalists
            for match in ko_result.matches.get("sf", []):
                if match.team1:
                    advancement["sf"][match.team1] += 1
                if match.team2:
                    advancement["sf"][match.team2] += 1
            
            # Track quarterfinalists
            for match in ko_result.matches.get("qf", []):
                if match.team1 and match.winner:
                    advancement["qf"][match.team1] += 1
                if match.team2 and match.winner:
                    advancement["qf"][match.team2] += 1
                # Losers also reached QF
                if match.team1 and match.winner == match.team2:
                    advancement["qf"][match.team1] += 1
                if match.team2 and match.winner == match.team1:
                    advancement["qf"][match.team2] += 1
            
            # Track R16 participants
            for match in ko_result.matches.get("r16", []):
                if match.team1:
                    advancement["r16"][match.team1] += 1
                if match.team2:
                    advancement["r16"][match.team2] += 1
            
            # Group qualifiers
            for group_name, group_result in group_results.items():
                for team in group_result.get_qualified(2):
                    advancement["group"][team] += 1
        
        # Calculate probabilities
        winner_probs = {team: count / runs for team, count in winner_counter.items()}
        runner_up_probs = {team: count / runs for team, count in runner_up_counter.items()}
        
        advancement_probs = {
            stage: {team: count / runs for team, count in counter.items()}
            for stage, counter in advancement.items()
        }
        
        # Top matchups
        common_finals = finals_matchups.most_common(10)
        
        return TournamentSimulation(
            runs=runs,
            timestamp=datetime.now().isoformat(),
            winner_probs=winner_probs,
            runner_up_probs=runner_up_probs,
            advancement_probs=advancement_probs,
            common_finals=common_finals,
            common_semis=semis_matchups
        )
    
    def predict_group_standings(self, group_name: str) -> Dict:
        """Predict standings for a specific group with uncertainty"""
        teams = self.groups[group_name]
        
        # Run multiple simulations
        position_counts = {team: {1: 0, 2: 0, 3: 0, 4: 0} for team in teams}
        qualified_counts = {team: 0 for team in teams}
        
        n_runs = 1000
        for _ in range(n_runs):
            result = self.group_sim.simulate_group(group_name, teams)
            for standing in result.standings:
                position_counts[standing.team][standing.rank] += 1
                if standing.advanced:
                    qualified_counts[standing.team] += 1
        
        # Format results
        predictions = []
        for team in teams:
            probs = {pos: count / n_runs for pos, count in position_counts[team].items()}
            predictions.append({
                "team": team,
                "expected_position": sum(pos * prob for pos, prob in probs.items()),
                "qualification_prob": qualified_counts[team] / n_runs,
                "position_probabilities": {k: round(v, 3) for k, v in probs.items()}
            })
        
        predictions.sort(key=lambda x: x["expected_position"])
        return {
            "group": group_name,
            "teams": predictions
        }
    
    def get_live_odds(self) -> Dict:
        """Get current winner odds from prediction"""
        result = self.run_monte_carlo(runs=1000)  # Quick run
        
        return {
            "timestamp": result.timestamp,
            "simulations": result.runs,
            "odds": {
                team: {
                    "implied_probability": prob,
                    "decimal_odds": 1 / prob if prob > 0 else 999
                }
                for team, prob in sorted(result.winner_probs.items(), 
                                        key=lambda x: x[1], reverse=True)[:20]
            }
        }


def print_results(simulation: TournamentSimulation, top_n: int = 15):
    """Pretty print tournament simulation results"""
    print("\n" + "="*80)
    print("WORLD CUP 2026 - TOURNAMENT SIMULATION RESULTS")
    print(f"Simulations: {simulation.runs:,} | Generated: {simulation.timestamp}")
    print("="*80)
    
    print("\n🏆 WINNER PROBABILITIES (Top {}):".format(top_n))
    print(f"{'Rank':<6} {'Team':<20} {'Win %':>10} {'Final %':>10} {'Semi %':>10} {'Odds':>10}")
    print("-"*80)
    
    sorted_teams = sorted(simulation.winner_probs.items(), 
                          key=lambda x: x[1], reverse=True)[:top_n]
    
    for rank, (team, win_prob) in enumerate(sorted_teams, 1):
        final_prob = simulation.advancement_probs.get("final", {}).get(team, 0)
        semi_prob = simulation.advancement_probs.get("sf", {}).get(team, 0)
        odds = 1 / win_prob if win_prob > 0 else 999
        
        print(f"{rank:<6} {team:<20} {win_prob*100:>9.2f}% {final_prob*100:>9.1f}% {semi_prob*100:>9.1f}% {odds:>10.1f}x")
    
    print("\n🥈 MOST LIKELY FINALS:")
    print("-"*80)
    for matchup, count in simulation.common_finals[:5]:
        prob = count / simulation.runs * 100
        print(f"  {' vs '.join(matchup)}: {prob:.2f}% ({count} times)")
    
    print("\n📊 BY STAGE:")
    print("-"*80)
    stages = [("Round of 16", "r16"), ("Quarter-Finals", "qf"), 
              ("Semi-Finals", "sf"), ("Final", "final")]
    
    for stage_name, stage_code in stages:
        teams_in_stage = len(simulation.advancement_probs.get(stage_code, {}))
        top_3 = sorted(simulation.advancement_probs.get(stage_code, {}).items(),
                      key=lambda x: x[1], reverse=True)[:3]
        print(f"  {stage_name}: {teams_in_stage} teams")
        print(f"    Top: {', '.join(f'{t} ({p*100:.1f}%)' for t, p in top_3)}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    import os
    import sys
    
    # Get data directory
    data_dir = os.environ.get('WC_DATA_DIR', './data')
    
    # Initialize
    predictor = MatchupPredictor()
    model = TournamentModel(predictor)
    
    # Run full tournament simulation
    print("Running WC 2026 Full Tournament Simulation...")
    simulation = model.run_monte_carlo(runs=5000)
    
    # Print results
    print_results(simulation, top_n=20)
    
    # Example: Group A prediction
    print("\n" + "="*80)
    print("GROUP A PREDICTION")
    print("="*80)
    
    group_a = model.predict_group_standings("A")
    print(f"\nGroup {group_a['group']}:")
    for team_pred in group_a['teams']:
        print(f"\n  {team_pred['team']}:")
        print(f"    Expected position: {team_pred['expected_position']:.2f}")
        print(f"    Qualification: {team_pred['qualification_prob']*100:.1f}%")
        print(f"    Position probs: {team_pred['position_probabilities']}")
    
    # Save results
    output_file = os.environ.get('WC_OUTPUT_DIR', './results') + '/tournament_simulation.json'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w') as f:
        json.dump({
            "runs": simulation.runs,
            "timestamp": simulation.timestamp,
            "winner_probs": simulation.winner_probs,
            "advancement_probs": simulation.advancement_probs,
            "common_finals": [(list(m), c) for m, c in simulation.common_finals]
        }, f, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
