#!/usr/bin/env python3
"""
Knockout Stage Simulation - WC 2026
====================================
Build and resolve knockout bracket from group results.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from collections import defaultdict
import random


@dataclass
class Matchup:
    """Single knockout matchup"""
    team1: Optional[str]  # None = not determined yet (e.g., "R16 Winner")
    team2: Optional[str]
    team1_source: str  # "1A", "2B", "R16-1", etc.
    team2_source: str
    round: str  # "r32", "r16", "qf", "sf", "final", "third_place"
    winner: Optional[str] = None
    extra_time: bool = False
    penalties: Tuple[int, int] = field(default_factory=tuple)
    
    def is_placeholder(self) -> bool:
        """Check if matchup has placeholder teams"""
        return self.team1 is None or self.team2 is None


@dataclass
class KnockoutResult:
    """Complete knockout stage results"""
    winner: str
    runner_up: str
    third_place: str
    matches: Dict[str, List[Matchup]]  # By round
    bracket: Dict
    
    def get_round_result(self, round_name: str) -> List[Matchup]:
        return self.matches.get(round_name, [])
    
    def get_team_path(self, team: str) -> List[str]:
        """Get path team took through bracket"""
        path = []
        for round_name, matches in self.matches.items():
            for m in matches:
                if team in [m.team1, m.team2]:
                    path.append(round_name)
                    if m.winner == team:
                        path.append(f"W {m.team1 if m.winner != m.team1 else m.team2}")
                    break
        return path


# WC 2026 48-team bracket mapping
# 12 groups → top 2 + 8 best 3rd → R32 → R16 → QF → SF → Final

# Round of 32 (16 matches)
R32_MAPPING = [
    ("1A", "3B/C/D/E"), ("1B", "3A/C/D/E"), ("1C", "3A/B/F/G"), ("1D", "3A/B/F/G"),
    ("1E", "3C/D/F/G/H"), ("1F", "3C/D/E/H"), ("1G", "3E/F/H/I"), ("1H", "3E/F/G/I"),
    ("1I", "3G/H/J/K"), ("1J", "3I/K/L"), ("1K", "3I/J/L"), ("1L", "3J/K/M"),
    ("2B", "2F"),  # Cross-group matchups
]

# Round of 16 (8 matches) - after R32
R16_MAPPING = [
    ("R32-1", "R32-2"),  # 1A vs Winner vs 1B vs Winner
    ("R32-3", "R32-4"),
    ("R32-5", "R32-6"),
    ("R32-7", "R32-8"),
    ("1C", "2D"),
    ("1D", "2C"),
    ("1E", "2F"),
    ("1F", "2E"),
]

# Simplified for 32-team format (R16 directly)
# Standard WC format: 8 groups of 4 → 16 teams
DIRECT_R16_MAPPING = [
    ("1A", "2B"), ("1C", "2D"), ("1E", "2F"), ("1G", "2H"),
    ("1B", "2A"), ("1D", "2C"), ("1F", "2E"), ("1H", "2G"),
]

QUARTERFINAL_MAPPING = [
    ("R16-1", "R16-2"), ("R16-3", "R16-4"), ("R16-5", "R16-6"), ("R16-7", "R16-8"),
]

SEMIFINAL_MAPPING = [
    ("QF-1", "QF-2"),
    ("QF-3", "QF-4"),
]

FINAL_MAPPING = ("SF-1", "SF-2")
THIRD_PLACE_MAPPING = ("SF-1L", "SF-2L")  # Losers of semis


class KnockoutSimulator:
    """
    Simulate knockout rounds from group qualifiers.
    """
    
    def __init__(self, predictor, format: str = "32team"):
        """
        Args:
            predictor: MatchPredictor for knockout probabilities
            format: "32team" (standard) or "48team" (2026 format)
        """
        self.predictor = predictor
        self.format = format
    
    def build_r16_bracket(self, group_results: Dict) -> List[Matchup]:
        """
        Build Round of 16 bracket from group qualifiers.
        
        Args:
            group_results: Dict[str, GroupResult] from group stage
            
        Returns:
            List of 8 R16 matchups
        """
        bracket = []
        
        for source1, source2 in DIRECT_R16_MAPPING:
            team1 = self._resolve_source(source1, group_results)
            team2 = self._resolve_source(source2, group_results)
            
            bracket.append(Matchup(
                team1=team1,
                team2=team2,
                team1_source=source1,
                team2_source=source2,
                round="r16"
            ))
        
        return bracket
    
    def _resolve_source(self, source: str, group_results: Dict) -> Optional[str]:
        """Resolve "1A", "2B" etc. to actual team name"""
        if len(source) == 2:
            pos = int(source[0])  # 1 or 2
            group = source[1]
            
            if group in group_results:
                qualified = group_results[group].get_qualified(2)
                if pos <= len(qualified):
                    return qualified[pos - 1]
        return None
    
    def simulate_knockout_match(self, matchup: Matchup) -> Matchup:
        """
        Simulate knockout match.
        Must have a winner (extra time/penalties).
        
        Returns modified matchup with winner
        """
        if matchup.team1 is None or matchup.team2 is None:
            return matchup
        
        # Get base prediction
        pred = self.predictor.predict(matchup.team1, matchup.team2, neutral_venue=True)
        
        # Knockout adjustment: slightly more likely to decide in 90 min
        # But draws go to extra time
        r = random.random()
        
        if r < pred.home_win_prob:
            matchup.winner = matchup.team1
            matchup.extra_time = False
        elif r < pred.home_win_prob + pred.draw_prob:
            # Go to extra time
            matchup.extra_time = True
            
            # Simulate extra time (favour team with better stamina/experience)
            # Simplified: 50/50 with slight favourite boost
            et_win_prob = 0.5 + (pred.home_win_prob - 0.33)
            et_win_prob = max(0.3, min(0.7, et_win_prob))
            
            if random.random() < et_win_prob:
                matchup.winner = matchup.team1
            else:
                matchup.winner = matchup.team2
            
            # Simulate penalties if still tied
            # Simplified: don't track exact score
        else:
            matchup.winner = matchup.team2
            matchup.extra_time = False
        
        return matchup
    
    def simulate_round(self, matchups: List[Matchup], round_name: str) -> List[Matchup]:
        """Simulate all matches in a round"""
        results = []
        i = 1
        
        for matchup in matchups:
            result = self.simulate_knockout_match(matchup)
            result.round = round_name
            results.append(result)
            i += 1
        
        return results
    
    def advance_winners(self, current_round: List[Matchup], 
                        next_mapping: List[Tuple]) -> List[Matchup]:
        """Create next round bracket from winners"""
        next_round = []
        
        for i, (source1, source2) in enumerate(next_mapping, 1):
            # Parse sources like "R16-1", "R16-2"
            team1 = self._resolve_winner_source(source1, current_round)
            team2 = self._resolve_winner_source(source2, current_round)
            
            next_round.append(Matchup(
                team1=team1,
                team2=team2,
                team1_source=source1,
                team2_source=source2,
                round="next"
            ))
        
        return next_round
    
    def _resolve_winner_source(self, source: str, 
                               current_round: List[Matchup]) -> Optional[str]:
        """Resolve "R16-1" to winner of first R16 match"""
        if source.startswith("R16-"):
            idx = int(source.split("-")[1]) - 1
            if idx < len(current_round):
                return current_round[idx].winner
        elif source == "R16-1L":
            return current_round[0].team2 if current_round[0].winner == current_round[0].team1 else current_round[0].team1
        elif source.startswith("QF-"):
            idx = int(source.split("-")[1]) - 1
            if idx < len(current_round):
                return current_round[idx].winner
        elif source.endswith("L"):
            # Loser of match
            base = source[:-1]
            idx = int(base.split("-")[1]) - 1
            if idx < len(current_round):
                m = current_round[idx]
                return m.team1 if m.winner == m.team2 else m.team2
        return None
    
    def simulate_tournament(self, group_results: Dict) -> KnockoutResult:
        """
        Simulate full knockout stage from group qualifiers.
        
        Args:
            group_results: Dict[str, GroupResult]
            
        Returns:
            KnockoutResult with complete bracket
        """
        all_matches = {}
        
        # Round of 16
        r16 = self.build_r16_bracket(group_results)
        r16_results = self.simulate_round(r16, "r16")
        all_matches["r16"] = r16_results
        
        # Quarterfinals
        qf_teams = self.advance_winners(r16_results, QUARTERFINAL_MAPPING)
        qf_results = self.simulate_round(qf_teams, "qf")
        all_matches["qf"] = qf_results
        
        # Semifinals
        sf_teams = self.advance_winners(qf_results, SEMIFINAL_MAPPING)
        sf_results = self.simulate_round(sf_teams, "sf")
        all_matches["sf"] = sf_results
        
        # Finals (separate simulation for both matches)
        final_matchup = Matchup(
            team1=sf_results[0].winner,
            team2=sf_results[1].winner,
            team1_source="SF-1",
            team2_source="SF-2",
            round="final"
        )
        final_result = self.simulate_knockout_match(final_matchup)
        all_matches["final"] = [final_result]
        
        # Third place
        third_matchup = Matchup(
            team1=sf_results[0].team1 if sf_results[0].winner == sf_results[0].team2 else sf_results[0].team2,
            team2=sf_results[1].team1 if sf_results[1].winner == sf_results[1].team2 else sf_results[1].team2,
            team1_source="SF-1L",
            team2_source="SF-2L",
            round="third_place"
        )
        third_result = self.simulate_knockout_match(third_matchup)
        all_matches["third_place"] = [third_result]
        
        return KnockoutResult(
            winner=final_result.winner,
            runner_up=final_result.team1 if final_result.winner == final_result.team2 else final_result.team2,
            third_place=third_result.winner,
            matches=all_matches,
            bracket=self._build_bracket_diagram(all_matches)
        )
    
    def _build_bracket_diagram(self, matches: Dict) -> Dict:
        """Build visualizable bracket structure"""
        bracket = {}
        
        for round_name, round_matches in matches.items():
            bracket[round_name] = []
            for i, m in enumerate(round_matches, 1):
                bracket[round_name].append({
                    "match": i,
                    "team1": m.team1,
                    "team2": m.team2,
                    "winner": m.winner,
                    "extra_time": m.extra_time
                })
        
        return bracket


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/ubuntu/wc-predictor-repo/src')
    from predictor import MatchupPredictor
    from group_stage import GroupStageSimulator
    
    predictor = MatchupPredictor()
    group_sim = GroupStageSimulator(predictor)
    
    # Example group results
    group_results = {
        "A": group_sim.simulate_group("A", ["Germany", "Scotland", "Hungary", "Switzerland"]),
        "B": group_sim.simulate_group("B", ["Spain", "Croatia", "Italy", "Albania"]),
        "C": group_sim.simulate_group("C", ["Slovenia", "Denmark", "Serbia", "England"]),
        "D": group_sim.simulate_group("D", ["Netherlands", "France", "Poland", "Austria"]),
        "E": group_sim.simulate_group("E", ["Romania", "Belgium", "Slovakia", "Ukraine"]),
        "F": group_sim.simulate_group("F", ["Portugal", "Turkey", "Georgia", "Czechia"]),
        "G": group_sim.simulate_group("G", ["Brazil", "Paraguay", "Ecuador", "Venezuela"]),
        "H": group_sim.simulate_group("H", ["Argentina", "Colombia", "Chile", "Peru"]),
    }
    
    knockout_sim = KnockoutSimulator(predictor)
    result = knockout_sim.simulate_tournament(group_results)
    
    print("\n=== KNOCKOUT STAGE ===\n")
    print(f"🏆 WINNER: {result.winner}")
    print(f"🥈 Runner-up: {result.runner_up}")
    print(f"🥉 Third place: {result.third_place}")
    
    print("\n--- BRACKET ---")
    for round_name, matches in result.matches.items():
        print(f"\n{round_name.upper()}:")
        for m in matches:
            winner_marker = f" → {m.winner}" if m.winner else ""
            print(f"  {m.team1 or '?'} vs {m.team2 or '?'}{winner_marker}")
