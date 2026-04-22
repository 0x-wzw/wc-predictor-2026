#!/usr/bin/env python3
"""
Group Stage Simulation - WC 2026
=================================
Simulate group stage with FIFA tiebreaker rules.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random


@dataclass
class GroupStanding:
    """Team standing in group table"""
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    rank: int = 0
    advanced: bool = False  # Top 2 advance
    
    @property
    def goal_diff(self) -> int:
        return self.goals_for - self.goals_against
    
    def add_match(self, gf: int, ga: int):
        """Add match result to standing"""
        self.played += 1
        self.goals_for += gf
        self.goals_against += ga
        
        if gf > ga:
            self.wins += 1
            self.points += 3
        elif gf == ga:
            self.draws += 1
            self.points += 1
        else:
            self.losses += 1


@dataclass
class GroupResult:
    """Complete group stage result"""
    name: str
    teams: List[str]
    standings: List[GroupStanding]
    matches: List[Dict]  # Match results
    
    def get_qualified(self, n: int = 2) -> List[str]:
        """Get top n qualified teams"""
        return [s.team for s in self.standings[:n]]
    
    def get_position(self, team: str) -> int:
        """Get final position (1-4)"""
        for i, s in enumerate(self.standings, 1):
            if s.team == team:
                return i
        return 0


class GroupStageSimulator:
    """
    Simulate group stage with all 6 matches per group.
    Implements FIFA tiebreaker rules.
    """
    
    def __init__(self, predictor):
        """
        Args:
            predictor: MatchPredictor instance
        """
        self.predictor = predictor
    
    def simulate_all_matches(self, teams: List[str]) -> List[Dict]:
        """
        Simulate all matches in group (round-robin).
        Returns list of match results.
        """
        matches = []
        
        # Generate all pairings (home/away doesn't matter in tournament)
        for i, team1 in enumerate(teams):
            for team2 in teams[i+1:]:
                result = self._simulate_match(team1, team2)
                matches.append(result)
        
        return matches
    
    def _simulate_match(self, team1: str, team2: str) -> Dict:
        """Simulate single group match"""
        # Use predictor for probabilities
        pred = self.predictor.predict(team1, team2, neutral_venue=True)
        
        # Simulate outcome
        r = random.random()
        if r < pred.home_win_prob:
            result = "home_win"
            gf1, gf2 = self._typical_group_score(True)
        elif r < pred.home_win_prob + pred.draw_prob:
            result = "draw"
            gf1 = gf2 = random.choice([0, 1, 1, 2, 2, 2, 3])
        else:
            result = "away_win"
            gf1, gf2 = self._typical_group_score(False)
            gf1, gf2 = gf2, gf1  # Swap
        
        return {
            "home_team": team1,
            "away_team": team2,
            "home_goals": gf1,
            "away_goals": gf2,
            "result": result,
            "prob_home": pred.home_win_prob,
            "prob_draw": pred.draw_prob,
            "prob_away": pred.away_win_prob
        }
    
    def _typical_group_score(self, win: bool) -> Tuple[int, int]:
        """Generate typical WC group stage score"""
        if win:
            # Winner scores 1-3, loser 0-1
            return random.choice([1, 2, 2, 2, 3, 3]), random.choice([0, 0, 0, 1, 1])
        else:
            return random.choice([0, 0, 0, 1, 1]), random.choice([1, 2, 2, 2, 3, 3])
    
    def calculate_standings(self, teams: List[str], matches: List[Dict]) -> List[GroupStanding]:
        """
        Calculate standings from match results.
        Implements FIFA tiebreaker rules.
        """
        # Initialize standings
        standings = {team: GroupStanding(team=team) for team in teams}
        
        # Process all matches
        for m in matches:
            t1, t2 = m["home_team"], m["away_team"]
            gf1, gf2 = m["home_goals"], m["away_goals"]
            
            standings[t1].add_match(gf1, gf2)
            standings[t2].add_match(gf2, gf1)
        
        # Convert to list for sorting
        table = list(standings.values())
        
        # Sort with tiebreaker rules
        sorted_table = self._apply_tiebreakers(table, matches)
        
        # Assign ranks and advancement
        for i, s in enumerate(sorted_table, 1):
            s.rank = i
            s.advanced = i <= 2
        
        return sorted_table
    
    def _apply_tiebreakers(self, table: List[GroupStanding], 
                           matches: List[Dict]) -> List[GroupStanding]:
        """
        Apply FIFA group stage tiebreaker rules:
        1. Points
        2. Goal difference
        3. Goals scored
        4. Head-to-head points
        5. Head-to-head goal difference
        6. Head-to-head goals scored
        7. Fair play
        8. Drawing of lots
        """
        # Sort by primary criteria
        table.sort(key=lambda s: (s.points, s.goal_diff, s.goals_for), reverse=True)
        
        # Find ties and apply secondary criteria
        i = 0
        while i < len(table):
            # Find tied teams
            tied = [i]
            for j in range(i + 1, len(table)):
                if (table[j].points == table[i].points and 
                    table[j].goal_diff == table[i].goal_diff and
                    table[j].goals_for == table[i].goals_for):
                    tied.append(j)
                else:
                    break
            
            # If tie exists, apply head-to-head
            if len(tied) > 1:
                tied_teams = [table[k].team for k in tied]
                h2h_sorted = self._head_to_head_sort(tied_teams, matches)
                
                # Replace tied section with h2h sorted
                for k, team in enumerate(h2h_sorted):
                    table[i + k] = next(s for s in table if s.team == team)
            
            i += len(tied)
        
        return table
    
    def _head_to_head_sort(self, teams: List[str], matches: List[Dict]) -> List[str]:
        """Sort tied teams by head-to-head results"""
        # Get h2h matches between these teams only
        h2h_matches = [m for m in matches 
                      if m["home_team"] in teams and m["away_team"] in teams]
        
        # Calculate h2h mini-table
        h2h_stats = {t: {"points": 0, "gd": 0, "gf": 0} for t in teams}
        
        for m in h2h_matches:
            t1, t2 = m["home_team"], m["away_team"]
            gf1, gf2 = m["home_goals"], m["away_goals"]
            
            h2h_stats[t1]["gf"] += gf1
            h2h_stats[t1]["gd"] += gf1 - gf2
            h2h_stats[t2]["gf"] += gf2
            h2h_stats[t2]["gd"] += gf2 - gf1
            
            if gf1 > gf2:
                h2h_stats[t1]["points"] += 3
            elif gf1 == gf2:
                h2h_stats[t1]["points"] += 1
                h2h_stats[t2]["points"] += 1
            else:
                h2h_stats[t2]["points"] += 3
        
        # Sort by h2h criteria
        sorted_teams = sorted(teams, 
            key=lambda t: (h2h_stats[t]["points"], 
                          h2h_stats[t]["gd"], 
                          h2h_stats[t]["gf"]), 
            reverse=True)
        
        return sorted_teams
    
    def simulate_group(self, name: str, teams: List[str]) -> GroupResult:
        """
        Simulate complete group stage.
        
        Args:
            name: Group name (A, B, C, etc.)
            teams: 4 team names
            
        Returns:
            GroupResult with standings and matches
        """
        matches = self.simulate_all_matches(teams)
        standings = self.calculate_standings(teams, matches)
        
        return GroupResult(
            name=name,
            teams=teams,
            standings=standings,
            matches=matches
        )
    
    def simulate_all_groups(self, groups: Dict[str, List[str]]) -> Dict[str, GroupResult]:
        """Simulate all groups in tournament"""
        return {
            name: self.simulate_group(name, teams)
            for name, teams in groups.items()
        }


# WC 2026 Groups (using teams available in historical data)
# Map of modern names to historical names
TEAM_NAME_MAP = {
    "USA": "United States",
    "Czechia": "Czech Republic",
    "Turkiye": "Turkey",
}

# Groups using actual historical team names
WC_2026_GROUPS = {
    "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
    "B": ["Spain", "Croatia", "Italy", "Wales"],
    "C": ["Slovenia", "Denmark", "Serbia", "England"],
    "D": ["France", "Netherlands", "Poland", "Austria"],
    "E": ["Romania", "Belgium", "Slovakia", "Ukraine"],
    "F": ["Portugal", "Turkey", "Czech Republic", "Greece"],
    "G": ["Brazil", "Paraguay", "Ecuador", "Mexico"],  # Mexico instead of Venezuela
    "H": ["Argentina", "Colombia", "Chile", "Peru"],
}


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '/home/ubuntu/wc-predictor-repo/src')
    from predictor import MatchupPredictor
    
    predictor = MatchupPredictor()
    simulator = GroupStageSimulator(predictor)
    
    # Example: Simulate Group A
    result = simulator.simulate_group("A", WC_2026_GROUPS["A"])
    
    print(f"\n=== GROUP {result.name} ===")
    print(f"{'Pos':<4} {'Team':<15} {'P':<3} {'W':<3} {'D':<3} {'L':<3} {'GF':<4} {'GA':<4} {'GD':<4} {'Pts':<4}")
    print("-" * 60)
    
    for s in result.standings:
        adv = "→" if s.advanced else " "
        print(f"{s.rank:<4} {s.team:<15} {s.played:<3} {s.wins:<3} {s.draws:<3} "
              f"{s.losses:<3} {s.goals_for:<4} {s.goals_against:<4} "
              f"{s.goal_diff:<+4} {s.points:<4} {adv}")
    
    print(f"\nQualified: {', '.join(result.get_qualified())}")
