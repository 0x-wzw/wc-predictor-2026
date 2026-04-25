#!/usr/bin/env python3
"""
Asian Handicap Calculation Engine v1.0
Poisson-based AH line calculator with quarter/fractional line support.
"""

import json
import math
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional
from pathlib import Path

# Pure Python Poisson implementation (no scipy dependency)

def poisson_pmf(k: int, lam: float) -> float:
    """Pure Python Poisson probability mass function."""
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return (lam ** k) * math.exp(-lam) / math.factorial(k)

def poisson_cdf(k_max: int, lam: float) -> float:
    """Cumulative probability P(X <= k_max) for Poisson."""
    if lam <= 0:
        return 1.0
    return sum(poisson_pmf(k, lam) for k in range(k_max + 1))

def _poisson_sample(lam: float) -> int:
    """Generate Poisson random sample using Knuth's algorithm."""
    if lam < 30:
        L = math.exp(-lam)
        k = 0
        p = 1.0
        while p > L:
            k += 1
            p *= random.random()
        return k - 1
    else:
        return max(0, int(random.gauss(lam, math.sqrt(lam))))

BASE = Path.home() / ".hermes"
MODEL_DIR = BASE / "models" / "wc2026"
AH_DIR = BASE / "models" / "ah_models"
DATA_DIR = BASE / "data" / "ah_models"

for d in [MODEL_DIR, AH_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class TeamRating:
    """Full team rating with uncertainty."""
    team: str
    attack: float          # Offensive rating (goals per match)
    defense: float         # Defensive rating (goals against per match)
    overall: float         # Elo-style composite
    variance: float = 0.1  # Rating uncertainty
    matches: int = 0     # Sample size for confidence

    @property
    def confidence(self) -> float:
        """Confidence based on sample size."""
        return min(1.0, self.matches / 50) if self.matches else 0.5


@dataclass
class PoissonPrediction:
    """Poisson-distributed match outcome probabilities."""
    home_expected: float
    away_expected: float
    home_win: float
    away_win: float
    draw: float
    confidence: float = 1.0

    def p_home_win_exact(self, margin: int) -> float:
        """Probability home wins by exactly margin goals."""
        total = 0.0
        for h in range(15):
            a = h - margin
            if a >= 0:
                total += (poisson_pmf(h, self.home_expected) *
                         poisson_pmf(a, self.away_expected))
        return total

    def get_margin_probs(self, max_margin: int = 5) -> Dict[int, float]:
        """Probability for each goal difference."""
        probs = {}
        for m in range(-max_margin, max_margin + 1):
            if m > 0:
                probs[m] = self.p_home_win_exact(m)
            elif m < 0:
                probs[m] = self.p_away_win_exact(abs(m))
            else:
                probs[m] = self.draw
        return probs

    def p_away_win_exact(self, margin: int) -> float:
        """Probability away wins by exactly margin goals."""
        return self.p_home_win_exact(margin)  # Symmetric for away


@dataclass
class AHLineProbability:
    """Probability for a specific AH line."""
    line: float
    home_win_prob: float   # Probability of beating the line (home side)
    away_win_prob: float   # Probability of beating the line (away side)
    push_prob: float       # Probability of push


@dataclass
class AHLineEdge:
    """Edge calculation for a specific AH line."""
    line: float
    side: str              # 'home' or 'away'
    model_prob: float
    market_prob: float
    market_odds: float
    edge: float
    expected_value: float
    kelly_fraction: float


@dataclass
class FairLines:
    """Complete fair line calculation for a match."""
    match_id: str
    home_team: str
    away_team: str
    match_date: str

    # Core probabilities
    home_win: float
    away_win: float
    draw: float

    # Fair lines
    fair_hdp: float        # Handicap (negative = home favorite)
    fair_goals: float      # Total expected goals
    fair_ou_line: float    # Fair over/under line

    # Confidence bands
    hdp_confidence_low: float
    hdp_confidence_high: float

    # Component predictions
    predicted_score: Tuple[float, float]  # (home, away)


# ─── Core Engine ───────────────────────────────────────────────────────────

class AsianHandicapEngine:
    """
    Core AH calculation using Poisson distributions.
    """

    # Rating difference to handicap conversion
    RATING_TO_SPREAD = [
        (500, float('inf'), -2.0),
        (400, 499, -1.75),
        (300, 399, -1.5),
        (200, 299, -1.25),
        (150, 199, -1.0),
        (100, 149, -0.75),
        (75, 99, -0.5),
        (50, 74, -0.25),
        (25, 49, 0.0),
        (-24, 24, 0.0),     # Pick'em
        (-49, -25, 0.0),
        (-74, -50, 0.25),
        (-99, -75, 0.5),
        (-149, -100, 0.75),
        (-199, -150, 1.0),
        (-299, -200, 1.25),
        (-399, -300, 1.5),
        (-499, -400, 1.75),
        (float('-inf'), -500, 2.0),
    ]

    # Variance in goal scoring
    GOAL_VARIANCE_FACTOR = 1.0  # Poisson variance = mean

    def __init__(self, ratings: Dict[str, TeamRating] = None):
        self.ratings = ratings or {}
        self._load_default_ratings()

    def _load_default_ratings(self):
        """Load team ratings from file or use defaults."""
        ratings_path = MODEL_DIR / "team_ratings.json"
        if ratings_path.exists():
            with open(ratings_path) as f:
                data = json.load(f)
                for team, r in data.items():
                    self.ratings[team] = TeamRating(
                        team=team,
                        attack=r.get("attack", 1.5),
                        defense=r.get("defense", 1.2),
                        overall=r.get("overall", 1000),
                        variance=r.get("variance", 0.1),
                        matches=r.get("matches", 0)
                    )
        else:
            # Default WC2026 ratings
            self._init_default_ratings()

    def _init_default_ratings(self):
        """Initialize with estimated WC2026 ratings."""
        defaults = {
            "Argentina": (2.1, 1.0, 1850),
            "France": (2.0, 0.9, 1820),
            "Brazil": (1.9, 1.0, 1800),
            "Germany": (1.8, 1.1, 1750),
            "Spain": (1.8, 0.9, 1740),
            "England": (1.9, 1.1, 1730),
            "Netherlands": (1.7, 1.0, 1700),
            "Portugal": (1.7, 1.1, 1680),
            "Belgium": (1.6, 1.0, 1650),
            "Uruguay": (1.5, 0.9, 1640),
            "Croatia": (1.4, 1.0, 1620),
            "Denmark": (1.4, 1.1, 1600),
            "Switzerland": (1.3, 1.0, 1580),
            "USA": (1.3, 1.2, 1550),
            "Mexico": (1.3, 1.2, 1540),
            "Senegal": (1.2, 0.9, 1530),
            "Poland": (1.3, 1.2, 1520),
            "Japan": (1.3, 1.1, 1510),
            "Morocco": (1.1, 0.9, 1500),
            "South_Korea": (1.2, 1.1, 1490),
            "Australia": (1.1, 1.2, 1480),
            "Ecuador": (1.2, 1.1, 1470),
            "Wales": (1.1, 1.1, 1460),
            "Ghana": (1.0, 1.1, 1450),
            "Cameroon": (1.0, 1.1, 1440),
            "Serbia": (1.1, 1.2, 1430),
            "Canada": (1.1, 1.3, 1420),
            "Tunisia": (0.9, 1.0, 1410),
            "Iran": (0.9, 1.1, 1400),
            "Costa_Rica": (0.8, 1.1, 1380),
            "Saudi_Arabia": (0.8, 1.2, 1350),
            "Qatar": (0.7, 1.2, 1300),
            "Italy": (1.6, 1.0, 1680),
            "Netherlands": (1.7, 1.0, 1700),
        }
        for team, (atk, def_, overall) in defaults.items():
            self.ratings[team] = TeamRating(
                team=team,
                attack=atk,
                defense=def_,
                overall=overall,
                matches=30
            )

    def calculate_match_prediction(
        self,
        home_team: str,
        away_team: str,
        neutral_venue: bool = True
    ) -> PoissonPrediction:
        """
        Calculate Poisson probabilities for a match.
        """
        home = self.ratings.get(home_team)
        away = self.ratings.get(away_team)

        if not home or not away:
            raise ValueError(f"Missing rating for {home_team} or {away_team}")

        # Expected goals
        home_expected = (home.attack + away.defense) / 2
        away_expected = (away.attack + home.defense) / 2

        if not neutral_venue:
            home_expected *= 1.08
            away_expected *= 0.92

        # Calculate probabilities via Poisson
        probs = self._poisson_monte_carlo(home_expected, away_expected)

        # Confidence based on both teams' data
        confidence = (home.confidence + away.confidence) / 2

        return PoissonPrediction(
            home_expected=home_expected,
            away_expected=away_expected,
            home_win=probs['home_win'],
            away_win=probs['away_win'],
            draw=probs['draw'],
            confidence=confidence
        )

    def _poisson_monte_carlo(
        self,
        home_lambda: float,
        away_lambda: float,
        simulations: int = 10000
    ) -> Dict[str, float]:
        """Monte Carlo simulation of Poisson-distributed match."""
        home_wins = 0
        away_wins = 0
        draws = 0

        for _ in range(simulations):
            home_goals = _poisson_sample(home_lambda)
            away_goals = _poisson_sample(away_lambda)

            if home_goals > away_goals:
                home_wins += 1
            elif away_goals > home_goals:
                away_wins += 1
            else:
                draws += 1

        total = simulations
        return {
            'home_win': home_wins / total,
            'away_win': away_wins / total,
            'draw': draws / total
        }

    def calculate_fair_lines(
        self,
        home_team: str,
        away_team: str,
        match_date: str = "",
        neutral_venue: bool = True
    ) -> FairLines:
        """
        Calculate all fair lines for a match.
        """
        prediction = self.calculate_match_prediction(home_team, away_team, neutral_venue)

        home = self.ratings[home_team]
        away = self.ratings[away_team]

        # Rating difference for handicap
        rating_diff = home.overall - away.overall
        fair_hdp = self._rating_to_spread(rating_diff)

        # Fair goal line
        fair_goals = prediction.home_expected + prediction.away_expected
        fair_ou = round(fair_goals * 2) / 2  # Round to nearest 0.5

        # Confidence intervals (95%)
        combined_variance = home.variance + away.variance
        hdp_low = self._rating_to_spread(rating_diff - 50 * math.sqrt(combined_variance))
        hdp_high = self._rating_to_spread(rating_diff + 50 * math.sqrt(combined_variance))

        return FairLines(
            match_id=f"{home_team}_vs_{away_team}_{match_date}",
            home_team=home_team,
            away_team=away_team,
            match_date=match_date,
            home_win=prediction.home_win,
            away_win=prediction.away_win,
            draw=prediction.draw,
            fair_hdp=fair_hdp,
            fair_goals=fair_goals,
            fair_ou_line=fair_ou,
            hdp_confidence_low=hdp_low,
            hdp_confidence_high=hdp_high,
            predicted_score=(prediction.home_expected, prediction.away_expected)
        )

    def _rating_to_spread(self, rating_diff: float) -> float:
        """Convert rating difference to handicap spread."""
        for low, high, spread in self.RATING_TO_SPREAD:
            if low <= rating_diff <= high:
                return spread
        return 0.0

    def get_ah_probabilities(self, lines: FairLines) -> Dict[float, AHLineProbability]:
        """
        Calculate AH probabilities for standard lines.
        """
        prediction = self.calculate_match_prediction(
            lines.home_team, lines.away_team
        )

        line_probs = {}
        standard_lines = [-1.5, -1.0, -0.75, -0.5, -0.25, 0.0,
                         0.25, 0.5, 0.75, 1.0, 1.5]

        for line in standard_lines:
            lp = self._calculate_line_probability(prediction, line)
            line_probs[line] = lp

        return line_probs

    def _calculate_line_probability(
        self,
        pred: PoissonPrediction,
        line: float
    ) -> AHLineProbability:
        """
        Calculate win/push/lose probability for specific AH line.

        Quarter lines (e.g., -0.75) are treated as split bets:
        -0.75 = half stake on -0.5, half on -1.0
        """
        # Handle quarter lines
        if abs(line % 0.25) > 0.001 and abs(line % 0.5) > 0.001:
            # Quarter line - split calculation
            if line == -0.75:
                # Half on -0.5, half on -1.0
                p_minus_05 = self._calculate_line_probability(pred, -0.5)
                p_minus_10 = self._calculate_line_probability(pred, -1.0)
                return AHLineProbability(
                    line=line,
                    home_win_prob=(p_minus_05.home_win_prob + p_minus_10.home_win_prob) / 2,
                    away_win_prob=(p_minus_05.away_win_prob + p_minus_10.away_win_prob) / 2,
                    push_prob=(p_minus_05.push_prob + p_minus_10.push_prob) / 2
                )
            elif line == -0.25:
                # Half on 0.0, half on -0.5
                p_00 = self._calculate_line_probability(pred, 0.0)
                p_minus_05 = self._calculate_line_probability(pred, -0.5)
                return AHLineProbability(
                    line=line,
                    home_win_prob=(p_00.home_win_prob + p_minus_05.home_win_prob) / 2,
                    away_win_prob=(p_00.away_win_prob + p_minus_05.away_win_prob) / 2,
                    push_prob=(p_00.push_prob + p_minus_05.push_prob) / 2
                )
            elif line == 0.25:
                # Half on 0.0, half on +0.5 (from away perspective)
                return self._calculate_line_probability(pred, -0.25)  # Mirror
            elif line == 0.75:
                return self._calculate_line_probability(pred, -0.75)  # Mirror

        # Standard lines
        if line == 0.0:
            # Draw no bet
            total_win = pred.home_win
            total_lose = pred.away_win
            push = pred.draw
            return AHLineProbability(
                line=line,
                home_win_prob=total_win / (total_win + total_lose),
                away_win_prob=1.0,  # Simplified
                push_prob=0.0  # Refund
            )

        elif line == -0.5:
            # Win by 1+ → win, draw or lose → lose
            return AHLineProbability(
                line=line,
                home_win_prob=pred.home_win,
                away_win_prob=pred.away_win + pred.draw,
                push_prob=0.0
            )

        elif line == 0.5:
            # Give +0.5 → win or draw = win
            return AHLineProbability(
                line=line,
                home_win_prob=pred.home_win + pred.draw,
                away_win_prob=pred.away_win,
                push_prob=0.0
            )

        elif line == -1.0:
            # Win by 2+ → full win, win by 1 → push
            p_win_by_exact_1 = pred.p_home_win_exact(1)
            p_win_by_2_plus = pred.home_win - p_win_by_exact_1
            return AHLineProbability(
                line=line,
                home_win_prob=p_win_by_2_plus,
                away_win_prob=1 - pred.home_win,
                push_prob=p_win_by_exact_1
            )

        elif line == 1.0:
            # Receive +1 → lose by 1 → push
            p_lose_by_exact_1 = pred.p_away_win_exact(1)
            return AHLineProbability(
                line=line,
                home_win_prob=pred.home_win,
                away_win_prob=1 - (pred.home_win + p_lose_by_exact_1),
                push_prob=p_lose_by_exact_1
            )

        # Default for other lines
        return AHLineProbability(line=line, home_win_prob=0.5, away_win_prob=0.5, push_prob=0.0)

    def calculate_edge(
        self,
        lines: FairLines,
        market_line: float,
        market_odds: float,
        side: str  # 'home' or 'away'
    ) -> AHLineEdge:
        """
        Calculate edge for a specific market line.
        """
        # Get model probability for this line
        ah_probs = self.get_ah_probabilities(lines)
        prob = ah_probs.get(market_line)

        if not prob:
            # Interpolated calculation
            prob = self._interpolate_prob(lines, market_line)

        model_prob = prob.home_win_prob if side == 'home' else prob.away_win_prob
        market_prob = 1.0 / market_odds

        edge = model_prob - market_prob

        # Expected value
        ev = (model_prob * (market_odds - 1)) - (1 - model_prob)

        # Kelly calculation (simplified for AH, accounts for push probability)
        if market_odds > 1:
            kelly = edge / (market_odds - 1)
            kelly = max(0, min(kelly * 0.25, 0.25))  # Quarter-Kelly
        else:
            kelly = 0

        return AHLineEdge(
            line=market_line,
            side=side,
            model_prob=model_prob,
            market_prob=market_prob,
            market_odds=market_odds,
            edge=edge,
            expected_value=ev,
            kelly_fraction=kelly
        )

    def _interpolate_prob(
        self,
        lines: FairLines,
        target_line: float
    ) -> AHLineProbability:
        """Interpolate probability between known lines."""
        ah_probs = self.get_ah_probabilities(lines)
        sorted_lines = sorted(ah_probs.keys(), key=lambda x: abs(x - target_line))

        # Get two closest lines
        if len(sorted_lines) >= 2:
            l1, l2 = sorted_lines[0], sorted_lines[1]
            p1, p2 = ah_probs[l1], ah_probs[l2]

            # Linear interpolation
            ratio = (target_line - l1) / (l2 - l1) if l2 != l1 else 0
            return AHLineProbability(
                line=target_line,
                home_win_prob=p1.home_win_prob + ratio * (p2.home_win_prob - p1.home_win_prob),
                away_win_prob=p1.away_win_prob + ratio * (p2.away_win_prob - p1.away_win_prob),
                push_prob=0.0  # Simplified
            )

        # Fallback
        return AHLineProbability(
            line=target_line,
            home_win_prob=0.5,
            away_win_prob=0.5,
            push_prob=0.0
        )


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AH Calculation Engine")
    sub = parser.add_subparsers(dest='cmd')

    # Predict command
    predict = sub.add_parser('predict', help='Predict fair lines for match')
    predict.add_argument('--home', required=True)
    predict.add_argument('--away', required=True)
    predict.add_argument('--date', default='')

    # Edge command
    edge = sub.add_parser('edge', help='Calculate edge vs market line')
    edge.add_argument('--home', required=True)
    edge.add_argument('--away', required=True)
    edge.add_argument('--line', type=float, required=True)
    edge.add_argument('--odds', type=float, required=True)
    edge.add_argument('--side', choices=['home', 'away'], required=True)

    # Batch command
    batch = sub.add_parser('batch', help='Batch process upcoming matches')

    args = parser.parse_args()

    engine = AsianHandicapEngine()

    if args.cmd == 'predict':
        lines = engine.calculate_fair_lines(args.home, args.away, args.date)
        print(json.dumps(asdict(lines), indent=2))

    elif args.cmd == 'edge':
        lines = engine.calculate_fair_lines(args.home, args.away)
        edge = engine.calculate_edge(lines, args.line, args.odds, args.side)
        print(json.dumps(asdict(edge), indent=2))

    elif args.cmd == 'batch':
        # Show top matches
        wc2026_matches = [
            ("Argentina", "France"),
            ("Brazil", "England"),
            ("Germany", "Spain"),
        ]
        print(f"{'Match':<30} {'Fair HDP':>10} {'Fair Goals':>12}")
        print("-" * 55)
        for home, away in wc2026_matches:
            lines = engine.calculate_fair_lines(home, away)
            match_name = f"{home} vs {away}"
            print(f"{match_name:<30} {lines.fair_hdp:>+10.2f} {lines.fair_ou_line:>12.1f}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
