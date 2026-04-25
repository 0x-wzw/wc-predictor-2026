#!/usr/bin/env python3
"""
AH Signal Providers (S10-S12)
Asian Handicap specific signals for line movement detection.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

BASE = Path.home() / ".hermes"
AH_DIR = BASE / "models" / "ah_models"
DATA_DIR = BASE / "data" / "ah_models"

for d in [AH_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Signal Result Types ────────────────────────────────────────────────────

@dataclass
class SignalResult:
    """Standard signal output format."""
    name: str
    fired: bool
    direction: str = "NEUTRAL"    # BUY, SELL, NEUTRAL
    strength: float = 0.0           # 0.0 to 1.0
    confidence: float = 0.0         # 0.0 to 1.0
    metadata: dict = None


@dataclass
class LineSnapshot:
    """Historical line data point."""
    timestamp: str
    home_spread: float
    home_odds: float
    away_odds: float
    total_line: float
    over_odds: float
    under_odds: float
    volume: float
    public_pct: float             # % betting on home/favorite


# ─── Signal S10: Line Movement ────────────────────────────────────────────

class S10LineMovement:
    """
    Signal 10: Detect line moves toward or away from model price.

    Logic:
    - Line moves TOWARD model fair line = opportunity (market converging to us)
    - Line moves AWAY from model fair line = avoid (market diverging)
    - Line moves WITH public but AGAINST sharp = S11 fires (contrarian)
    """

    def __init__(self, model_lines: Dict[str, float]):
        """
        Args:
            model_lines: {match_id: fair_home_spread}
        """
        self.model_lines = model_lines

    def evaluate(
        self,
        match_id: str,
        current: LineSnapshot,
        history: List[LineSnapshot]
    ) -> SignalResult:
        """
        Evaluate line movement signal.
        """
        fair_line = self.model_lines.get(match_id)
        if fair_line is None:
            return SignalResult(name="S10", fired=False, metadata={"error": "no_model_line"})

        if len(history) < 2:
            return SignalResult(name="S10", fired=False, metadata={"error": "insufficient_history"})

        # Compare to historical baseline (5 samples ago or oldest)
        baseline_idx = min(5, len(history) - 1)
        baseline = history[-baseline_idx]

        # Calculate distances from fair
        current_distance = abs(current.home_spread - fair_line)
        baseline_distance = abs(baseline.home_spread - fair_line)

        # Line moved toward model
        toward_model = current_distance < baseline_distance
        away_from_model = current_distance > baseline_distance

        # Line movement speed (quarter point per unit time)
        line_delta = current.home_spread - baseline.home_spread
        movement_speed = abs(line_delta) / baseline_idx

        # Signal logic
        if toward_model:
            # Market converging to our view - good for entry
            fired = True
            direction = "BUY" if current.home_spread < fair_line else "SELL"
            strength = min(1.0, (baseline_distance - current_distance) / 0.5)
        elif away_from_model and current_distance > 0.5:
            # Market diverging significantly - potential fade if overextended
            fired = True
            direction = "SELL" if current.home_spread < fair_line else "BUY"
            strength = min(0.7, (current_distance - baseline_distance) / 0.5)
        else:
            fired = False
            direction = "NEUTRAL"
            strength = 0.0

        return SignalResult(
            name="S10",
            fired=fired,
            direction=direction,
            strength=strength,
            confidence=min(1.0, (baseline_distance + 0.1) * 2),
            metadata={
                "fair_line": fair_line,
                "current_line": current.home_spread,
                "baseline_line": baseline.home_spread,
                "line_delta": line_delta,
                "movement_speed": movement_speed,
                "toward_model": toward_model
            }
        )


# ─── Signal S11: Sharp/Retail Divergence ────────────────────────────────────

class S11SharpDivergence:
    """
    Signal 11: Detect when public money != smart money.

    Mechanics:
    - Public % on Home: 70%
    - Line moves TOWARD Home (more negative spread)
    = Sharp money agrees with public

    - Public % on Home: 70%
    - Line moves AWAY from Home (less negative/more positive spread)
    = Sharp money fading public

    We follow smart money (line movement) when it contradicts public %."""

    PUBLIC_THRESHOLD = 70.0       # % public on one side
    LINE_MOVE_THRESHOLD = 0.25    # Quarter point

    def evaluate(
        self,
        match_id: str,
        current: LineSnapshot,
        history: List[LineSnapshot]
    ) -> SignalResult:
        """
        Evaluate sharp/retail divergence.
        """
        if len(history) < 2:
            return SignalResult(name="S11", fired=False)

        prev = history[-1]

        public_home_pct = current.public_pct
        line_moved_home = current.home_spread < prev.home_spread - self.LINE_MOVE_THRESHOLD
        line_moved_away = current.home_spread > prev.home_spread + self.LINE_MOVE_THRESHOLD

        # Public heavily on home
        if public_home_pct >= self.PUBLIC_THRESHOLD:
            # But line moved away from home = smart money on away
            if line_moved_away:
                return SignalResult(
                    name="S11",
                    fired=True,
                    direction="BUY" if current.home_spread < 0 else "SELL",
                    strength=0.7,
                    confidence=0.65,
                    metadata={
                        "scenario": "smart_fading_public_home",
                        "public_pct": public_home_pct,
                        "line_delta": current.home_spread - prev.home_spread
                    }
                )
            # Line moved with public = sharp agrees
            elif line_moved_home:
                return SignalResult(
                    name="S11",
                    fired=True,
                    direction="BUY" if current.home_spread < 0 else "SELL",
                    strength=0.5,
                    confidence=0.55,
                    metadata={
                        "scenario": "sharp_agrees_public_home",
                        "public_pct": public_home_pct
                    }
                )

        # Public heavily on away (implied by low public_home_pct)
        elif public_home_pct <= 30.0:
            # But line moved toward home = smart money on home
            if line_moved_home:
                return SignalResult(
                    name="S11",
                    fired=True,
                    direction="BUY" if current.home_spread < 0 else "SELL",
                    strength=0.7,
                    confidence=0.65,
                    metadata={
                        "scenario": "smart_fading_public_away",
                        "public_pct": public_home_pct,
                        "line_delta": current.home_spread - prev.home_spread
                    }
                )

        return SignalResult(
            name="S11",
            fired=False,
            direction="NEUTRAL",
            strength=0.0,
            confidence=0.5
        )


# ─── Signal S12: Goal Line Correlation ─────────────────────────────────────

class S12GoalLineCorrelation:
    """
    Signal 12: Detect inconsistency between AH spread and total line.

    Logic:
    - High spread (-1.0 or more) + Low total (U 2.0) = defensive mismatch
      (favorite should win 1-0 or 2-0 - value on under or favorite AH)
    - Low spread (0.0 or +0.5) + High total (O 2.5+) = open attacking game
      (close match with goals - value on over or underdog AH)

    Implied goals from spread + line should align. When they don't, value exists."""

    def evaluate(
        self,
        match_id: str,
        ah_line: float,
        goals_line: float,
        home_odds: float,
        over_odds: float,
        model: Dict  # Contains model's fair lines
    ) -> SignalResult:
        """
        Evaluate AH/total correlation.
        """
        fair_hdp = model.get('fair_hdp', 0)
        fair_ou = model.get('fair_goals', 2.5)

        # Calculate implied spread from goals line
        # Rough heuristic: spread ≈ (goals_line / 2 - 0.5) for competitive odds
        implied_spread = (goals_line / 2) - 1.0
        spread_from_ah = abs(ah_line)

        # Detect mismatch
        spread_rating = self._rate_spread(abs(ah_line))
        goals_rating = self._rate_goals(goals_line)

        # Mismatch scenarios
        if spread_rating == "high" and goals_rating == "low":
            # Heavy favorite, low total = favorite can win low scoring
            fired = True
            direction = "BUY" if ah_line < 0 else "SELL"  # Take favorite
            scenario = "defensive_matchup"
        elif spread_rating == "low" and goals_rating == "high":
            # Close match, high total = anyone can score
            fired = True
            direction = "BUY" if ah_line > 0 else "SELL"  # Take underdog
            scenario = "open_attacking_game"
        elif spread_rating == "high" and goals_rating == "high":
            # Heavy favorite, high total = potential blowout
            fired = True
            direction = "BUY" if ah_line < 0 else "SELL"  # Aggressive favorite
            scenario = "blowout_potential"
        else:
            fired = False
            direction = "NEUTRAL"
            scenario = "correlated"

        # Strength based on divergence magnitude
        if fired:
            hdp_divergence = abs(fair_hdp - ah_line)
            ou_divergence = abs(fair_ou - goals_line)
            strength = min(1.0, (hdp_divergence + ou_divergence) / 1.0)
            confidence = 0.5 + (strength * 0.3)
        else:
            strength = 0.0
            confidence = 0.5

        return SignalResult(
            name="S12",
            fired=fired,
            direction=direction,
            strength=strength,
            confidence=confidence,
            metadata={
                "scenario": scenario,
                "spread_rating": spread_rating,
                "goals_rating": goals_rating,
                "fair_hdp": fair_hdp,
                "fair_ou": fair_ou,
                "implied_spread": implied_spread
            }
        )

    def _rate_spread(self, spread: float) -> str:
        """Categorize spread magnitude."""
        if spread >= 1.0:
            return "high"
        elif spread >= 0.5:
            return "medium"
        else:
            return "low"

    def _rate_goals(self, line: float) -> str:
        """Categorize goals line."""
        if line <= 2.0:
            return "low"
        elif line >= 3.0:
            return "high"
        else:
            return "medium"


# ─── AH Signal Filter ─────────────────────────────────────────────────────

class AHSignalFilter:
    """
    Combined AH signal filter integrating S10, S11, S12.
    Filters trades based on AH-specific signals.
    """

    def __init__(self, model_lines: Dict[str, float]):
        self.s10 = S10LineMovement(model_lines)
        self.s11 = S11SharpDivergence()
        self.s12 = S12GoalLineCorrelation()

    def evaluate_all(
        self,
        match_id: str,
        line_data: LineSnapshot,
        history: List[LineSnapshot],
        model: Dict
    ) -> Dict[str, SignalResult]:
        """
        Run all AH signals.
        """
        # S10 and S11 evaluate line movement
        s10_result = self.s10.evaluate(match_id, line_data, history)
        s11_result = self.s11.evaluate(match_id, line_data, history)

        # S12 evaluates correlation
        s12_result = self.s12.evaluate(
            match_id,
            line_data.home_spread,
            line_data.total_line,
            line_data.home_odds,
            line_data.over_odds,
            model
        )

        return {
            "S10": s10_result,
            "S11": s11_result,
            "S12": s12_result
        }

    def get_trade_confirmation(
        self,
        results: Dict[str, SignalResult],
        min_signals: int = 2
    ) -> Tuple[bool, float, str]:
        """
        Determine if signals confirm trade.

        Returns:
            (fired, confidence, direction)
        """
        fired_signals = [r for r in results.values() if r.fired]
        count = len(fired_signals)

        if count < min_signals:
            return False, 0.0, "NEUTRAL"

        # Aggregate direction
        directions = [r.direction for r in fired_signals]
        direction = self._consensus_direction(directions)

        # Aggregate confidence
        avg_strength = sum(r.strength for r in fired_signals) / count
        avg_conf = sum(r.confidence for r in fired_signals) / count

        return True, avg_conf * avg_strength, direction

    def _consensus_direction(self, directions: List[str]) -> str:
        """Find consensus direction."""
        if not directions:
            return "NEUTRAL"
        buys = sum(1 for d in directions if d == "BUY")
        sells = sum(1 for d in directions if d == "SELL")

        if buys > sells:
            return "BUY"
        elif sells > buys:
            return "SELL"
        return "NEUTRAL"


# ─── Utility Functions ─────────────────────────────────────────────────────

def load_line_history(match_id: str, limit: int = 20) -> List[LineSnapshot]:
    """Load historical line data for a match."""
    history_path = DATA_DIR / "line_history" / f"{match_id}.jsonl"

    if not history_path.exists():
        return []

    snapshots = []
    with open(history_path) as f:
        for line in f:
            data = json.loads(line)
            snapshots.append(LineSnapshot(**data))

    return snapshots[-limit:]


def save_signal_results(match_id: str, results: Dict[str, SignalResult]):
    """Log signal results for tracking."""
    log_path = DATA_DIR / "ah_signals.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.now().isoformat(),
        "match_id": match_id,
        "signals": {name: {
            "fired": r.fired,
            "direction": r.direction,
            "strength": r.strength,
            "confidence": r.confidence,
            "metadata": r.metadata
        } for name, r in results.items()}
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AH Signal Providers")
    sub = parser.add_subparsers(dest='cmd')

    # Test command
    test = sub.add_parser('test', help='Test signals on sample data')
    test.add_argument('--match-id', default='TEST001')
    test.add_argument('--line', type=float, default=-0.5)
    test.add_argument('--goals', type=float, default=2.5)

    args = parser.parse_args()

    if args.cmd == 'test':
        # Create sample data
        model_lines = {args.match_id: -0.75}
        filter_ = AHSignalFilter(model_lines)

        # Create line history
        history = [
            LineSnapshot("2026-04-20T09:00:00", -0.25, 1.90, 1.90, 2.5, 1.90, 1.90, 100000, 55),
            LineSnapshot("2026-04-20T10:00:00", -0.5, 1.95, 1.85, 2.5, 1.90, 1.90, 120000, 60),
            LineSnapshot("2026-04-20T11:00:00", -0.5, 1.95, 1.85, 2.5, 1.90, 1.90, 150000, 65),
            LineSnapshot("2026-04-20T12:00:00", -0.75, 1.90, 1.90, 2.5, 1.90, 1.90, 200000, 72),
        ]

        current = LineSnapshot(
            datetime.now().isoformat(),
            args.line, 1.90, 1.90, args.goals, 1.90, 1.90, 250000, 70
        )

        model = {"fair_hdp": -0.75, "fair_goals": 2.5}

        results = filter_.evaluate_all(args.match_id, current, history, model)

        print("=" * 70)
        print("AH SIGNAL TEST RESULTS")
        print("=" * 70)
        for name, result in results.items():
            print(f"\n{name}:")
            print(f"  Fired: {result.fired}")
            print(f"  Direction: {result.direction}")
            print(f"  Strength: {result.strength:.2f}")
            print(f"  Confidence: {result.confidence:.2f}")
            if result.metadata:
                print(f"  Metadata: {result.metadata}")

        confirmed, conf, direction = filter_.get_trade_confirmation(results)
        print("\n" + "=" * 70)
        print(f"TRADE CONFIRMATION: {confirmed}")
        print(f"  Direction: {direction}")
        print(f"  Confidence: {conf:.2f}")
        print("=" * 70)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
