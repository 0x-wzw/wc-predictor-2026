#!/usr/bin/env python3
"""
AH Backtest Validator
Validate Asian Handicap model accuracy against historical match data.
"""
import json
import math
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

BASE = Path.home() / ".hermes"
AH_DIR = BASE / "models" / "ah_models"
DATA_DIR = BASE / "data" / "ah_models"

for d in [AH_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class HistoricalMatch:
    """Historical match result for validation."""
    match_id: str
    date: str
    home_team: str
    away_team: str
    home_goals: int
    away_goals: int
    rating_home: float
    rating_away: float


@dataclass
class LinePrediction:
    """Model prediction for a line (backtest record)."""
    match_id: str
    match_date: str
    prediction_timestamp: str
    line: float
    side: str        # 'home' or 'away'
    model_prob: float
    market_prob: float
    market_odds: float
    result: Optional[str] = None  # 'win', 'loss', 'push'
    actual_return: float = 0.0    # Stake-normalized return (-1 to +odds)


@dataclass
class BacktestResult:
    """Aggregated backtest metrics."""
    total_predictions: int
    wins: int
    losses: int
    pushes: int
    accuracy: float
    roi: float
    brier_score: float
    sharpe_ratio: float
    kelly_bankroll: float         # Ending bankroll from Kelly sizing


@dataclass
class LineAccuracy:
    """Accuracy of line direction prediction."""
    total: int
    correct_direction: int    # Model said -0.5, favorite won
    wrong_direction: int
    push_count: int
    line_accuracy: float        # Correct direction rate
    closing_edge: float         # Model vs closing line accuracy


# ─── Backtest Engine ───────────────────────────────────────────────────────

class AHBacktester:
    """
    Validate AH model against historical matches.
    """

    def __init__(self, predictions_file: str = None):
        self.predictions: List[LinePrediction] = []
        self.results: List[BacktestResult] = []

        self.prediction_log = DATA_DIR / "prediction_log.jsonl"
        self.results_file = DATA_DIR / "backtest_results.json"

    def load_historical_matches(self, filepath: str) -> List[HistoricalMatch]:
        """Load historical match results."""
        matches = []
        with open(filepath) as f:
            for line in f:
                data = json.loads(line)
                matches.append(HistoricalMatch(**data))
        return matches

    def validate_line_prediction(
        self,
        prediction: LinePrediction,
        match: HistoricalMatch
    ) -> LinePrediction:
        """
        Check if prediction beat the line.
        """
        goal_diff = match.home_goals - match.away_goals

        # Determine result against the line
        line = prediction.line
        side = prediction.side

        if side == 'home':
            # Home bet with handicap line
            if line == -0.5:
                # Home must win
                result = 'win' if goal_diff > 0 else 'loss'
                actual_return = prediction.market_odds - 1 if result == 'win' else -1
            elif line == 0.0:
                # Draw no bet
                result = 'win' if goal_diff > 0 else ('push' if goal_diff == 0 else 'loss')
                actual_return = prediction.market_odds - 1 if result == 'win' else (0 if result == 'push' else -1)
            elif line == -1.0:
                # Win by 2+ for full win, win by 1 for push
                if goal_diff > 1:
                    result = 'win'
                    actual_return = prediction.market_odds - 1
                elif goal_diff == 1:
                    result = 'push'
                    actual_return = 0
                else:
                    result = 'loss'
                    actual_return = -1
            elif line == -0.25:
                # Half on 0, half on -0.5
                if goal_diff > 0:
                    result = 'win'
                    actual_return = prediction.market_odds - 1
                elif goal_diff == 0:
                    result = 'push'
                    actual_return = -0.5  # Lose half
                else:
                    result = 'loss'
                    actual_return = -1
            elif line == -0.75:
                # Half on -0.5, half on -1.0
                if goal_diff > 1:
                    result = 'win'
                    actual_return = prediction.market_odds - 1
                elif goal_diff == 1:
                    result = 'win_half'  # Win half
                    actual_return = (prediction.market_odds - 1) / 2
                else:
                    result = 'loss'
                    actual_return = -1
            else:
                result = 'unknown'
                actual_return = 0
        else:
            # Away bet (mirror of home)
            if line == 0.5:
                # Away +0.5 = win or draw
                result = 'win' if goal_diff < 0 else ('loss' if goal_diff > 0 else 'win')
                actual_return = prediction.market_odds - 1 if result == 'win' else -1
            elif line == 1.0:
                # Away +1
                if goal_diff < -1:
                    result = 'win'
                    actual_return = prediction.market_odds - 1
                elif goal_diff == -1:
                    result = 'push'
                    actual_return = 0
                else:
                    result = 'loss'
                    actual_return = -1
            else:
                result = 'unknown'
                actual_return = 0

        prediction.result = result
        prediction.actual_return = actual_return
        return prediction

    def run_backtest(
        self,
        predictions: List[LinePrediction],
        matches: List[HistoricalMatch]
    ) -> BacktestResult:
        """
        Run full backtest across all predictions.
        """
        match_lookup = {m.match_id: m for m in matches}

        validated = []
        for pred in predictions:
            match = match_lookup.get(pred.match_id)
            if match:
                validated_pred = self.validate_line_prediction(pred, match)
                validated.append(validated_pred)

        # Calculate metrics
        wins = sum(1 for p in validated if p.result == 'win')
        losses = sum(1 for p in validated if p.result == 'loss')
        pushes = sum(1 for p in validated if p.result in ['push', 'win_half'])

        total = len(validated) - pushes  # Exclude pushes from accuracy
        accuracy = wins / total if total > 0 else 0

        # ROI calculation
        total_return = sum(p.actual_return for p in validated)
        roi = total_return / len(validated) if validated else 0

        # Brier score for probability calibration
        brier = self._calculate_brier(validated)

        # Kelly bankroll simulation
        bankroll = self._kelly_bankroll(validated)

        # Sharpe ratio (return / std)
        returns = [p.actual_return for p in validated if p.result != 'unknown']
        if len(returns) > 1:
            mean_ret = statistics.mean(returns)
            std_ret = statistics.stdev(returns) if len(returns) > 1 else 0.01
            if std_ret > 0:
                sharpe = mean_ret / std_ret * (len(returns) ** 0.5)
            else:
                sharpe = 0
        else:
            sharpe = 0

        result = BacktestResult(
            total_predictions=len(validated),
            wins=wins,
            losses=losses,
            pushes=pushes,
            accuracy=accuracy,
            roi=roi,
            brier_score=brier,
            sharpe_ratio=sharpe,
            kelly_bankroll=bankroll
        )

        self._save_result(result)
        return result

    def _calculate_brier(self, predictions: List[LinePrediction]) -> float:
        """Calculate Brier score."""
        brier_sum = 0
        n = 0
        for p in predictions:
            if p.result == 'unknown':
                continue
            outcome = 1.0 if p.result == 'win' else 0.0
            brier_sum += (p.model_prob - outcome) ** 2
            n += 1
        return brier_sum / n if n > 0 else 0

    def _kelly_bankroll(self, predictions: List[LinePrediction]) -> float:
        """Simulate Kelly-bet bankroll growth."""
        bankroll = 1.0
        for p in predictions:
            if p.result == 'unknown':
                continue

            # Quarter-Kelly sizing
            kelly_fraction = max(0, (p.model_prob * p.market_odds - 1) / (p.market_odds - 1))
            kelly_fraction = min(kelly_fraction * 0.25, 0.25)  # Cap at 25%

            stake = bankroll * kelly_fraction
            bankroll += stake * p.actual_return

            if bankroll <= 0:
                bankroll = 0.01  # Minimum

        return bankroll

    def validate_line_direction(
        self,
        model_lines: Dict[str, float],  # match_id -> model fair line
        matches: List[HistoricalMatch]
    ) -> LineAccuracy:
        """
        Did model line direction match actual result?
        """
        total = 0
        correct = 0
        wrong = 0
        pushes = 0

        for match in matches:
            mid = match.match_id
            if mid not in model_lines:
                continue

            total += 1
            model_line = model_lines[mid]
            actual_diff = match.home_goals - match.away_goals

            # Expected winner based on line
            if model_line < 0:
                expected_winner = 'home'
            elif model_line > 0:
                expected_winner = 'away'
            else:
                expected_winner = 'draw'

            # Actual winner
            if actual_diff > 0:
                actual_winner = 'home'
            elif actual_diff < 0:
                actual_winner = 'away'
            else:
                actual_winner = 'draw'

            # Check line accuracy
            if expected_winner == 'draw' and actual_winner == 'draw':
                pushes += 1
            elif expected_winner == actual_winner:
                correct += 1
            else:
                wrong += 1

        return LineAccuracy(
            total=total,
            correct_direction=correct,
            wrong_direction=wrong,
            push_count=pushes,
            line_accuracy=correct / (correct + wrong) if (correct + wrong) > 0 else 0,
            closing_edge=0.0  # TODO: compare vs closing lines
        )

    def _save_result(self, result: BacktestResult):
        """Log backtest result."""
        record = {
            "timestamp": datetime.now().isoformat(),
            **asdict(result)
        }

        with open(self.results_file, "a") as f:
            f.write(json.dumps(record) + "\n")

    def generate_report(self, results: List[BacktestResult]) -> str:
        """Generate human-readable backtest report."""
        if not results:
            return "No backtest results to report."

        latest = results[-1]

        report = f"""
AH MODEL BACKTEST REPORT
{'=' * 60}
Generated: {datetime.now().isoformat()}

OVERALL PERFORMANCE
{'─' * 60}
Total Predictions: {latest.total_predictions:,}
Wins: {latest.wins:,}
Losses: {latest.losses:,}
Pushes: {latest.pushes:,}

Accuracy (excl. pushes): {latest.accuracy:.1%}
ROI per Bet: {latest.roi:+.2%}
Brier Score: {latest.brier_score:.4f}
Sharpe Ratio: {latest.sharpe_ratio:.2f}
Kelly Bankroll: {latest.kelly_bankroll:.2f}x

{'=' * 60}

INTERPRETATION
{'─' * 60}
- Accuracy > 55%: Model has predictive edge
- ROI > 3%: Beating vig and variance
- Brier < 0.22: Well-calibrated probabilities
- Sharpe > 0.5: Risk-adjusted returns
- Kelly > 2.0x: Profitable Kelly sizing

Status: {"✓ MODEL VALIDATED" if latest.accuracy > 0.55 and latest.roi > 0.03 else "⚠ MODEL NEEDS IMPROVEMENT"}
{'=' * 60}
"""
        return report


# ─── Simulated Historical Data Generator ─────────────────────────────────

def generate_sample_historical_data(num_matches: int = 100, output_path: str = None):
    """Generate sample historical data for testing backtester."""
    matches = []
    teams = ["Argentina", "Brazil", "Germany", "France", "Spain",
             "England", "Netherlands", "Portugal", "Belgium", "Italy"]

    np.random.seed(42)

    for i in range(num_matches):
        t1, t2 = np.random.choice(teams, 2, replace=False)

        # Generate somewhat realistic scores based on random strength
        s1, s2 = np.random.randint(0, 5, 2)
        if np.random.random() < 0.3:  # 30% draws
            s2 = s1

        matches.append(HistoricalMatch(
            match_id=f"M{i:04d}",
            date=f"2026-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}",
            home_team=t1,
            away_team=t2,
            home_goals=s1,
            away_goals=s2,
            rating_home=np.random.randint(1400, 1900),
            rating_away=np.random.randint(1400, 1900)
        ))

    if output_path:
        with open(output_path, 'w') as f:
            for m in matches:
                f.write(json.dumps({
                    "match_id": m.match_id,
                    "date": m.date,
                    "home_team": m.home_team,
                    "away_team": m.away_team,
                    "home_goals": m.home_goals,
                    "away_goals": m.away_goals,
                    "rating_home": m.rating_home,
                    "rating_away": m.rating_away
                }) + "\n")

    return matches


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AH Backtest Validator")
    sub = parser.add_subparsers(dest='cmd')

    # Generate command
    gen = sub.add_parser('generate', help='Generate sample historical data')
    gen.add_argument('--matches', type=int, default=100)
    gen.add_argument('--output', default=str(DATA_DIR / "historical_sample.jsonl"))

    # Run command
    run = sub.add_parser('run', help='Run backtest on predictions')
    run.add_argument('--predictions', required=True)
    run.add_argument('--matches', required=True)

    # Report command
    report = sub.add_parser('report', help='Generate backtest report')
    report.add_argument('--results', default=str(DATA_DIR / "backtest_results.json"))

    args = parser.parse_args()

    if args.cmd == 'generate':
        matches = generate_sample_historical_data(args.matches, args.output)
        print(f"Generated {len(matches)} sample matches to {args.output}")

    elif args.cmd == 'run':
        bt = AHBacktester()

        # Load data
        with open(args.predictions) as f:
            preds = [LinePrediction(**json.loads(line)) for line in f]

        matches = bt.load_historical_matches(args.matches)
        result = bt.run_backtest(preds, matches)

        print(bt.generate_report([result]))

    elif args.cmd == 'report':
        bt = AHBacktester()

        results = []
        if Path(args.results).exists():
            with open(args.results) as f:
                for line in f:
                    data = json.loads(line)
                    del data['timestamp']  # Remove non-dataclass field
                    results.append(BacktestResult(**data))

        print(bt.generate_report(results))

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
