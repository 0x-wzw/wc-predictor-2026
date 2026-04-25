#!/usr/bin/env python3
"""
AH Unified Pipeline v2.0
Asian Handicap model integrated with WC2026 unified pipeline.
Layer 2.A: AH-specific edge generation and signal filtering.
"""

import json
import sys
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add scripts to path for imports
BASE = Path.home() / ".hermes"
SYS_PATH = str(BASE / "scripts")
if SYS_PATH not in sys.path:
    sys.path.insert(0, SYS_PATH)

from ah_engine import AsianHandicapEngine, TeamRating, FairLines, AHLineEdge
from ah_signals import AHSignalFilter, LineSnapshot, SignalResult

DATA_DIR = BASE / "data" / "ah_models"
OUTPUT_DIR = BASE / "models" / "ah_models"
WC_DIR = BASE / "data" / "wc2026_signals"

for d in [DATA_DIR, OUTPUT_DIR, WC_DIR]:
    d.mkdir(parents=True, exist_ok=True)


# ─── Configuration ──────────────────────────────────────────────────────────

@dataclass
class AHConfig:
    """AH pipeline configuration."""
    min_edge: float = 0.025       # 2.5% minimum edge
    strong_edge: float = 0.05     # 5% strong override
    min_odds: float = 1.70        # Don't bet short odds
    max_odds: float = 3.00        # Value traps get rarer at extremes
    min_signals: int = 2          # Require 2+ AH signals
    max_position_pct: float = 0.05  # Max 5% per position
    account_size: float = 10000   # Bankroll


# ─── AH Unified Pipeline ───────────────────────────────────────────────────

class AHUnifiedPipeline:
    """
    Unified Asian Handicap prediction pipeline.
    Integrates with WC2026 pipeline via shared state files.
    """

    def __init__(self, config: AHConfig = None):
        self.config = config or AHConfig()
        self.engine = AsianHandicapEngine()
        self.signal_filter = None  # Initialized per match

        # State
        self.matches: List[Tuple[str, str, str]] = []  # (home, away, date)
        self.fair_lines: Dict[str, FairLines] = {}
        self.market_lines: Dict[str, Dict] = {}
        self.edges: List[AHLineEdge] = []
        self.trades: List[Dict] = []

    def set_upcoming_matches(self, matches: List[Tuple[str, str, str]]):
        """Set matches to analyze."""
        self.matches = matches

    def generate_fair_lines(self) -> Dict[str, FairLines]:
        """
        Stage 1: Calculate fair lines for all matches.
        """
        print("┌─" + "─" * 68 + "┐")
        print("│ ASIAN HANDICAP FAIR LINE CALCULATION" + " " * 31 + "│")
        print("└─" + "─" * 68 + "┘")
        print(f"\n{'Match':<30} {'Fair HDP':>10} {'Fair OU':>10} {'Conf':>8}")
        print("─" * 60)

        for home, away, date in self.matches:
            try:
                lines = self.engine.calculate_fair_lines(home, away, date)
                match_id = f"{home}_vs_{away}_{date}"
                self.fair_lines[match_id] = lines

                conf = lines.away_win  # Simplified confidence display
                print(f"{home:<14} vs {away:<14} {lines.fair_hdp:>+9.2f} "
                      f"{lines.fair_ou_line:>9.1f} {lines.home_win:>7.1%}")

            except Exception as e:
                print(f"  ⚠ Error calculating {home} vs {away}: {e}")

        # Save fair lines
        self._save_fair_lines()
        return self.fair_lines

    def ingest_market_lines(self, source: str = "simulated") -> Dict[str, Dict]:
        """
        Stage 2: Fetch or simulate market AH lines.
        """
        print("\n┌─" + "─" * 68 + "┐")
        print("│ MARKET LINE INGESTION" + " " * 46 + "│")
        print("└─" + "─" * 68 + "┘")

        if source == "simulated":
            # Simulate realistic market lines with some noise
            import random
            random.seed(42)

            for match_id, lines in self.fair_lines.items():
                # Add noise to fair lines
                hdp_noise = random.choice([-0.25, 0, 0.25]) + random.gauss(0, 0.1)
                fair_hdp = lines.fair_hdp

                market_hdp = round((fair_hdp + hdp_noise) * 4) / 4
                market_odds = 1.90 + random.gauss(0, 0.05)

                self.market_lines[match_id] = {
                    "home_line": market_hdp,
                    "home_odds": max(1.65, min(2.30, market_odds)),
                    "away_odds": max(1.65, min(2.30, 3.90 - market_odds)),
                    "ou_line": lines.fair_ou_line + random.choice([-0.25, 0, 0.25]),
                    "over_odds": 1.90,
                    "under_odds": 1.90,
                    "source": "simulated",
                    "timestamp": datetime.now().isoformat()
                }

        # TODO: Add support for Odds API, Betfair feeds

        print(f"  Ingested market lines for {len(self.market_lines)} matches")
        return self.market_lines

    def calculate_edges(self) -> List[AHLineEdge]:
        """
        Stage 3: Calculate edge vs market.
        """
        print("\n┌─" + "─" * 68 + "┐")
        print("│ EDGE CALCULATION" + " " * 53 + "│")
        print("└─" + "─" * 68 + "┘")
        print(f"\n{'Match':<25} {'Line':>8} {'Model%':>8} {'Mkt%':>8} {'Edge':>8} {'Size':>8}")
        print("─" * 75)

        self.edges = []

        for match_id, fair in self.fair_lines.items():
            if match_id not in self.market_lines:
                continue

            market = self.market_lines[match_id]

            # Calculate edge for home side
            try:
                edge = self.engine.calculate_edge(
                    fair,
                    market["home_line"],
                    market["home_odds"],
                    "home"
                )

                if abs(edge.edge) >= self.config.min_edge:
                    self.edges.append(edge)
                    print(f"{match_id[:24]:<25} {edge.line:>+8.2f} "
                          f"{edge.model_prob*100:>7.1f}% {edge.market_prob*100:>7.1f}% "
                          f"{edge.edge*100:>+7.1f}% {edge.kelly_fraction*100:>7.1f}%")

            except Exception as e:
                print(f"  ⚠ Error calculating edge: {e}")

        # Sort by edge magnitude
        self.edges.sort(key=lambda x: abs(x.edge), reverse=True)
        return self.edges

    def filter_signals(self) -> List[Dict]:
        """
        Stage 4: Run AH-specific signals (S10, S11, S12).
        """
        print("\n┌─" + "─" * 68 + "┐")
        print("│ AH SIGNAL FILTER (S10-S12)" + " " * 41 + "│")
        print("└─" + "─" * 68 + "┘")

        # Initialize signal filter with model fair lines
        model_hdp = {mid: lines.fair_hdp for mid, lines in self.fair_lines.items()}
        self.signal_filter = AHSignalFilter(model_hdp)

        self.trades = []

        for edge in self.edges:
            if abs(edge.edge) < self.config.min_edge:
                continue

            # Find match_id from edge
            match_id = self._find_match_id(edge)
            if not match_id:
                # Try direct lookup by iterating
                for mid in self.fair_lines:
                    lines = self.fair_lines[mid]
                    # Check if this edge could belong to this match
                    if abs(lines.fair_hdp - edge.line) < 2.0:  # Within reasonable range
                        match_id = mid
                        break
            if not match_id:
                continue

            # Get line snapshot
            market = self.market_lines.get(match_id, {})
            line_snapshot = LineSnapshot(
                timestamp=datetime.now().isoformat(),
                home_spread=market.get("home_line", 0),
                home_odds=market.get("home_odds", 0),
                away_odds=market.get("away_odds", 0),
                total_line=market.get("ou_line", 2.5),
                over_odds=market.get("over_odds", 0),
                under_odds=market.get("under_odds", 0),
                volume=100000,  # Simulated
                public_pct=60   # Simulated
            )

            # Create model dict for S12
            model = {
                "fair_hdp": self.fair_lines.get(match_id, FairLines(
                    match_id, "", "", "", 0, 0, 0, 0, 0, 0, 0, 0, (0, 0)
                )).fair_hdp,
                "fair_goals": 2.5
            }

            # Evaluate signals
            results = self.signal_filter.evaluate_all(
                match_id, line_snapshot, [], model
            )

            # Check confirmation
            confirmed, confidence, direction = self.signal_filter.get_trade_confirmation(
                results, self.config.min_signals
            )
            # For demo: relax signal requirements
            if not confirmed and (abs(edge.edge) > self.config.min_edge):
                confirmed = True
                confidence = 0.5
                direction = "BUY" if edge.edge > 0 else "SELL"

            # Override for strong edges
            if abs(edge.edge) >= self.config.strong_edge:
                confirmed = True
                confidence = min(confidence + 0.2, 0.95)

            if confirmed:
                trade = {
                    "match_id": match_id,
                    "line": edge.line,
                    "side": edge.side,
                    "odds": edge.market_odds,
                    "model_prob": edge.model_prob,
                    "market_prob": edge.market_prob,
                    "edge": edge.edge,
                    "kelly": edge.kelly_fraction,
                    "confidence": confidence,
                    "direction": direction,
                    "signals": {k: v.fired for k, v in results.items()},
                    "signal_strength": {k: v.strength for k, v in results.items()}
                }
                self.trades.append(trade)

        return self.trades

    def finalize_trades(self) -> List[Dict]:
        """
        Stage 5: Apply position sizing and finalize trades.
        """
        print("\n┌─" + "─" * 68 + "┐")
        print("│ FINAL TRADES" + " " * 55 + "│")
        print("└─" + "─" * 68 + "┘")
        print(f"\n{'Match':<25} {'Line':>8} {'Odds':>6} {'Edge':>7} {'Kelly':>7} {'Size':>8} {'Conf':>6}")
        print("─" * 75)

        finalized = []
        for trade in self.trades:
            # Calculate position size
            raw_size = self.config.account_size * trade["kelly"]
            size = min(raw_size, self.config.account_size * self.config.max_position_pct)
            size = max(0, size)

            if size > 0:
                finalized_trade = {
                    **trade,
                    "position_size": size,
                    "expected_value": trade["edge"] * size
                }
                finalized.append(finalized_trade)

                match_display = trade["match_id"][:24]
                print(f"{match_display:<25} {trade['line']:>+8.2f} "
                      f"{trade['odds']:>6.2f} {trade['edge']*100:>+6.1f}% "
                      f"{trade['kelly']*100:>6.1f}% ${size:>6.0f} "
                      f"{trade['confidence']*100:>5.0f}%")

        self._save_trades(finalized)
        return finalized

    def generate_full_report(self) -> Dict:
        """
        Generate complete AH pipeline report.
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "matches_analyzed": len(self.matches),
            "fair_lines_generated": len(self.fair_lines),
            "edges_calculated": len(self.edges),
            "trades_filtered": len(self.trades),
            "trades_finalized": 0,
            "total_expected_value": 0.0,
            "trades": []
        }

        if self.trades:
            report["trades_finalized"] = len(self.trades)
            report["total_expected_value"] = sum(t.get("expected_value", 0) for t in self.trades)
            report["trades"] = self.trades

        return report

    def _find_match_id(self, edge: AHLineEdge) -> Optional[str]:
        """Find match_id for an edge."""
        for mid in self.fair_lines:
            lines = self.fair_lines[mid]
            if lines.fair_hdp == edge.line:  # Simplified match
                return mid
        return None

    def _save_fair_lines(self):
        """Save fair lines to file."""
        path = OUTPUT_DIR / f"fair_lines_{datetime.now():%Y%m%d_%H%M}.json"
        data = {mid: {
            "home_team": lines.home_team,
            "away_team": lines.away_team,
            "fair_hdp": lines.fair_hdp,
            "fair_ou": lines.fair_ou_line,
            "home_win": lines.home_win,
            "draw": lines.fair_goals,  # Simplified
            "away_win": lines.away_win
        } for mid, lines in self.fair_lines.items()}

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\n  → Saved fair lines to {path}")

    def _save_trades(self, trades: List[Dict]):
        """Log trades for tracking."""
        path = DATA_DIR / "trades.jsonl"
        with open(path, "a") as f:
            for t in trades:
                f.write(json.dumps(t) + "\n")

        print(f"\n  → Logged {len(trades)} trades to {path}")

    # ─── Workflow Orchestration ────────────────────────────────────────────────

    def run_full_pipeline(self) -> Dict:
        """Execute complete AH pipeline."""
        print("\n" + "=" * 70)
        print("  ASIAN HANDICAP UNIFIED PIPELINE v2.0")
        print("  Model-First | Edge Detection | Signal Filter | Trade")
        print("=" * 70)

        # Stage 1: Fair lines
        self.generate_fair_lines()

        # Stage 2: Market lines
        self.ingest_market_lines()

        # Stage 3: Edge calculation
        self.calculate_edges()

        # Stage 4: Signal filter
        self.filter_signals()

        # Stage 5: Finalize
        finalized = self.finalize_trades()

        # Generate report
        report = self.generate_full_report()

        print("\n┌─" + "─" * 68 + "┐")
        print("│ PIPELINE SUMMARY" + " " * 53 + "│")
        print("└─" + "─" * 68 + "┘")
        print(f"  Matches analyzed: {report['matches_analyzed']}")
        print(f"  Edges calculated: {report['edges_calculated']}")
        print(f"  Trades finalized: {report['trades_finalized']}")
        print(f"  Total EV: ${report['total_expected_value']:.2f}")
        print("=" * 70)

        return report


# ─── Integration with WC2026 Pipeline ──────────────────────────────────────

class AHIntegration:
    """
    Bridge to integrate AH pipeline with main WC2026 pipeline.
    """

    STATE_FILE = WC_DIR / "ah_integration_state.json"

    def __init__(self):
        self.state = self._load_state()

    def _load_state(self) -> Dict:
        if self.STATE_FILE.exists():
            with open(self.STATE_FILE) as f:
                return json.load(f)
        return {"last_run": None, "trades_open": 0}

    def _save_state(self):
        with open(self.STATE_FILE, "w") as f:
            json.dump(self.state, f)

    def get_upcoming_matches_from_wc(self) -> List[Tuple[str, str, str]]:
        """
        Load upcoming matches from WC2026 schedule.
        Returns: [(home, away, date), ...]
        """
        # TODO: Integrate with actual WC2026 schedule file
        # For now, return sample matches
        return [
            ("Argentina", "France", "2026-07-15"),
            ("Brazil", "England", "2026-07-16"),
            ("Germany", "Spain", "2026-07-17"),
            ("Netherlands", "Italy", "2026-07-18"),
            ("Portugal", "Belgium", "2026-07-19"),
            ("USA", "Mexico", "2026-07-20"),
        ]

    def sync_with_wc_signals(self, ah_trades: List[Dict]) -> Dict:
        """
        Sync AH trades with main WC2026 signal pipeline.
        Returns enriched signal data.
        """
        wc_signals = []

        for trade in ah_trades:
            wc_signal = {
                "type": "AH",
                "match": trade["match_id"],
                "direction": trade["direction"],
                "edge": trade["edge"],
                "confidence": trade["confidence"],
                "size": trade.get("position_size", 0),
                # Add as S10-S12 signal flags
                "s10": trade["signals"].get("S10", False),
                "s11": trade["signals"].get("S11", False),
                "s12": trade["signals"].get("S12", False),
            }
            wc_signals.append(wc_signal)

        return {
            "timestamp": datetime.now().isoformat(),
            "ah_signals": wc_signals,
            "signal_count": len(wc_signals)
        }

    def run(self):
        """Execute integrated AH + WC pipeline."""
        print("═" * 70)
        print("  AH + WC2026 INTEGRATED PIPELINE")
        print("═" * 70)

        # Get WC2026 matches
        matches = self.get_upcoming_matches_from_wc()

        # Run AH pipeline
        config = AHConfig()
        pipeline = AHUnifiedPipeline(config)
        pipeline.set_upcoming_matches(matches)
        report = pipeline.run_full_pipeline()

        # Sync with WC
        if report.get("trades"):
            sync = self.sync_with_wc_signals(report["trades"])
            print("\n" + "─" * 70)
            print("  SYNC WITH WC SIGNALS")
            print("─" * 70)
            print(f"  AH signals exported: {sync['signal_count']}")

            # Save to WC data dir
            sync_path = WC_DIR / f"ah_signals_{datetime.now():%Y%m%d}.json"
            with open(sync_path, "w") as f:
                json.dump(sync, f, indent=2)
            print(f"  → Saved to {sync_path}")

        # Update state
        self.state["last_run"] = datetime.now().isoformat()
        self.state["trades_open"] = len(report.get("trades", []))
        self._save_state()

        return report


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="AH Unified Pipeline")
    sub = parser.add_subparsers(dest='cmd')

    # Run command
    run = sub.add_parser('run', help='Run full AH pipeline')
    run.add_argument('--mode', choices=['standalone', 'integrated'], default='standalone')

    # Predict command
    pred = sub.add_parser('predict', help='Predict specific match')
    pred.add_argument('--home', required=True)
    pred.add_argument('--away', required=True)
    pred.add_argument('--line', type=float, required=True)
    pred.add_argument('--odds', type=float, required=True)
    pred.add_argument('--side', choices=['home', 'away'], default='home')

    args = parser.parse_args()

    if args.cmd == 'run':
        if args.mode == 'integrated':
            integration = AHIntegration()
            integration.run()
        else:
            config = AHConfig()
            pipeline = AHUnifiedPipeline(config)

            # Use default matches
            matches = [
                ("Argentina", "France", "2026-07-15"),
                ("Brazil", "England", "2026-07-16"),
                ("Germany", "Spain", "2026-07-17"),
                ("Netherlands", "Italy", "2026-07-18"),
            ]
            pipeline.set_upcoming_matches(matches)
            pipeline.run_full_pipeline()

    elif args.cmd == 'predict':
        engine = AsianHandicapEngine()
        lines = engine.calculate_fair_lines(args.home, args.away)
        edge = engine.calculate_edge(lines, args.line, args.odds, args.side)

        print("\n" + "=" * 70)
        print(f"  AH EDGE CALCULATION: {args.home} vs {args.away}")
        print("=" * 70)
        print(f"  Market Line: {args.line}")
        print(f"  Market Odds: {args.odds}")
        print(f"  Model Fair HDP: {lines.fair_hdp}")
        print(f"  Model Fair Goals: {lines.fair_ou_line}")
        print(f"\n  Model Probability: {edge.model_prob:.1%}")
        print(f"  Market Implied: {edge.market_prob:.1%}")
        print(f"  Edge: {edge.edge:+.2%}")
        print(f"  Expected Value: {edge.expected_value:+.2%}")
        print(f"  Kelly Fraction: {edge.kelly_fraction:.2%}")
        print("=" * 70)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
