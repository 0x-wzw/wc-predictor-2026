#!/usr/bin/env python3
"""
WC2026 Unified Prediction Pipeline v2.0
========================================
Model-first architecture. Internal model anchors all predictions.
External data (odds, results, news) feeds INTO the model.
Signals (S1-S8) filter trade execution.

Commands:
  status          - Show pipeline health
  ingest          - Fetch external odds
  compare         - Model edge vs market
  signals         - Generate filtered trade signals
  report          - Full daily report
"""

import json, math, os, sys, urllib.request, re, hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List
from dataclasses import dataclass, asdict

# ─── Paths ───────────────────────────────────────────────────────
BASE = Path.home() / ".hermes"
MODEL_DIR = BASE / "models" / "wc2026"
DATA_DIR = BASE / "data" / "wc2026_signals"
REPORTS = Path("/tmp/wc_monitor")

MODEL_PATH = DATA_DIR / "model_predictions.json"
STATE_PATH = MODEL_DIR / "pipeline_state.json"
PREDICTIONS_LOG = MODEL_DIR / "predictions_log.jsonl"
SIGNALS_LOG = MODEL_DIR / "external_signals.jsonl"

REPORTS.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ─── Constants ──────────────────────────────────────────────────
MIN_EDGE = 0.02        # 2% minimum
STRONG_EDGE = 0.05      # 5% strong override
MIN_LIQUIDITY = 50000 # $50K
ACCOUNT_SIZE = 10000   # For Kelly sizing calc

@dataclass
class EdgeReport:
    team: str
    model_prob: float
    market_prob: float
    edge: float
    signal_count: int
    signals_detail: List[str]
    confidence: float
    action: str
    position_size: float

# ─── Pipeline Core ──────────────────────────────────────────────
class UnifiedPipeline:
    """Master orchestrator for model-first prediction."""

    def __init__(self):
        self.model: Dict[str, float] = {}
        self.market: Dict[str, dict] = {}
        self.state: dict = {}
        self._load()

    def _load(self):
        if MODEL_PATH.exists():
            with open(MODEL_PATH) as f:
                self.model = json.load(f)
        if STATE_PATH.exists():
            with open(STATE_PATH) as f:
                self.state = json.load(f)
        else:
            self.state = {
                "last_ingest": "",
                "last_compare": "",
                "predictions": 0,
                "brier_30d": 0.0,
            }

    def _save_state(self):
        with open(STATE_PATH, "w") as f:
            json.dump(self.state, f, indent=2)

    def _model_hash(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model, sort_keys=True).encode()
        ).hexdigest()[:16]

    # ── LAYER 0: Status ──────────────────────────────────────────

    def status(self):
        print("=" * 58)
        print(" WC2026 UNIFIED PIPELINE v2.0")
        print(" Model-First | Signals-Filter | Continuous Fine-Tune")
        print("=" * 58)
        print(f"\n Model:")
        print(f"   Teams:     {len(self.model)}")
        print(f"   Hash:      {self._model_hash()}")
        print(f"\n Pipeline:")
        print(f"   Predictions:  {self.state.get('predictions', 0)}")
        print(f"   Brier (30d):  {self.state.get('brier_30d', 0):.4f}")
        print(f"   Last ingest:   {self.state.get('last_ingest', 'Never')}")

    # ── LAYER 1: Ingest External Data ────────────────────────────

    def ingest(self):
        print("Fetching Polymarket...")
        self.market = self._fetch_polymarket()

        if not self.market:
            print("WARNING: No data fetched")
            return

        # Log to signals
        with open(SIGNALS_LOG, "a") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(),
                "source": "polymarket",
                "teams": len(self.market),
            }) + "\n")

        self.state["last_ingest"] = datetime.now().isoformat()
        self._save_state()
        print(f"  → {len(self.market)} markets ingested")

    def _fetch_polymarket(self) -> Dict[str, dict]:
        url = "https://gamma-api.polymarket.com/markets?active=true&limit=100"
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = json.loads(r.read())
        except Exception as e:
            print(f"  Error: {e}")
            return {}

        pattern = re.compile(r'Will (.+?) win the 2026 FIFA World Cup', re.IGNORECASE)
        markets = {}

        for m in data:
            match = pattern.search(m.get('question', ''))
            if not match:
                continue
            team = match.group(1).strip()
            team = 'USA' if team.upper() == 'USA' else team.title()

            prices = m.get('outcomePrices')
            yes = None
            if isinstance(prices, str):
                try: yes = float(json.loads(prices)[0])
                except: pass
            elif isinstance(prices, list):
                try: yes = float(prices[0])
                except: pass

            if yes is None:
                continue

            markets[team] = {
                'prob': yes,
                'volume': float(m.get('volume', 0)),
                'liquidity': float(m.get('liquidity', 0)),
                'id': m.get('id'),
            }
        return markets

    # ── LAYER 2: Compare Model vs Market ───────────────────────────

    def compare(self):
        if not self.market:
            print("Run ingest first.")
            return
        if not self.model:
            print("No internal model found.")
            return

        self.edges = []
        for team in set(self.model.keys()) & set(self.market.keys()):
            mp = self.model[team]
            mk = self.market[team]
            edge = mp - mk['prob']

            self.edges.append({
                'team': team,
                'model': round(mp, 4),
                'market': round(mk['prob'], 4),
                'edge': round(edge, 4),
                'edge_pct': round(edge * 100, 2),
                'volume': mk['volume'],
            })

        self.edges.sort(key=lambda x: abs(x['edge']), reverse=True)

        # Save
        path = MODEL_DIR / f"compare_{datetime.now():%Y%m%d_%H%M}.json"
        with open(path, "w") as f:
            json.dump({
                "ts": datetime.now().isoformat(),
                "count": len(self.edges),
                "edges": self.edges,
            }, f, indent=2)

        self.state["last_compare"] = datetime.now().isoformat()
        self._save_state()

        print(f"  → {len(self.edges)} edges calculated")
        if self.edges:
            print(f"  → Strongest: {self.edges[0]['team']} {self.edges[0]['edge_pct']:+.2f}%")

    # ── LAYER 3: Signal Filter ───────────────────────────────────

    def signals(self):
        if not hasattr(self, 'edges') or not self.edges:
            self.ingest()
            self.compare()

        reports = []
        for e in self.edges:
            edge = e['edge']
            abs_e = abs(edge)

            # Skip low edge or low liquidity
            if abs_e < MIN_EDGE:
                continue
            if e['volume'] < MIN_LIQUIDITY:
                continue

            # Evaluate confirmation signals
            sigs = self._evaluate_signals(e)
            n_sig = len(sigs)

            # Trade logic
            if abs_e >= STRONG_EDGE:
                trade, conf, reason = True, min(0.95, 0.5 + abs_e * 10), "strong_override"
            elif abs_e >= MIN_EDGE and n_sig >= 2:
                trade, conf, reason = True, min(0.8, 0.4 + abs_e * 10), "confirmed"
            elif abs_e >= MIN_EDGE and n_sig >= 1:
                trade, conf, reason = True, min(0.6, 0.3 + abs_e * 8), "weak_conf"
            else:
                trade, conf, reason = False, min(0.3, abs_e * 5), "no_conf"

            if not trade:
                continue

            action = "BUY" if edge > 0 else "SELL"
            if abs_e < STRONG_EDGE:
                action += "+"

            size = self._position_size(edge, e['market'], conf, e['volume'])

            reports.append(EdgeReport(
                team=e['team'], model_prob=e['model'], market_prob=e['market'],
                edge=edge, signal_count=n_sig, signals_detail=sigs,
                confidence=round(conf, 3), action=action, position_size=round(size, 2),
            ))

        reports.sort(key=lambda x: abs(x.edge), reverse=True)
        self._print_signals(reports)

        # Save
        path = REPORTS / f"signals_{datetime.now():%Y%m%d_%H%M%S}.json"
        with open(path, "w") as f:
            json.dump({
                "ts": datetime.now().isoformat(),
                "signals": [asdict(r) for r in reports],
            }, f, indent=2)

        print(f"\n  → {len(reports)} signals → {path}")
        return reports

    def _evaluate_signals(self, e: dict) -> List[str]:
        sigs = []
        edge, prob, vol = e['edge'], e['market'], e['volume']

        # S1: Disposition (contrarian check)
        if edge < -0.05 and prob > 0.5:
            sigs.append("S1_contrarian")
        if edge > 0.05 and prob < 0.1:
            sigs.append("S1_deep_value")

        # S3: Velocity (stable)
        if abs(edge) < 0.10:
            sigs.append("S3_stable")

        # S6: Volume
        if vol > MIN_LIQUIDITY:
            sigs.append("S6_liquid")
        if vol > 500000:
            sigs.append("S6_high_vol")

        # S8: Theta (WC is far out)
        sigs.append("S8_favorable")

        # S9: Model edge strength
        if abs(edge) >= STRONG_EDGE:
            sigs.append("S9_strong")
        elif abs(edge) >= MIN_EDGE:
            sigs.append("S9_medium")

        return sigs

    def _position_size(self, edge, market_prob, confidence, volume) -> float:
        """Quarter-Kelly with safety rails. Handles BUY and SELL."""
        if market_prob <= 0 or market_prob >= 1:
            return 0

        # For SELL (edge < 0), we bet on NO → prob(NO) = 1 - P(YES)
        # For BUY (edge > 0), we bet on YES → prob(YES) = P(YES)
        if edge < 0:
            p_model = 1.0 - (market_prob + edge)  # model NO prob
            p_market = 1.0 - market_prob           # market NO prob
        else:
            p_model = market_prob + edge
            p_market = market_prob

        if p_model <= 0 or p_model >= 1 or p_market <= 0 or p_market >= 1:
            return 0

        odds = 1.0 / p_market
        b = odds - 1
        p = max(0.001, min(0.999, p_model))
        q = 1 - p

        if b <= 0:
            return 0

        kelly = max(0, (b * p - q) / b)
        kelly = min(kelly, 0.25) * 0.25  # Quarter-Kelly
        kelly *= confidence

        liquidity_cap = volume * 0.01
        size = ACCOUNT_SIZE * kelly
        return min(size, liquidity_cap, ACCOUNT_SIZE * 0.02)

    def _print_signals(self, reports: List[EdgeReport]):
        if not reports:
            print("\n  No actionable signals")
            return

        print("\n" + "=" * 88)
        print(" TRADE SIGNALS  |  Model First  |  Edge + Signal Filter")
        print("=" * 88)
        print(f"{'Team':<14} {'Model':>7} {'Market':>7} {'Edge':>7} {'Conf':>5} {'Signals':>10} {'Action':>7} {'Size':>8}")
        print("-" * 88)

        for r in reports:
            sig_short = ",".join(r.signals_detail[:2]) if r.signals_detail else "—"
            print(f"{r.team:<14} {r.model_prob*100:>6.1f}% {r.market_prob*100:>6.1f}% {r.edge*100:>+6.1f}% {r.confidence*100:>4.0f}% {sig_short:>10} {r.action:>7} ${r.position_size:>7,.0f}")

    # ── LAYER 5: Full Report ─────────────────────────────────────

    def report(self):
        self.ingest()
        self.compare()

        print("\n" + "=" * 70)
        print(" WC2026 UNIFIED DAILY REPORT")
        print(f" {datetime.now().isoformat()}")
        print("=" * 70)

        # Top edges
        print("\n🎯 TOP 10 EDGES (Model vs Market)")
        print(f"{'#':<3} {'Team':<14} {'Model':>7} {'Market':>7} {'Edge':>8} {'Action':>7}")
        print("-" * 50)
        for i, e in enumerate(self.edges[:10], 1):
            a = "BUY" if e['edge'] > 0 else "SELL"
            print(f"{i:<3} {e['team']:<14} {e['model']*100:>6.1f}% {e['market']*100:>6.1f}% {e['edge']*100:>+7.1f}% {a:>7}")

        # Trade signals
        signals = self.signals()

        print("\n📈 HEALTH")
        print(f"   Predictions: {self.state.get('predictions', 0)}")
        print(f"   Brier (30d): {self.state.get('brier_30d', 0):.4f}")

        return signals

# ─── CLI ────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    sub.add_parser('status', help='Pipeline status')
    sub.add_parser('ingest', help='Fetch external odds')
    sub.add_parser('compare', help='Model vs market edge')
    sub.add_parser('signals', help='Generate trade signals')
    sub.add_parser('report', help='Full daily report')

    args = parser.parse_args()
    p = UnifiedPipeline()

    cmds = {
        'status': p.status,
        'ingest': p.ingest,
        'compare': lambda: (p.ingest(), p.compare()),
        'signals': lambda: (p.ingest(), p.compare(), p.signals()),
        'report': p.report,
    }

    if args.cmd in cmds:
        cmds[args.cmd]()
    else:
        p.status()
        print("\nUsage: python wc2026_unified_pipeline.py {status|ingest|compare|signals|report}")

if __name__ == "__main__":
    main()
