#!/usr/bin/env python3
"""WC 2026 Daily Odds Monitor — live Polymarket vs v3 model"""
import json
import os
import sys
import urllib.request
import re
from datetime import datetime

OUTPUT_DIR = "/tmp/wc_monitor"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = '/home/ubuntu/.hermes/data/wc2026_signals/model_predictions.json'

# ---------- helpers ----------
def parse_prices(prices):
    if prices is None:
        return None, None
    if isinstance(prices, list):
        try:
            return float(prices[0]), float(prices[1])
        except Exception:
            return None, None
    if isinstance(prices, str):
        try:
            p = json.loads(prices)
            return float(p[0]), float(p[1])
        except Exception:
            return None, None
    return None, None


def fetch_polymarket_wc_odds():
    """Fetch active WC 2026 winner markets from Polymarket Gamma API."""
    url = "https://gamma-api.polymarket.com/markets?active=true&limit=100"
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())

    wc_pattern = re.compile(r'Will (.+?) win the 2026 FIFA World Cup', re.IGNORECASE)
    odds = {}
    for m in data:
        q = m.get('question', '')
        match = wc_pattern.search(q)
        if not match:
            continue
        team = match.group(1).strip()
        yes_price, _ = parse_prices(m.get('outcomePrices'))
        if yes_price is None:
            continue
        # Normalize team names to title case, except known acronyms
        if team.upper() == 'USA':
            team = 'USA'
        else:
            team = team.title()
        odds[team] = {
            'implied_probability': yes_price,
            'volume': float(m.get('volume', '0')),
            'liquidity': float(m.get('liquidity', '0')),
            'market_id': m.get('id'),
            'question': q,
        }
    return odds


def load_model():
    with open(MODEL_PATH, 'r') as f:
        raw = json.load(f)
    model = {}
    for k, v in raw.items():
        name = k.title() if k.lower() != 'usa' else 'USA'
        model[name] = float(v)
    return model


# ---------- main ----------
print("Fetching live Polymarket odds ...", file=sys.stderr)
try:
    live = fetch_polymarket_wc_odds()
    print(f"Fetched {len(live)} live WC 2026 markets from Polymarket", file=sys.stderr)
except Exception as e:
    print(f"ERROR fetching Polymarket: {e}", file=sys.stderr)
    sys.exit(1)

model = load_model()
print(f"Loaded {len(model)} team probabilities from {MODEL_PATH}", file=sys.stderr)

# Align and compare
results = []
for team in sorted(set(live.keys()) | set(model.keys())):
    market_prob = live.get(team, {}).get('implied_probability', None)
    mod_prob = model.get(team, None)
    if market_prob is None or mod_prob is None:
        continue
    market_pct = market_prob * 100
    mod_pct = mod_prob * 100
    delta = mod_pct - market_pct
    if delta > 3:
        sig, action = "BUY++", "ACCUMULATE"
    elif delta > 1:
        sig, action = "BUY+", "ACCUMULATE"
    elif delta < -3:
        sig, action = "SELL++", "REDUCE"
    elif delta < -1:
        sig, action = "SELL+", "REDUCE"
    else:
        sig = action = "FAIR"
    results.append({
        'team': team,
        'market': market_pct,
        'model': mod_pct,
        'delta': delta,
        'signal': sig,
        'action': action,
        'volume': live[team]['volume'],
    })

# Sort by |delta| descending for display
results.sort(key=lambda x: abs(x['delta']), reverse=True)

# Console output
print("=" * 80)
print(f"WC 2026 DAILY MONITOR - {datetime.now().isoformat()}")
print("Live Polymarket  vs  v3 Historical Model")
print("=" * 80)
print(f"{'Team':<18} {'Market':>8} {'Model':>8} {'Delta':>8} {'Signal':>8} {'Volume':>12}")
print("-" * 80)
for r in results:
    print(f"{r['team']:<18} {r['market']:>7.2f}% {r['model']:>7.2f}% {r['delta']:>+7.2f}% {r['signal']:>8} ${r['volume']:>10,.0f}")

signals = [r for r in results if r['signal'] != 'FAIR']

print("\n" + "=" * 80)
print("TOP TRADING SIGNALS")
print("=" * 80)
if signals:
    signals.sort(key=lambda x: abs(x['delta']), reverse=True)
    for s in signals[:8]:
        dir_arrow = "↑" if s['delta'] > 0 else "↓"
        print(f"{s['signal']:>6}  {s['team']:<16}  {s['delta']:>+6.2f}% edge  {dir_arrow}  {s['action']}")
else:
    print("No significant signals today")

# Save report
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = f"{OUTPUT_DIR}/report_{timestamp}.json"
with open(report_path, 'w') as f:
    json.dump({
        'timestamp': datetime.now().isoformat(),
        'teams_tracked': len(results),
        'signals': len(signals),
        'results': results,
        'signals_detail': signals[:8],
    }, f, indent=2)

print(f"\nReport saved: {report_path}", file=sys.stderr)
