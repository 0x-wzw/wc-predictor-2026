#!/usr/bin/env python3
"""
WC2026 Signal Checker — Pre-script for cron delivery.
Checks for latest signals and formats Telegram output.
"""

import json, glob, os
from datetime import datetime

def check():
    pattern = os.path.expanduser("/tmp/wc_monitor/signals_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return "📊 WC2026 Daily\nNo actionable signals today."
    
    latest = files[-1]
    with open(latest) as f:
        data = json.load(f)
    
    signals = data.get('signals', [])
    
    lines = [
        f"⚡ WC2026 Daily — {data.get('ts', datetime.now().isoformat())[:10]}",
        f"Signals: {len(signals)}",
    ]
    
    for s in signals[:5]:
        emoji = "🟢" if s['edge'] > 0 else "🔴"
        lines.append(f"{emoji} {s['action']} {s['team']}: ${s['position_size']:,.0f} ({s['edge']*100:+.1f}% edge)")
    
    return "\n".join(lines)

if __name__ == "__main__":
    print(check())
