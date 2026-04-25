#!/usr/bin/env python3
import os
import json
import random
from collections import defaultdict
from datetime import datetime

DATA_DIR = os.environ.get('WC_DATA_DIR', './data')
OUTPUT_DIR = os.environ.get('WC_OUTPUT_DIR', './results')

print("="*70)
print("WORLD CUP 2026 PREDICTION MODEL v3 - FIXED")
print("="*70)

with open(f"{DATA_DIR}/world_cup_unified_dataset.json", 'r') as f:
    data = json.load(f)

matches = [m for m in data['matches'] if isinstance(m.get('home_goals'), int)]

# Team stats by phase
team_stats = defaultdict(lambda: {
    'group': {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0},
    'knockout': {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0},
    'overall': {'titles': 0, 'finals': 0, 'semis': 0, 'r16': 0, 'quarters': 0}
})

KO_STAGES = {'Round of 16', 'Quarter-final', 'Semi-final', 'Final', 'Third place'}

for m in matches:
    home, away = m['home_team'], m['away_team']
    stage = m['stage']
    hg, ag = m['home_goals'], m['away_goals']
    
    if hg > ag:
        hr, ar = 'win', 'loss'
    elif hg < ag:
        hr, ar = 'loss', 'win'
    else:
        hr, ar = 'draw', 'draw'
    
    if stage == 'Group stage':
        team_stats[home]['group']['played'] += 1
        team_stats[away]['group']['played'] += 1
        if hr == 'win':
            team_stats[home]['group']['wins'] += 1
            team_stats[away]['group']['losses'] += 1
        elif hr == 'loss':
            team_stats[home]['group']['losses'] += 1
            team_stats[away]['group']['wins'] += 1
        else:
            team_stats[home]['group']['draws'] += 1
            team_stats[away]['group']['draws'] += 1
    elif stage in KO_STAGES:
        team_stats[home]['knockout']['played'] += 1
        team_stats[away]['knockout']['played'] += 1
        if hr == 'win':
            team_stats[home]['knockout']['wins'] += 1
            team_stats[away]['knockout']['losses'] += 1
        elif hr == 'loss':
            team_stats[home]['knockout']['losses'] += 1
            team_stats[away]['knockout']['wins'] += 1
    
    if stage == 'Final':
        team_stats[home]['overall']['finals'] += 1
        team_stats[away]['overall']['finals'] += 1
        if hr == 'win':
            team_stats[home]['overall']['titles'] += 1
    elif stage == 'Semi-final':
        team_stats[home]['overall']['semis'] += 1
        team_stats[away]['overall']['semis'] += 1
    elif stage == 'Quarter-final':
        team_stats[home]['overall']['quarters'] += 1
        team_stats[away]['overall']['quarters'] += 1
    elif stage == 'Round of 16':
        team_stats[home]['overall']['r16'] += 1
        team_stats[away]['overall']['r16'] += 1

# Calculate phase-based win probabilities
team_probs = {}
for team, stats in team_stats.items():
    gs = stats['group']
    ko = stats['knockout']
    
    gs_wr = (gs['wins'] / gs['played']) if gs['played'] > 0 else 0.3
    ko_wr = (ko['wins'] / ko['played']) if ko['played'] > 0 else 0.4
    
    titles = stats['overall']['titles']
    finals = stats['overall']['finals']
    semis = stats['overall']['semis']
    quarters = stats['overall']['quarters']
    r16 = stats['overall']['r16']
    
    total_appearances = max(finals, 1)
    
    # Advancement probabilities
    p_qual = min(max(gs_wr * 1.2, 0.5), 0.95)
    p_r16 = min(max(r16 / total_appearances if total_appearances else 0.5, 0.3), 0.9)
    p_qf = min(max(quarters / max(r16, 1), 0.3), 0.9)
    p_sf = min(max(semis / max(quarters, 1), 0.3), 0.9)
    p_final = min(max(finals / max(semis, 1), 0.3), 0.9)
    p_win = titles / max(finals, 1) if finals > 0 else 0.5
    
    team_probs[team] = {
        'group_wr': gs_wr, 'ko_wr': ko_wr, 'titles': titles, 'finals': finals,
        'qualify': p_qual, 'r16': p_r16 * p_qual, 'qf': p_qf * p_r16 * p_qual,
        'sf': p_sf * p_qf * p_r16 * p_qual, 'final': p_final * p_sf * p_qf * p_r16 * p_qual,
        'win': p_win * p_final * p_sf * p_qf * p_r16 * p_qual
    }

# Monte Carlo simulation
def sim_tournament(teams, n=10000):
    results = {t: {'group': 0, 'r16': 0, 'qf': 0, 'sf': 0, 'final': 0, 'win': 0} for t in teams}
    
    for _ in range(n):
        # Group stage
        qualified = [t for t in teams if random.random() < team_probs.get(t, {}).get('qualify', 0.3)]
        for t in qualified[:16]:
            results[t]['group'] += 1
            
            # R16
            if random.random() < team_probs.get(t, {}).get('r16', 0.3) / max(team_probs.get(t, {}).get('qualify', 0.3), 0.01):
                results[t]['r16'] += 1
                # QF (simplified)
                if random.random() < team_probs.get(t, {}).get('qf', 0.2) / max(team_probs.get(t, {}).get('r16', 0.2), 0.01):
                    results[t]['qf'] += 1
                    # SF
                    if random.random() < 0.5:
                        results[t]['sf'] += 1
                        # Final
                        if random.random() < team_probs.get(t, {}).get('final', 0.3) / max(team_probs.get(t, {}).get('sf', 0.3), 0.01):
                            results[t]['final'] += 1
                            # Win
                            if random.random() < 0.5:
                                results[t]['win'] += 1
    
    return {t: {k: v/n for k, v in r.items()} for t, r in results.items()}

qualified = ["Spain", "France", "England", "Argentina", "Brazil", "Portugal", "Germany", "Netherlands",
             "Norway", "Japan", "Belgium", "Morocco", "Colombia", "USA", "Mexico", "Switzerland",
             "Uruguay", "Croatia", "Turkiye", "Ecuador", "Senegal"]

print("\nRunning 10,000 simulation Monte Carlo...")
sim_results = sim_tournament(qualified, n=10000)

print("\n" + "="*70)
print("v3 SIMULATION RESULTS (Phase-Dependent)")
print("="*70)

print(f"\n{'Team':<15} {'Win%':>8} {'Final%':>8} {'SF%':>8} {'QF%':>8}")
print("-"*55)

sorted_sim = sorted(sim_results.items(), key=lambda x: x[1]['win'], reverse=True)
for team, probs in sorted_sim[:15]:
    print(f"{team:<15} {probs['win']*100:>7.2f}% {probs['final']*100:>7.2f}% {probs['sf']*100:>7.2f}% {probs['qf']*100:>7.2f}%")

# Market comparison
market = {'Spain': 0.163, 'France': 0.161, 'England': 0.111, 'Argentina': 0.089, 'Brazil': 0.086,
          'Portugal': 0.080, 'Germany': 0.055, 'Netherlands': 0.034, 'Norway': 0.024, 'Japan': 0.022,
          'Belgium': 0.022, 'Morocco': 0.019, 'Colombia': 0.017, 'USA': 0.013, 'Mexico': 0.011,
          'Switzerland': 0.010, 'Uruguay': 0.010, 'Croatia': 0.009, 'Turkiye': 0.007, 'Ecuador': 0.007, 'Senegal': 0.007}

print("\n" + "="*70)
print("v3 MARKET COMPARISON")
print("="*70)
print(f"\n{'Team':<15} {'Market%':>10} {'v3 Model%':>10} {'Delta%':>10} {'Signal'}")
print("-"*65)

signals = []
for team in market:
    mkt = market[team]
    model = sim_results.get(team, {}).get('win', 0)
    delta = model - mkt
    
    if delta > 0.02:
        sig = "📈 BUY"
    elif delta < -0.02:
        sig = "📉 SELL"
    else:
        sig = "➡️ FAIR"
    
    signals.append({'team': team, 'market': mkt, 'model': model, 'delta': delta, 'signal': sig})
    print(f"{team:<15} {mkt*100:>9.2f}% {model*100:>9.2f}% {delta*100:>+9.2f}% {sig}")

# Save outputs
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(f"{OUTPUT_DIR}/v3_team_stats.json", 'w') as f:
    json.dump({k: dict(v) for k, v in team_stats.items()}, f, indent=2)

with open(f"{OUTPUT_DIR}/v3_phase_probs.json", 'w') as f:
    json.dump(team_probs, f, indent=2)

with open(f"{OUTPUT_DIR}/v3_sim_results.json", 'w') as f:
    json.dump(dict(sim_results), f, indent=2)

with open(f"{OUTPUT_DIR}/v3_signals.json", 'w') as f:
    json.dump(signals, f, indent=2)

print("\n" + "="*70)
print("TOP v3 TRADING SIGNALS")
print("="*70)

buys = [s for s in signals if s['signal'] == '📈 BUY']
sells = [s for s in signals if s['signal'] == '📉 SELL']

print("\n📈 BUY (Model > Market):")
for s in sorted(buys, key=lambda x: x['delta'], reverse=True)[:5]:
    print(f"  {s['team']:<15} +{s['delta']*100:.2f}% edge (model: {s['model']*100:.1f}% vs market: {s['market']*100:.1f}%)")

print("\n📉 SELL (Market > Model):")
for s in sorted(sells, key=lambda x: x['delta'])[:5]:
    print(f"  {s['team']:<15} {s['delta']*100:.2f}% overpriced (model: {s['model']*100:.1f}% vs market: {s['market']*100:.1f}%)")

print("\n✅ v3 Phase-Dependent Simulator Complete!")
print(f"Outputs saved to {OUTPUT_DIR}")
