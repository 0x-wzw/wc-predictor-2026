#!/usr/bin/env python3
"""
World Cup Prediction Model v3 - Tournament Simulator
Phase-dependent performance with group qualification + knockout advancement
"""

import json
import random
from collections import defaultdict
from datetime import datetime

DATA_DIR = "/tmp/wc_data/final"
OUTPUT_DIR = "/tmp/wc_model"

print("="*70)
print("WORLD CUP 2026 PREDICTION MODEL v3 - TOURNAMENT SIMULATOR")
print("="*70)
print(f"Generated: {datetime.now().isoformat()}")
print()

# Load match data
with open(f"{DATA_DIR}/world_cup_unified_dataset.json", 'r') as f:
    data = json.load(f)

matches = [m for m in data['matches'] if isinstance(m.get('home_goals'), int)]
print(f"Loaded {len(matches)} historical matches")

# Define stages
GROUP_STAGE = 'Group stage'
KO_STAGES = ['Round of 16', 'Quarter-final', 'Semi-final', 'Final', 'Third place']
ALL_KO = set(KO_STAGES)

# Calculate team statistics by phase
team_stats = defaultdict(lambda: {
    'group': {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0},
    'knockout': {'played': 0, 'wins': 0, 'draws': 0, 'losses': 0, 'goals_for': 0, 'goals_against': 0},
    'overall': {'played': 0, 'wins': 0, 'titles': 0, 'finals': 0, 'semis': 0, 'quarters': 0},
    'advancement': {'group_exits': 0, 'r16_exits': 0, 'qf_exits': 0, 'sf_exits': 0, 'runner_up': 0, 'wins': 0}
})

# Process all matches
for m in matches:
    home, away = m['home_team'], m['away_team']
    stage = m['stage']
    hg, ag = m['home_goals'], m['away_goals']
    
    # Determine result
    if hg > ag:
        home_result, away_result = 'win', 'loss'
    elif hg < ag:
        home_result, away_result = 'loss', 'win'
    else:
        home_result, away_result = 'draw', 'draw'
    
    # Group stage stats
    if stage == GROUP_STAGE:
        for team, goals_for, goals_against, result in [(home, hg, ag, home_result), (away, ag, hg, away_result)]:
            team_stats[team]['group']['played'] += 1
            team_stats[team]['group']['goals_for'] += goals_for
            team_stats[team]['group']['goals_against'] += goals_against
            if result == 'win':
                team_stats[team]['group']['wins'] += 1
            elif result == 'draw':
                team_stats[team]['group']['draws'] += 1
            else:
                team_stats[team]['group']['losses'] += 1
    
    # Knockout stats
    elif stage in ALL_KO:
        for team, goals_for, goals_against, result in [(home, hg, ag, home_result), (away, ag, hg, away_result)]:
            team_stats[team]['knockout']['played'] += 1
            team_stats[team]['knockout']['goals_for'] += goals_for
            team_stats[team]['knockout']['goals_against'] += goals_against
            if result == 'win':
                team_stats[team]['knockout']['wins'] += 1
            elif result == 'draw':
                team_stats[team]['knockout']['draws'] += 1
            else:
                team_stats[team]['knockout']['losses'] += 1
    
    # Track advancement
    team_stats[home]['overall']['played'] += 1
    team_stats[away]['overall']['played'] += 1
    
    if home_result == 'win':
        team_stats[home]['overall']['wins'] += 1
    elif away_result == 'win':
        team_stats[away]['overall']['wins'] += 1
    
    # Title/achievement tracking
    if stage == 'Final':
        if home_result == 'win':
            team_stats[home]['overall']['titles'] += 1
            team_stats[home]['overall']['finals'] += 1
            team_stats[away]['overall']['finals'] += 1
            team_stats[away]['advancement']['runner_up'] += 1
            team_stats[home]['advancement']['wins'] += 1
        else:
            team_stats[away]['overall']['titles'] += 1
            team_stats[away]['overall']['finals'] += 1
            team_stats[home]['overall']['finals'] += 1
            team_stats[home]['advancement']['runner_up'] += 1
            team_stats[away]['advancement']['wins'] += 1
    elif stage == 'Semi-final':
        team_stats[home]['overall']['semis'] += 1
        team_stats[away]['overall']['semis'] += 1
        if home_result == 'loss':
            team_stats[home]['advancement']['sf_exits'] += 1
        else:
            team_stats[away]['advancement']['sf_exits'] += 1
    elif stage == 'Quarter-final':
        team_stats[home]['overall']['quarters'] += 1
        team_stats[away]['overall']['quarters'] += 1
        if home_result == 'loss':
            team_stats[home]['advancement']['qf_exits'] += 1
        else:
            team_stats[away]['advancement']['qf_exits'] += 1
    elif stage == 'Round of 16':
        if home_result == 'loss':
            team_stats[home]['advancement']['r16_exits'] += 1
        else:
            team_stats[away]['advancement']['r16_exits'] += 1

# Calculate derived metrics
def calc_stats(s):
    """Calculate win rate, goal diff, etc from raw stats"""
    if s['played'] == 0:
        return {'win_rate': 0, 'draw_rate': 0, 'loss_rate': 0, 'goals_per_game': 0, 'goals_against_per_game': 0}
    
    return {
        'win_rate': s['wins'] / s['played'],
        'draw_rate': s['draws'] / s['played'] if 'draws' in s else 0,
        'loss_rate': s['losses'] / s['played'] if 'losses' in s else 0,
        'goals_per_game': s['goals_for'] / s['played'],
        'goals_against_per_game': s['goals_against'] / s['played'],
        'goal_diff_per_game': (s['goals_for'] - s['goals_against']) / s['played'],
        'played': s['played']
    }

# Build team metrics
team_metrics = {}
for team, stats in team_stats.items():
    gs = calc_stats(stats['group'])
    ko = calc_stats(stats['knockout'])
    ov = stats['overall']
    adv = stats['advancement']
    
    total_ko_games = sum(stats['knockout'].values()) - stats['knockout'].get('goals_for', 0) - stats['knockout'].get('goals_against', 0)
    total_tournaments = ov.get('titles', 0) + adv.get('runner_up', 0) + adv.get('sf_exits', 0) + adv.get('qf_exits', 0) + adv.get('r16_exits', 0)
    
    team_metrics[team] = {
        'team': team,
        'group_stage': gs,
        'knockout': ko,
        'overall': {
            'win_rate': ov['wins'] / ov['played'] if ov['played'] > 0 else 0,
            'titles': ov.get('titles', 0),
            'finals': ov.get('finals', 0),
            'semis': ov.get('semis', 0),
            'quarters': ov.get('quarters', 0),
            'total_matches': ov['played']
        },
        'advancement': {
            'titles': adv.get('wins', 0),
            'runner_up': adv.get('runner_up', 0),
            'semis': adv.get('sf_exits', 0),
            'quarters': adv.get('qf_exits', 0),
            'r16': adv.get('r16_exits', 0),
            'total_deep_runs': adv.get('wins', 0) + adv.get('runner_up', 0) + adv.get('sf_exits', 0)
        },
        'ko_stage_presence': total_tournaments
    }

print(f"\nProcessed stats for {len(team_metrics)} teams")

# Print phase-dependent performance
print("\n" + "="*70)
print("PHASE-DEPENDENT PERFORMANCE (Top Nations)")
print("="*70)
print(f"\n{'Team':<15} {'GS Win%':>10} {'KO Win%':>10} {'KO Boost':>10}")
print("-"*50)

phase_data = []
for team in ['Brazil', 'Germany', 'West Germany', 'Italy', 'Argentina', 'France', 'England', 'Spain', 'Netherlands', 'Uruguay']:
    if team in team_metrics:
        m = team_metrics[team]
        gs_wr = m['group_stage']['win_rate'] * 100
        ko_wr = m['knockout']['win_rate'] * 100
        boost = ko_wr - gs_wr
        phase_data.append((team, gs_wr, ko_wr, boost))

phase_data.sort(key=lambda x: x[3], reverse=True)
for team, gs, ko, boost in phase_data:
    print(f"{team:<15} {gs:>9.1f}% {ko:>9.1f}% {boost:>+9.1f}%")

print("\n" + "="*70)
print("TOURNAMENT ADVANCEMENT PROBABILITIES")
print("="*70)

# Calculate advancement probabilities for 2026 qualified teams
qualified_2026 = ["Spain", "France", "England", "Argentina", "Brazil", "Portugal", 
                   "Germany", "Netherlands", "Norway", "Japan", "Belgium", "Morocco",
                   "Colombia", "USA", "Mexico", "Switzerland", "Uruguay", "Croatia",
                   "Turkiye", "Ecuador", "Senegal"]

def calculate_advancement_prob(team):
    """Calculate probability of reaching each stage"""
    if team not in team_metrics:
        return {'group_exit': 0.95, 'r16': 0.03, 'qf': 0.01, 'sf': 0.005, 'final': 0.002, 'win': 0.001}
    
    m = team_metrics[team]
    gs = m['group_stage']
    ko = m['knockout']
    
    # Group stage qualification (top 2 of 4)
    # Based on win rate and historical group performance
    gs_win_rate = gs['win_rate']
    gs_draw_rate = gs.get('draw_rate', 0)
    
    # Expected points in group (3 games): 3*win + 1*draw
    expected_points = 3 * gs_win_rate + gs_draw_rate
    # Qualification threshold ~4-5 points typically
    group_qual_prob = min(max(gs_win_rate * 1.5 + gs_draw_rate * 0.5, 0.1), 0.95)
    
    # Knockout advancement (conditional on reaching that stage)
    ko_win_rate = ko['win_rate'] if ko['played'] > 0 else 0.4
    
    # Historical advancement rates
    total_deep = m['advancement']['total_deep_runs']
    total_tournaments = m['ko_stage_presence'] if m['ko_stage_presence'] > 0 else 1
    
    titles = m['advancement']['titles']
    finals = titles + m['advancement']['runner_up']
    semis = finals + m['advancement']['semis']
    quarters = semis + m['advancement']['quarters']
    r16s = quarters + m['advancement'].get('r16', 0)
    
    # Probabilities (conditional on reaching previous stage)
    p_r16_given_qual = min(r16s / total_tournaments, 0.95) if total_tournaments > 0 else 0.5
    p_qf_given_r16 = min(quarters / r16s, 0.95) if r16s > 0 else 0.5
    p_sf_given_qf = min(semis / quarters, 0.95) if quarters > 0 else 0.5
    p_final_given_sf = min(finals / semis, 0.95) if semis > 0 else 0.5
    p_win_given_final = titles / finals if finals > 0 else 0.5
    
    # Absolute probabilities
    p_qual = group_qual_prob
    p_r16 = p_qual * p_r16_given_qual
    p_qf = p_r16 * p_qf_given_r16
    p_sf = p_qf * p_sf_given_qf
    p_final = p_sf * p_final_given_sf
    p_win = p_final * p_win_given_final
    
    return {
        'qualify': p_qual,
        'r16': p_r16,
        'qf': p_qf,
        'sf': p_sf,
        'final': p_final,
        'win': p_win,
        'group_exit': 1 - p_qual
    }

# Calculate for all qualified teams
advancement_probs = {}
for team in qualified_2026:
    advancement_probs[team] = calculate_advancement_prob(team)

# Display
print(f"\n{'Team':<15} {'Qual%':>8} {'R16%':>8} {'QF%':>8} {'SF%':>8} {'Final%':>8} {'Win%':>8}")
print("-"*75)

sorted_teams = sorted(advancement_probs.items(), key=lambda x: x[1]['win'], reverse=True)
for team, probs in sorted_teams[:15]:
    print(f"{team:<15} {probs['qualify']*100:>7.1f}% {probs['r16']*100:>7.1f}% {probs['qf']*100:>7.1f}% {probs['sf']*100:>7.1f}% {probs['final']*100:>7.1f}% {probs['win']*100:>7.1f}%")

# Monte Carlo Simulation
def simulate_tournament(teams, n_simulations=10000):
    """Run Monte Carlo simulation of tournament"""
    results = defaultdict(lambda: {'qualify': 0, 'r16': 0, 'qf': 0, 'sf': 0, 'final': 0, 'win': 0})
    
    for _ in range(n_simulations):
        # Simulate group stage (simplified - just use qualification probability)
        qualified = []
        for team in teams:
            if random.random() < advancement_probs[team]['qualify']:
                qualified.append(team)
                results[team]['qualify'] += 1
        
        # Simulate knockout rounds
        r16 = []
        for team in qualified[:16]:  # Top 16 qualified
            if random.random() < advancement_probs[team]['r16'] / advancement_probs[team]['qualify']:
                r16.append(team)
                results[team]['r16'] += 1
        
        # Fill remaining spots if needed
        while len(r16) < 16 and len(qualified) > len(r16):
            r16.append(qualified[len(r16)])
        
        qf = []
        for team in r16[:8]:
            if team in advancement_probs:
                if random.random() < 0.5:  # Simplified - 50/50 for now
                    qf.append(team)
                    results[team]['qf'] += 1
        
        sf = []
        for team in qf[:4]:
            if random.random() < 0.5:
                sf.append(team)
                results[team]['sf'] += 1
        
        final = []
        for team in sf[:2]:
            if random.random() < 0.5:
                final.append(team)
                results[team]['final'] += 1
        
        if final:
            winner = final[0] if random.random() < advancement_probs[final[0]]['win'] / advancement_probs[final[0]]['final'] else final[-1]
            results[winner]['win'] += 1
    
    # Normalize to probabilities
    for team in results:
        for stage in results[team]:
            results[team][stage] /= n_simulations
    
    return results

print("\n" + "="*70)
print("MONTE CARLO SIMULATION (10,000 runs)")
print("="*70)

# Run simulation
sim_results = simulate_tournament(qualified_2026, n_simulations=10000)

print(f"\n{'Team':<15} {'Qual%':>8} {'R16%':>8} {'QF%':>8} {'SF%':>8} {'Final%':>8} {'Win%':>8}")
print("-"*75)

sorted_sim = sorted(sim_results.items(), key=lambda x: x[1]['win'], reverse=True)
for team, probs in sorted_sim[:15]:
    print(f"{team:<15} {probs['qualify']*100:>7.1f}% {probs['r16']*100:>7.1f}% {probs['qf']*100:>7.1f}% {probs['sf']*100:>7.1f}% {probs['final']*100:>7.1f}% {probs['win']*100:>7.1f}%")

# Compare with market odds
print("\n" + "="*70)
print("MARKET COMPARISON (v3 Model vs Polymarket)")
print("="*70)

market_odds = {
    'Spain': 0.163, 'France': 0.161, 'England': 0.111, 'Argentina': 0.089,
    'Brazil': 0.086, 'Portugal': 0.080, 'Germany': 0.055, 'Netherlands': 0.034,
    'Norway': 0.024, 'Japan': 0.022, 'Belgium': 0.022, 'Morocco': 0.019,
    'Colombia': 0.017, 'USA': 0.013, 'Mexico': 0.011, 'Switzerland': 0.010,
    'Uruguay': 0.010, 'Croatia': 0.009, 'Turkiye': 0.007, 'Ecuador': 0.007,
    'Senegal': 0.007
}

print(f"\n{'Team':<15} {'Market':>10} {'v3 Model':>10} {'Delta':>10} {'Signal'}")
print("-"*65)

signals = []
for team in market_odds:
    market = market_odds[team]
    model = sim_results.get(team, {}).get('win', 0)
    delta = model - market
    
    if delta > 0.02:
        sig = "📈 BUY"
    elif delta < -0.02:
        sig = "📉 SELL"
    else:
        sig = "➡️ FAIR"
    
    signals.append({'team': team, 'market': market, 'model': model, 'delta': delta, 'signal': sig})
    print(f"{team:<15} {market*100:>9.2f}% {model*100:>9.2f}% {delta*100:>+9.2f}% {sig}")

# Save outputs
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(f"{OUTPUT_DIR}/v3_team_metrics.json", 'w') as f:
    json.dump(team_metrics, f, indent=2)

with open(f"{OUTPUT_DIR}/v3_advancement_probs.json", 'w') as f:
    json.dump(advancement_probs, f, indent=2)

with open(f"{OUTPUT_DIR}/v3_simulation_results.json", 'w') as f:
    json.dump(dict(sim_results), f, indent=2)

with open(f"{OUTPUT_DIR}/v3_market_comparison.json", 'w') as f:
    json.dump(signals, f, indent=2)

print("\n" + "="*70)
print("OUTPUTS SAVED")
print("="*70)
print("Files saved to /tmp/wc_model/:")
print("  - v3_team_metrics.json")
print("  - v3_advancement_probs.json")
print("  - v3_simulation_results.json")
print("  - v3_market_comparison.json")

print("\n" + "="*70)
print("TOP v3 TRADING SIGNALS")
print("="*70)

buys = [s for s in signals if s['signal'] == '📈 BUY']
sells = [s for s in signals if s['signal'] == '📉 SELL']

print("\n📈 BUY (Model > Market):")
for s in sorted(buys, key=lambda x: x['delta'], reverse=True)[:5]:
    print(f"  {s['team']:<15} +{s['delta']*100:.2f}% edge")

print("\n📉 SELL (Market > Model):")
for s in sorted(sells, key=lambda x: abs(x['delta']), reverse=True)[:5]:
    print(f"  {s['team']:<15} {s['delta']*100:.2f}% overpriced")

print("\n✅ v3 Tournament Simulator Complete!")
