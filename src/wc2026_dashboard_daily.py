#!/usr/bin/env python3
"""WC2026 Dashboard Updater - Daily scrape + deploy to Cloudflare Pages"""

import json
import subprocess
from datetime import datetime
from pathlib import Path

DATA_DIR = Path('/home/ubuntu/.hermes/data/wc2026_tracker')
REPO_DIR = Path('/home/ubuntu/intelligence-teoh')

def run(cmd, cwd=None):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=cwd)
    return r.stdout + r.stderr, r.returncode

def gen_html(tracker):
    groups = tracker['groups']
    knockout = tracker['knockout']
    meta = tracker['meta']
    
    status_class = 'status-prep' if meta['status'] == 'pre-tournament' else 'status-live'
    status_label = meta['status'].replace('-', ' ').title()
    
    html = '<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">'
    html += '<meta name="viewport" content="width=device-width, initial-scale=1.0">'
    html += '<title>WC 2026 Tracker | Full Tournament Metrics</title>'
    html += '<style>'
    html += ':root { --bg: #0d1117; --surface: #161b22; --surface-2: #21262d; --surface-3: #30363d; --text: #c9d1d9; --text-muted: #8b949e; --text-dim: #6e7681; --accent: #58a6ff; --accent-2: #3fb950; --accent-3: #f85149; --accent-4: #d29922; --border: #30363d; }'
    html += '* { margin: 0; padding: 0; box-sizing: border-box; }'
    html += 'body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; background: var(--bg); color: var(--text); line-height: 1.6; max-width: 1400px; margin: 0 auto; padding: 2rem; }'
    html += 'header { margin-bottom: 2rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem; }'
    html += '.brand { display: flex; align-items: center; gap: 0.5rem; font-size: 1.25rem; font-weight: 600; text-decoration: none; color: var(--text); }'
    html += '.brand span { color: var(--accent); font-weight: 700; }'
    html += 'h1 { font-size: 1.5rem; margin: 1rem 0; }'
    html += 'h2 { font-size: 1.1rem; margin: 2rem 0 1rem; color: var(--accent); font-weight: 500; text-transform: uppercase; letter-spacing: 0.05em; }'
    html += 'h3 { font-size: 0.95rem; margin: 1.5rem 0 0.5rem; color: var(--text-muted); font-weight: 500; }'
    html += '.meta { color: var(--text-muted); font-size: 0.85rem; margin-bottom: 1rem; }'
    html += '.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 1rem; margin: 1.5rem 0; }'
    html += '.stat-box { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 1rem; text-align: center; }'
    html += '.stat-value { font-size: 1.75rem; font-weight: 600; color: var(--accent); }'
    html += '.stat-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; }'
    html += '.stat-sub { font-size: 0.75rem; color: var(--text-dim); margin-top: 0.25rem; }'
    html += '.groups-container { display: grid; grid-template-columns: repeat(auto-fit, minmax(380px, 1fr)); gap: 1.5rem; margin: 1rem 0; }'
    html += '.group-card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; overflow: hidden; }'
    html += '.group-header { background: var(--surface-2); padding: 0.75rem 1rem; font-weight: 600; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }'
    html += '.group-header span { color: var(--accent); }'
    html += 'table.standings { width: 100%; font-size: 0.85rem; border-collapse: collapse; }'
    html += 'table.standings th { background: var(--surface-2); color: var(--text-muted); font-weight: 500; text-transform: uppercase; font-size: 0.65rem; letter-spacing: 0.05em; padding: 0.5rem; text-align: center; }'
    html += 'table.standings th:first-child { text-align: left; padding-left: 1rem; }'
    html += 'table.standings td { padding: 0.5rem; text-align: center; border-bottom: 1px solid var(--border); }'
    html += 'table.standings td:first-child { text-align: left; padding-left: 1rem; font-weight: 500; }'
    html += 'table.standings tr:nth-child(1) td:first-child, table.standings tr:nth-child(2) td:first-child { color: var(--accent-2); }'
    html += 'table.standings tr:nth-child(3) td:first-child, table.standings tr:nth-child(4) td:first-child { color: var(--accent-3); }'
    html += 'table.standings tr:hover { background: var(--surface-2); }'
    html += '.points { font-weight: 700; color: var(--accent); }'
    html += '.form-cell { font-size: 0.75rem; color: var(--text-dim); }'
    html += '.form-w { color: var(--accent-2); font-weight: 600; }'
    html += '.form-d { color: var(--accent-4); font-weight: 600; }'
    html += '.form-l { color: var(--accent-3); font-weight: 600; }'
    html += '.fixtures-list { padding: 0.75rem 1rem; font-size: 0.8rem; border-top: 1px solid var(--border); }'
    html += '.fixture-item { display: flex; justify-content: space-between; padding: 0.35rem 0; border-bottom: 1px dashed var(--surface-3); }'
    html += '.fixture-item:last-child { border-bottom: none; }'
    html += '.fixture-matchday { color: var(--text-dim); font-size: 0.7rem; min-width: 25px; }'
    html += '.fixture-teams { flex: 1; text-align: center; }'
    html += '.fixture-vs { color: var(--text-dim); margin: 0 0.5rem; }'
    html += '.knockout-section { margin: 2rem 0; }'
    html += '.knockout-round { margin: 1rem 0; }'
    html += '.knockout-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1rem; }'
    html += '.knockout-match { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 0.75rem 1rem; display: flex; justify-content: space-between; align-items: center; }'
    html += '.knockout-match .vs { color: var(--text-dim); font-size: 0.8rem; }'
    html += '.knockout-match .placeholder { color: var(--text-dim); font-style: italic; }'
    html += '.metrics-legend { display: flex; gap: 2rem; flex-wrap: wrap; margin: 1rem 0; padding: 1rem; background: var(--surface); border: 1px solid var(--border); border-radius: 6px; font-size: 0.8rem; }'
    html += '.metric-def { display: flex; gap: 0.5rem; align-items: center; }'
    html += '.metric-def code { background: var(--surface-2); padding: 0.15rem 0.4rem; border-radius: 3px; font-family: monospace; color: var(--accent); }'
    html += 'footer { margin-top: 3rem; padding-top: 1rem; border-top: 1px solid var(--border); font-size: 0.8rem; color: var(--text-muted); }'
    html += '.status-badge { display: inline-block; padding: 0.15rem 0.5rem; border-radius: 12px; font-size: 0.7rem; font-weight: 500; text-transform: uppercase; }'
    html += '.status-prep { background: var(--accent-4); color: #000; }'
    html += '.status-live { background: var(--accent-2); color: #000; }'
    html += '.status-complete { background: var(--accent); color: #000; }'
    html += '</style></head><body>'
    
    html += f'<header><a href="/" class="brand"><span>⚽</span> WC 2026 Tracker</a>'
    html += f'<h1>FIFA World Cup 2026 — Full Tournament Metrics</h1>'
    html += f'<div class="meta"><span class="status-badge {status_class}">{status_label}</span> • '
    html += f'Kickoff: June 11, 2026 • Hosts: Canada, Mexico, USA • {meta["days_to_kickoff"]} Days Remaining</div></header>'
    
    html += '<div class="stats-grid">'
    html += f'<div class="stat-box"><div class="stat-value">{meta["groups"]}</div><div class="stat-label">Groups</div><div class="stat-sub">(12 planned full)</div></div>'
    html += f'<div class="stat-box"><div class="stat-value">{meta["teams"]}</div><div class="stat-label">Teams</div><div class="stat-sub">(48 planned full)</div></div>'
    html += '<div class="stat-box"><div class="stat-value">48</div><div class="stat-label">Group Matches</div></div>'
    html += '<div class="stat-box"><div class="stat-value">16</div><div class="stat-label">Knockout</div><div class="stat-sub">R16 to Final</div></div>'
    html += '<div class="stat-box"><div class="stat-value">64</div><div class="stat-label">Total Matches</div><div class="stat-sub">104 for 48 teams</div></div>'
    html += f'<div class="stat-box"><div class="stat-value">0</div><div class="stat-label">Goals Scored</div></div>'
    html += '</div>'
    
    html += '<div class="metrics-legend">'
    html += '<div class="metric-def"><code>P</code> Matches Played</div>'
    html += '<div class="metric-def"><code>W</code> Wins</div>'
    html += '<div class="metric-def"><code>D</code> Draws</div>'
    html += '<div class="metric-def"><code>L</code> Losses</div>'
    html += '<div class="metric-def"><code>GF</code> Goals For</div>'
    html += '<div class="metric-def"><code>GA</code> Goals Against</div>'
    html += '<div class="metric-def"><code>GD</code> Goal Difference</div>'
    html += '<div class="metric-def"><code>Pts</code> Points</div>'
    html += '<div class="metric-def"><code>Form</code> Last 5 (W/D/L)</div>'
    html += '</div>'
    
    html += '<h2>▸ Group Stage Standings</h2><div class="groups-container">'
    
    for group_name, group_data in groups.items():
        md_completed = max(t['played'] for t in group_data['teams']) if any(t['played'] > 0 for t in group_data['teams']) else 0
        html += f'<div class="group-card"><div class="group-header"><span>Group {group_name}</span><small style="color: var(--text-dim);">Matchday {md_completed}/3</small></div>'
        html += '<table class="standings"><thead><tr><th>Team</th><th>P</th><th>W</th><th>D</th><th>L</th><th>GF</th><th>GA</th><th>GD</th><th>Pts</th><th>Form</th></tr></thead><tbody>'
        
        sorted_teams = sorted(group_data['teams'], key=lambda t: (t['points'], t['goals_for'] - t['goals_against'], t['goals_for']), reverse=True)
        
        for team in sorted_teams:
            parts = []
            for f in team.get('form', []):
                cls = f'form-{f.lower()}'
                parts.append(f'<span class="{cls}">{f}</span>')
            form = ' '.join(parts) if parts else '-'
            gd = team['goals_for'] - team['goals_against']
            html += f'<tr><td>{team["team"]}</td><td>{team["played"]}</td><td>{team["won"]}</td><td>{team["drawn"]}</td><td>{team["lost"]}</td><td>{team["goals_for"]}</td><td>{team["goals_against"]}</td><td>{gd:+d}</td><td class="points">{team["points"]}</td><td class="form-cell">{form}</td></tr>'
        
        html += '</tbody></table><div class="fixtures-list">'
        for fixture in group_data.get('fixtures', []):
            html += f'<div class="fixture-item"><span class="fixture-matchday">MD{fixture["matchday"]}</span><span class="fixture-teams">{fixture["home"]} <span class="fixture-vs">vs</span> {fixture["away"]}</span><span style="color: var(--text-dim); font-size: 0.7rem;">TBD</span></div>'
        html += '</div></div>'
    
    html += '</div><div class="knockout-section"><h2>▸ Knockout Bracket</h2>'
    
    for round_name, round_matches in knockout.items():
        round_label = round_name.replace('_', ' ').title()
        if round_name == 'third_place':
            round_label = '3rd Place Playoff'
        html += f'<div class="knockout-round"><h3>{round_label}</h3><div class="knockout-grid">'
        for match in round_matches:
            if round_name == 'final':
                html += f'<div class="knockout-match" style="border-color: var(--accent-4);"><span style="font-weight:600;">🏆 {round_label}</span><span class="vs">{match["home"]} vs {match["away"]}</span></div>'
            else:
                html += f'<div class="knockout-match"><span class="placeholder">{match["home"]}</span><span class="vs">vs</span><span class="placeholder">{match["away"]}</span></div>'
        html += '</div></div>'
    
    html += f'</div><footer>Generated by Hermes Agent • Data: FIFA Official • Updates: Daily • Count: {meta["update_count"]}</footer></body></html>'
    return html

def main():
    print(f"=== WC2026 Dashboard Update {datetime.now().isoformat()} ===")
    
    tracker_path = DATA_DIR / 'wc2026_tracker.json'
    with open(tracker_path) as f:
        tracker = json.load(f)
    
    tracker['meta']['last_updated'] = datetime.now().isoformat()
    tracker['meta']['update_count'] += 1
    tracker['meta']['days_to_kickoff'] = max(0, (datetime(2026, 6, 11) - datetime.now()).days)
    
    now = datetime.now()
    if now < datetime(2026, 6, 11):
        tracker['meta']['status'] = 'pre-tournament'
    elif now < datetime(2026, 7, 19):
        tracker['meta']['status'] = 'in-progress'
    else:
        tracker['meta']['status'] = 'completed'
    
    with open(tracker_path, 'w') as f:
        json.dump(tracker, f, indent=2)
    
    html = gen_html(tracker)
    output_path = REPO_DIR / 'wc2026-tracker-full.html'
    with open(output_path, 'w') as f:
        f.write(html)
    
    print(f"Dashboard written: {output_path}")
    
    out, code = run('git add -A && git commit -m "WC2026 daily update"', cwd=REPO_DIR)
    print(out)
    
    if code == 0:
        out2, code2 = run('git push origin master', cwd=REPO_DIR)
        print(out2)
        if code2 == 0:
            print("✅ Deployed to Cloudflare Pages")
        else:
            print(f"Push exit: {code2}")
    else:
        print("Nothing to commit")
    
    print(f"\n📊 https://intelligence.teoh.my/wc2026-tracker-full.html")

if __name__ == "__main__":
    main()
