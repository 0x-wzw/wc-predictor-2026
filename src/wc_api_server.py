#!/usr/bin/env python3
"""
WC 2026 Real-Time Prediction API
FastAPI-style endpoint for live match predictions
"""

import json
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from datetime import datetime
import threading

# Add wc_model to path
sys.path.insert(0, '/tmp/wc_model')

try:
    from matchup_predictor import WCMatchupPredictor
except ImportError:
    print("Loading predictor module...")
    # Fallback import
    import importlib.util
    spec = importlib.util.spec_from_file_location("matchup_predictor", "/tmp/wc_model/matchup_predictor.py")
    matchup_module = importlib.util.module_from_spec(spec)
    sys.modules["matchup_predictor"] = matchup_module
    spec.loader.exec_module(matchup_module)
    WCMatchupPredictor = matchup_module.WCMatchupPredictor

# Initialize predictor globally
print("Initializing WC Matchup Predictor...")
PREDICTOR = WCMatchupPredictor()
print(f"Loaded {len(PREDICTOR.team_profiles)} team profiles")

class WCPredictionHandler(BaseHTTPRequestHandler):
    """HTTP Request handler for WC predictions"""
    
    def log_message(self, format, *args):
        """Custom logging"""
        print(f"[{datetime.now().isoformat()}] {args[0]}")
    
    def _set_headers(self, status=200, content_type='application/json'):
        """Set response headers"""
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
    
    def _send_json(self, data, status=200):
        """Send JSON response"""
        self._set_headers(status)
        self.wfile.write(json.dumps(data, indent=2).encode())
    
    def _send_error(self, message, status=400):
        """Send error response"""
        self._send_json({'error': message, 'status': 'error'}, status)
    
    def do_OPTIONS(self):
        """Handle CORS preflight"""
        self._set_headers()
    
    def do_GET(self):
        """Handle GET requests"""
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)
        
        # Health check
        if path == '/health':
            self._send_json({
                'status': 'healthy',
                'service': 'wc2026-prediction-api',
                'timestamp': datetime.now().isoformat(),
                'teams_loaded': len(PREDICTOR.team_profiles),
                'version': '1.0.0'
            })
            return
        
        # Predict match endpoint
        if path == '/predict':
            self._handle_predict(params)
            return
        
        # Get team info
        if path == '/team':
            self._handle_team_info(params)
            return
        
        # List all teams
        if path == '/teams':
            self._handle_list_teams()
            return
        
        # API documentation
        if path == '/' or path == '/docs':
            self._handle_docs()
            return
        
        self._send_error('Endpoint not found. Use /docs for API documentation.', 404)
    
    def do_POST(self):
        """Handle POST requests"""
        parsed = urlparse(self.path)
        path = parsed.path
        
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            body = self.rfile.read(content_length)
            try:
                data = json.loads(body.decode())
            except json.JSONDecodeError:
                self._send_error('Invalid JSON body')
                return
        else:
            data = {}
        
        if path == '/predict':
            self._handle_predict_post(data)
            return
        
        if path == '/simulate/group':
            self._handle_simulate_group(data)
            return
        
        if path == '/compare':
            self._handle_compare(data)
            return
        
        self._send_error('Endpoint not found', 404)
    
    def _handle_predict(self, params):
        """Handle GET /predict"""
        home = params.get('home', [''])[0]
        away = params.get('away', [''])[0]
        stage = params.get('stage', ['Group stage'])[0]
        form_home = float(params.get('form_home', [0.5])[0])
        form_away = float(params.get('form_away', [0.5])[0])
        
        if not home or not away:
            self._send_error('Required parameters: home, away')
            return
        
        try:
            prediction = PREDICTOR.predict_match(home, away, stage, form_home, form_away)
            self._send_json({
                'status': 'success',
                'prediction': {
                    'home_team': prediction.home_team,
                    'away_team': prediction.away_team,
                    'stage': prediction.stage,
                    'probabilities': {
                        'home_win': round(prediction.home_win_prob * 100, 2),
                        'draw': round(prediction.draw_prob * 100, 2),
                        'away_win': round(prediction.away_win_prob * 100, 2)
                    },
                    'expected_goals': {
                        'home': prediction.expected_home_goals,
                        'away': prediction.expected_away_goals
                    },
                    'confidence': round(prediction.confidence * 100, 1),
                    'key_factors': prediction.key_factors
                },
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self._send_error(f'Prediction error: {str(e)}', 500)
    
    def _handle_predict_post(self, data):
        """Handle POST /predict"""
        home = data.get('home')
        away = data.get('away')
        stage = data.get('stage', 'Group stage')
        form_home = data.get('form_home', 0.5)
        form_away = data.get('form_away', 0.5)
        must_win_home = data.get('must_win_home', False)
        must_win_away = data.get('must_win_away', False)
        
        if not home or not away:
            self._send_error('Required fields: home, away')
            return
        
        try:
            prediction = PREDICTOR.predict_match(
                home, away, stage, form_home, form_away,
                must_win_home, must_win_away
            )
            self._send_json({
                'status': 'success',
                'prediction': {
                    'home_team': prediction.home_team,
                    'away_team': prediction.away_team,
                    'stage': prediction.stage,
                    'probabilities': {
                        'home_win': round(prediction.home_win_prob * 100, 2),
                        'draw': round(prediction.draw_prob * 100, 2),
                        'away_win': round(prediction.away_win_prob * 100, 2)
                    },
                    'expected_goals': {
                        'home': prediction.expected_home_goals,
                        'away': prediction.expected_away_goals
                    },
                    'confidence': round(prediction.confidence * 100, 1),
                    'analysis': {
                        'home_style': PREDICTOR.team_profiles.get(home, {}).style if hasattr(PREDICTOR.team_profiles.get(home), 'style') else 'unknown',
                        'away_style': PREDICTOR.team_profiles.get(away, {}).style if hasattr(PREDICTOR.team_profiles.get(away), 'style') else 'unknown',
                        'h2h_advantage': prediction.key_factors.get('h2h_advantage', 0),
                        'experience_diff': prediction.key_factors.get('experience_diff', 0)
                    }
                },
                'timestamp': datetime.now().isoformat()
            })
        except Exception as e:
            self._send_error(f'Prediction error: {str(e)}', 500)
    
    def _handle_team_info(self, params):
        """Handle GET /team"""
        team = params.get('name', [''])[0]
        if not team:
            self._send_error('Required parameter: name')
            return
        
        profile = PREDICTOR.team_profiles.get(team)
        if not profile:
            self._send_error(f'Team not found: {team}', 404)
            return
        
        self._send_json({
            'status': 'success',
            'team': {
                'name': profile.name,
                'group_stage_win_rate': round(profile.group_win_rate * 100, 1),
                'knockout_win_rate': round(profile.ko_win_rate * 100, 1),
                'goals_per_game': profile.goals_for_pg,
                'goals_conceded_per_game': profile.goals_against_pg,
                'world_cup_titles': profile.titles,
                'world_cup_finals': profile.finals,
                'total_wc_matches': profile.total_wc_matches,
                'style': profile.style
            }
        })
    
    def _handle_list_teams(self):
        """Handle GET /teams"""
        teams = []
        for name, profile in PREDICTOR.team_profiles.items():
            teams.append({
                'name': name,
                'titles': profile.titles,
                'matches': profile.total_wc_matches,
                'group_win_rate': round(profile.group_win_rate * 100, 1)
            })
        
        teams.sort(key=lambda x: (x['titles'], x['group_win_rate']), reverse=True)
        
        self._send_json({
            'status': 'success',
            'count': len(teams),
            'teams': teams
        })
    
    def _handle_simulate_group(self, data):
        """Handle POST /simulate/group"""
        group = data.get('teams', [])
        if len(group) != 4:
            self._send_error('Exactly 4 teams required for group simulation')
            return
        
        try:
            result = PREDICTOR.simulate_group_stage(group)
            self._send_json({
                'status': 'success',
                'group': group,
                'standings': [
                    {
                        'position': i + 1,
                        'team': team,
                        'points': result['results'][team]['points'],
                        'record': f"{result['results'][team]['wins']}-{result['results'][team]['draws']}-{result['results'][team]['losses']}",
                        'goal_difference': result['results'][team]['gf'] - result['results'][team]['ga']
                    }
                    for i, team in enumerate(result['standings'])
                ],
                'qualifiers': result['qualifiers']
            })
        except Exception as e:
            self._send_error(f'Simulation error: {str(e)}', 500)
    
    def _handle_compare(self, data):
        """Handle POST /compare - compare multiple matchups"""
        matchups = data.get('matchups', [])
        if not matchups:
            self._send_error('matchups array required')
            return
        
        results = []
        for matchup in matchups:
            try:
                pred = PREDICTOR.predict_match(
                    matchup.get('home'),
                    matchup.get('away'),
                    matchup.get('stage', 'Group stage'),
                    matchup.get('form_home', 0.5),
                    matchup.get('form_away', 0.5)
                )
                results.append({
                    'matchup': f"{pred.home_team} vs {pred.away_team}",
                    'probabilities': {
                        'home_win': round(pred.home_win_prob * 100, 1),
                        'draw': round(pred.draw_prob * 100, 1),
                        'away_win': round(pred.away_win_prob * 100, 1)
                    }
                })
            except Exception as e:
                results.append({
                    'matchup': f"{matchup.get('home')} vs {matchup.get('away')}",
                    'error': str(e)
                })
        
        self._send_json({
            'status': 'success',
            'comparisons': results
        })
    
    def _handle_docs(self):
        """Show API documentation"""
        docs = {
            'service': 'WC 2026 Prediction API',
            'version': '1.0.0',
            'description': 'Real-time match predictions for FIFA World Cup 2026',
            'endpoints': {
                'GET /health': {
                    'description': 'Health check',
                    'response': 'Service status and metrics'
                },
                'GET /predict': {
                    'description': 'Predict single match',
                    'parameters': {
                        'home': 'Home team name (required)',
                        'away': 'Away team name (required)',
                        'stage': 'Group stage|Round of 16|Quarter-final|Semi-final|Final',
                        'form_home': 'Current form 0-1 (default: 0.5)',
                        'form_away': 'Current form 0-1 (default: 0.5)'
                    }
                },
                'POST /predict': {
                    'description': 'Predict single match with body',
                    'body': {
                        'home': 'Team name',
                        'away': 'Team name',
                        'stage': 'Tournament stage',
                        'form_home': 'Float 0-1',
                        'form_away': 'Float 0-1',
                        'must_win_home': 'Boolean',
                        'must_win_away': 'Boolean'
                    }
                },
                'GET /teams': {
                    'description': 'List all available teams'
                },
                'GET /team': {
                    'description': 'Get team info',
                    'parameters': {'name': 'Team name'}
                },
                'POST /simulate/group': {
                    'description': 'Simulate group stage',
                    'body': {'teams': ['Team1', 'Team2', 'Team3', 'Team4']}
                },
                'POST /compare': {
                    'description': 'Compare multiple matchups',
                    'body': {'matchups': [{'home': 'A', 'away': 'B', 'stage': '...'}, ...]}
                }
            },
            'examples': {
                'curl_predict': 'curl "http://localhost:8080/predict?home=France&away=Brazil&stage=Quarter-final"',
                'curl_teams': 'curl http://localhost:8080/teams',
                'curl_health': 'curl http://localhost:8080/health'
            }
        }
        self._send_json(docs)


def run_server(port=8080):
    """Run the prediction API server"""
    server_address = ('', port)
    httpd = HTTPServer(server_address, WCPredictionHandler)
    print(f"\n{'='*70}")
    print(f"WC 2026 Prediction API Server")
    print(f"{'='*70}")
    print(f"Server running on http://0.0.0.0:{port}")
    print(f"Teams loaded: {len(PREDICTOR.team_profiles)}")
    print(f"Endpoints:")
    print(f"  - GET  http://localhost:{port}/health")
    print(f"  - GET  http://localhost:{port}/predict?home=France&away=Brazil")
    print(f"  - GET  http://localhost:{port}/teams")
    print(f"  - POST http://localhost:{port}/predict")
    print(f"  - POST http://localhost:{port}/simulate/group")
    print(f"{'='*70}\n")
    print("Press Ctrl+C to stop\n")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        httpd.server_close()
        print("Server stopped.")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='WC 2026 Prediction API')
    parser.add_argument('--port', type=int, default=8080, help='Server port (default: 8080)')
    args = parser.parse_args()
    
    run_server(args.port)
ENDSCRIPT