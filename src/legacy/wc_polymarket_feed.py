#!/usr/bin/env python3
"""
WC 2026 Polymarket Real-Time Integration
Fetches live odds from Polymarket Gamma API and updates predictions
"""

import json
import time
import asyncio
import httpx
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict
import threading
import queue

# Polymarket Gamma API endpoints
GAMMA_API_URL = "https://gamma-api.polymarket.com"
CLOB_API_URL = "https://clob.polymarket.com"

WC2026_EVENT_ID = "2026-fifa-world-cup-winner-595"  # Polymarket event slug

@dataclass
class MarketData:
    """Polymarket market data structure"""
    market_id: str
    question: str
    outcome_name: str
    probability: float  # 0-1
    volume: float
    liquidity: float
    last_updated: str
    spread: float  # bid-ask spread
    odds_history: List[Dict] = None
    
    def to_dict(self):
        return asdict(self)


class PolymarketClient:
    """Polymarket API client for fetching live odds"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.cache = {}
        self.cache_ttl = 30  # 30 second cache
        self.last_fetch = {}
        self.session = None
        
    async def _init_session(self):
        """Initialize HTTP session"""
        if self.session is None:
            self.session = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    'User-Agent': 'WC2026-Monitor/1.0',
                    'Accept': 'application/json'
                }
            )
    
    async def fetch_wc2026_markets(self) -> List[MarketData]:
        """Fetch all WC 2026 winner markets (separate market per team)"""
        await self._init_session()
        
        cache_key = "wc2026_markets"
        if self._is_cached(cache_key):
            return self.cache[cache_key]
        
        try:
            # Fetch markets from Gamma API
            url = f"{GAMMA_API_URL}/markets"
            response = await self.session.get(url, params={
                "active": "true",
                "limit": 100,
                "archived": "false"
            })
            
            if response.status_code != 200:
                print(f"API returned status {response.status_code}")
                return []
                
            data = response.json()
            markets_raw = data if isinstance(data, list) else data.get('markets', [])
            
            print(f"Fetched {len(markets_raw)} total markets")
            
            # Filter WC 2026 win markets
            wc_markets = []
            for m in markets_raw:
                # Ensure m is a dict
                if not isinstance(m, dict):
                    continue
                    
                question = m.get('question', '').lower()
                if 'win the 2026 fifa world cup' in question:
                    # Extract team name from question
                    team = self._extract_team_name(question)
                    if team:
                        outcomes = m.get('outcomes', [])
                        # Parse outcomes if it's a JSON string
                        if isinstance(outcomes, str):
                            import json
                            try:
                                outcomes = json.loads(outcomes)
                            except:
                                outcomes = ["Yes", "No"]  # Default
                        
                        # Get "Yes" probability from outcomePrices (if available)
                        yes_prob = 0
                        outcome_prices = m.get('outcomePrices', [])
                        if isinstance(outcome_prices, str):
                            import json
                            try:
                                outcome_prices = json.loads(outcome_prices)
                            except:
                                outcome_prices = []
                        
                        if outcome_prices and len(outcome_prices) >= 2:
                            # First price is "Yes"
                            yes_prob = float(outcome_prices[0]) if isinstance(outcome_prices[0], (int, float, str)) else 0
                            if isinstance(yes_prob, str):
                                yes_prob = float(yes_prob)
                        
                        # Fallback to other fields
                        if not yes_prob:
                            yes_prob = m.get('probability', 0) or m.get('midPrice', 0) or m.get('lastTradePrice', 0)
                        
                        market_data = MarketData(
                            market_id=m.get('id', ''),
                            question=m.get('question', ''),
                            outcome_name=team,
                            probability=yes_prob,
                            volume=m.get('volume', 0),
                            liquidity=m.get('liquidity', 0),
                            last_updated=datetime.now().isoformat(),
                            spread=0.02,
                            odds_history=[]
                        )
                        wc_markets.append(market_data)
            
            self._set_cache(cache_key, wc_markets)
            return wc_markets
            
        except Exception as e:
            print(f"Error fetching WC markets: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def _extract_team_name(self, question: str) -> str:
        """Extract team name from Polymarket question"""
        # Regex pattern: "Will X win the..."
        import re
        match = re.search(r'will ([^.]+?) win the 2026', question.lower())
        if match:
            return match.group(1).strip().title()
        return None

    async def fetch_event(self, slug: str) -> Optional[Dict]:
        """Fetch event data from Gamma API"""
        # WC 2026 not in events, markets are standalone
        return None
    
    async def fetch_markets(self, event_slug: str) -> List[MarketData]:
        """Fetch all markets for an event"""
        event = await self.fetch_event(event_slug)
        if not event:
            return []
        
        markets = []
        for market in event.get('markets', []):
            for outcome in market.get('outcomes', []):
                market_data = MarketData(
                    market_id=market.get('id', ''),
                    question=market.get('question', ''),
                    outcome_name=outcome.get('name', ''),
                    probability=outcome.get('probability', 0),
                    volume=market.get('volume', 0),
                    liquidity=market.get('liquidity', 0),
                    last_updated=datetime.now().isoformat(),
                    spread=outcome.get('spread', 0.02),
                    odds_history=[]
                )
                markets.append(market_data)
        
        return markets
    
    async def fetch_order_book(self, market_id: str) -> Optional[Dict]:
        """Fetch order book from CLOB API"""
        await self._init_session()
        
        url = f"{CLOB_API_URL}/book"
        try:
            response = await self.session.get(url, params={"market": market_id})
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Error fetching order book: {e}")
        
        return None
    
    def _is_cached(self, key: str) -> bool:
        """Check if cache entry is valid"""
        if key not in self.last_fetch:
            return False
        return (time.time() - self.last_fetch[key]) < self.cache_ttl
    
    def _set_cache(self, key: str, data: any):
        """Set cache entry"""
        self.cache[key] = data
        self.last_fetch[key] = time.time()
    
    async def close(self):
        """Close HTTP session"""
        if self.session:
            await self.session.aclose()


class WC2026RealTimeFeed:
    """
    Real-time feed for WC 2026 odds from Polymarket
    """
    
    def __init__(self):
        self.client = PolymarketClient()
        self.current_odds: Dict[str, MarketData] = {}
        self.odds_history: Dict[str, List[Dict]] = {}
        self.update_callbacks = []
        self.running = False
        self.update_interval = 60  # Update every 60 seconds
        self._thread = None
        
    def register_callback(self, callback):
        """Register callback for odds updates"""
        self.update_callbacks.append(callback)
        
    def _notify_callbacks(self, team: str, old_odds: float, new_odds: float):
        """Notify registered callbacks"""
        for callback in self.update_callbacks:
            try:
                callback(team, old_odds, new_odds)
            except Exception as e:
                print(f"Callback error: {e}")
    
    async def fetch_current_odds(self) -> Dict[str, MarketData]:
        """Fetch current WC 2026 winner odds"""
        print(f"[{datetime.now().isoformat()}] Fetching WC 2026 odds...")
        
        markets = await self.client.fetch_wc2026_markets()
        
        odds = {}
        for market in markets:
            team_name = self._normalize_team_name(market.outcome_name)
            if team_name:
                # Store in history
                if team_name not in self.odds_history:
                    self.odds_history[team_name] = []
                
                entry = {
                    'timestamp': market.last_updated,
                    'probability': market.probability,
                    'odds': self._probability_to_odds(market.probability),
                    'volume': market.volume
                }
                self.odds_history[team_name].append(entry)
                
                # Keep last 100 entries
                if len(self.odds_history[team_name]) > 100:
                    self.odds_history[team_name] = self.odds_history[team_name][-100:]
                
                # Check for significant change
                old_odds = self.current_odds.get(team_name, MarketData("", "", "", 0, 0, 0, "", 0)).probability
                if abs(old_odds - market.probability) > 0.01:  # 1% change
                    self._notify_callbacks(team_name, old_odds, market.probability)
                
                odds[team_name] = market
                
        self.current_odds = odds
        print(f"Fetched odds for {len(odds)} teams")
        return odds
    
    def _normalize_team_name(self, name: str) -> Optional[str]:
        """Normalize team name from Polymarket format"""
        name_map = {
            'USA': 'USA',
            'United States': 'USA',
            'Turkey': 'Turkiye',
            'Türkiye': 'Turkiye',
            'England': 'England',
            'Great Britain': 'England',  # Sometimes incorrectly labeled
        }
        
        normalized = name.strip()
        return name_map.get(normalized, normalized)
    
    def _probability_to_odds(self, prob: float) -> float:
        """Convert probability to decimal odds"""
        if prob <= 0:
            return 999.0
        return 1.0 / prob
    
    def get_odds_movement(self, team: str, hours: int = 24) -> Dict:
        """Get odds movement over time"""
        history = self.odds_history.get(team, [])
        if not history:
            return {}
        
        cutoff = datetime.now() - timedelta(hours=hours)
        recent = [h for h in history if datetime.fromisoformat(h['timestamp']) > cutoff]
        
        if len(recent) < 2:
            return {'movement': 0, 'trend': 'flat'}
        
        first = recent[0]['probability']
        last = recent[-1]['probability']
        movement = last - first
        
        # Calculate trend
        if movement > 0.02:
            trend = 'up'
        elif movement < -0.02:
            trend = 'down'
        else:
            trend = 'flat'
        
        return {
            'movement': movement,
            'trend': trend,
            'start': first,
            'current': last,
            'highest': max(h['probability'] for h in recent),
            'lowest': min(h['probability'] for h in recent),
            'data_points': len(recent)
        }
    
    def get_arbitrage_opportunities(self, model_probs: Dict[str, float]) -> List[Dict]:
        """
        Find arbitrage opportunities between model and market
        """
        opportunities = []
        
        for team, market_data in self.current_odds.items():
            market_prob = market_data.probability
            model_prob = model_probs.get(team, 0)
            
            delta = model_prob - market_prob
            
            if delta > 0.05:  # Model thinks team is undervalued
                movement = self.get_odds_movement(team, hours=24)
                
                opportunities.append({
                    'team': team,
                    'type': 'undervalued',
                    'market_prob': market_prob,
                    'model_prob': model_prob,
                    'edge': delta,
                    'trend': movement.get('trend', 'unknown'),
                    'recommendation': 'BUY' if movement.get('trend') != 'down' else 'HOLD'
                })
            
            elif delta < -0.05:  # Market overvalued
                movement = self.get_odds_movement(team, hours=24)
                
                opportunities.append({
                    'team': team,
                    'type': 'overvalued',
                    'market_prob': market_prob,
                    'model_prob': model_prob,
                    'edge': abs(delta),
                    'trend': movement.get('trend', 'unknown'),
                    'recommendation': 'SELL' if movement.get('trend') != 'up' else 'HOLD'
                })
        
        return sorted(opportunities, key=lambda x: x['edge'], reverse=True)
    
    async def start_realtime_feed(self):
        """Start real-time odds feed"""
        self.running = True
        print("Starting WC 2026 real-time feed...")
        
        while self.running:
            try:
                await self.fetch_current_odds()
                await asyncio.sleep(self.update_interval)
            except Exception as e:
                print(f"Feed error: {e}")
                await asyncio.sleep(5)
        
        await self.client.close()
    
    def stop(self):
        """Stop real-time feed"""
        self.running = False
        print("WC 2026 real-time feed stopped")


class PredictionAPIIntegration:
    """
    Integration layer between Polymarket feed and Prediction API
    """
    
    def __init__(self):
        self.feed = WC2026RealTimeFeed()
        self.signals_queue = queue.Queue()
        
        # Register callback
        self.feed.register_callback(self._on_odds_change)
    
    def _on_odds_change(self, team: str, old_odds: float, new_odds: float):
        """Handle significant odds changes"""
        change = new_odds - old_odds
        direction = "up" if change > 0 else "down"
        
        print(f"[ALERT] {team}: {old_odds:.1%} -> {new_odds:.1%} ({change:+.1%})")
        
        # Add to signals queue
        self.signals_queue.put({
            'team': team,
            'old_odds': old_odds,
            'new_odds': new_odds,
            'timestamp': datetime.now().isoformat(),
            'direction': direction
        })
    
    async def get_live_signals(self, model_probs: Dict[str, float]) -> Dict:
        """Get live trading signals"""
        
        # Update odds
        current_odds = await self.feed.fetch_current_odds()
        
        # Generate signals
        signals = []
        for team in current_odds:
            market_prob = current_odds[team].probability
            model_prob = model_probs.get(team, 0)
            
            delta = model_prob - market_prob
            
            if delta > 0.02:
                signal_type = 'BUY'
                strength = min(delta * 10, 1.0)
            elif delta < -0.02:
                signal_type = 'SELL'
                strength = min(abs(delta) * 10, 1.0)
            else:
                signal_type = 'HOLD'
                strength = 0.0
            
            movement = self.feed.get_odds_movement(team, hours=1)
            
            signals.append({
                'team': team,
                'market_probability': market_prob,
                'model_probability': model_prob,
                'delta': delta,
                'signal': signal_type,
                'strength': round(strength, 2),
                'trend': movement.get('trend', 'flat'),
                'last_updated': current_odds[team].last_updated
            })
        
        # Sort by strength
        signals.sort(key=lambda x: x['strength'], reverse=True)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'signals_count': len([s for s in signals if s['signal'] != 'HOLD']),
            'signals': signals[:10]  # Top 10
        }


# Demo execution
async def demo():
    """Demo the real-time feed"""
    print("="*70)
    print("WC 2026 POLYMARKET REAL-TIME FEED DEMO")
    print("="*70)
    
    feed = WC2026RealTimeFeed()
    integration = PredictionAPIIntegration()
    
    # Load model probabilities
    try:
        with open('/tmp/wc_model/v3_sim_results.json', 'r') as f:
            model_data = json.load(f)
            model_probs = {
                team: data.get('win', 0) if isinstance(data, dict) else 0
                for team, data in model_data.items()
            }
    except:
        print("Warning: Could not load model probabilities")
        model_probs = {}
    
    # Fetch once
    print("\nFetching current odds...")
    odds = await feed.fetch_current_odds()
    
    print(f"\n{'Team':<15} {'Market%':>10} {'Model%':>10} {'Delta':>8} {'Signal'}")
    print("-"*60)
    
    for team, data in sorted(odds.items(), key=lambda x: x[1].probability, reverse=True)[:15]:
        market_prob = data.probability * 100
        model_prob = model_probs.get(team, 0) * 100
        delta = model_prob - market_prob
        
        if delta > 3:
            sig = "📈 BUY"
        elif delta < -3:
            sig = "📉 SELL"
        else:
            sig = "➡️ HOLD"
        
        print(f"{team:<15} {market_prob:>9.1f}% {model_prob:>9.1f}% {delta:>+7.1f}% {sig}")
    
    # Find arbitrage opportunities
    print("\n" + "="*70)
    print("ARBITRAGE OPPORTUNITIES")
    print("="*70)
    
    opportunities = feed.get_arbitrage_opportunities(model_probs)
    for opp in opportunities[:5]:
        print(f"\n{opp['team']}:")
        print(f"  Type: {opp['type']}")
        print(f"  Market: {opp['market_prob']:.1%} | Model: {opp['model_prob']:.1%}")
        print(f"  Edge: {opp['edge']:.1%} | Trend: {opp['trend']}")
        print(f"  Recommendation: {opp['recommendation']}")
    
    await feed.client.close()
    print("\n✅ Demo complete")


if __name__ == "__main__":
    asyncio.run(demo())
