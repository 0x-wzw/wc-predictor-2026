#!/usr/bin/env python3
"""
WC2026 Model Fine-Tuning Engine
=================================
Continuously update team ratings from:
  1. Match results (highest weight) — Bayesian Elo update
  2. External odds disagreement (medium weight) — Market-implied strength nudge
  3. News/catalyst events (low weight) — Conditional adjustment

Usage:
  python wc2026_finetune.py result --match "France:2-1:Brazil"
  python wc2026_finetune.py odds --nudge 0.05 --team France
  python wc2026_finetune.py recalibrate
  python wc2026_finetune.py status
"""

import json, math, os, sys, pickle, hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass, asdict

# ─── Paths ───────────────────────────────────────────────────────
BASE = Path.home() / ".hermes"
MODELS = BASE / "models" / "wc2026"
DATA = BASE / "data" / "wc2026_signals"

MODEL_PROBS = DATA / "model_predictions.json"
ENGINE_STATE = MODELS / "engine_state_v1.pkl"
RATING_LOG = MODELS / "rating_log.jsonl"
MATCH_LOG = MODELS / "match_history.jsonl"

MODELS.mkdir(parents=True, exist_ok=True)

# ─── Tuning Parameters ────────────────────────────────────────────
K_ELO = 16          # Elo K-factor for match results (responsive but stable)
K_ODDS = 4          # K-factor for odds nudges (weak)
K_NEWS = 2          # K-factor for news events (very weak)
DRAW_HOME_BOOST = 0.06  # Dixon-Coles tau for low-scoring draw adjustment

LOG_SCALE_BASE = 2000  # Rating for best team reference

@dataclass
class RatingRecord:
    team: str
    rating: float          # Elo-like rating
    offensive: float       # xG above/below average
    defensive: float       # xG conceded below/above average
    matches: int
    last_updated: str
    source: str           # "match" | "odds" | "news" | "init"
    delta: float           # Rating change
    confidence: float      # 0-1 based on match volume

class FineTuneEngine:
    """Bayesian updater for team ratings."""

    def __init__(self):
        self.probs: Dict[str, float] = {}
        self.ratings: Dict[str, float] = {}  # team -> rating
        self.offensive: Dict[str, float] = {}
        self.defensive: Dict[str, float] = {}
        self.matches: Dict[str, int] = {}
        self._load()

    def _load(self):
        if MODEL_PROBS.exists():
            with open(MODEL_PROBS) as f:
                self.probs = json.load(f)
            self._init_ratings()

        # Also load any saved engine state
        if ENGINE_STATE.exists():
            try:
                with open(ENGINE_STATE, 'rb') as f:
                    state = pickle.load(f)
                if 'ratings' in state:
                    self.ratings.update(state['ratings'])
                if 'offensive' in state:
                    self.offensive.update(state['offensive'])
                if 'defensive' in state:
                    self.defensive.update(state['defensive'])
                if 'matches' in state:
                    self.matches.update(state['matches'])
            except Exception:
                pass

    def _init_ratings(self):
        """Convert win probabilities to Elo ratings."""
        if not self.probs:
            return
        max_prob = max(self.probs.values())
        for team, prob in self.probs.items():
            if prob > 0:
                strength = math.log(prob / max_prob + 0.01)
                self.ratings[team] = max(1300, min(2100, LOG_SCALE_BASE + 200 * strength))
            else:
                self.ratings[team] = 1300
            self.offensive.setdefault(team, 0.0)
            self.defensive.setdefault(team, 0.0)
            self.matches.setdefault(team, 0)

    def _expected_score(self, r1: float, r2: float) -> float:
        """Elo expected score for team 1 vs team 2."""
        return 1.0 / (1.0 + 10 ** ((r2 - r1) / 400))

    def _log_rating_change(self, record: RatingRecord):
        """Log rating change for audit trail."""
        with open(RATING_LOG, "a") as f:
            f.write(json.dumps(asdict(record)) + "\n")

    # ── LAYER 1: Match Result Updates ────────────────────────────

    def update_from_match(self, team1: str, g1: int, team2: str, g2: int,
                          location: str = "neutral", competition: str = "WC2026"):
        """Update ratings from match result."""
        r1 = self.ratings.get(team1, 1500)
        r2 = self.ratings.get(team2, 1500)

        # Expected scores
        exp1 = self._expected_score(r1, r2)
        exp2 = 1 - exp1

        # Actual scores (1 for win, 0.5 for draw, 0 for loss)
        if g1 > g2:
            s1, s2 = 1.0, 0.0
        elif g1 == g2:
            s1, s2 = 0.5, 0.5
        else:
            s1, s2 = 0.0, 1.0

        # Dixon-Coles draw adjustment for low-scoring
        if g1 == g2 and g1 <= 1:
            dc_factor = 1 + DRAW_HOME_BOOST
            if s1 == 0.5:  # draw
                # Slightly adjust expected to account for low-scoring draws being more likely
                pass

        # Update ratings
        delta1 = K_ELO * (s1 - exp1)
        delta2 = K_ELO * (s2 - exp2)

        self.ratings[team1] = r1 + delta1
        self.ratings[team2] = r2 + delta2

        # Update offensive/defensive from goals
        avg_goals = 1.35
        self.offensive[team1] += (g1 - avg_goals) * 0.1
        self.defensive[team1] += (avg_goals - g2) * 0.1
        self.offensive[team2] += (g2 - avg_goals) * 0.1
        self.defensive[team2] += (avg_goals - g1) * 0.1

        self.matches[team1] = self.matches.get(team1, 0) + 1
        self.matches[team2] = self.matches.get(team2, 0) + 1

        # Log
        self._log_rating_change(RatingRecord(
            team=team1, rating=self.ratings[team1], offensive=self.offensive[team1],
            defensive=self.defensive[team1], matches=self.matches[team1],
            last_updated=datetime.now().isoformat(), source="match", delta=delta1,
            confidence=min(0.95, 0.5 + self.matches[team1] * 0.05)
        ))
        self._log_rating_change(RatingRecord(
            team=team2, rating=self.ratings[team2], offensive=self.offensive[team2],
            defensive=self.defensive[team2], matches=self.matches[team2],
            last_updated=datetime.now().isoformat(), source="match", delta=delta2,
            confidence=min(0.95, 0.5 + self.matches[team2] * 0.05)
        ))

        # Save match
        with open(MATCH_LOG, "a") as f:
            f.write(json.dumps({
                "ts": datetime.now().isoformat(),
                "team1": team1, "g1": g1,
                "team2": team2, "g2": g2,
                "location": location, "competition": competition,
            }) + "\n")

        print(f"  {team1} {g1}-{g2} {team2}")
        print(f"  → {team1}: {r1:.0f} → {self.ratings[team1]:.0f} ({delta1:+.1f})")
        print(f"  → {team2}: {r2:.0f} → {self.ratings[team2]:.0f} ({delta2:+.1f})")

    # ── LAYER 2: External Odds Nudge ─────────────────────────────

    def nudge_from_odds(self, team: str, market_prob: float, model_prob: float,
                        weight: float = 1.0):
        """
        Nudge rating based on odds disagreement.
        If market consistently prices team differently than model,
        weakly adjust model toward market (learning from informed money).
        """
        if team not in self.ratings:
            return

        # Convert market prob to implied rating difference from best team
        # This is weak evidence — stronger when volume is high
        prob_diff = market_prob - model_prob

        # Convert to rating change
        # A 1% prob difference ≈ 2 Elo points (very weak)
        rating_delta = K_ODDS * weight * prob_diff * 100

        old_rating = self.ratings[team]
        self.ratings[team] += rating_delta

        self._log_rating_change(RatingRecord(
            team=team, rating=self.ratings[team],
            offensive=self.offensive.get(team, 0),
            defensive=self.defensive.get(team, 0),
            matches=self.matches.get(team, 0),
            last_updated=datetime.now().isoformat(),
            source="odds", delta=rating_delta,
            confidence=0.3,  # low confidence from odds alone
        ))

        print(f"  {team}: {old_rating:.0f} → {self.ratings[team]:.0f} (odds nudge {rating_delta:+.1f})")

    # ── LAYER 3: News/Catalyst Update ────────────────────────────

    def update_from_news(self, team: str, event_type: str, impact: float):
        """
        Conditional update from news events.
        impact: +1.0 to -1.0 (positive = team improved)
        """
        if team not in self.ratings:
            return

        delta = K_NEWS * impact
        old = self.ratings[team]
        self.ratings[team] += delta

        self._log_rating_change(RatingRecord(
            team=team, rating=self.ratings[team],
            offensive=self.offensive.get(team, 0),
            defensive=self.defensive.get(team, 0),
            matches=self.matches.get(team, 0),
            last_updated=datetime.now().isoformat(),
            source="news", delta=delta,
            confidence=0.15,
        ))

        print(f"  {team}: {old:.0f} → {self.ratings[team]:.0f} (news: {event_type}, impact {impact:+.2f})")

    # ── LAYER 4: Recalibrate Model ───────────────────────────────

    def recalibrate(self):
        """Recalibrate tournament probabilities from updated ratings."""
        if not self.ratings:
            print("No ratings available")
            return

        # Convert ratings directly to win probabilities using softmax
        # Tournament win probability ≈ relative team strength
        # For 48 teams in WC2026, we need to account for the bracket structure
        # But for simplicity, we normalize by exponential of ratings

        # Raw strengths
        strengths = {}
        for team, rating in self.ratings.items():
            strengths[team] = 10 ** (rating / 400)

        total = sum(strengths.values())
        winner_probs = {
            team: round(strength / total, 8)
            for team, strength in strengths.items()
        }

        with open(MODEL_PROBS, "w") as f:
            json.dump(winner_probs, f, indent=2)

        # Save engine state
        with open(ENGINE_STATE, "wb") as f:
            pickle.dump({
                "ratings": self.ratings,
                "offensive": self.offensive,
                "defensive": self.defensive,
                "matches": self.matches,
                "last_updated": datetime.now().isoformat(),
            }, f)

        # Print summary
        print(f"\nRecalibrated {len(winner_probs)} teams")
        top = sorted(winner_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        print("Top 5:")
        for team, prob in top:
            print(f"  {team}: {prob*100:.2f}%")

        return winner_probs

    def status(self):
        print("=" * 50)
        print(" FINE-TUNE ENGINE STATUS")
        print("=" * 50)
        print(f"Teams: {len(self.ratings)}")
        print(f"Ratings range: {min(self.ratings.values()):.0f} - {max(self.ratings.values()):.0f}")
        print(f"Model path: {MODEL_PROBS}")

        if self.ratings:
            print("\nTop 10 by rating:")
            for team, r in sorted(self.ratings.items(), key=lambda x: x[1], reverse=True)[:10]:
                m = self.matches.get(team, 0)
                print(f"  {team}: {r:.0f} (matches: {m})")

def main():
    import argparse
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='cmd')

    # Match result update
    result = sub.add_parser('result', help='Update from match result')
    result.add_argument('--match', required=True, help='Team1:G1-G2:Team2')
    result.add_argument('--location', default='neutral', choices=['neutral', 'home', 'away'])

    # Odds nudge
    nudge = sub.add_parser('nudge', help='Nudge from odds disagreement')
    nudge.add_argument('--team', required=True)
    nudge.add_argument('--market', type=float, required=True, help='Market prob')
    nudge.add_argument('--model', type=float, required=True, help='Model prob')
    nudge.add_argument('--weight', type=float, default=1.0)

    # News update
    news = sub.add_parser('news', help='Update from news event')
    news.add_argument('--team', required=True)
    news.add_argument('--event', required=True)
    news.add_argument('--impact', type=float, required=True)

    # Recalibrate
    recal = sub.add_parser('recalibrate', help='Recalibrate model from ratings')

    # Status
    sub.add_parser('status', help='Show engine status')

    args = parser.parse_args()
    engine = FineTuneEngine()

    if args.cmd == 'result':
        parts = args.match.split(':')
        if len(parts) != 3:
            print("Format: Team1:G1-G2:Team2")
            sys.exit(1)
        t1, score, t2 = parts
        g1, g2 = map(int, score.split('-'))
        engine.update_from_match(t1, g1, t2, g2, args.location)
        engine.recalibrate()

    elif args.cmd == 'nudge':
        engine.nudge_from_odds(args.team, args.market, args.model, args.weight)
        engine.recalibrate()

    elif args.cmd == 'news':
        engine.update_from_news(args.team, args.event, args.impact)
        engine.recalibrate()

    elif args.cmd == 'recalibrate':
        engine.recalibrate()

    elif args.cmd == 'status':
        engine.status()

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
