# WC 2026 Full Tournament Prediction Model Plan

**Date:** 2026-04-22  
**Goal:** Build comprehensive WC 2026 prediction system covering group stage → knockout → finals

---

## 1. Current State Assessment

### What exists:
- `src/simulator.py` - Basic tournament simulation
- `src/predictor.py` - Match-level predictions
- `src/polymarket_feed.py` - Market odds integration
- Historical team data loaded

### What's missing:
- Complete group stage simulation with 1st/2nd placement logic
- Knockout bracket progression tracking
- Dynamic matchup predictions based on tournament phase
- Full tournament path probability tree

---

## 2. Proposed Architecture

```
┌─────────────────┐
│ TournamentModel │  ← Orchestrates full simulation
└────────┬────────┘
         │
    ┌────┴────┬────────────┬──────────────┐
    ↓         ↓            ↓              ↓
┌────────┐ ┌────────┐ ┌──────────┐ ┌─────────┐
│ Group   │ │R16     │ │QF/SF     │ │Final    │
│Stage    │ │Matchup │ │Sim       │ │Prediction
│Simulator│ │Resolver│ │          │ │         
└────────┘ └────────┘ └──────────┘ └─────────┘
    │         │
    └────┬────┘
         ↓
┌─────────────────┐
│ MatchPredictor │  ← Phase-aware match logic
└─────────────────┘
```

---

## 3. Implementation Plan

### Phase 1: Group Stage Simulation System

**File:** `src/group_stage.py`

```python
@dataclass
class GroupResult:
    name: str  # Group A, B, etc.
    teams: List[str]
    standings: List[GroupStanding]  # Sorted by rank

@dataclass  
class GroupStanding:
    team: str
    rank: int  # 1-4
    played: int
    wins: int
    draws: int
    losses: int
    goals_for: int
    goals_against: int
    goal_diff: int
    points: int
    advanced: bool  # Top 2 advance
```

**Key Functions:**
- `simulate_group(group_teams: List[str]) -> GroupResult`
- `calculate_standings(matches: List[MatchResult]) -> List[GroupStanding]`
- `resolve_ties(teams: List[GroupStanding]) -> List[GroupStanding]`
  - FIFA tiebreakers: GD → GS → H2H

**Output:** Groups A-F complete with 1st and 2nd place advancement

---

### Phase 2: Knockout Bracket Resolution

**File:** `src/knockout.py`

```python
@dataclass
class KnockoutBracket:
    r16: List[Matchup]  # 8 matches
    quarterfinals: List[Matchup]  # 4 matches
    semifinals: List[Matchup]  # 2 matches
    final: Matchup
    third_place: Matchup

@dataclass
class Matchup:
    team1: str
    team2: str
    team1_source: str  # "1A", "2B", etc.
    team2_source: str
```

**Bracket Mapping (WC 2026 format):**
```
R16: 1A vs 2B, 1C vs 2D, 1E vs 2F, 1G vs 2H, 1B vs 2A, 1D vs 2C, 1F vs 2E, 1H vs 2G
QF: R16-1 winner vs R16-2 winner, etc.
SF: QF winners
Final: SF winners
```

**Key Functions:**
- `build_bracket(group_results: Dict[str, GroupResult]) -> KnockoutBracket`
- `resolve_matchup(matchup: Matchup, winner: str) -> str`
- `simulate_knockout(bracket: KnockoutBracket) -> TournamentResult`

---

### Phase 3: Phase-Aware Match Predictor

**Enhancement to:** `src/predictor.py`

```python
class PhaseAwarePredictor(MatchupPredictor):
    def predict_group_match(self, team1: str, team2: str) -> PredictionResult:
        # Group stage: draws possible
        # Conservative predictions (teams play for qualification)
        pass
    
    def predict_knockout_match(self, team1: str, team2: str) -> PredictionResult:
        # Knockout: extra time/penalties
        # Higher pressure, conservative tactics
        # Boost favorites slightly
        pass
```

**Phase Adjustments:**
- Group stage: +5% draw probability (teams play safe)
- Knockout: -5% draw (settled by ET), slight favorite boost
- Final: Form factor multiplied by 1.2x

---

### Phase 4: Full Tournament Monte Carlo Simulation

**File:** `src/tournament_model.py`

```python
@dataclass
class TournamentSimulation:
    runs: int
    group_results: Dict[str, GroupResult]  # Aggregated over all runs
    advancement_probs: Dict[str, Dict[str, float]]  # p(team reaches stage)
    winner_probs: Dict[str, float]  # p(team wins tournament)
    final_matchups: Counter  # Most common finals

class TournamentModel:
    def run_monte_carlo(self, runs: int = 10000) -> TournamentSimulation:
        results = []
        for i in range(runs):
            # 1. Simulate groups
            groups = {name: simulate_group(teams) 
                     for name, teams in GROUPS_2026.items()}
            
            # 2. Build bracket
            bracket = build_bracket(groups)
            
            # 3. Simulate knockout
            winner = simulate_knockout(bracket)
            
            results.append(winner)
        
        return aggregate_results(results)
```

---

### Phase 5: Output & API Integration

**New Endpoints in `src/api_server.py`:**

```python
@app.get("/tournament/simulate")
def simulate_tournament(runs: int = 1000):
    """Run full tournament Monte Carlo"""
    result = model.run_monte_carlo(runs)
    return {
        "winner_probabilities": result.winner_probs,
        "advancement_probabilities": result.advancement_probs,
        "most_likely_finals": result.final_matchups.most_common(5)
    }

@app.get("/tournament/group/{group_id}")
def predict_group(group_id: str):
    """Predict group standings"""
    return simulate_group(GROUPS_2026[group_id])

@app.get("/tournament/bracket/{scenario}")
def get_bracket(scenario: str = "most_likely"):
    """Get bracket for a specific scenario"""
    pass
```

---

## 4. Files to Create/Modify

| File | Action | Purpose |
|------|--------|---------|
| `src/group_stage.py` | Create | Group simulation logic |
| `src/knockout.py` | Create | Knockout bracket resolution |
| `src/tournament_model.py` | Create | Full tournament orchestration |
| `src/predictor.py` | Modify | Add phase-aware methods |
| `src/data.py` | Create | WC 2026 groups + schedule |
| `src/api_server.py` | Modify | Add tournament endpoints |

---

## 5. Data Requirements

**WC 2026 Group Structure:**
```json
{
  "groups": {
    "A": ["Germany", "Scotland", "Hungary", "Switzerland"],
    "B": ["Spain", "Croatia", "Italy", "Albania"],
    "C": ["Slovenia", "Denmark", "Serbia", "England"],
    "D": ["Netherlands", "France", "Poland", "Austria"],
    "E": ["Belgium", "Slovakia", "Romania", "Ukraine"],
    "F": ["Turkey", "Georgia", "Portugal", "Czechia"], 
    "G": ["Morocco", "Zambia", "DR Congo", "Tanzania"],
    "H": ["Egypt", "Burkina Faso", "Guinea-Bissau", "Ethiopia"]
  }
}
```

*(Note: Confirm actual teams for WC 2026 once qualified)*

---

## 6. Testing Strategy

**Unit Tests:**
- Group tiebreaker scenarios (all FIFA rules)
- Knockout bracket mapping correctness
- Monte Carlo convergence

**Integration Tests:**
- Full simulation run produces valid bracket
- API endpoints return correct schema
- Polymarket comparison still works

**Validation:**
- Compare against betting odds
- Historical backtest: predict 2022 WC

---

## 7. Risks & Tradeoffs

| Risk | Mitigation |
|------|-----------|
| Group tiebreaker complexity | Implement full FIFA rules |
| Monte Carlo noise | Run 10k+ iterations, track variance |
| Missing qualified teams | Make groups configurable |
| Computation time | Cache group results, parallelize |

---

## 8. Success Metrics

- [ ] Can simulate 10k tournaments in < 30 seconds
- [ ] API returns winner probabilities with confidence intervals
- [ ] Group standings match FIFA rules exactly
- [ ] Bracket visualizable (text or JSON)
- [ ] Monte Carlo errors < 0.5% for top teams

---

## 9. Execution Steps

1. Create group stage module with standings calculation
2. Implement FIFA tiebreaker rules
3. Create knockout bracket resolver
4. Build full tournament orchestrator
5. Add phase-aware prediction adjustments
6. Create API endpoints
7. Write tests
8. Validate with historical data

**Estimated effort:** 2-3 days for complete implementation

---

*Plan created: 2026-04-22*
