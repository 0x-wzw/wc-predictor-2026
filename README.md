# WC2026 Prediction Engine v3.0

Model-first prediction system for FIFA World Cup 2026 outcomes, market edge detection, and Asian Handicap line calculation.

## 🎯 Architecture Philosophy

```
[Internal Model] ← external data (odds, results, news)
       ↓
[Edge Calculation] = Model Prob - Market Prob
       ↓
[Signal Filter] = S1-S12 confirm edge → trade
       ↓
[Position Sizing] = Quarter-Kelly adjusted by signals + liquidity
```

**Core Principle**: Internal model anchors ALL predictions. External data feeds INTO the model. Signals are confirmation filters, not primary alpha generators.

## 📁 Repository Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── wc2026_engine.py              # Core Elo + Poisson engine
│   ├── wc2026_model.py               # Production match predictor
│   ├── wc2026_unified_pipeline.py    # Main orchestration (S1-S9)
│   ├── wc2026_finetune.py            # Bayesian model updater
│   ├── wc2026_master.py              # Master CLI entry point
│   ├── wc2026_check_signals.py       # Telegram report formatter
│   ├── wc2026_enhanced_predictor.py  # Advanced prediction features
│   ├── wc2026_match_predictor.py     # Standalone match predictor
│   ├── wc_daily_monitor.py           # Daily Polymarket odds monitor
│   ├── wc2026_dashboard_daily.py     # Dashboard generation
│   ├── ah/                           # Asian Handicap subpackage
│   │   ├── ah_engine.py              # Poisson AH calculator
│   │   ├── ah_signals.py             # S10-S12 AH signals
│   │   ├── ah_unified_pipeline.py    # AH orchestration
│   │   └── ah_backtest.py            # AH backtesting framework
│   ├── predictor.py                  # Legacy match predictor
│   ├── api_server.py                 # FastAPI server
│   └── polymarket_feed.py            # Polymarket integration
├── data/
│   ├── wc2026_signals/               # Signal history & predictions
│   └── ah_models/                    # AH fair line data
├── models/
│   ├── wc2026/                       # Model artifacts
│   └── ah_models/                    # AH model outputs
└── docs/                             # Architecture documentation
```

## 🚀 Quick Start

```bash
pip install -r requirements.txt

# Full orchestration
python src/wc2026_master.py daily

# Direct commands
python src/wc2026_master.py full_report

# Pipeline stages
python src/wc2026_unified_pipeline.py report     # Full report
python src/wc2026_unified_pipeline.py signals    # Trade signals only

# Asian Handicap
python src/ah/ah_unified_pipeline.py run

# Fine-tune after match result
python src/wc2026_finetune.py result --match "Spain:2-1:Brazil"
```

## 📊 Components

### Core Engine (`wc2026_engine.py`)
- **Team Ratings**: Elo-like system anchored to historical FIFA rankings
- **Match Prediction**: Elo-based expected goals + Poisson scoring
- **Bayesian Update**: Ratings update after each match result
- **Tournament Sim**: Monte Carlo to derive winner probabilities

### Unified Pipeline (`wc2026_unified_pipeline.py`)
| Layer | Purpose |
|-------|---------|
| Ingest | Fetch Polymarket odds (public Gamma API) |
| Compare | Calculate model vs market edge |
| Signals | Apply S1-S9 filters to trades |
| Report | Output formatted daily report |

### Signal Framework (S1-S12)

**Outcome Signals (S1-S9)**:
- S1: Disposition/contrarian check
- S3: Velocity stability
- S6: Volume/liquidity
- S8: Theta (time to expiry)
- S9: Model edge strength

**Asian Handicap Signals (S10-S12)**:
- S10: Line movement toward model fair line
- S11: Sharp/retail divergence detection
- S12: Goal line/Asian Handicap correlation

### Asian Handicap Engine (`ah/ah_engine.py`)
- Pure Python Poisson implementation (no scipy)
- Quarter/fractional line support (-0.25, -0.75, etc.)
- Fair line calculator from team ratings
- Edge detection vs market lines

### Fine-Tuning (`wc2026_finetune.py`)
```python
# Match results: Bayesian Elo update (K=16, highest weight)
# Odds disagreement: Rating nudge (K=4, learns from market)
# News events: Weak adjustment (K=2)
```

## ⚙️ Configuration

Edit paths in scripts or use environment variables:

```bash
export WC2026_MODEL_DIR="~/.hermes/models/wc2026"
export WC2026_DATA_DIR="~/.hermes/data/wc2026_signals"
export WC2026_ACCOUNT_SIZE=10000
```

## 📈 Trading Signal Thresholds

| Delta | Signal | Action |
|-------|--------|--------|
| ≥3% | BUY++ | ACCUMULATE (aggressive) |
| ≥1% | BUY+ | ACCUMULATE |
| ≤-3% | SELL++ | REDUCE (aggressive) |
| ≤-1% | SELL+ | REDUCE |
| -1%~+1% | FAIR | HOLD |

**Position Sizing**: Quarter-Kelly with liquidity caps

## 🔧 API Usage

```python
from wc2026_engine import WC2026Engine

engine = WC2026Engine()
engine.init_ratings()

# Predict match
result = engine.predict_match("France", "Brazil")
print(f"France: {result.t1_win_p:.1%}, Draw: {result.draw_p:.1%}, Brazil: {result.t2_win_p:.1%}")

# Simulate tournament
probs = engine.simulate_tournament(runs=100000)
for team, prob in sorted(probs.items(), key=lambda x: -x[1])[:5]:
    print(f"{team}: {prob:.1%}")
```

```python
# Asian Handicap
from src.ah.ah_engine import AsianHandicapEngine

engine = AsianHandicapEngine()
lines = engine.calculate_fair_lines("Argentina", "France")
print(f"Fair HDP: {lines.fair_hdp:+.2f}")
```

## 📡 Cron Job Setup

Daily monitoring runs at 09:00 UTC:

```bash
# Add to crontab
0 9 * * * cd ~/wc-predictor-repo && python src/wc2026_check_signals.py
```

## 📝 License

MIT - See LICENSE file

## 🤝 Credits

Model architecture inspired by:
- FiveThirtyEight's World Cup model
- Elo rating systems (Chess/FIFA)
- Dixon-Coles Poisson modifications
- Kelly criterion position sizing
