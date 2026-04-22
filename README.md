# World Cup 2026 Prediction Model

Predictive model for FIFA World Cup 2026 match outcomes and tournament winner probabilities.

## Quick Start

```bash
pip install -r requirements.txt
python src/simulator.py
```

## Architecture

- **src/simulator.py**: Tournament simulation using historical WC data
- **src/predictor.py**: Matchup prediction using team profiles
- **src/polymarket_feed.py**: Real-time odds from Polymarket (public API)
- **src/api_server.py**: FastAPI server for predictions

## Data Sources

- Historical World Cup match data (public)
- FIFA rankings (public)
- Polymarket odds (public Gamma API)

## Methodology

1. **Phase-aware modeling**: Separate group vs knockout performance
2. **Team profiles**: Historical win rates, advancement odds
3. **Matchup prediction**: Experience gap + head-to-head + home advantage
4. **Signal generation**: Model vs market edge detection (Kelly criterion)

## API Usage

```python
from src.predictor import MatchupPredictor

p = MatchupPredictor()
result = p.predict("France", "Germany", neutral_venue=True)
print(f"France win: {result.home_win_prob:.1%}")
```

## License

MIT
