/
├── README.md                           # Main documentation
├── LICENSE                             # MIT License
├── requirements.txt                    # Python dependencies
├── .gitignore                          # Git ignore patterns
├── DIRECTORY.md                       # This file - repo structure
├── src/                               # Source code
│   ├── __init__.py                    # Package init
│   ├── wc2026_engine.py              # Core prediction engine (Elo + Poisson)
│   ├── wc2026_model.py               # Production match predictor
│   ├── wc2026_unified_pipeline.py    # Main orchestration pipeline
│   ├── wc2026_finetune.py            # Bayesian fine-tuning engine
│   ├── wc2026_master.py              # Master CLI entry point
│   ├── wc2026_check_signals.py       # Signal checking formatter
│   ├── wc2026_match_predictor.py     # Standalone match predictor
│   ├── wc2026_enhanced_predictor.py  # Advanced predictions
│   ├── wc_daily_monitor.py           # Daily Polymarket monitor
│   ├── wc2026_dashboard_daily.py     # Dashboard generator
│   ├── legacy/                        # Legacy/backward compatibility
│   │   ├── predictor.py              # Original predictor
│   │   ├── simulator.py            # Original simulator
│   │   ├── api_server.py            # Original API server
│   │   ├── wc_api_server.py         # WorldCup API server
│   │   └── polymarket_feed.py       # Polymarket integration
│   └── ah/                          # Asian Handicap package
│       ├── __init__.py              # AH package init
│       ├── ah_engine.py             # Poisson AH calculator
│       ├── ah_signals.py            # AH-specific signals (S10-S12)
│       ├── ah_unified_pipeline.py   # AH orchestration
│       └── ah_backtest.py           # AH backtesting
├── data/                            # Data directory
│   ├── wc2026_signals/              # Signal history & predictions
│   │   └── model_predictions.json   # Current model predictions
│   └── ah_models/                   # AH fair line data
├── models/                          # Model artifacts
│   ├── wc2026/                      # WC2026 model storage
│   └── ah_models/                   # AH model outputs
├── docs/                            # Documentation
│   ├── ARCHITECTURE.md              # System architecture
│   ├── API.md                       # API reference
│   └── SIGNALS.md                   # Signal documentation (S1-S12)
└── tests/                           # Unit tests
    ├── test_engine.py               # Engine tests
    ├── test_model.py                # Model tests
    ├── test_pipeline.py           # Pipeline tests
    └── test_ah.py                   # AH tests
