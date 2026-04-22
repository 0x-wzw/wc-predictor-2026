/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── simulator.py          # Core tournament simulator
│   ├── predictor.py          # Matchup predictor
│   ├── api_server.py         # FastAPI server
│   ├── polymarket_feed.py    # Polymarket integration (no keys)
│   ├── signal_engine.py      # Signal generation
│   └── utils.py              # Helper functions
├── data/
│   ├── README.md             # Data source documentation
│   ├── historical/           # Historical match data (public)
│   └── wc2026_groups.json    # 2026 tournament structure
├── results/
│   ├── probabilities.json    # Sample output
│   and signals/              # Signal records
└── tests/
    ├── test_simulator.py
    └── test_predictor.py
