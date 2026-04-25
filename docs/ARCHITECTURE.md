# WC2026 Prediction Engine Documentation

## Architecture Overview

### Model-First Philosophy

The WC2026 prediction engine follows a strict model-first architecture where:

1. **Internal Model** generates ALL probability estimates
2. **External Data** (odds, results, news) feeds INTO the model
3. **Signals** (S1-S12) filter trade execution, they do NOT generate alpha

This prevents circular dependencies and ensures model integrity.

```
┌─────────────────────────────────────────────────────────┐
│                    EXTERNAL DATA                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐ │
│  │Polymarket│  │Match Res │  │   News   │  │  Rankings│ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘ │
│       │             │             │             │      │
└───────┼─────────────┼─────────────┼─────────────┼──────┘
        │             │             │             │
        └─────────────┴──────┬──────┴─────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────┐
│                    INTERNAL MODEL                        │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────┐ │
│  │  Elo Ratings │────▶│ Poisson Goals│────▶│ Match Prob│ │
│  └──────────────┘     └──────────────┘     └────┬─────┘ │
└────────────────────────────────────────────────┼────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────┐
│                     EDGE DETECTION                       │
│              Edge = Model Prob - Market Prob            │
└─────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────┐
│              SIGNAL FILTER (S1-S12)                    │
│     ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐          │
│     │ S1  │ │ S3  │ │ S6  │ │ S8  │ │ S9  │  ← Outcome│
│     └─────┘ └─────┘ └─────┘ └─────┘ └─────┘          │
│     ┌─────┐ ┌─────┐ ┌─────┐                          │
│     │ S10 │ │ S11 │ │ S12 │      ← Asian Handicap     │
│     └─────┘ └─────┘ └─────┘                            │
└─────────────────────────────────────────────────────────┘
                                                  │
                                                  ▼
┌─────────────────────────────────────────────────────────┐
│               POSITION SIZING                          │
│         Quarter-Kelly × Signal Confidence               │
└─────────────────────────────────────────────────────────┘
