# ⚾ Home Run Prediction

This project estimates the probability of a home run occurring during a given MLB plate appearance using a **Bayesian hierarchical model**. The model incorporates batter-specific and pitcher-specific effects to make matchup-level predictions and is developed using **PyMC** and **ArviZ**. 

```
├── Diagnostics and Prediction.ipynb
├── Game Preprocessing 2024.ipynb
├── Game Preprocessing 2025.ipynb
├── Home Run Prediction.ipynb
├── README.md
├── artifacts
│   ├── logs
│   │   ├── 2025-10-01_03
│   │   │   ├── events.log
│   │   │   ├── full
│   │   │   │   └── config.yaml
│   │   │   ├── metrics.jsonl
│   │   │   └── noprev
│   │   │       └── config.yaml
│   │   └── 2025-10-02_01
│   │       ├── events.log
│   │       ├── full
│   │       │   └── config.yaml
│   │       ├── metrics.jsonl
│   │       └── noprev
│   │           └── config.yaml
└── src
    └── hrmodel
        ├── __init__.py
        ├── diagnostics.py
        ├── evaluation.py
        ├── model.py
        ├── prediction.py
        ├── preprocessing.py
        ├── save_ppc.py
        └── utils
            ├── __init__.py
            ├── logging_utils.py
            └── paths.py
```