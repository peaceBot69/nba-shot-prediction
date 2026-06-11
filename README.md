# NBA Shot Prediction — End-to-End ML Classification Pipeline

Binary classification pipeline predicting NBA shot success (made/missed) using shot characteristics, defensive pressure, and game context features from the 2014–15 NBA season.

---

## Results

| Model | ROC-AUC | Accuracy | Precision (Made) | Recall (Made) | F1 (Made) |
|---|---|---|---|---|---|
| Logistic Regression (Baseline) | 0.634 | 61% | 0.60 | 0.42 | 0.49 |
| Random Forest | **0.641** | 61% | 0.59 | 0.46 | 0.52 |
| XGBoost | 0.633 | 61% | 0.58 | 0.47 | 0.52 |

> ROC-AUC of ~0.64 is consistent with domain literature — shot success is inherently stochastic beyond measurable features. Random Forest achieved the best AUC.

---

## Features

- Exploratory Data Analysis (EDA) — shot distribution, distance, shot clock, defender proximity
- Feature Engineering — target encoding, shot clock median imputation, correlated feature removal
- Baseline Model — Logistic Regression
- Advanced Models — Random Forest, XGBoost
- Model Evaluation — ROC-AUC, Precision, Recall, F1-Score across all 3 classifiers
- SHAP Feature Importance — TreeExplainer on RF and XGBoost; shot distance and defender proximity identified as dominant predictors
- Streamlit Dashboard — real-time shot make-probability prediction with serialised model

---

## Dataset

NBA Shot Logs (2014–15 season) — 128,069 shot attempts across the full season. Source: [Kaggle NBA Shot Logs Dataset](https://www.kaggle.com/datasets/dansbecker/nba-shot-logs).

Key features used:
- `SHOT_DIST` — distance from basket
- `SHOT_CLOCK` — time remaining on shot clock (5,567 nulls → median imputed)
- `CLOSE_DEF_DIST` — closest defender distance
- `DRIBBLES` — dribbles before shot
- `TOUCH_TIME` — time holding ball before shot
- `LOCATION`, `PERIOD`, `FINAL_MARGIN` — game context

Dropped: `FGM`, `PTS` (target leakage), `PTS_TYPE` (collinear with `SHOT_DIST`)

---

## Tech Stack

- Python — pandas, NumPy, scikit-learn, XGBoost, matplotlib, seaborn
- SHAP — model interpretability
- Streamlit — interactive prediction dashboard
- pickle — model serialisation (`final_model.pkl`, `scaler.pkl`)

---

## Project Structure

```
nba-shot-predictor/
│
├── app.py                  # Streamlit dashboard
├── notebooks/
│   └── Untitled.ipynb      # Full pipeline: EDA → features → models → SHAP
├── data/
│   └── shot_logs.csv
├── final_model.pkl         # Serialised Random Forest
├── final_model_xgb.pkl     # Serialised XGBoost
├── scaler.pkl              # StandardScaler
├── requirements.txt
└── README.md
```

---

## Running the Project

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## Key Findings

- **Shot distance** is the strongest predictor — shots under 8ft have significantly higher make probability
- **Defender proximity** has clear signal — open shots (>6ft defender distance) convert at higher rates
- **Shot clock** contributes weakly alone but improves model when combined with distance
- All 3 models converge around 0.63–0.64 AUC — ceiling likely imposed by inherent shot randomness

---

## Future Improvements

- Player-level historical shooting % as feature
- Hyperparameter tuning via GridSearchCV / Optuna
- Multi-season data for generalisation
- Live NBA data integration via NBA Stats API
