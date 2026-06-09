# NBA Shot Prediction

## Overview

This project predicts whether an NBA shot will be made or missed using machine learning techniques. The model uses shot characteristics, defensive pressure, and game context features to estimate shot success probability.

## Features

* Exploratory Data Analysis (EDA)
* Feature Engineering
* Logistic Regression Baseline
* Random Forest Classifier
* XGBoost Classifier
* Model Evaluation using ROC-AUC, Accuracy, Precision, Recall, and F1-Score
* Streamlit Web Application for Real-Time Predictions

## Dataset

NBA Shot Logs dataset containing information such as:

* Shot Distance
* Shot Clock
* Defender Distance
* Touch Time
* Shot Type
* Game Context

## Tech Stack

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Matplotlib
* Seaborn
* Streamlit

## Project Structure

```text
nba-shot-predictor/
│
├── app.py
├── notebooks/
├── data/
├── final_model.pkl
├── scaler.pkl
├── requirements.txt
└── README.md
```

## Results

The project compares multiple machine learning models and identifies the most important factors affecting shot success. Feature importance analysis shows that shot distance, shot zone, and defensive pressure are among the strongest predictors.

## Running the Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Future Improvements

* Hyperparameter tuning
* Additional player-level features
* Advanced ensemble methods
* Live NBA data integration

```
```
