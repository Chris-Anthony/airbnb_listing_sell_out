# What Makes an Airbnb Listing Sell Out?

Analysis of 10,000 synthetic short-term rental listings across 12 cities to predict **high-demand listings** (classification) and **annual revenue** (regression).

## Dataset

[Kaggle - What Makes an Airbnb Listing Sell Out?](https://www.kaggle.com/datasets/vedauv/what-makes-an-airbnb-listing-sell-out-10k-stays) -- 10,000 rows, 21 columns covering listing attributes, host metrics, and booking outcomes across 12 global cities.

## Analysis Overview

| Section | Description |
|---|---|
| EDA | Target distributions, feature distributions, correlation heatmap, city-level analysis |
| Feature Engineering | `total_cost`, `price_per_guest`, `quality_score` + multicollinearity (VIF) and interaction screening |
| Classification | 5-model comparison (Logistic Regression, Random Forest, XGBoost, Gradient Boosting, LightGBM) with hyperparameter tuning and threshold optimization |
| Regression | Same 5-model comparison predicting log-transformed annual revenue, evaluated in original USD scale |

## Key Results

### Classification (High Demand)

| Model | Test AUC |
|---|---|
| Random Forest (Tuned) | **0.9342** |
| Random Forest | 0.9331 |
| Logistic Regression | 0.9327 |
| Gradient Boosting | 0.9327 |
| XGBoost | 0.9324 |
| LightGBM | 0.9289 |

Threshold optimization (0.50 -> 0.38) improved F1 from 0.70 to 0.76 by trading slight precision for a large recall gain.

### Regression (Annual Revenue)

| Model | R² (log) | RMSE (USD) | MAE (USD) |
|---|---|---|---|
| XGBoost (Tuned) | **0.80** | ~$3,700 | ~$2,680 |
| Gradient Boosting | 0.78 | $3,724 | $2,664 |
| XGBoost | 0.78 | $3,732 | $2,683 |
| LightGBM | 0.78 | $3,774 | $2,685 |
| Random Forest | 0.77 | $3,801 | $2,761 |
| Ridge | 0.75 | $5,507 | $3,272 |

### Business Insights

- Bangkok and Cape Town have the highest demand rates (~40%) despite the lowest prices -- affordability drives occupancy
- The only beneficial interaction feature was `price x distance_to_center`, suggesting central listings sustain higher prices without hurting demand
- London and New York lead in revenue (~$18K/yr) but have the lowest demand rates (~20%) -- high prices compensate for lower occupancy

## Setup

```bash
conda env create -f environment.yml
conda activate airbnb
jupyter notebook airbnb_analysis.ipynb
```

## Tools

Python 3.11, pandas, NumPy, matplotlib, seaborn, scikit-learn, XGBoost, LightGBM, statsmodels
