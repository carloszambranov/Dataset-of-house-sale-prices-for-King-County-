# King County House Price Prediction

Predicting house sale prices in King County, WA (May 2014 – May 2015) using machine learning.

## Dataset
- **21,613 records**, 21 features
- Source: `data/king_county_houses.csv`
- Target: `price` (sale price in USD)

## Project Structure
```
project-king-county-house-prices/
├── data/
│   └── king_county_houses.csv
├── king_county_analysis.ipynb
└── README.md
```

## ML Process

### 1. EDA
- Price distribution (right-skewed → log transform helpful)
- Correlation analysis: `sqft_living`, `grade`, `bathrooms` are top correlated features
- Geographic heatmap of prices
- Premium properties (≥$650K) deep-dive: 24% of dataset, higher grade, larger area, more waterfront

### 2. Feature Engineering
| Feature | Description |
|---|---|
| `house_age` | Year of sale minus year built |
| `was_renovated` | Binary flag for renovation |
| `years_since_renovation` | Age or years since last reno |
| `has_basement` | Binary flag |
| `living_space_ratio` | sqft_living vs nearest 15 neighbors |
| `sale_month` | Seasonal effect |

### 3. Models Trained
| Model | Notes |
|---|---|
| Linear Regression | Baseline, scaled features |
| Ridge (α=10) | L2 regularization |
| Lasso (α=100) | L1 regularization + feature selection |
| Random Forest | 100 trees, no scaling needed |
| **Gradient Boosting** | **Best model**, 200 estimators |

### 4. Results
Gradient Boosting achieved the best performance:
- **R² ≈ 0.87+** on test set
- RMSE ~$120K
- Cross-validated with 5-fold CV for robustness

### 5. Key Findings
- `sqft_living`, `grade`, and `lat` (location) are the most impactful features
- Waterfront properties command a ~3x premium
- Properties with grade ≥ 10 are almost exclusively in the premium segment
- Geographic location (lat/long/zipcode) is highly predictive

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
jupyter notebook king_county_analysis.ipynb
```
