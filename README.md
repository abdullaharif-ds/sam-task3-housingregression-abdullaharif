# 🏠 Simple Linear Regression on Housing

**SAM AI Technologies Data Science Internship — Task 3**

## 📌 Problem Statement

Build a linear regression model to predict house prices based on features like number of rooms, income, and other housing characteristics, using proper data preprocessing (feature selection and normalization).

## 📊 Dataset Details

- **Name:** California Housing Prices Dataset
- **Note on dataset choice:** the task references "a housing prices dataset (like the Boston Housing dataset)." The classic Boston Housing dataset was removed from scikit-learn and is no longer distributed due to ethical concerns about one of its features (it encoded a racially-motivated variable). **California Housing is the standard, actively-maintained modern replacement** for exactly this kind of regression exercise, so it was used here instead.
- **Size:** 20,640 district-level records
- **Features used:** `total_rooms`, `total_bedrooms`, `population`, `households`, `median_income`
- **Target:** `median_house_value`
- **Data quality:** 207 missing values (~1%) in `total_bedrooms`, handled via median imputation

## 🧠 Approach

1. **Feature selection:** Chose five numeric features most relevant to predicting house value — including room/bedroom counts as the task specifically asks for.
2. **Preprocessing:** Imputed missing values with the median, then applied `StandardScaler` to **normalize** all features to zero mean / unit variance (required by the task).
3. **Simple linear regression:** Built a true single-feature regression using `median_income` (the feature most correlated with price) as the predictor, with the fitted line plotted directly against the actual data.
4. **Multi-feature linear regression:** Extended to all five selected, normalized features, to demonstrate how much predictive power is gained over the single-feature version.
5. **Evaluation:** Compared both models using RMSE and R² Score on a held-out 20% test set.

## 📈 Results

| Model | RMSE | R² Score |
|---|---|---|
| Simple (median_income only) | $84,209 | 0.4589 |
| Multi-feature (5 features) | $79,802 | 0.5140 |

- The simple single-feature model already captures a substantial portion of the price variation (R² = 0.46), confirming income is a strong standalone predictor.
- Adding four more features improves R² by about 5.5 percentage points and reduces RMSE by roughly $4,400 — a real but modest gain, showing that income dominates the other selected features in predictive power.
- Among the multi-feature coefficients (on normalized scale), `median_income` had by far the largest positive coefficient, while `total_rooms` and `population` had negative coefficients once income was accounted for — an interesting effect that likely reflects densely-populated, smaller-home urban areas.

## 🖼️ Output

See the `output/` folder for:
- `1_simple_regression_income_vs_price.png` — single-feature regression with fitted line
- `2_multi_feature_predicted_vs_actual.png` — multi-feature model predictions vs. actual values

## 📦 Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## 🚀 Usage

```bash
pip install -r requirements.txt
python housing_regression.py
```

## 🏗️ Project Structure

```
sam-task3-housingregression-abdullaharif/
│
├── housing_regression.py
├── housing.csv
├── requirements.txt
├── README.md
└── output/
    ├── 1_simple_regression_income_vs_price.png
    └── 2_multi_feature_predicted_vs_actual.png
```

## 👤 Author

**Abdullah Arif** — SAM AI Technologies Data Science Internship

---
*Tagged: @SAM AI Technologies*
