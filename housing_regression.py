"""
Task 3 — Simple Linear Regression on Housing
------------------------------------------------
Uses a housing prices dataset (California Housing — used here in place of
the older, now-deprecated Boston Housing dataset, which was removed from
scikit-learn due to ethical concerns about one of its features; California
Housing is the standard modern replacement for this exact kind of exercise).

Performs data preprocessing (feature selection, normalization) and builds a
linear regression model to predict house prices based on features like
number of rooms and other housing characteristics.

Author: Abdullah Arif
Internship: SAM AI Technologies — Data Science Internship
Task: Task 3
Dataset: California Housing Prices
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42


def load_data() -> pd.DataFrame:
    return pd.read_csv(os.path.join(BASE_DIR, "housing.csv"))


def print_summary(df: pd.DataFrame):
    print("=" * 55)
    print("CALIFORNIA HOUSING DATASET SUMMARY")
    print("=" * 55)
    print(f"\nShape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nMissing values per column:")
    print(df.isnull().sum())
    print("\nFirst 5 rows:")
    print(df.head())
    print()


def preprocess(df: pd.DataFrame):
    """
    Feature selection + normalization, as required by the task.
    Selects rooms, bedrooms, population, households, and income as
    predictors of house value, imputes missing values, and standardizes
    (normalizes) all features to zero mean / unit variance.
    """
    features = ["total_rooms", "total_bedrooms", "population", "households", "median_income"]
    target = "median_house_value"

    X = df[features].copy()
    y = df[target].copy()

    imputer = SimpleImputer(strategy="median")
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=features)

    return X_scaled, y, features


def plot_simple_regression(df: pd.DataFrame):
    """
    A true SIMPLE linear regression: one predictor (median_income, the
    single feature most correlated with price) against the target,
    with the fitted regression line drawn directly on the scatter plot.
    """
    X_simple = df[["median_income"]].values
    y = df["median_house_value"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X_simple, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("--- Simple Linear Regression (median_income -> median_house_value) ---")
    print(f"Coefficient (slope): {model.coef_[0]:.2f}")
    print(f"Intercept: {model.intercept_:.2f}")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R^2 Score: {r2:.4f}\n")

    plt.figure(figsize=(8, 6))
    plt.scatter(X_test, y_test, alpha=0.25, s=15, color="#4C72B0", label="Actual data")
    x_line = np.linspace(X_simple.min(), X_simple.max(), 100).reshape(-1, 1)
    y_line = model.predict(x_line)
    plt.plot(x_line, y_line, color="red", linewidth=2.5, label="Regression line")
    plt.title("Simple Linear Regression: Median Income vs. House Value", fontsize=13, fontweight="bold")
    plt.xlabel("Median Income (tens of thousands $)")
    plt.ylabel("Median House Value ($)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "1_simple_regression_income_vs_price.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved: {path}")

    return rmse, r2


def plot_multi_feature_regression(X_scaled, y, features):
    """
    Extends to a multi-feature linear regression using all selected,
    normalized features — showing how much predictive power is gained
    over the single-feature version above.
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=RANDOM_STATE
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print("--- Multi-Feature Linear Regression (all selected, normalized features) ---")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"R^2 Score: {r2:.4f}\n")

    print("Feature coefficients (on normalized scale):")
    for feature, coef in zip(features, model.coef_):
        print(f"  {feature:<20}: {coef:>10.2f}")

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, predictions, alpha=0.25, s=15, color="#55A868")
    max_val = max(y_test.max(), predictions.max())
    plt.plot([0, max_val], [0, max_val], color="red", linestyle="--", label="Perfect prediction")
    plt.title("Multi-Feature Regression: Predicted vs. Actual", fontsize=13, fontweight="bold")
    plt.xlabel("Actual House Value ($)")
    plt.ylabel("Predicted House Value ($)")
    plt.legend()
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "2_multi_feature_predicted_vs_actual.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"\nSaved: {path}")

    return rmse, r2


def main():
    df = load_data()
    print_summary(df)

    simple_rmse, simple_r2 = plot_simple_regression(df)

    X_scaled, y, features = preprocess(df)
    multi_rmse, multi_r2 = plot_multi_feature_regression(X_scaled, y, features)

    print("\n" + "=" * 55)
    print("COMPARISON: Simple vs. Multi-Feature Regression")
    print("=" * 55)
    print(f"{'Model':<25}{'RMSE':>15}{'R2':>10}")
    print(f"{'Simple (1 feature)':<25}{'$' + format(simple_rmse, ',.0f'):>15}{simple_r2:>10.4f}")
    print(f"{'Multi-feature (5)':<25}{'$' + format(multi_rmse, ',.0f'):>15}{multi_r2:>10.4f}")
    print("\nAll visualizations generated successfully in the 'output/' folder.")


if __name__ == "__main__":
    main()
