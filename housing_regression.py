"""
Housing Price Prediction using Linear Regression
==================================================
Steps: Data Exploration -> Cleaning -> Feature Selection -> Model Training
       -> Evaluation -> Visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

# ----------------------------------------------------------------------
# 1. DATA COLLECTION
# ----------------------------------------------------------------------
df = pd.read_csv("data/Housing.csv")
print("Dataset shape:", df.shape)
print(df.head())

# ----------------------------------------------------------------------
# 2. DATA EXPLORATION & CLEANING
# ----------------------------------------------------------------------
print("\nMissing values per column:\n", df.isnull().sum())
print("\nDuplicate rows:", df.duplicated().sum())
df = df.drop_duplicates()

# Encode binary yes/no columns
binary_cols = ["mainroad", "guestroom", "basement", "hotwaterheating",
                "airconditioning", "prefarea"]
for col in binary_cols:
    df[col] = df[col].map({"yes": 1, "no": 0})

# One-hot encode furnishingstatus (3 categories)
df = pd.get_dummies(df, columns=["furnishingstatus"], drop_first=True)
bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

print("\nCleaned dataset info:")
print(df.dtypes)

# Correlation with target
corr = df.corr(numeric_only=True)["price"].sort_values(ascending=False)
print("\nCorrelation with price:\n", corr)

# ----------------------------------------------------------------------
# 3. FEATURE SELECTION
# ----------------------------------------------------------------------
# Use all engineered features; area, bathrooms, stories, airconditioning etc.
# show the strongest correlation with price (see corr above).
features = [c for c in df.columns if c != "price"]
X = df[features]
y = df["price"]

# ----------------------------------------------------------------------
# 4. MODEL TRAINING
# ----------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ----------------------------------------------------------------------
# 5. MODEL EVALUATION
# ----------------------------------------------------------------------
y_pred_train = model.predict(X_train_scaled)
y_pred_test = model.predict(X_test_scaled)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test)

print("\n===== MODEL PERFORMANCE =====")
print(f"Train R^2: {train_r2:.4f}")
print(f"Test  R^2: {test_r2:.4f}")
print(f"Test  MSE: {test_mse:,.0f}")
print(f"Test  RMSE: {test_rmse:,.0f}")
print(f"Test  MAE: {test_mae:,.0f}")

coef_df = pd.DataFrame({
    "feature": features,
    "coefficient": model.coef_
}).sort_values("coefficient", key=abs, ascending=False)
print("\nFeature coefficients (standardized):\n", coef_df)

# ----------------------------------------------------------------------
# 6. VISUALIZATION
# ----------------------------------------------------------------------

# (a) Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm",
            cbar_kws={"label": "correlation"})
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("outputs/plots/01_correlation_heatmap.png")
plt.close()

# (b) Actual vs Predicted scatter
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred_test, alpha=0.6, edgecolor="k", color="#4C72B0")
lims = [min(y_test.min(), y_pred_test.min()), max(y_test.max(), y_pred_test.max())]
plt.plot(lims, lims, "r--", label="Perfect Prediction")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title(f"Actual vs Predicted Price (Test Set)\nR² = {test_r2:.3f}")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/plots/02_actual_vs_predicted.png")
plt.close()

# (c) Residual plot
residuals = y_test - y_pred_test
plt.figure(figsize=(7, 6))
plt.scatter(y_pred_test, residuals, alpha=0.6, edgecolor="k", color="#55A868")
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Price")
plt.ylabel("Residual (Actual - Predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("outputs/plots/03_residual_plot.png")
plt.close()

# (d) Residual distribution
plt.figure(figsize=(7, 5))
sns.histplot(residuals, kde=True, color="#C44E52")
plt.xlabel("Residual")
plt.title("Distribution of Residuals")
plt.tight_layout()
plt.savefig("outputs/plots/04_residual_distribution.png")
plt.close()

# (e) Feature coefficient importance
plt.figure(figsize=(8, 6))
colors = ["#4C72B0" if v > 0 else "#C44E52" for v in coef_df["coefficient"]]
plt.barh(coef_df["feature"], coef_df["coefficient"], color=colors)
plt.xlabel("Standardized Coefficient")
plt.title("Feature Importance (Linear Regression Coefficients)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/plots/05_feature_importance.png")
plt.close()

print("\nAll plots saved to outputs/plots/")

# Save cleaned dataset & metrics for reporting
df.to_csv("outputs/cleaned_housing.csv", index=False)
with open("outputs/metrics.txt", "w") as f:
    f.write(f"train_r2={train_r2:.4f}\n")
    f.write(f"test_r2={test_r2:.4f}\n")
    f.write(f"test_mse={test_mse:.2f}\n")
    f.write(f"test_rmse={test_rmse:.2f}\n")
    f.write(f"test_mae={test_mae:.2f}\n")
coef_df.to_csv("outputs/coefficients.csv", index=False)