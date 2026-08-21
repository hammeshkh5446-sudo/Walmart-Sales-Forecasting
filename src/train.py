import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# Load Dataset
# =========================

df = pd.read_csv("data/Walmart.csv")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Sort by Store and Date
df = df.sort_values(["Store", "Date"])


# =========================
# Feature Engineering
# =========================

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Previous week sales
df["Previous_Week_Sales"] = (
    df.groupby("Store")["Weekly_Sales"].shift(1)
)

# Sales from 4 weeks ago
df["Previous_4_Weeks_Sales"] = (
    df.groupby("Store")["Weekly_Sales"].shift(4)
)

# Remove rows where lag features are unavailable
df = df.dropna(
    subset=[
        "Previous_Week_Sales",
        "Previous_4_Weeks_Sales"
    ]
).copy()


print("Dataset shape:", df.shape)
print(
    "Date range:",
    df["Date"].min(),
    "to",
    df["Date"].max()
)


# =========================
# Time-Based Train/Test Split
# =========================

split_date = df["Date"].quantile(0.80)

train = df[df["Date"] <= split_date].copy()
test = df[df["Date"] > split_date].copy()

print("\nSplit Date:", split_date)

print("\nTraining set:")
print(train.shape)
print(train["Date"].min(), "to", train["Date"].max())

print("\nTest set:")
print(test.shape)
print(test["Date"].min(), "to", test["Date"].max())


# =========================
# Features and Target
# =========================

features = [
    "Store",
    "Holiday_Flag",
    "Temperature",
    "Fuel_Price",
    "CPI",
    "Unemployment",
    "Year",
    "Month",
    "Week",
    "Previous_Week_Sales",
    "Previous_4_Weeks_Sales",
]

target = "Weekly_Sales"

X_train = train[features]
y_train = train[target]

X_test = test[features]
y_test = test[target]

print("\nX_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# =========================
# Baseline
# =========================

baseline_predictions = X_test["Previous_Week_Sales"]

baseline_mae = mean_absolute_error(
    y_test,
    baseline_predictions
)

print("\nBaseline MAE:", baseline_mae)


# =========================
# Random Forest
# =========================

rf_model = RandomForestRegressor(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# =========================
# Save Model
# =========================

model_data = {
    "model": rf_model,
    "features": features
}

joblib.dump(
    model_data,
    "models/walmart_sales_model.joblib"
)

print("\nModel saved successfully!")

rf_predictions = rf_model.predict(X_test)

rf_mae = mean_absolute_error(
    y_test,
    rf_predictions
)

print("\nRandom Forest MAE:", rf_mae)


# =========================
# Linear Regression
# =========================

linear_model = LinearRegression()

linear_model.fit(X_train, y_train)

linear_predictions = linear_model.predict(X_test)

linear_mae = mean_absolute_error(
    y_test,
    linear_predictions
)

print("\nLinear Regression MAE:", linear_mae)

# =========================
# Random Forest Evaluation
# =========================

rf_rmse = mean_squared_error(
    y_test,
    rf_predictions
) ** 0.5

rf_r2 = r2_score(
    y_test,
    rf_predictions
)

print("\nRandom Forest Evaluation:")
print("MAE:", rf_mae)
print("RMSE:", rf_rmse)
print("R2 Score:", rf_r2)

# =========================
# Feature Importance
# =========================

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\nFeature Importance:")
print(feature_importance)
