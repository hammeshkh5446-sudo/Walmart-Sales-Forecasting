import pandas as pd

# Load dataset
df = pd.read_csv("data/Walmart.csv")

# Convert Date to datetime
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

# Sort by Store and Date
df = df.sort_values(["Store", "Date"])

# -----------------------------
# Time-based Features
# -----------------------------

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week.astype(int)

# Year-Month for analysis/aggregation
df["Year_Month"] = df["Date"].dt.to_period("M")

# -----------------------------
# Lag Feature
# -----------------------------

df["Previous_Week_Sales"] = (
    df.groupby("Store")["Weekly_Sales"].shift(1)
)

# Check the engineered features
print("\nFeature Engineering Result:")
print(
    df[
        [
            "Store",
            "Date",
            "Weekly_Sales",
            "Year",
            "Month",
            "Week",
            "Year_Month",
            "Previous_Week_Sales",
        ]
    ].head(10)
)

print("\nMissing values after feature engineering:")
print(df.isnull().sum())

df = df.dropna(subset=["Previous_Week_Sales"]).copy()

print("\nFinal dataset shape:")
print(df.shape)
