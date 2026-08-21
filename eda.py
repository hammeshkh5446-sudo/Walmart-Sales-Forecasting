import pandas as pd

df = pd.read_csv("data/Walmart.csv")

df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)

print(df.head())
print("\nData type of Date:")
print(df["Date"].dtype)

print("\nWeekly Sales Summary:")
print(df["Weekly_Sales"].describe())

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.hist(df["Weekly_Sales"], bins=30)
plt.xlabel("Weekly Sales")
plt.ylabel("Frequency")
plt.title("Distribution of Weekly Sales")
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

weekly_sales = df.groupby("Date")["Weekly_Sales"].sum()

plt.plot(weekly_sales.index, weekly_sales.values)

plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.title("Total Weekly Sales Over Time")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

holiday_sales = df.groupby("Holiday_Flag")["Weekly_Sales"].mean()

print("\nAverage Weekly Sales by Holiday Flag:")
print(holiday_sales)

store_sales = df.groupby("Store")["Weekly_Sales"].mean().sort_values(ascending=False)

print("\nAverage Weekly Sales by Store:")
print(store_sales)

plt.figure(figsize=(12, 6))

store_sales.sort_values().plot(kind="bar")

plt.xlabel("Store")
plt.ylabel("Average Weekly Sales")
plt.title("Average Weekly Sales by Store")
plt.tight_layout()
plt.show()

print("\nCorrelation with Weekly Sales:")
print(
    df[
        [
            "Weekly_Sales",
            "Temperature",
            "Fuel_Price",
            "CPI",
            "Unemployment",
        ]
    ].corr()["Weekly_Sales"].sort_values(ascending=False)
)

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

print("\nDate Features:")
print(df[["Date", "Year", "Month"]].head())

monthly_sales = df.groupby("Month")["Weekly_Sales"].mean()

print("\nAverage Weekly Sales by Month:")
print(monthly_sales)