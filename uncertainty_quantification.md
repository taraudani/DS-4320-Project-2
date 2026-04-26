```python

import pandas as pd

#load the clean data
df = pd.read_csv("pjm_demand_clean.csv", parse_dates=["datetime"])

#add hour column for hourly-level stats
df["hour"] = df["datetime"].dt.hour

#daily top-level stats
df["date"] = df["datetime"].dt.date
daily = df.groupby("date")["demand_mw"].agg(
    daily_avg="mean",
    daily_peak="max"
).reset_index()

print("Daily Uncertainty")
print(daily[["daily_avg", "daily_peak"]].describe().round(1))

#hourly features
print("Hourly Uncertainty")
print(df[["demand_mw", "solar_mw", "wind_mw"]].describe().round(1))

#null rates
print("Null Rates")
for col in ["demand_mw", "solar_mw", "wind_mw"]:
    null_pct = df[col].isna().sum() / len(df) * 100
    print(f"{col}: {null_pct:.2f}% null")

```
