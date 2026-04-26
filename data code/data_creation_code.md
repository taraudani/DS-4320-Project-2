```python
import pandas as pd
import logging

#configuring logging for data creation
logging.basicConfig(
    filename="pipeline.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

urls = [
    "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jan_Jun.csv",
    "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jul_Dec.csv",
    "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jan_Jun.csv",
    "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2024_Jul_Dec.csv",
    "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2025_Jan_Jun.csv",
    "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2025_Jul_Dec.csv",
]

dfs = []
for url in urls:
    try:
        print(f"Loading {url.split('/')[-1]}...")
        df = pd.read_csv(url, low_memory=False)
        dfs.append(df)
        logging.info(f"Loaded {url.split('/')[-1]}: {len(df):,} rows")
    except Exception as e:
        logging.error(f"Failed to load {url}: {e}")
        raise

raw = pd.concat(dfs, ignore_index=True)
print(f"Total rows before filter: {len(raw):,}")
logging.info(f"Total raw rows: {len(raw):,}")

#filter to PJM balancing authority only
pjm = raw[raw["Balancing Authority"] == "PJM"].copy()
print(f"PJM rows: {len(pjm):,}")

#parse datetime
pjm["datetime"] = pd.to_datetime(pjm["UTC Time at End of Hour"], utc=True)
pjm = pjm.sort_values("datetime").reset_index(drop=True)

#keeping only relevant columns
pjm_clean = pjm[[
    "datetime",
    "Demand (MW)",
    "Net Generation (MW) from Solar",
    "Net Generation (MW) from Wind"
]].copy()

pjm_clean.columns = ["datetime", "demand_mw", "solar_mw", "wind_mw"]
pjm_clean = pjm_clean.dropna(subset=["demand_mw"])

print(f"Clean rows: {len(pjm_clean):,}")
print(f"Date range: {pjm_clean.datetime.min()} to {pjm_clean.datetime.max()}")
logging.info(f"Clean rows: {len(pjm_clean):,}")

#save to csv file
pjm_clean.to_csv("pjm_demand_clean.csv", index=False)
```
