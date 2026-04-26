# DS-4320-Project-2

### Executive Summary 
This repository contains the full pipeline for a machine learning project that predicts hourly electricity demand on the PJM Interconnection 24 hours in advance. Using three years of publicly available hourly data from the U.S. Energy Information Administration's EIA-930 dataset, the project builds a MongoDB document database of daily demand records, engineers a set of time-based and lag features, and trains an XGBoost model for prediction. The repository includes the data creation and cleaning pipeline, MongoDB upload and query code, the full model training and evaluation workflow, model predictions stored back to a second MongoDB collection, and a visualization of forecast accuracy. All materials are written in Python and Markdown following the DS 4320 project standards. 

**Name** - Tara Udani

**NetID** - hav7tz

**DOI** - [![DOI]()

**Press release** - [Link to Press Release](https://github.com/taraudani/DS-4320-Project-2/blob/b18096cfc5d453dab2bf010f9e406e3a67d8c28f/press_release.md)

**Data** - [Link to data folder](https://myuva-my.sharepoint.com/:f:/r/personal/hav7tz_virginia_edu/Documents/DS%204320%20Project%202/Data?csf=1&web=1&e=nTiHlx)

**Pipeline** - [Link to Pipeline Folder](https://github.com/taraudani/DS-4320-Project-2/tree/2e3aa440de3e02b023fc5a30eea9b98a70051af3/pipeline)

**License** - [MIT License](https://github.com/taraudani/DS-4320-Project-2/blob/2e3aa440de3e02b023fc5a30eea9b98a70051af3/LICENSE)

## Problem Definition

### General and Specific Problem:

- **General Problem:**

Electricity grids must continuously balance supply and demand in real time, making predictions for usage to understand the balance. Inaccurate demand forecasts lead to either wasted energy or dangerous supply shortfalls that risk grid instability and blackouts, especially with renewable sources.

- **Specific Problem:**

Can hourly electricity demand for the PJM Interconnection be predicted 24 hours in advance using historical load data and time-based patterns to support day-ahead energy scheduling decisions?

### Motivation:
The United States electricity grid is undergoing a fundamental transformation. As solar and wind generation capacity expands rapidly across the PJM region, which serves roughly 65 million people across 13 states on the East Coast, grid operators face an increasingly complex need to balance supply and demand. Unlike coal or natural gas plants, renewable sources cannot be turned up on demand, as their output depends entirely on weather conditions (sun, wind, etc.). This means that accurate day-ahead demand forecasting is critical because if operators overestimate demand, clean renewable energy gets wasted, and if they underestimate it, they must spin up expensive and carbon-intensive plants on short notice. Better forecasting directly translates to lower electricity costs, reduced carbon emissions, and a more stable grid, making this problem both economically significant and environmentally urgent.

### Rationale:
The general problem of grid balancing is vast and involves generation planning, transmission constraints, and market pricing, which is far too broad for a focused data project. The scope was refined to PJM specifically because it is the largest competitive wholesale electricity market in the world, and publishes clean and complete hourly data through the EIA-930 dataset. It also  has a documented day-ahead scheduling process that gives our prediction horizon a concrete, real-world motivation. It was further narrowed to the 24-hour ahead horizon because it directly maps to how PJM's day-ahead energy market actually operates. Operators must commit generation resources the day before, making next-day hourly demand forecasts an actionable output to produce. This refinement gives the project a clear evaluation context, as our model succeeds if its 24-hour forecasts are accurate enough to meaningfully inform day-ahead renewable scheduling.

### Press Release Headline:
[Eastern U.S. Electricity Grid (PJM) Demand Forecasting By ML Model Unlocks New Energy Strategy](https://github.com/taraudani/DS-4320-Project-2/blob/b18096cfc5d453dab2bf010f9e406e3a67d8c28f/press_release.md)

## Domain Exposition

### Terminology:

| Term | Definition |
|------|------------|
| **PJM Interconnection** | The regional transmission organization managing the grid across 13 mid-Atlantic and midwest states |
| **Load / Demand (MW)** | The total amount of electricity being consumed at a given moment, measured in megawatts |
| **Day-Ahead Market** | Energy market where generators and buyers commit to supply/demand schedules 24 hours in advance |
| **Renewable Intermittency** | The variability of solar and wind generation due to weather, making output unpredictable |
| **MAPE** | Mean Absolute Percentage Error, the standard accuracy metric in energy forecasting |

### Domain:
This project lives at the intersection of energy systems engineering and applied machine learning. The electricity sector is one of the most operationally complex industries in the world, as the grid must balance supply and demand instantaneously, at massive scale and with almost zero tolerance for error. PJM Interconnection operates as both a grid manager and a competitive wholesale market, coordinating hundreds of generators, transmission lines, and utilities across states from Illinois to New Jersey. The rapid growth of renewable energy has introduced new forecasting challenges because solar and wind output fluctuates with weather in ways that dispatchable generators do not, making renewable energy yet another domain this project covers. Understanding these domains require familiarity with both the physical constraints of power systems and the economic mechanisms of electricity markets.

### Background Reading:
 [Link to OneDrive Folder with Readings](https://myuva-my.sharepoint.com/:f:/r/personal/hav7tz_virginia_edu/Documents/DS%204320%20Project%202/Readings?csf=1&web=1&e=jOpmEk)

### Reading Summary:
| Title | Description | Link |
|-------|-------------|------|
| EIA Hourly Electric Grid Monitor | Official documentation of the EIA-930 dataset including column definitions and methodology | [Link](https://myuva-my.sharepoint.com/:b:/r/personal/hav7tz_virginia_edu/Documents/DS%204320%20Project%202/Readings/Real-time%20Operating%20Grid%20-%20U.S.%20Energy%20Information%20Administration%20(EIA).pdf?csf=1&web=1&e=jBYN1A) |
| Short-Term Load Forecasting with ML | Academic overview of machine learning approaches to electricity demand forecasting, covering feature engineering and model benchmarks | [Link](https://myuva-my.sharepoint.com/:b:/r/personal/hav7tz_virginia_edu/Documents/DS%204320%20Project%202/Readings/Short%20Term%20Load%20Forecasting%20using%20Machine%20Learning%20Techniques%20_%20IEEE%20Conference%20Publication%20_%20IEEE%20Xplore.pdf?csf=1&web=1&e=Oic6JE) |
| Understanding the Differences Among PJM's Markets | PJM's description of how they generate day-ahead and real-time load forecasts for grid operations | [Link](https://myuva-my.sharepoint.com/:b:/r/personal/hav7tz_virginia_edu/Documents/DS%204320%20Project%202/Readings/understanding-the-difference-among-pjms-markets.pdf?csf=1&web=1&e=559v1K) |
| Confronting the Duck Curve: How to Address Over-Generation of Solar Energy | Explanation of how solar generation creates a characteristic demand curve shape that challenges grid operators in the afternoon | [Link](https://myuva-my.sharepoint.com/:b:/r/personal/hav7tz_virginia_edu/Documents/DS%204320%20Project%202/Readings/Confronting%20the%20Duck%20Curve_%20How%20to%20Address%20Over-Generation%20of%20Solar%20Energy%20_%20Department%20of%20Energy.pdf?csf=1&web=1&e=WxktsP) |
| Integrating Variable Renewable Energy Resources Into the Grid: Key Issues | NREL report detailing the operational and forecasting challenges introduced by high penetrations of wind and solar on regional grids | [Link](https://myuva-my.sharepoint.com/:b:/r/personal/hav7tz_virginia_edu/Documents/DS%204320%20Project%202/Readings/Integrating%20Variable%20Renewable%20ENergy.pdf?csf=1&web=1&e=WWy51K) |

## Data Creation

### Provenance:
The primary dataset for this project comes from the U.S. Energy Information
Administration's EIA-930 Hourly Electric Grid Monitor, which is a publicly available dataset updated hourly and downloadable as CSV files in six-month increments from eia.gov/electricity/gridmonitor. The data covers hourly electricity demand, net generation by fuel source, and interchange between regions for all major U.S. balancing authorities. For this project, six files covering January 2023 through December 2025 were downloaded and filtered all records to the PJM Interconnection balancing authority.

### Code:
| File | Description | Link |
|------|-------------|------|
| pjm_demand_clean.csv code | Code to transform 6 EIA-930 CSV files by concatenating and filtering to PJM balancing authority, parsing UTC datetime, selecting demand/solar/wind columns, and saving to `pjm_demand_clean.csv` | [Link](https://github.com/taraudani/DS-4320-Project-2/blob/1d8e7b59b73059ce874293a9235b5d8e31701e58/data_creation_code.md) |

### Rationale:
A 24-hour prediction horizon was chosen because it directly maps to PJM's day-ahead energy market, giving the model output a concrete operational use case. 2023-2024 were chosen as
training data and 2025 as the test set rather than a random split because
time-series data must be split chronologically, and random splitting would
allow the model to "see the future" during training and produce artificially
inflated accuracy scores. XGBoost was chosen as the primary model because it
handles tabular data with engineered lag features better than neural
networks at this data scale, and because its feature importance output
provides interpretable insight into which patterns the model relies on most. MAPE is used as the evaluation metric because it is the industry standard in
energy forecasting and expresses error as a percentage of actual demand,
making it intuitive to interpret regardless of the absolute MW scale.

### Bias Identification:
Several sources of bias could affect this dataset. First, the EIA-930 data
includes imputed and adjusted values for hours where direct metering data
was missing or anomalous, and these imputed values are estimates, not
measurements, so they could systematically understate demand variability during
extreme weather events when meters are most likely to fail. Second, the
2023-2025 window captures a period of post-COVID economic recovery, meaning
demand patterns may not represent long-run normal behavior and may not be entirely accurate predictors for a growing economy. Finally, PJM's grid footprint has changed over time as utilities join or leave the
interconnection, meaning the "PJM demand" figure in 2023 may not represent
exactly the same geographic area as in 2025.

### Bias Mitigation:
To mitigate bias, the raw "Demand (MW)" column is used rather
than the adjusted version where possible, and hours marked as imputed are flagged so they can be excluded from error metric calculations if needed. To address the economic recovery period bias, year is included as a feature to the model so it can learn any underlying trend rather than treating all years as
identical. Geographic changes to PJM's footprint are difficult to fully
correct for without detailed records, but their impact on hourly totals is small relative to seasonal and daily variation and therefore treated as acceptable noise to this project.

## Metadata

### Implicit Schema:
The EIA-930 PJM demand data is stored in MongoDB as one document per day
with 24 hourly readings embedded as a nested array. These are the guidelines to follow for all contributors:

**Top-Level Fields (required on every document):**
- `date` — string, format YYYY-MM-DD, must be unique in the collection
- `year` — integer, four digits
- `month` — integer, 1-12
- `dow` — integer, 0-6 (0=Monday, 6=Sunday, day of week)
- `is_holiday` — integer, must be exactly 0 or 1, never null
- `daily_avg_demand_mw` — float, rounded to 1 decimal, never null
- `daily_peak_demand_mw` — float, rounded to 1 decimal, never null
- `hours` — array, must always contain exactly 24 objects sorted by hour

**Nested Fields (required inside each object in the hours array):**
- `hour` — integer, 0-23, must appear exactly once per document
- `demand_mw` — float, never null
- `solar_mw` — float, null permitted only for missing sensor data
- `wind_mw` — float, null permitted only for missing sensor data

**Naming Rules:**
- All field names must be lowercase_with_underscores, no camelCase or
  capitalized variants (ex. `demand_mw` not `DemandMW`)
- The `hours` array must always be sorted ascending (hour 0 first)
- No new fields may be added without updating this guide

### Data Summary:
| Collection | Documents | Date Range | Hourly Readings | Purpose |
|------------|-----------|------------|-----------------|---------|
| `pjm_demand` | ~1,095 | Jan 2023 - Dec 2025 | ~26,280 total (24 per document) | Raw cleaned EIA data used for exploration and model training |
| `pjm_predictions` | ~365 | Jan 2025 - Dec 2025 | ~8,760 total (24 per document) | Model output, predicted vs actual demand for 2025 test year |

### Data Dictionary:

 | Feature | Type | Description | Example | Mean | Std Dev | Min | Max | Null Rate | Notes |
|---------|------|-------------|---------|------|---------|-----|-----|-----------|-------|
| `date` | string | Calendar date in YYYY-MM-DD (UTC) | `"2024-07-15"` | — | — | — | — | 0% | Unique per document |
| `year` | integer | Four-digit calendar year | `2024` | — | — | — | — | 0% | — |
| `month` | integer | Month of year (1–12) | `7` | — | — | — | — | 0% | — |
| `dow` | integer | Day of week, 0=Monday through 6=Sunday | `0` | — | — | — | — | 0% | — |
| `is_holiday` | integer | 1 if US federal holiday, 0 otherwise | `0` | — | — | — | — | 0% | — |
| `hour` | integer | UTC hour of day, 0–23 | `14` | — | — | — | — | 0% | Nested inside hours array |
| `daily_avg_demand_mw` | float | Mean of all 24 hourly demand values for the day (MW) | `92,896` | 92,896 MW | 12,564 MW | 70,853 MW | 136,916 MW | 0% | Moderate variance driven by seasonal heating and cooling cycles |
| `daily_peak_demand_mw` | float | Maximum single-hour demand value for the day (MW) | `105,745` | 105,745 MW | 16,770 MW | 78,226 MW | 160,560 MW | 0% | Higher variance than daily average; extreme peaks driven by heat waves and cold snaps |
| `demand_mw` | float | Total PJM grid consumption at this hour (MW) | `92,933` | 92,933 MW | 15,507 MW | 56,260 MW | 160,560 MW | 0% | Most reliable feature; highest variance at morning ramp (7–9am) and evening peak (5–8pm) |
| `solar_mw` | float | Net PJM solar generation at this hour (MW) | `1,253` | 1,253 MW | 1,732 MW | 0 MW | 7,672 MW | 50.09% | Zero every night; high null rate due to missing EIA source data for certain periods |
| `wind_mw` | float | Net PJM wind generation at this hour (MW) | `3,605` | 3,605 MW | 2,320 MW | 107 MW | 9,891 MW | 50.09% | More consistent than solar; null rate matches solar due to same missing data windows |

**Note:** The 50% null rate for `solar_mw` and `wind_mw` reflects missing data in the raw EIA-930
source files for certain time periods, not a data collection error. These features are used only
when available and rows with null generation values are excluded from any features that depend on them.
