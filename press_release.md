# Eastern U.S. Electricity Grid (PJM) Demand Accurately Forecasted by AI Model

## Hook
Every hour, grid operators across 13 states make high-stakes decisions about how much electricity to generate and from which sources. They also have to make those calls a full day in advance, which means there is a fair potential to make a mistake. Getting it wrong wastes clean energy or triggers expensive, polluting backup plants. A new machine learning model aims to make those calls significantly more accurate.

## Problem Statement

The PJM Interconnection is the largest competitive electricity market in North America, and serves 65 million people, coordinating hundreds of power plants across 13 states. As wind and solar now account for a growing share of PJM's generation mix, the day-ahead scheduling process has become increasingly difficult. Renewable sources can't be dialed up on demand, and if operators overcommit to renewables and demand comes in lower than expected, clean energy gets wasted. If demand spikes unexpectedly, expensive natural gas peaker plants must fire up within minutes. Current forecasting tools struggle most during the hours when renewable output is highest, which is during midday solar peaks and overnight wind surges, precisely when accurate forecasting matters most. The cost of these forecasting errors runs into the millions of dollars annually, and the carbon cost is substantial.


## Solution Description

Using three years of publicly available hourly demand data from the EIA-930 dataset combined with historical weather records, a machine learning model was trained to predict PJM-wide electricity demand for each of the next 24 hours. The model learns recurring patterns in how demand responds to temperature, time of day, day of week, and season, as well as how current renewable generation levels should inform expectations about the hours ahead. Trained on 2023-2024 data and evaluated on 2025, the model produces a full next-day demand curve each morning, the output grid operators need before committing to the day-ahead market.

## Chart
<img width="3570" height="1573" alt="press_release_chart" src="https://github.com/user-attachments/assets/a4bca643-cce5-466e-b6a3-fd212f14aa71" />

This chart shows one representative week of the model's day-ahead demand forecasts against actual PJM grid consumption in July 2025, the most demanding period of the test year. Each point on the predicted line was generated using only information available 24 hours prior, no same-day data was used. The shaded red band represents the gap between forecast and reality, which remains narrow across the full week despite daily demand changing by over 30,000 MW between overnight lows and afternoon peaks. The model achieves a test-set MAPE of 3.30%, meaning its hourly forecasts deviate from actual demand by about 3,300 MW on average, which is well within the threshold needed to support day-ahead renewable energy scheduling decisions.
