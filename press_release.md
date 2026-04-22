# Eastern U.S. Electricity Grid (PJM) Demand Forecasting By ML Model Unlocks New Energy Strategy

## Hook
Every hour, grid operators across 13 states make high-stakes decisions about how much electricity to generate and from which sources. They also have to make those calls a full day in advance, which means there is a fair potential to make a mistake. Getting it wrong wastes clean energy or triggers expensive, polluting backup plants. A new machine learning model aims to make those calls significantly more accurate.

## Problem Statement

The PJM Interconnection is the largest competitive electricity market in North America, and serves 65 million people, coordinating hundreds of power plants across 13 states. As wind and solar now account for a growing share of PJM's generation mix, the day-ahead scheduling process has become increasingly difficult. Renewable sources can't be dialed up on demand, and if operators overcommit to renewables and demand comes in lower than expected, clean energy gets wasted. If demand spikes unexpectedly, expensive natural gas peaker plants must fire up within minutes. Current forecasting tools struggle most during the hours when renewable output is highest, which is during midday solar peaks and overnight wind surges, precisely when accurate forecasting matters most. The cost of these forecasting errors runs into the millions of dollars annually, and the carbon cost is substantial.


## Solution Description

Using three years of publicly available hourly demand data from the EIA-930 dataset combined with historical weather records, a machine learning model was trained to predict PJM-wide electricity demand for each of the next 24 hours. The model learns recurring patterns in how demand responds to temperature, time of day, day of week, and season, as well as how current renewable generation levels should inform expectations about the hours ahead. Trained on 2023-2024 data and evaluated on 2025, the model produces a full next-day demand curve each morning, the output grid operators need before committing to the day-ahead market.

## Chart
