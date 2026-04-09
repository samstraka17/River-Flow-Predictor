# River Flow Prediction Agent

## Project Overview
Predicts future river flow conditions for the upper Arkansas River basin using 
historical snowpack, river flow, and weather data.

## Data Sources
- USGS stream gauge data (waterservices.usgs.gov, and https://api.waterdata.usgs.gov/statistics/v0/)
- NRCS SNOTEL snowpack data (https://wcc.sc.egov.usda.gov/awdbRestApi/, and wcc.sc.egov.usda.gov)
- NOAA weather data (api.weather.gov)

## Conventions
- Use pandas DataFrames for all data handling
- Date range: 1990–present
- Prediction horizon: 30 days out