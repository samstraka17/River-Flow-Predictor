"""NOAA weather data fetcher for the upper Arkansas River basin.

Two data sources:
  - api.weather.gov  : 7-day point forecasts (no API key needed)
  - NOAA CDO API     : historical daily observations (requires free API token)
                       Set env var NOAA_CDO_TOKEN to enable historical fetch.
                       Token registration: https://www.ncdc.noaa.gov/cdo-web/token
"""

import os
import warnings
import requests
import pandas as pd
from datetime import date, timedelta
from typing import Optional

# Coordinates for the Salida, CO area (central upper Arkansas basin)
DEFAULT_LAT = 38.535
DEFAULT_LON = -105.999

# NOAA GHCND station IDs near the Arkansas basin
ARKANSAS_WEATHER_STATIONS: dict[str, str] = {
    "USW00093058": "Pueblo Memorial Airport, CO",
    "USC00057936": "Salida, CO",
    "USW00003017": "Leadville/Lake County Airport, CO",
}
DEFAULT_CDO_STATION = "GHCND:USW00093058"  # Pueblo – longest record

_POINTS_URL  = "https://api.weather.gov/points/{lat},{lon}"
_CDO_URL     = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"
_OBS_URL     = "https://api.weather.gov/stations/{station_id}/observations"


# ---------------------------------------------------------------------------
# Forecast (no auth required)
# ---------------------------------------------------------------------------

def fetch_forecast(
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
) -> pd.DataFrame:
    """Fetch the NWS 7-day gridpoint forecast for a lat/lon.

    Returns
    -------
    DataFrame indexed by date with columns:
        temp_max_f, temp_min_f, precip_chance_pct, short_forecast
    One row per calendar day (up to ~7 days out).
    """
    # Step 1: resolve grid info from coordinates
    meta = requests.get(
        _POINTS_URL.format(lat=lat, lon=lon),
        headers={"User-Agent": "RiverFlowPredictor/1.0"},
        timeout=15,
    )
    meta.raise_for_status()
    props = meta.json()["properties"]
    forecast_url = props["forecast"]

    # Step 2: fetch the actual forecast
    fc = requests.get(
        forecast_url,
        headers={"User-Agent": "RiverFlowPredictor/1.0"},
        timeout=15,
    )
    fc.raise_for_status()
    periods = fc.json()["properties"]["periods"]

    # Collapse day/night period pairs into single daily rows
    daily: dict[str, dict] = {}
    for period in periods:
        day_str = period["startTime"][:10]
        row = daily.setdefault(day_str, {"temp_max_f": float("nan"), "temp_min_f": float("nan"),
                                          "precip_chance_pct": 0.0, "short_forecast": ""})
        temp = period.get("temperature", float("nan"))
        if period.get("isDaytime", True):
            row["temp_max_f"] = temp
            row["short_forecast"] = period.get("shortForecast", "")
        else:
            row["temp_min_f"] = temp

        # probabilityOfPrecipitation may be None
        pop = (period.get("probabilityOfPrecipitation") or {}).get("value") or 0
        row["precip_chance_pct"] = max(row["precip_chance_pct"], float(pop))

    df = pd.DataFrame.from_dict(daily, orient="index")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df.sort_index()


# ---------------------------------------------------------------------------
# Historical weather (NOAA CDO, requires free API token)
# ---------------------------------------------------------------------------

def fetch_historical_weather(
    station_id: str = DEFAULT_CDO_STATION,
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
    token: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily historical weather observations from NOAA CDO.

    Requires a free API token from https://www.ncdc.noaa.gov/cdo-web/token
    Pass it via the ``token`` argument or set the ``NOAA_CDO_TOKEN``
    environment variable.

    Parameters
    ----------
    station_id: GHCND station identifier (e.g. ``GHCND:USW00093058``).
    start_date: ISO date string, inclusive.
    end_date:   ISO date string, inclusive (defaults to yesterday).
    token:      CDO API token.  Falls back to ``NOAA_CDO_TOKEN`` env var.

    Returns
    -------
    DataFrame indexed by date with columns:
        tmax_f, tmin_f, prcp_in
    Returns an empty DataFrame with a warning if no token is available.
    """
    if end_date is None:
        end_date = (date.today() - timedelta(days=1)).isoformat()

    token = token or os.environ.get("NOAA_CDO_TOKEN")
    if not token:
        warnings.warn(
            "NOAA_CDO_TOKEN is not set – historical weather data unavailable. "
            "Get a free token at https://www.ncdc.noaa.gov/cdo-web/token",
            stacklevel=2,
        )
        return pd.DataFrame(columns=["tmax_f", "tmin_f", "prcp_in"])

    headers = {"token": token}
    # CDO limits responses to 1000 records per request; paginate by year
    start = pd.to_datetime(start_date)
    end   = pd.to_datetime(end_date)

    all_records: list[dict] = []
    chunk_start = start
    while chunk_start <= end:
        chunk_end = min(chunk_start + pd.DateOffset(years=1) - pd.Timedelta(days=1), end)
        params = {
            "datasetid":  "GHCND",
            "stationid":  station_id,
            "startdate":  chunk_start.date().isoformat(),
            "enddate":    chunk_end.date().isoformat(),
            "datatypeid": "TMAX,TMIN,PRCP",
            "units":      "standard",   # Fahrenheit / inches
            "limit":      1000,
        }
        resp = requests.get(_CDO_URL, headers=headers, params=params, timeout=30)
        if resp.status_code == 429:
            warnings.warn("[noaa] CDO rate limit hit – partial data returned.")
            break
        resp.raise_for_status()
        results = resp.json().get("results", [])
        all_records.extend(results)
        chunk_start = chunk_end + pd.Timedelta(days=1)

    if not all_records:
        return pd.DataFrame(columns=["tmax_f", "tmin_f", "prcp_in"])

    raw = pd.DataFrame(all_records)
    raw["date"] = pd.to_datetime(raw["date"].str[:10])

    pivot = raw.pivot_table(index="date", columns="datatype", values="value", aggfunc="first")
    pivot = pivot.rename(columns={"TMAX": "tmax_f", "TMIN": "tmin_f", "PRCP": "prcp_in"})
    pivot.index.name = "date"

    for col in ["tmax_f", "tmin_f", "prcp_in"]:
        if col not in pivot.columns:
            pivot[col] = float("nan")

    return pivot[["tmax_f", "tmin_f", "prcp_in"]].sort_index()
