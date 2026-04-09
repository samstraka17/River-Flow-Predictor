"""USGS NWIS daily streamflow fetcher for the upper Arkansas River basin.

Data source: https://waterservices.usgs.gov/nwis/dv/
Statistics API: https://api.waterdata.usgs.gov/statistics/v0/
"""

import requests
import pandas as pd
from datetime import date
from typing import Optional

# Key gauges in the upper Arkansas River basin, ordered upstream → downstream
ARKANSAS_GAUGES: dict[str, str] = {
    "07086000": "Arkansas River at Leadville, CO",
    "07091200": "Arkansas River at Granite, CO",
    "07096000": "Arkansas River at Parkdale, CO",
    "07099970": "Arkansas River near Avondale, CO",
}

# Default primary gauge (Granite sits above most major tributaries)
DEFAULT_GAUGE = "07091200"

_NWIS_URL = "https://waterservices.usgs.gov/nwis/dv/"
_DISCHARGE_CD = "00060"  # mean daily discharge, cfs
_MEAN_STAT = "00003"     # statistical code for daily mean


def fetch_flow(
    site_no: str = DEFAULT_GAUGE,
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily mean discharge (cfs) from a single USGS gauge.

    Parameters
    ----------
    site_no:    8-digit USGS site number.
    start_date: ISO date string, inclusive.
    end_date:   ISO date string, inclusive (defaults to today).

    Returns
    -------
    DataFrame indexed by date with columns [site_no, discharge_cfs].
    Rows with sentinel/negative values are removed.
    """
    if end_date is None:
        end_date = date.today().isoformat()

    params = {
        "format": "json",
        "sites": site_no,
        "startDT": start_date,
        "endDT": end_date,
        "parameterCd": _DISCHARGE_CD,
        "statCd": _MEAN_STAT,
    }

    resp = requests.get(_NWIS_URL, params=params, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    time_series = payload.get("value", {}).get("timeSeries", [])
    if not time_series:
        raise ValueError(
            f"No discharge data returned for site {site_no} "
            f"({start_date} – {end_date})."
        )

    raw_values = time_series[0]["values"][0]["value"]

    records = []
    for entry in raw_values:
        try:
            cfs = float(entry["value"])
        except (ValueError, TypeError):
            cfs = float("nan")
        records.append({"date": entry["dateTime"][:10], "discharge_cfs": cfs})

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["site_no"] = site_no
    df = df.set_index("date").sort_index()

    # Drop USGS sentinel values (e.g. -999999) and physically impossible negatives
    df = df[df["discharge_cfs"] >= 0]

    return df[["site_no", "discharge_cfs"]]


def fetch_multiple_gauges(
    site_nos: Optional[list[str]] = None,
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily flow for multiple gauges and return a wide DataFrame.

    Each gauge becomes a column named ``flow_<site_no>``.  Missing dates
    for a gauge are left as NaN rather than dropping the row.

    Parameters
    ----------
    site_nos:   List of 8-digit USGS site numbers.  Defaults to all
                ``ARKANSAS_GAUGES``.
    start_date: ISO date string, inclusive.
    end_date:   ISO date string, inclusive (defaults to today).

    Returns
    -------
    Wide DataFrame indexed by date, columns = [flow_<site_no>, ...].
    """
    if site_nos is None:
        site_nos = list(ARKANSAS_GAUGES.keys())

    frames: list[pd.DataFrame] = []
    for site in site_nos:
        try:
            df = fetch_flow(site, start_date, end_date)
            df = df[["discharge_cfs"]].rename(columns={"discharge_cfs": f"flow_{site}"})
            frames.append(df)
        except Exception as exc:
            print(f"[usgs] Warning – skipping site {site}: {exc}")

    if not frames:
        raise RuntimeError("No USGS gauge data could be retrieved.")

    return pd.concat(frames, axis=1).sort_index()
