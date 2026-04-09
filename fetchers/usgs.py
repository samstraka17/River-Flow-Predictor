"""USGS streamflow data from the USGS Water Data OGC API.

API docs: https://api.waterdata.usgs.gov/ogcapi/v0
Auth:     Bearer token via API_USGS_PAT env var (optional; bypasses rate limits).
          Register at https://api.waterdata.usgs.gov/signup/
"""

import os
from datetime import date
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

_BASE = "https://api.waterdata.usgs.gov/ogcapi/v0"
_PAGE_SIZE = 10_000

# Gauges used in the upper Arkansas River basin
ARKANSAS_GAUGES: dict[str, str] = {
    "07083710": "Arkansas River Below Empire Gulch near Malta, CO",
    "07091200": "Arkansas River near Nathrop, CO",
}

DEFAULT_GAUGE = "07091200"


def _headers() -> dict[str, str]:
    token = os.getenv("API_USGS_PAT")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def _get_all_pages(url: str, params: dict) -> list[dict]:
    """Fetch all pages from an OGC API endpoint and return the combined feature list."""
    features: list[dict] = []
    params = {**params, "limit": _PAGE_SIZE, "offset": 0}

    while True:
        resp = requests.get(url, params=params, headers=_headers(), timeout=60)
        resp.raise_for_status()
        payload = resp.json()

        page_features = payload.get("features", [])
        features.extend(page_features)

        if len(page_features) < _PAGE_SIZE:
            break

        params["offset"] += _PAGE_SIZE

    return features


def fetch_flow(
    site_no: str = DEFAULT_GAUGE,
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily mean discharge (cfs) from the USGS Water Data OGC API.

    Parameters
    ----------
    site_no:    8-digit USGS site number.
    start_date: ISO date string, inclusive.
    end_date:   ISO date string, inclusive (defaults to today).

    Returns
    -------
    DataFrame indexed by date with columns [site_no, discharge_cfs].
    """
    end = end_date or date.today().isoformat()
    params = {
        "monitoring_location_id": f"USGS-{site_no}",
        "parameter_code": "00060",   # discharge
        "statistic_id": "00003",     # mean
        "datetime": f"{start_date}/{end}",
        "f": "json",
    }

    features = _get_all_pages(f"{_BASE}/collections/daily/items", params)

    if not features:
        raise RuntimeError(
            f"No daily discharge data returned for site {site_no} "
            f"({start_date} – {end})."
        )

    _SENTINEL = {"", None, "Ice", "Ssn", "Bkw", "Eqp", "Rat", "***"}
    records = [
        {
            "date": feat["properties"]["time"],
            "discharge_cfs": feat["properties"]["value"],
            "site_no": site_no,
        }
        for feat in features
        if feat.get("properties", {}).get("value") not in _SENTINEL
    ]

    df = pd.DataFrame(records)
    df["date"] = pd.to_datetime(df["date"])
    df["discharge_cfs"] = pd.to_numeric(df["discharge_cfs"], errors="coerce")
    df = df.dropna(subset=["discharge_cfs"])
    df = df[df["discharge_cfs"] >= 0]

    return df.set_index("date").sort_index()[["site_no", "discharge_cfs"]]


def fetch_doy_statistics(
    site_no: str = "07083710",
    start_date: str = "1990-01-01",
) -> pd.DataFrame:
    """Compute day-of-year mean discharge statistics from the live daily API.

    Fetches all daily records since start_date and computes the arithmetic mean
    discharge per calendar day (MM-DD).

    Parameters
    ----------
    site_no:    8-digit USGS site number.
    start_date: Earliest record to include in the averages.

    Returns
    -------
    DataFrame indexed by 'MM-DD' day-of-year strings with columns
    [discharge_cfs, sample_count].
    """
    df = fetch_flow(site_no=site_no, start_date=start_date)

    df["day_of_year"] = df.index.strftime("%m-%d")
    stats = (
        df.groupby("day_of_year")["discharge_cfs"]
        .agg(discharge_cfs="mean", sample_count="count")
        .reset_index()
        .set_index("day_of_year")
    )
    return stats
