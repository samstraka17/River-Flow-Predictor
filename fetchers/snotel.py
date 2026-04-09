"""NRCS SNOTEL snowpack fetcher for the upper Arkansas River basin.

Data source: https://wcc.sc.egov.usda.gov/awdbRestApi/
Swagger docs: https://wcc.sc.egov.usda.gov/awdbRestApi/swagger-ui/index.html
"""

import requests
import pandas as pd
from datetime import date
from typing import Optional

# SNOTEL stations in the upper Arkansas River basin.
# Format: "ID:STATE:NETWORK"
ARKANSAS_SNOTEL_SITES: dict[str, str] = {
    "1012:CO:SNTL": "Fremont Pass, CO  (11,800 ft)",
    "589:CO:SNTL":  "Porphyry Creek, CO (10,900 ft)",
    "369:CO:SNTL":  "Independence Pass, CO (10,600 ft)",
    "622:CO:SNTL":  "Monarch Pass, CO   (11,300 ft)",
    "838:CO:SNTL":  "Spruce, CO          (9,840 ft)",
}

# AWDB element codes
SWE_CODE        = "WTEQ"    # Snow water equivalent, inches
SNOW_DEPTH_CODE = "SNWD"    # Snow depth, inches
PRECIP_CODE     = "PRCPSA"  # Cumulative season precipitation, inches

_BASE_URL = "https://wcc.sc.egov.usda.gov/awdbRestApi/services/v1/data"


def _parse_station_response(station_data: dict, element_cd: str) -> pd.DataFrame:
    """Convert a single station's AWDB response dict into a dated DataFrame."""
    triplet = station_data.get("stationTriplet", "UNKNOWN")
    station_id = triplet.split(":")[0]
    col_name = f"{element_cd.lower()}_{station_id}"

    begin_str = station_data.get("beginDate")
    values = station_data.get("values", [])

    if not begin_str or not values:
        return pd.DataFrame()

    begin = pd.to_datetime(begin_str)
    records = []
    for i, val in enumerate(values):
        dt = begin + pd.Timedelta(days=i)
        try:
            v = float(val) if val is not None else float("nan")
        except (ValueError, TypeError):
            v = float("nan")
        records.append({"date": dt, col_name: v})

    return pd.DataFrame(records).set_index("date")


def fetch_snotel(
    station_triplets: Optional[list[str]] = None,
    element_cd: str = SWE_CODE,
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Fetch daily SNOTEL data for Arkansas basin stations.

    Parameters
    ----------
    station_triplets: List of ``ID:STATE:NETWORK`` strings.  Defaults to all
                      ``ARKANSAS_SNOTEL_SITES``.
    element_cd:       AWDB element code (``WTEQ``, ``SNWD``, ``PRCPSA``).
    start_date:       ISO date string, inclusive.
    end_date:         ISO date string, inclusive (defaults to today).

    Returns
    -------
    Wide DataFrame indexed by date with one column per station named
    ``<element_cd>_<station_id>`` (lower-cased) plus a ``<element>_basin_avg``
    column containing the mean across all stations on each date.
    """
    if end_date is None:
        end_date = date.today().isoformat()
    if station_triplets is None:
        station_triplets = list(ARKANSAS_SNOTEL_SITES.keys())

    params = {
        "stationTriplets": ",".join(station_triplets),
        "elementCd": element_cd,
        "beginDate": start_date,
        "endDate": end_date,
        "duration": "DAILY",
        "getFlags": "false",
        "returnOriginalValues": "false",
        "returnSuspectData": "false",
    }

    resp = requests.get(_BASE_URL, params=params, timeout=60)
    resp.raise_for_status()
    payload = resp.json()

    frames: list[pd.DataFrame] = []
    for station_data in payload:
        df = _parse_station_response(station_data, element_cd)
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError(
            f"No SNOTEL {element_cd} data retrieved for stations: "
            + ", ".join(station_triplets)
        )

    combined = pd.concat(frames, axis=1).sort_index()

    # Negative SWE is physically impossible; replace with NaN
    combined = combined.clip(lower=0)

    # Basin-average across whichever stations reported on each date
    avg_col = f"{element_cd.lower()}_basin_avg"
    combined[avg_col] = combined.mean(axis=1)

    return combined
