"""NRCS SNOTEL snowpack data loader for the upper Arkansas River basin.

Reads from local CSV files in the SNOTEL_DATA/ directory.
"""

import pathlib
import pandas as pd
from typing import Optional

# Path to SNOTEL CSV data directory (relative to this file's package root)
_DATA_DIR = pathlib.Path(__file__).parent.parent / "SNOTEL_DATA"

# SNOTEL stations in the upper Arkansas River basin (station ID → display name)
ARKANSAS_SNOTEL_SITES: dict[str, str] = {
    "369": "Brumley, CO",
    "485": "Fremont Pass, CO",
    "589": "Lone Cone, CO",
    "622": "Mesa Lakes, CO",
    "838": "University Camp, CO",
}

_CSV_FILES: dict[str, str] = {
    "369": "SNOTEL 369- Brumley, CO.csv",
    "485": "SNOTEL 485- Fremont Pass, CO.csv",
    "589": "SNOTEL 589- Lone Cone, CO.csv",
    "622": "SNOTEL 622- Mesa Lakes, CO.csv",
    "838": "SNOTEL 838- University Camp, CO.csv",
}

_SWE_COL = "Snow Water Equivalent (in) Start of Day Values"


def _load_station_swe(station_id: str) -> pd.DataFrame:
    """Load SWE for a single station from its CSV file."""
    csv_path = _DATA_DIR / _CSV_FILES[station_id]
    df = pd.read_csv(csv_path, comment="#", parse_dates=["Date"], index_col="Date")
    df.index.name = "date"
    return df[[_SWE_COL]].rename(columns={_SWE_COL: f"wteq_{station_id}"})


def fetch_snotel(
    start_date: str = "1990-01-01",
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load daily SWE data for Arkansas basin SNOTEL stations from local CSV files.

    Parameters
    ----------
    start_date: ISO date string, inclusive (default 1990-01-01).
    end_date:   ISO date string, inclusive (defaults to today's date).

    Returns
    -------
    Wide DataFrame indexed by date with one column per station named
    ``wteq_{station_id}`` plus a ``wteq_basin_avg`` column.
    """
    frames = [_load_station_swe(sid) for sid in ARKANSAS_SNOTEL_SITES]
    frames = [df for df in frames if not df.empty]

    if not frames:
        raise RuntimeError("No SNOTEL SWE data loaded — check SNOTEL_DATA/ directory.")

    combined = pd.concat(frames, axis=1).sort_index()

    combined = combined.loc[start_date:]
    if end_date:
        combined = combined.loc[:end_date]

    # Negative SWE is physically impossible; clip to zero
    combined = combined.clip(lower=0)

    combined["wteq_basin_avg"] = combined.mean(axis=1)

    return combined
