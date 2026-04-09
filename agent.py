"""River Flow Prediction Agent – main orchestration loop.

Uses the Anthropic SDK with tool use to coordinate data fetching, model
training, and 30-day flow prediction for the upper Arkansas River basin.

Usage:
    python agent.py
    python agent.py --query "What will river flow look like in 30 days?"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from typing import Any

from dotenv import load_dotenv
load_dotenv()

import anthropic
import pandas as pd

from fetchers.usgs import fetch_flow, ARKANSAS_GAUGES
from fetchers.snotel import fetch_snotel, ARKANSAS_SNOTEL_SITES
from fetchers.noaa import fetch_forecast, fetch_historical_weather
from models.predictor import FlowPredictor

# ---------------------------------------------------------------------------
# Tool definitions exposed to Claude
# ---------------------------------------------------------------------------

TOOLS: list[dict] = [
    {
        "name": "fetch_usgs_flow",
        "description": (
            "Fetch daily mean streamflow (cfs) from USGS gauges on the "
            "upper Arkansas River. Returns a JSON summary of recent and "
            "historical discharge data."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_no": {
                    "type": "string",
                    "description": (
                        "8-digit USGS site number. Defaults to 07091200 "
                        "(Arkansas River near Nathrop, CO). "
                        "Available: " + ", ".join(f"{k} ({v})" for k, v in ARKANSAS_GAUGES.items())
                    ),
                    "default": "07091200",
                },
                "start_date": {
                    "type": "string",
                    "description": "Start date ISO-8601 (e.g. 1990-01-01).",
                    "default": "1990-01-01",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date ISO-8601 (defaults to today).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "fetch_snotel_snowpack",
        "description": (
            "Load daily Snow Water Equivalent (SWE, inches) from local NRCS "
            "SNOTEL CSV files for stations in the upper Arkansas River basin."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "start_date": {
                    "type": "string",
                    "description": "Start date ISO-8601 (e.g. 1990-01-01).",
                    "default": "1990-01-01",
                },
                "end_date": {
                    "type": "string",
                    "description": "End date ISO-8601 (defaults to today).",
                },
            },
            "required": [],
        },
    },
    {
        "name": "fetch_weather_forecast",
        "description": (
            "Fetch the 7-day NWS weather forecast for the upper Arkansas "
            "River basin (near Salida, CO). Returns daily high/low temps "
            "and precipitation probability."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "lat": {
                    "type": "number",
                    "description": "Latitude (default 38.535 = Salida, CO).",
                    "default": 38.535,
                },
                "lon": {
                    "type": "number",
                    "description": "Longitude (default -105.999 = Salida, CO).",
                    "default": -105.999,
                },
            },
            "required": [],
        },
    },
    {
        "name": "train_flow_model",
        "description": (
            "Train a gradient-boosted regression model correlating snowpack "
            "and weather data to 30-day ahead river flow. Automatically "
            "fetches 1990–present data from USGS and SNOTEL. "
            "Returns cross-validated performance metrics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_no": {
                    "type": "string",
                    "description": "USGS gauge site number to use as the flow target.",
                    "default": "07091200",
                },
                "save_model": {
                    "type": "boolean",
                    "description": "Whether to save the trained model to disk.",
                    "default": True,
                },
            },
            "required": [],
        },
    },
    {
        "name": "predict_flow",
        "description": (
            "Use the trained regression model to predict mean river flow "
            "(cfs) over the next 30 days, based on current snowpack, "
            "recent flow, and weather forecast."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "site_no": {
                    "type": "string",
                    "description": "USGS gauge site number.",
                    "default": "07091200",
                },
            },
            "required": [],
        },
    },
    {
        "name": "get_basin_summary",
        "description": (
            "Return a concise summary of current basin conditions: "
            "latest streamflow, basin-average SWE, and SWE percentile "
            "relative to historical median."
        ),
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

_predictor: FlowPredictor | None = None  # cached across calls in a session


def _tool_fetch_usgs_flow(site_no: str = "07091200",
                          start_date: str = "1990-01-01",
                          end_date: str | None = None) -> str:
    end = end_date or date.today().isoformat()
    df = fetch_flow(site_no=site_no, start_date=start_date, end_date=end)
    recent = df.tail(14)
    result = {
        "site_no": site_no,
        "site_name": ARKANSAS_GAUGES.get(site_no, "Unknown"),
        "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
        "total_records": len(df),
        "latest_discharge_cfs": round(float(df["discharge_cfs"].iloc[-1]), 1),
        "14d_mean_cfs": round(float(df["discharge_cfs"].tail(14).mean()), 1),
        "recent_14_days": recent["discharge_cfs"].round(1).to_dict(),
    }
    return json.dumps(result, default=str)


def _tool_fetch_snotel(start_date: str = "1990-01-01",
                       end_date: str | None = None) -> str:
    df = fetch_snotel(start_date=start_date, end_date=end_date)
    latest = df.iloc[-1]
    result = {
        "date_range": f"{df.index.min().date()} to {df.index.max().date()}",
        "stations": list(ARKANSAS_SNOTEL_SITES.values()),
        "latest_date": str(df.index[-1].date()),
        "latest_basin_avg_swe_in": round(float(latest["wteq_basin_avg"]), 2) if "wteq_basin_avg" in latest else None,
        "latest_by_station": {
            col: round(float(latest[col]), 2)
            for col in df.columns
            if col != "wteq_basin_avg" and not pd.isna(latest[col])
        },
    }
    return json.dumps(result, default=str)


def _tool_fetch_weather_forecast(lat: float = 38.535, lon: float = -105.999) -> str:
    df = fetch_forecast(lat=lat, lon=lon)
    result = {
        "location": f"{lat}°N, {abs(lon)}°W",
        "forecast_days": len(df),
        "forecast": df.to_dict(orient="index"),
    }
    return json.dumps(result, default=str)


def _tool_train_model(site_no: str = "07091200", save_model: bool = True) -> str:
    global _predictor

    print("[agent] Fetching historical flow data (1990–present)…")
    flow_df = fetch_flow(site_no=site_no, start_date="1990-01-01")

    print("[agent] Fetching historical SNOTEL SWE data…")
    snotel_df = fetch_snotel(start_date="1990-01-01")

    _predictor = FlowPredictor()
    print("[agent] Engineering features and training model…")
    X, y = _predictor.build_training_data(flow_df, snotel_df)
    metrics = _predictor.train(X, y)

    if save_model:
        path = _predictor.save()
        metrics["saved_to"] = str(path)

    return json.dumps(metrics, default=str)


def _tool_predict_flow(site_no: str = "07091200") -> str:
    global _predictor

    # Load saved model if not already in memory
    if _predictor is None:
        _predictor = FlowPredictor()
        try:
            _predictor.load()
            print("[agent] Loaded saved model from disk.")
        except FileNotFoundError:
            return json.dumps({
                "error": "No trained model found. Call train_flow_model first.",
            })

    # Fetch recent data for feature construction (last 90 days is sufficient)
    lookback = "2025-01-01"
    print("[agent] Fetching recent flow and snowpack for prediction…")
    flow_df   = fetch_flow(site_no=site_no, start_date=lookback)
    snotel_df = fetch_snotel(start_date=lookback)

    try:
        forecast_cfs = _predictor.predict_one(flow_df, snotel_df)
    except Exception as exc:
        return json.dumps({"error": str(exc)})

    current_cfs = float(flow_df["discharge_cfs"].iloc[-1])
    result = {
        "prediction_date": date.today().isoformat(),
        "site_no": site_no,
        "site_name": ARKANSAS_GAUGES.get(site_no, "Unknown"),
        "current_discharge_cfs": round(current_cfs, 1),
        "predicted_30d_mean_cfs": round(forecast_cfs, 1),
        "change_pct": round((forecast_cfs - current_cfs) / current_cfs * 100, 1),
        "horizon_days": 30,
    }
    return json.dumps(result, default=str)


def _tool_basin_summary() -> str:
    today = date.today().isoformat()
    # Fetch last 30 days
    lookback = pd.Timestamp.today() - pd.DateOffset(days=30)
    lb_str = lookback.date().isoformat()

    flow_df = fetch_flow(site_no="07091200", start_date=lb_str)
    snotel_df = fetch_snotel(start_date=lb_str)

    current_flow = float(flow_df["discharge_cfs"].iloc[-1])
    swe_now = float(snotel_df["wteq_basin_avg"].iloc[-1]) if "wteq_basin_avg" in snotel_df.columns else float("nan")

    # Historical median SWE for current day-of-year (using all available data)
    full_snotel = fetch_snotel(start_date="1990-01-01")
    doy = date.today().timetuple().tm_yday
    if "wteq_basin_avg" in full_snotel.columns:
        historical_doy = full_snotel[full_snotel.index.day_of_year == doy]["wteq_basin_avg"]
        median_swe = float(historical_doy.median()) if not historical_doy.empty else float("nan")
        pct_of_median = round(swe_now / median_swe * 100, 1) if median_swe > 0 else None
    else:
        median_swe = float("nan")
        pct_of_median = None

    result = {
        "as_of": today,
        "current_flow_cfs": round(current_flow, 1),
        "current_basin_swe_in": round(swe_now, 2),
        "historical_median_swe_in": round(median_swe, 2) if not pd.isna(median_swe) else None,
        "swe_pct_of_median": pct_of_median,
        "30d_avg_flow_cfs": round(float(flow_df["discharge_cfs"].mean()), 1),
    }
    return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

_TOOL_MAP = {
    "fetch_usgs_flow":       _tool_fetch_usgs_flow,
    "fetch_snotel_snowpack": _tool_fetch_snotel,
    "fetch_weather_forecast":_tool_fetch_weather_forecast,
    "train_flow_model":      _tool_train_model,
    "predict_flow":          _tool_predict_flow,
    "get_basin_summary":     _tool_basin_summary,
}


def dispatch_tool(name: str, inputs: dict[str, Any]) -> str:
    """Call the appropriate tool function and return a JSON string result."""
    fn = _TOOL_MAP.get(name)
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {name}"})
    try:
        return fn(**inputs)
    except Exception as exc:
        return json.dumps({"error": str(exc), "tool": name})


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = f"""\
You are an expert hydrology assistant specializing in the upper Arkansas River \
basin in Colorado. Today's date is {date.today().isoformat()}.

You have access to real-time and historical data tools for USGS streamflow, \
NRCS SNOTEL snowpack, and NOAA weather. Use them to answer questions accurately.

When making flow predictions:
1. First check current basin conditions (snowpack + flow).
2. Fetch the 7-day weather forecast.
3. If a trained model exists, use predict_flow; otherwise train one first.
4. Explain your reasoning in plain language, including confidence caveats.

Always cite data sources and dates. Express flows in cfs (cubic feet per second) \
and SWE in inches.
"""


def run_agent(user_query: str, verbose: bool = True) -> str:
    """Run the agentic loop for a single user query.

    Parameters
    ----------
    user_query: Natural language question or request.
    verbose:    Print tool calls and intermediate steps to stdout.

    Returns
    -------
    Final assistant response text.
    """
    client = anthropic.Anthropic()

    messages: list[dict] = [{"role": "user", "content": user_query}]

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Extract final text
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return "(no text response)"

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                if verbose:
                    print(f"\n[tool] → {block.name}({json.dumps(block.input, default=str)})")
                result = dispatch_tool(block.name, block.input)
                if verbose:
                    # Pretty-print a truncated result
                    try:
                        parsed = json.loads(result)
                        preview = json.dumps(parsed, indent=2)[:600]
                    except Exception:
                        preview = result[:600]
                    print(f"[tool] ← {preview}{'…' if len(result) > 600 else ''}")
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                })

            messages.append({"role": "user", "content": tool_results})
        else:
            # Unexpected stop reason
            break

    return "(agent loop ended unexpectedly)"


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="River Flow Prediction Agent – upper Arkansas River basin"
    )
    parser.add_argument(
        "--query",
        default=(
            "Give me a complete basin conditions report: current snowpack, "
            "streamflow, and your best 30-day flow prediction with explanation."
        ),
        help="Natural language query for the agent.",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress intermediate tool call output.",
    )
    args = parser.parse_args()

    print(f"Query: {args.query}\n{'='*60}")
    answer = run_agent(args.query, verbose=not args.quiet)
    print(f"\n{'='*60}\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
