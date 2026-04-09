# River Flow Predictor

Predicts 30-day mean streamflow for the upper Arkansas River basin using historical snowpack (SNOTEL), USGS gauge data, and NOAA weather forecasts. An Anthropic Claude agent orchestrates data fetching, model training, and inference via tool use.

## Data Sources

| Source | Data | Location |
|--------|------|----------|
| USGS NWIS | Daily discharge — site 07091200 (Arkansas River near Nathrop, CO) | fetched live via OGC API |
| NRCS SNOTEL | Daily Snow Water Equivalent — 5 basin stations | `SNOTEL_DATA/` |
| NOAA NWS | 7-day weather forecast (api.weather.gov) | fetched live |

## Setup

### 1. Activate the virtual environment

```bash
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your Anthropic API key

```bash
export ANTHROPIC_API_KEY="your-key-here"
```

## Running the agent

**Default query** — full basin conditions report + 30-day prediction:

```bash
python agent.py
```

**Custom query:**

```bash
python agent.py --query "What will river flow look like in 30 days?"
python agent.py --query "What is the current snowpack compared to historical median?"
```

**Quiet mode** (suppress tool call output):

```bash
python agent.py --quiet
python agent.py --query "Current flow at Nathrop?" --quiet
```

## Project structure

```
River-Flow-Predictor/
├── agent.py                  # Claude agent loop + tool definitions
├── fetchers/
│   ├── usgs.py               # USGS Water Data OGC API (daily discharge + DOY stats)
│   ├── snotel.py             # SNOTEL static CSV loader (SWE)
│   └── noaa.py               # NOAA NWS forecast + CDO historical weather
├── models/
│   └── predictor.py          # GradientBoostingRegressor, 30-day horizon
├── SNOTEL_DATA/              # Static SNOTEL SWE CSV files
└── requirements.txt
```
