"""Scikit-learn regression model: snowpack + weather → 30-day river flow.

The model treats this as a supervised regression problem:
  - Features: current SWE at each SNOTEL station, lagged flow, day-of-year,
              temperature, precipitation.
  - Target:   mean discharge (cfs) over the next PREDICTION_HORIZON days.

Training uses time-series cross-validation to prevent data leakage.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PREDICTION_HORIZON = 30  # days ahead
_DEFAULT_MODEL_PATH = Path("models/flow_predictor.pkl")


class FlowPredictor:
    """Gradient-boosted regression model for 30-day flow forecasting."""

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path: Path = model_path or _DEFAULT_MODEL_PATH
        self.pipeline: Optional[Pipeline] = None
        self.feature_names: list[str] = []

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def build_training_data(
        self,
        flow_df: pd.DataFrame,
        snotel_df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Merge data sources and engineer features for model training.

        Parameters
        ----------
        flow_df:    DataFrame with a discharge/flow column, DatetimeIndex.
        snotel_df:  DataFrame with SWE columns, DatetimeIndex.
        weather_df: Optional DataFrame with tmax_f/tmin_f/prcp_in, DatetimeIndex.

        Returns
        -------
        (X, y) where X is the feature DataFrame and y is the target Series.
        Target = mean discharge over the next PREDICTION_HORIZON days.
        Rows with NaN in target or key features are dropped.
        """
        # Identify the primary flow column
        flow_cols = [c for c in flow_df.columns if "flow_" in c or "discharge" in c]
        if not flow_cols:
            raise ValueError("flow_df must contain a 'flow_*' or 'discharge*' column.")
        primary_flow = flow_cols[0]

        df = flow_df[[primary_flow]].copy().rename(columns={primary_flow: "flow_cfs"})

        # Merge SNOTEL data
        df = df.join(snotel_df, how="left")

        # Forward-fill SNOTEL columns to cover sensor outages and trailing
        # edge gaps where CSV data ends before the USGS record.  Limit to 14
        # days so genuinely missing stations don't silently propagate further.
        swe_raw_cols = [c for c in df.columns if c.startswith("wteq_") or c.startswith("swe_")]
        df[swe_raw_cols] = df[swe_raw_cols].ffill(limit=14)

        # Merge weather if provided
        if weather_df is not None and not weather_df.empty:
            df = df.join(weather_df, how="left")

        # ---- Feature engineering ----
        df["day_of_year"]   = df.index.day_of_year
        df["month"]         = df.index.month
        df["flow_7d_avg"]   = df["flow_cfs"].rolling(7, min_periods=4).mean()
        df["flow_30d_avg"]  = df["flow_cfs"].rolling(30, min_periods=15).mean()
        df["flow_lag7"]     = df["flow_cfs"].shift(7)
        df["flow_lag14"]    = df["flow_cfs"].shift(14)
        df["flow_lag30"]    = df["flow_cfs"].shift(30)

        # SWE rate of change (snowmelt signal)
        swe_cols = [c for c in df.columns if c.startswith("wteq_") or c.startswith("swe_")]
        for col in swe_cols:
            df[f"{col}_delta7"] = df[col].diff(7)

        # ---- Target ----
        # Mean flow over the next PREDICTION_HORIZON days (forward-looking)
        df["target"] = (
            df["flow_cfs"]
            .shift(-PREDICTION_HORIZON)
            .rolling(PREDICTION_HORIZON, min_periods=PREDICTION_HORIZON // 2)
            .mean()
        )

        df = df.dropna(subset=["target", "flow_7d_avg", "flow_lag7", "flow_lag30"])

        # Drop the raw flow_cfs from features to avoid target leakage
        # (lagged versions are fine as predictors)
        feature_cols = [c for c in df.columns if c != "target" and c != "flow_cfs"]
        self.feature_names = feature_cols

        return df[feature_cols], df["target"]

    def prepare_prediction_row(
        self,
        flow_df: pd.DataFrame,
        snotel_df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Build a single-row feature DataFrame for a live prediction.

        Calls ``build_training_data`` on recent data and returns the last row,
        which represents today's feature state.
        """
        X, _ = self.build_training_data(flow_df, snotel_df, weather_df)
        return X.iloc[[-1]]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        n_cv_splits: int = 5,
    ) -> dict:
        """Fit the model with time-series cross-validation.

        Parameters
        ----------
        X:            Feature DataFrame (from ``build_training_data``).
        y:            Target Series.
        n_cv_splits:  Number of CV folds.

        Returns
        -------
        Dict with cross-validated MAE, RMSE, R², sample count, and feature list.
        """
        self.pipeline = Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=300,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        min_samples_leaf=10,
                        random_state=42,
                    ),
                ),
            ]
        )

        tscv = TimeSeriesSplit(n_splits=n_cv_splits)
        cv_mae, cv_rmse, cv_r2 = [], [], []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X), start=1):
            X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
            self.pipeline.fit(X_tr, y_tr)
            preds = self.pipeline.predict(X_val)
            cv_mae.append(mean_absolute_error(y_val, preds))
            cv_rmse.append(root_mean_squared_error(y_val, preds))
            cv_r2.append(r2_score(y_val, preds))

        # Final fit on all data
        self.pipeline.fit(X, y)

        return {
            "cv_mae_mean":  round(float(np.mean(cv_mae)), 1),
            "cv_mae_std":   round(float(np.std(cv_mae)), 1),
            "cv_rmse_mean": round(float(np.mean(cv_rmse)), 1),
            "cv_r2_mean":   round(float(np.mean(cv_r2)), 3),
            "cv_r2_std":    round(float(np.std(cv_r2)), 3),
            "n_samples":    len(X),
            "n_features":   len(self.feature_names),
            "features":     self.feature_names,
        }

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted 30-day mean discharge (cfs) for each row of X."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained. Call train() or load() first.")
        return self.pipeline.predict(X)

    def predict_one(
        self,
        flow_df: pd.DataFrame,
        snotel_df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> float:
        """Convenience wrapper: build features from raw DataFrames and predict."""
        row = self.prepare_prediction_row(flow_df, snotel_df, weather_df)
        return float(self.predict(row)[0])

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Optional[Path] = None) -> Path:
        """Serialize the fitted pipeline to disk with joblib."""
        save_path = path or self.model_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "features": self.feature_names}, save_path)
        return save_path

    def load(self, path: Optional[Path] = None) -> "FlowPredictor":
        """Deserialize a previously saved model."""
        load_path = path or self.model_path
        obj = joblib.load(load_path)
        self.pipeline = obj["pipeline"]
        self.feature_names = obj["features"]
        return self

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending."""
        if self.pipeline is None:
            raise RuntimeError("Model not trained.")
        imp = self.pipeline.named_steps["model"].feature_importances_
        return pd.Series(imp, index=self.feature_names).sort_values(ascending=False)

    def summary(self) -> str:
        """Human-readable model summary."""
        if self.pipeline is None:
            return "FlowPredictor (untrained)"
        top5 = self.feature_importance().head(5)
        lines = [
            f"FlowPredictor – GradientBoostingRegressor",
            f"  Prediction horizon : {PREDICTION_HORIZON} days",
            f"  Features           : {len(self.feature_names)}",
            f"  Top-5 importances  :",
        ] + [f"    {name}: {val:.3f}" for name, val in top5.items()]
        return "\n".join(lines)
