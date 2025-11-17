"""
forecasting_utils.py

Time series forecasting utilities using Prophet and ARIMA.
"""

from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA


# ---------------------------
# Preprocessing
# ---------------------------

def prepare_time_series(
    df: pd.DataFrame,
    date_col: str = "date",
    value_col: str = "close",
) -> pd.DataFrame:
    """
    Prepare a univariate time series for Prophet:
    - rename columns to 'ds' and 'y'
    - drop NA
    - sort by date

    Parameters
    ----------
    df : pd.DataFrame
    date_col : str
    value_col : str

    Returns
    -------
    pd.DataFrame
    """
    ts = df[[date_col, value_col]].copy()
    ts = ts.rename(columns={date_col: "ds", value_col: "y"})
    ts = ts.dropna(subset=["ds", "y"])
    ts = ts.sort_values("ds").reset_index(drop=True)
    return ts


# ---------------------------
# Prophet utilities
# ---------------------------

def fit_prophet_model(
    ts: pd.DataFrame,
    yearly_seasonality: bool = True,
    weekly_seasonality: bool = True,
    daily_seasonality: bool = False,
    changepoint_prior_scale: float = 0.05,
) -> Prophet:
    """
    Fit a Prophet model to time series.

    Parameters
    ----------
    ts : pd.DataFrame
        DataFrame with columns 'ds' (datetime) and 'y' (value).
    yearly_seasonality : bool
    weekly_seasonality : bool
    daily_seasonality : bool
    changepoint_prior_scale : float

    Returns
    -------
    Prophet
    """
    model = Prophet(
        yearly_seasonality=yearly_seasonality,
        weekly_seasonality=weekly_seasonality,
        daily_seasonality=daily_seasonality,
        changepoint_prior_scale=changepoint_prior_scale,
    )
    model.fit(ts)
    return model


def forecast_with_prophet(
    model: Prophet,
    periods: int,
    freq: str = "D",
) -> pd.DataFrame:
    """
    Generate forecasts using a fitted Prophet model.

    Parameters
    ----------
    model : Prophet
    periods : int
        Number of periods into the future.
    freq : str
        Pandas offset alias (e.g. 'D', 'H').

    Returns
    -------
    pd.DataFrame
        Forecast DataFrame with columns including 'ds', 'yhat', 'yhat_lower', 'yhat_upper'.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


# ---------------------------
# ARIMA utilities
# ---------------------------

def fit_arima_model(
    ts: pd.Series,
    order: Tuple[int, int, int] = (1, 1, 1),
) -> ARIMA:
    """
    Fit an ARIMA model.

    Parameters
    ----------
    ts : pd.Series
        Time series indexed by date.
    order : tuple
        (p, d, q) ARIMA order.

    Returns
    -------
    ARIMAResults
    """
    model = ARIMA(ts, order=order)
    fitted = model.fit()
    return fitted


def forecast_with_arima(
    model,
    steps: int,
) -> pd.Series:
    """
    Forecast future values using a fitted ARIMA model.

    Parameters
    ----------
    model : ARIMAResults
        Fitted ARIMA model.
    steps : int
        Forecast horizon.

    Returns
    -------
    pd.Series
    """
    forecast = model.forecast(steps=steps)
    return forecast
