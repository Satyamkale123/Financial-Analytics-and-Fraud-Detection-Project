"""
data_cleaning.py

Utility functions for loading and cleaning stock price data
and fraud transaction datasets.
"""

from __future__ import annotations

import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd


# ---------------------------
# Generic helpers
# ---------------------------

def list_stock_files(directory: str, extensions: Tuple[str, ...] = (".csv", ".xlsx")) -> List[str]:
    """
    List stock data files in a given directory.

    Parameters
    ----------
    directory : str
        Path to directory containing stock files.
    extensions : tuple of str
        Allowed file extensions.

    Returns
    -------
    List[str]
        List of full file paths.
    """
    files = []
    for fname in os.listdir(directory):
        if fname.lower().endswith(extensions):
            files.append(os.path.join(directory, fname))
    return files


def load_stock_file(filepath: str, date_col: str = "date") -> pd.DataFrame:
    """
    Load a single stock data file (CSV or XLSX).

    Parameters
    ----------
    filepath : str
        Path to stock file.
    date_col : str
        Name of the date column.

    Returns
    -------
    pd.DataFrame
    """
    if filepath.lower().endswith(".csv"):
        df = pd.read_csv(filepath)
    elif filepath.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(filepath)
    else:
        raise ValueError(f"Unsupported file type: {filepath}")

    if date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.sort_values(date_col).reset_index(drop=True)

    return df


def load_all_stocks(
    directory: str,
    date_col: str = "date",
    symbol_col: str = "symbol",
    infer_symbol_from_filename: bool = True,
) -> pd.DataFrame:
    """
    Load and concatenate multiple stock files into a single DataFrame.

    Parameters
    ----------
    directory : str
        Folder containing stock files.
    date_col : str
        Name of the date column.
    symbol_col : str
        Column name to store the stock symbol.
    infer_symbol_from_filename : bool
        If True, symbol is guessed from filename (before first dot/underscore).

    Returns
    -------
    pd.DataFrame
        Combined stock data with a symbol column.
    """
    files = list_stock_files(directory)
    all_dfs = []

    for path in files:
        df = load_stock_file(path, date_col=date_col)

        if infer_symbol_from_filename:
            fname = os.path.basename(path)
            symbol_guess = fname.split(".")[0].split("_")[0].upper()
        else:
            symbol_guess = None

        if symbol_col not in df.columns:
            df[symbol_col] = symbol_guess

        all_dfs.append(df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, axis=0, ignore_index=True)
    combined = combined.sort_values([symbol_col, date_col]).reset_index(drop=True)
    return combined


# ---------------------------
# Stock data cleaning
# ---------------------------

def clean_stock_data(
    df: pd.DataFrame,
    price_cols: Optional[List[str]] = None,
    volume_col: str = "volume",
    min_volume: Optional[float] = 0.0,
) -> pd.DataFrame:
    """
    Basic cleaning for stock price data.

    Operations:
    - Drop duplicate rows
    - Remove rows with null dates
    - Forward-fill missing price values
    - Remove negative/zero volumes if requested

    Parameters
    ----------
    df : pd.DataFrame
        Raw stock data.
    price_cols : list of str, optional
        Columns that contain price info (e.g., ['open', 'high', 'low', 'close']).
        If None, attempts to infer common names.
    volume_col : str
        Volume column name.
    min_volume : float, optional
        Minimum allowed volume. If not None, rows with volume < min_volume are removed.

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    if "date" in df.columns:
        df = df.dropna(subset=["date"])
        df = df.sort_values("date")
    df = df.drop_duplicates()

    if price_cols is None:
        price_cols = [c for c in df.columns if c.lower() in ["open", "high", "low", "close", "adj_close"]]

    # Forward/backward fill price data
    if price_cols:
        df[price_cols] = df[price_cols].ffill().bfill()

    # Remove bad volume values
    if volume_col in df.columns and min_volume is not None:
        df = df[df[volume_col] >= min_volume]

    df = df.reset_index(drop=True)
    return df


# ---------------------------
# Fraud dataset cleaning
# ---------------------------

def load_fraud_data(filepath: str) -> pd.DataFrame:
    """
    Load fraud dataset CSV.

    Parameters
    ----------
    filepath : str
        Path to fraud_dataset.csv

    Returns
    -------
    pd.DataFrame
    """
    df = pd.read_csv(filepath)
    return df


def clean_fraud_data(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    drop_cols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Basic cleaning for fraud transaction data.

    Operations:
    - Drop duplicate transaction IDs (if 'transaction_id' exists)
    - Drop user-specified columns
    - Handle missing values (simple imputation or row drop)
    - Ensure binary target column

    Parameters
    ----------
    df : pd.DataFrame
        Raw fraud dataset.
    target_col : str
        Name of the binary target column.
    drop_cols : list of str, optional
        Columns to drop (e.g., high cardinality IDs).

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()

    # Drop duplicates by transaction_id if present
    if "transaction_id" in df.columns:
        df = df.drop_duplicates(subset=["transaction_id"])

    # Drop user-defined columns
    if drop_cols is not None:
        existing = [c for c in drop_cols if c in df.columns]
        df = df.drop(columns=existing)

    # Basic missing-value handling
    # Numeric: fill with median, Categorical: fill with mode
    for col in df.columns:
        if df[col].isna().sum() == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Ensure binary target
    if target_col in df.columns:
        df[target_col] = df[target_col].astype(int)

    df = df.reset_index(drop=True)
    return df


# ---------------------------
# IO helpers
# ---------------------------

def save_cleaned_data(df: pd.DataFrame, filepath: str, index: bool = False) -> None:
    """
    Save cleaned dataset to CSV.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save.
    filepath : str
        Output path.
    index : bool
        Whether to save index.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=index)
