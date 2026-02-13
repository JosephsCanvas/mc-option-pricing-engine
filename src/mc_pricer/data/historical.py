"""Historical price data and technical indicators from Yahoo Finance.

This module provides functionality to fetch historical OHLCV data and compute
common technical analysis indicators. Requires yfinance to be installed.

WARNING: This module makes network calls. Yahoo Finance data is unofficial
and provided for educational and research purposes only.

Note: This module is part of the mc-option-pricing-engine and is designed
for options pricing research. For LLM signal extraction and point-in-time
financial prediction, see the separate llm-signals project.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class HistoricalData:
    """Historical OHLCV data with metadata.

    Attributes
    ----------
    ticker : str
        Ticker symbol (e.g., 'AAPL', 'SPY').
    df : pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, and any
        computed indicators.
    start_date : str
        Start date of data in YYYY-MM-DD format.
    end_date : str
        End date of data in YYYY-MM-DD format.
    timestamp_utc : str
        ISO8601 timestamp when data was fetched (UTC).
    """

    ticker: str
    df: pd.DataFrame
    start_date: str
    end_date: str
    timestamp_utc: str


def fetch_historical(
    ticker: str,
    period: str | None = None,
    start: str | None = None,
    end: str | None = None,
    interval: Literal["1d", "1wk", "1mo"] = "1d",
) -> HistoricalData:
    """Fetch historical OHLCV data from Yahoo Finance.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'AAPL', 'SPY').
    period : str | None
        Period to fetch (e.g., '1mo', '3mo', '1y', '2y', '5y', 'max').
        If provided, start/end are ignored.
    start : str | None
        Start date in YYYY-MM-DD format. Required if period is None.
    end : str | None
        End date in YYYY-MM-DD format. Defaults to today if not provided.
    interval : str
        Data interval: '1d' (daily), '1wk' (weekly), '1mo' (monthly).

    Returns
    -------
    HistoricalData
        Object containing OHLCV DataFrame and metadata.

    Raises
    ------
    ImportError
        If yfinance is not installed.
    ValueError
        If ticker is invalid or no data is returned.

    Examples
    --------
    >>> data = fetch_historical('AAPL', period='1y')  # doctest: +SKIP
    >>> print(f"Fetched {len(data.df)} days of data")  # doctest: +SKIP
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required to fetch market data. Install with: pip install yfinance"
        ) from e

    stock = yf.Ticker(ticker)

    try:
        if period is not None:
            df = stock.history(period=period, interval=interval)
        else:
            if start is None:
                raise ValueError("Either 'period' or 'start' must be provided")
            df = stock.history(start=start, end=end, interval=interval)
    except Exception as e:
        raise ValueError(f"Failed to fetch history for '{ticker}': {e}") from e

    if df.empty:
        raise ValueError(f"No historical data returned for '{ticker}'")

    # Clean up DataFrame
    df = df.reset_index()
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
        df = df.set_index("Date")
    elif "Datetime" in df.columns:
        df["Datetime"] = pd.to_datetime(df["Datetime"]).dt.tz_localize(None)
        df = df.set_index("Datetime")

    # Keep only OHLCV columns
    cols_to_keep = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in cols_to_keep if c in df.columns]]

    timestamp = datetime.now(timezone.utc).isoformat()
    start_date = str(df.index.min().date())
    end_date = str(df.index.max().date())

    return HistoricalData(
        ticker=ticker,
        df=df,
        start_date=start_date,
        end_date=end_date,
        timestamp_utc=timestamp,
    )


# =============================================================================
# Technical Indicators
# =============================================================================


def add_sma(
    df: pd.DataFrame, column: str = "Close", periods: list[int] | None = None
) -> pd.DataFrame:
    """Add Simple Moving Average indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data.
    column : str
        Column to compute SMA on (default: 'Close').
    periods : list[int] | None
        List of periods for SMA (default: [10, 20, 50, 200]).

    Returns
    -------
    pd.DataFrame
        DataFrame with SMA columns added (e.g., SMA_10, SMA_20).
    """
    if periods is None:
        periods = [10, 20, 50, 200]

    df = df.copy()
    for p in periods:
        df[f"SMA_{p}"] = df[column].rolling(window=p).mean()
    return df


def add_ema(
    df: pd.DataFrame, column: str = "Close", periods: list[int] | None = None
) -> pd.DataFrame:
    """Add Exponential Moving Average indicators.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data.
    column : str
        Column to compute EMA on (default: 'Close').
    periods : list[int] | None
        List of periods for EMA (default: [12, 26, 50]).

    Returns
    -------
    pd.DataFrame
        DataFrame with EMA columns added (e.g., EMA_12, EMA_26).
    """
    if periods is None:
        periods = [12, 26, 50]

    df = df.copy()
    for p in periods:
        df[f"EMA_{p}"] = df[column].ewm(span=p, adjust=False).mean()
    return df


def add_rsi(df: pd.DataFrame, column: str = "Close", period: int = 14) -> pd.DataFrame:
    """Add Relative Strength Index (RSI).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data.
    column : str
        Column to compute RSI on (default: 'Close').
    period : int
        RSI period (default: 14).

    Returns
    -------
    pd.DataFrame
        DataFrame with RSI_{period} column added.

    Notes
    -----
    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over the period.
    When there are no losses, RSI = 100. When there are no gains, RSI = 0.
    """
    df = df.copy()
    delta = df[column].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    # Handle edge case where avg_loss is 0 (all gains, no losses) -> RSI = 100
    # Handle edge case where avg_gain is 0 (all losses, no gains) -> RSI = 0
    rsi = pd.Series(index=df.index, dtype=float)

    # Where both are valid and avg_loss > 0, compute normal RSI
    valid_mask = avg_gain.notna() & avg_loss.notna()
    nonzero_loss = avg_loss > 0

    # Normal case: RS = gain/loss, RSI = 100 - 100/(1+RS)
    rs = avg_gain / avg_loss.where(nonzero_loss, np.nan)
    rsi = 100 - (100 / (1 + rs))

    # Edge case: no losses (avg_loss == 0 and avg_gain >= 0) -> RSI = 100
    rsi = rsi.where(~(valid_mask & ~nonzero_loss & (avg_gain >= 0)), 100.0)

    # Edge case: no gains (avg_gain == 0 and avg_loss > 0) -> RSI = 0
    no_gains = (avg_gain == 0) & (avg_loss > 0)
    rsi = rsi.where(~(valid_mask & no_gains), 0.0)

    df[f"RSI_{period}"] = rsi
    return df


def add_macd(
    df: pd.DataFrame,
    column: str = "Close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Add MACD (Moving Average Convergence Divergence).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data.
    column : str
        Column to compute MACD on (default: 'Close').
    fast : int
        Fast EMA period (default: 12).
    slow : int
        Slow EMA period (default: 26).
    signal : int
        Signal line EMA period (default: 9).

    Returns
    -------
    pd.DataFrame
        DataFrame with MACD, MACD_Signal, and MACD_Hist columns added.
    """
    df = df.copy()
    ema_fast = df[column].ewm(span=fast, adjust=False).mean()
    ema_slow = df[column].ewm(span=slow, adjust=False).mean()

    df["MACD"] = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    return df


def add_bollinger_bands(
    df: pd.DataFrame,
    column: str = "Close",
    period: int = 20,
    std_dev: float = 2.0,
) -> pd.DataFrame:
    """Add Bollinger Bands.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data.
    column : str
        Column to compute bands on (default: 'Close').
    period : int
        SMA period for middle band (default: 20).
    std_dev : float
        Number of standard deviations for bands (default: 2.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with BB_Middle, BB_Upper, BB_Lower columns added.
    """
    df = df.copy()
    df["BB_Middle"] = df[column].rolling(window=period).mean()
    rolling_std = df[column].rolling(window=period).std()
    df["BB_Upper"] = df["BB_Middle"] + (std_dev * rolling_std)
    df["BB_Lower"] = df["BB_Middle"] - (std_dev * rolling_std)

    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add Average True Range (ATR).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with High, Low, Close columns.
    period : int
        ATR period (default: 14).

    Returns
    -------
    pd.DataFrame
        DataFrame with ATR_{period} column added.
    """
    df = df.copy()

    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift()).abs()
    low_close = (df["Low"] - df["Close"].shift()).abs()

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df[f"ATR_{period}"] = true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    return df


def add_realized_volatility(
    df: pd.DataFrame,
    column: str = "Close",
    period: int = 20,
    annualize: bool = True,
) -> pd.DataFrame:
    """Add realized volatility (historical volatility).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with price data.
    column : str
        Column to compute volatility on (default: 'Close').
    period : int
        Rolling window period (default: 20).
    annualize : bool
        Whether to annualize (multiply by sqrt(252)) (default: True).

    Returns
    -------
    pd.DataFrame
        DataFrame with RealizedVol_{period} column added.
    """
    df = df.copy()
    log_returns = np.log(df[column] / df[column].shift(1))
    vol = log_returns.rolling(window=period).std()

    if annualize:
        vol = vol * np.sqrt(252)

    df[f"RealizedVol_{period}"] = vol
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add a standard set of technical indicators.

    Adds: SMA(10,20,50,200), EMA(12,26), RSI(14), MACD, Bollinger Bands,
    ATR(14), and Realized Volatility(20).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with OHLCV data.

    Returns
    -------
    pd.DataFrame
        DataFrame with all indicators added.
    """
    df = add_sma(df, periods=[10, 20, 50, 200])
    df = add_ema(df, periods=[12, 26])
    df = add_rsi(df)
    df = add_macd(df)
    df = add_bollinger_bands(df)
    df = add_atr(df)
    df = add_realized_volatility(df)
    return df


# =============================================================================
# Signal Generation (for research)
# =============================================================================


def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Generate basic trading signals from indicators.

    This is for research/backtesting purposes only. Not investment advice.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with indicators already computed.

    Returns
    -------
    pd.DataFrame
        DataFrame with signal columns added:
        - Signal_SMA_Cross: 1 if SMA_20 > SMA_50, -1 otherwise
        - Signal_RSI: 1 if RSI < 30 (oversold), -1 if RSI > 70 (overbought)
        - Signal_MACD: 1 if MACD > Signal line, -1 otherwise
        - Signal_BB: 1 if price < lower band, -1 if price > upper band
    """
    df = df.copy()

    # SMA crossover signal
    if "SMA_20" in df.columns and "SMA_50" in df.columns:
        df["Signal_SMA_Cross"] = np.where(df["SMA_20"] > df["SMA_50"], 1, -1)

    # RSI signal
    if "RSI_14" in df.columns:
        df["Signal_RSI"] = np.where(df["RSI_14"] < 30, 1, np.where(df["RSI_14"] > 70, -1, 0))

    # MACD signal
    if "MACD" in df.columns and "MACD_Signal" in df.columns:
        df["Signal_MACD"] = np.where(df["MACD"] > df["MACD_Signal"], 1, -1)

    # Bollinger Band signal
    if all(c in df.columns for c in ["Close", "BB_Upper", "BB_Lower"]):
        df["Signal_BB"] = np.where(
            df["Close"] < df["BB_Lower"],
            1,
            np.where(df["Close"] > df["BB_Upper"], -1, 0),
        )

    return df
