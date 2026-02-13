"""Tests for historical data and technical indicators module."""

import numpy as np
import pandas as pd
import pytest

from mc_pricer.data.historical import (
    HistoricalData,
    add_all_indicators,
    add_atr,
    add_bollinger_bands,
    add_ema,
    add_macd,
    add_realized_volatility,
    add_rsi,
    add_sma,
    generate_signals,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """Create sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100

    # Generate random walk price series
    returns = np.random.randn(n) * 0.02
    close = 100 * np.exp(np.cumsum(returns))

    # Generate consistent OHLC
    high = close * (1 + np.abs(np.random.randn(n) * 0.01))
    low = close * (1 - np.abs(np.random.randn(n) * 0.01))
    open_ = low + (high - low) * np.random.rand(n)
    volume = np.random.randint(1000000, 10000000, n)

    dates = pd.date_range("2024-01-01", periods=n, freq="D")

    return pd.DataFrame(
        {
            "Open": open_,
            "High": high,
            "Low": low,
            "Close": close,
            "Volume": volume,
        },
        index=dates,
    )


@pytest.fixture
def small_df() -> pd.DataFrame:
    """Create small DataFrame for edge case testing."""
    return pd.DataFrame(
        {
            "Open": [100, 101, 102, 103, 104],
            "High": [101, 102, 103, 104, 105],
            "Low": [99, 100, 101, 102, 103],
            "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
            "Volume": [1000, 1100, 1200, 1300, 1400],
        },
        index=pd.date_range("2024-01-01", periods=5, freq="D"),
    )


# =============================================================================
# SMA Tests
# =============================================================================


class TestSMA:
    """Tests for Simple Moving Average."""

    def test_sma_adds_columns(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """SMA adds expected columns."""
        df = add_sma(sample_ohlcv_df, periods=[10, 20])
        assert "SMA_10" in df.columns
        assert "SMA_20" in df.columns

    def test_sma_values_correct(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """SMA values match manual calculation."""
        df = add_sma(sample_ohlcv_df, periods=[5])

        # Check a specific value (first valid)
        expected = sample_ohlcv_df["Close"].iloc[:5].mean()
        actual = df["SMA_5"].iloc[4]
        assert np.isclose(actual, expected, rtol=1e-10)

    def test_sma_nan_at_start(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """SMA has NaN values before enough data."""
        df = add_sma(sample_ohlcv_df, periods=[10])
        assert df["SMA_10"].iloc[:9].isna().all()
        assert not df["SMA_10"].iloc[9:].isna().any()

    def test_sma_default_periods(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """SMA uses default periods [10, 20, 50, 200]."""
        df = add_sma(sample_ohlcv_df)
        assert "SMA_10" in df.columns
        assert "SMA_20" in df.columns
        assert "SMA_50" in df.columns
        assert "SMA_200" in df.columns


# =============================================================================
# EMA Tests
# =============================================================================


class TestEMA:
    """Tests for Exponential Moving Average."""

    def test_ema_adds_columns(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """EMA adds expected columns."""
        df = add_ema(sample_ohlcv_df, periods=[12, 26])
        assert "EMA_12" in df.columns
        assert "EMA_26" in df.columns

    def test_ema_no_nan(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """EMA should not have NaN (ewm handles this)."""
        df = add_ema(sample_ohlcv_df, periods=[12])
        # EMA starts from first value
        assert not df["EMA_12"].isna().any()

    def test_ema_converges_to_sma(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """EMA should roughly follow price trend."""
        df = add_ema(sample_ohlcv_df, periods=[10])
        # EMA should be between min and max of close
        assert df["EMA_10"].min() >= df["Close"].min() * 0.9
        assert df["EMA_10"].max() <= df["Close"].max() * 1.1


# =============================================================================
# RSI Tests
# =============================================================================


class TestRSI:
    """Tests for Relative Strength Index."""

    def test_rsi_adds_column(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """RSI adds expected column."""
        df = add_rsi(sample_ohlcv_df, period=14)
        assert "RSI_14" in df.columns

    def test_rsi_bounded(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """RSI values should be between 0 and 100."""
        df = add_rsi(sample_ohlcv_df, period=14)
        valid_rsi = df["RSI_14"].dropna()
        assert (valid_rsi >= 0).all()
        assert (valid_rsi <= 100).all()

    def test_rsi_extreme_up(self) -> None:
        """RSI should be high for mostly rising prices."""
        # Create a series that mostly goes up with small fluctuations
        np.random.seed(42)
        # Need enough data points for RSI warmup (period=14 requires 14+ points)
        base = np.linspace(100, 200, 50)  # 50 points going up
        noise = np.random.randn(len(base)) * 0.5  # small noise
        prices = pd.DataFrame({"Close": base + noise})
        df = add_rsi(prices, period=14)
        # After warmup, RSI should be high (strong uptrend)
        valid_rsi = df["RSI_14"].dropna()
        assert len(valid_rsi) > 10, f"Expected valid RSI values, got {len(valid_rsi)}"
        assert valid_rsi.iloc[-1] > 60, f"Expected RSI > 60 for uptrend, got {valid_rsi.iloc[-1]}"

    def test_rsi_extreme_down(self) -> None:
        """RSI should be near 0 for consistently falling prices."""
        prices = pd.DataFrame({"Close": np.arange(150, 100, -1.0)})
        df = add_rsi(prices, period=14)
        # After warmup, RSI should be very low
        assert df["RSI_14"].iloc[-1] < 10


# =============================================================================
# MACD Tests
# =============================================================================


class TestMACD:
    """Tests for MACD indicator."""

    def test_macd_adds_columns(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """MACD adds expected columns."""
        df = add_macd(sample_ohlcv_df)
        assert "MACD" in df.columns
        assert "MACD_Signal" in df.columns
        assert "MACD_Hist" in df.columns

    def test_macd_histogram_is_difference(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """MACD histogram equals MACD minus Signal."""
        df = add_macd(sample_ohlcv_df)
        diff = df["MACD"] - df["MACD_Signal"]
        assert np.allclose(df["MACD_Hist"], diff, rtol=1e-10)


# =============================================================================
# Bollinger Bands Tests
# =============================================================================


class TestBollingerBands:
    """Tests for Bollinger Bands."""

    def test_bb_adds_columns(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Bollinger Bands adds expected columns."""
        df = add_bollinger_bands(sample_ohlcv_df)
        assert "BB_Middle" in df.columns
        assert "BB_Upper" in df.columns
        assert "BB_Lower" in df.columns

    def test_bb_upper_above_lower(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Upper band should always be above lower band."""
        df = add_bollinger_bands(sample_ohlcv_df)
        valid = df[["BB_Upper", "BB_Lower"]].dropna()
        assert (valid["BB_Upper"] > valid["BB_Lower"]).all()

    def test_bb_middle_is_sma(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Middle band should equal SMA."""
        df = add_bollinger_bands(sample_ohlcv_df, period=20)
        df = add_sma(df, periods=[20])
        valid = df[["BB_Middle", "SMA_20"]].dropna()
        assert np.allclose(valid["BB_Middle"], valid["SMA_20"], rtol=1e-10)


# =============================================================================
# ATR Tests
# =============================================================================


class TestATR:
    """Tests for Average True Range."""

    def test_atr_adds_column(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """ATR adds expected column."""
        df = add_atr(sample_ohlcv_df, period=14)
        assert "ATR_14" in df.columns

    def test_atr_positive(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """ATR should always be positive."""
        df = add_atr(sample_ohlcv_df, period=14)
        valid_atr = df["ATR_14"].dropna()
        assert (valid_atr > 0).all()


# =============================================================================
# Realized Volatility Tests
# =============================================================================


class TestRealizedVolatility:
    """Tests for realized volatility calculation."""

    def test_rv_adds_column(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Realized volatility adds expected column."""
        df = add_realized_volatility(sample_ohlcv_df, period=20)
        assert "RealizedVol_20" in df.columns

    def test_rv_positive(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Realized volatility should be positive."""
        df = add_realized_volatility(sample_ohlcv_df, period=20)
        valid = df["RealizedVol_20"].dropna()
        assert (valid >= 0).all()

    def test_rv_annualization(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Annualized vol should be ~sqrt(252) times daily vol."""
        df_ann = add_realized_volatility(sample_ohlcv_df, period=20, annualize=True)
        df_daily = add_realized_volatility(sample_ohlcv_df, period=20, annualize=False)

        ratio = df_ann["RealizedVol_20"].dropna() / df_daily["RealizedVol_20"].dropna()
        expected_ratio = np.sqrt(252)
        assert np.allclose(ratio, expected_ratio, rtol=1e-10)


# =============================================================================
# All Indicators Tests
# =============================================================================


class TestAllIndicators:
    """Tests for add_all_indicators convenience function."""

    def test_adds_all_expected_columns(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """All expected indicator columns are added."""
        df = add_all_indicators(sample_ohlcv_df)

        expected_cols = [
            "SMA_10",
            "SMA_20",
            "SMA_50",
            "SMA_200",
            "EMA_12",
            "EMA_26",
            "RSI_14",
            "MACD",
            "MACD_Signal",
            "MACD_Hist",
            "BB_Middle",
            "BB_Upper",
            "BB_Lower",
            "ATR_14",
            "RealizedVol_20",
        ]

        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_does_not_modify_original(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Original DataFrame is not modified."""
        original_cols = list(sample_ohlcv_df.columns)
        _ = add_all_indicators(sample_ohlcv_df)
        assert list(sample_ohlcv_df.columns) == original_cols


# =============================================================================
# Signal Generation Tests
# =============================================================================


class TestSignalGeneration:
    """Tests for trading signal generation."""

    def test_signals_adds_columns(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """Signal generation adds expected columns."""
        df = add_all_indicators(sample_ohlcv_df)
        df = generate_signals(df)

        assert "Signal_SMA_Cross" in df.columns
        assert "Signal_RSI" in df.columns
        assert "Signal_MACD" in df.columns
        assert "Signal_BB" in df.columns

    def test_sma_cross_values(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """SMA cross signal should be -1 or 1."""
        df = add_all_indicators(sample_ohlcv_df)
        df = generate_signals(df)

        unique_vals = df["Signal_SMA_Cross"].dropna().unique()
        assert set(unique_vals).issubset({-1, 1})

    def test_rsi_signal_values(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """RSI signal should be -1, 0, or 1."""
        df = add_all_indicators(sample_ohlcv_df)
        df = generate_signals(df)

        unique_vals = df["Signal_RSI"].dropna().unique()
        assert set(unique_vals).issubset({-1, 0, 1})


# =============================================================================
# HistoricalData Dataclass Tests
# =============================================================================


class TestHistoricalData:
    """Tests for HistoricalData dataclass."""

    def test_dataclass_creation(self, sample_ohlcv_df: pd.DataFrame) -> None:
        """HistoricalData can be created."""
        data = HistoricalData(
            ticker="TEST",
            df=sample_ohlcv_df,
            start_date="2024-01-01",
            end_date="2024-04-10",
            timestamp_utc="2024-04-10T12:00:00Z",
        )

        assert data.ticker == "TEST"
        assert len(data.df) == len(sample_ohlcv_df)
        assert data.start_date == "2024-01-01"


# =============================================================================
# Integration Test (requires network - skipped by default)
# =============================================================================


@pytest.mark.skip(reason="Requires network access to Yahoo Finance")
class TestFetchHistorical:
    """Integration tests for fetch_historical (requires network)."""

    def test_fetch_spy_1mo(self) -> None:
        """Fetch SPY data for 1 month."""
        from mc_pricer.data.historical import fetch_historical

        data = fetch_historical("SPY", period="1mo")
        assert data.ticker == "SPY"
        assert len(data.df) > 15  # At least 15 trading days
        assert "Close" in data.df.columns

    def test_fetch_with_indicators(self) -> None:
        """Fetch data and add indicators."""
        from mc_pricer.data.historical import fetch_historical

        data = fetch_historical("AAPL", period="3mo")
        df = add_all_indicators(data.df)
        df = generate_signals(df)

        assert "RSI_14" in df.columns
        assert "Signal_MACD" in df.columns
