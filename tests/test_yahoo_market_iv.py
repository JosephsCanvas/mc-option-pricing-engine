"""Tests for market data ingestion and implied volatility computation.

These tests are fully deterministic and do not make real network calls.
Network dependencies are mocked using pytest fixtures.
"""

from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from mc_pricer.analytics.black_scholes import bs_price
from mc_pricer.analytics.market_iv import expiry_to_years, quote_to_iv
from mc_pricer.data.yahoo_options import OptionQuote, compute_mids, filter_quotes


class TestOptionQuote:
    """Test OptionQuote dataclass and methods."""

    def test_mid_valid(self):
        """Test mid price computation with valid bid/ask."""
        quote = OptionQuote(
            ticker="SPY",
            expiry="2026-01-16",
            option_type="call",
            strike=450.0,
            bid=10.0,
            ask=10.5,
            last=10.2,
            iv_yahoo=0.20,
            open_interest=1000,
            volume=50,
            underlying_spot=450.0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )
        assert quote.mid() == 10.25

    def test_mid_invalid_bid_zero(self):
        """Test mid returns None when bid is zero."""
        quote = OptionQuote(
            ticker="SPY",
            expiry="2026-01-16",
            option_type="call",
            strike=450.0,
            bid=0.0,
            ask=10.5,
            last=None,
            iv_yahoo=None,
            open_interest=None,
            volume=None,
            underlying_spot=450.0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )
        assert quote.mid() is None

    def test_mid_invalid_ask_less_than_bid(self):
        """Test mid returns None when ask < bid."""
        quote = OptionQuote(
            ticker="SPY",
            expiry="2026-01-16",
            option_type="call",
            strike=450.0,
            bid=10.5,
            ask=10.0,
            last=None,
            iv_yahoo=None,
            open_interest=None,
            volume=None,
            underlying_spot=450.0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )
        assert quote.mid() is None

    def test_rel_spread(self):
        """Test relative spread computation."""
        quote = OptionQuote(
            ticker="SPY",
            expiry="2026-01-16",
            option_type="call",
            strike=450.0,
            bid=10.0,
            ask=11.0,
            last=None,
            iv_yahoo=None,
            open_interest=None,
            volume=None,
            underlying_spot=450.0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )
        rel_spread = quote.rel_spread()
        assert rel_spread is not None
        # spread = 1.0, mid = 10.5, rel_spread = 1.0/10.5 ≈ 0.0952
        assert abs(rel_spread - 1.0 / 10.5) < 1e-10

    def test_rel_spread_none_when_mid_none(self):
        """Test rel_spread returns None when mid is None."""
        quote = OptionQuote(
            ticker="SPY",
            expiry="2026-01-16",
            option_type="call",
            strike=450.0,
            bid=0.0,
            ask=10.0,
            last=None,
            iv_yahoo=None,
            open_interest=None,
            volume=None,
            underlying_spot=450.0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )
        assert quote.rel_spread() is None


class TestFilterQuotes:
    """Test filtering utilities."""

    def make_quote(self, bid: float, ask: float, volume: int | None = None, oi: int | None = None):
        """Helper to create test quotes."""
        return OptionQuote(
            ticker="SPY",
            expiry="2026-01-16",
            option_type="call",
            strike=450.0,
            bid=bid,
            ask=ask,
            last=None,
            iv_yahoo=None,
            open_interest=oi,
            volume=volume,
            underlying_spot=450.0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )

    def test_filter_removes_zero_bid(self):
        """Test that zero bid quotes are filtered out."""
        quotes = [
            self.make_quote(0.0, 10.0),  # Should be filtered (zero bid)
            self.make_quote(9.5, 10.0),  # Should pass (rel_spread = 0.5/9.75 ≈ 0.051)
        ]
        filtered = filter_quotes(quotes, min_bid=0.01)
        assert len(filtered) == 1
        assert filtered[0].bid == 9.5

    def test_filter_removes_zero_ask(self):
        """Test that zero ask quotes are filtered out."""
        quotes = [
            self.make_quote(5.0, 0.0),  # Should be filtered (zero ask)
            self.make_quote(9.5, 10.0),  # Should pass (rel_spread = 0.5/9.75 ≈ 0.051)
        ]
        filtered = filter_quotes(quotes, min_bid=0.01)
        assert len(filtered) == 1
        assert filtered[0].ask == 10.0

    def test_filter_removes_ask_less_than_bid(self):
        """Test that inverted bid/ask are filtered out."""
        quotes = [
            self.make_quote(10.0, 5.0),  # Should be filtered (ask < bid)
            self.make_quote(9.5, 10.0),  # Should pass (rel_spread = 0.5/9.75 ≈ 0.051)
        ]
        filtered = filter_quotes(quotes, min_bid=0.01)
        assert len(filtered) == 1

    def test_filter_min_bid(self):
        """Test minimum bid filter."""
        quotes = [
            self.make_quote(0.005, 0.01),  # Should be filtered (bid too low)
            self.make_quote(0.095, 0.10),  # Should pass (rel_spread = 0.005/0.0975 ≈ 0.051)
        ]
        filtered = filter_quotes(quotes, min_bid=0.01)
        assert len(filtered) == 1
        assert filtered[0].bid == 0.095

    def test_filter_max_rel_spread(self):
        """Test maximum relative spread filter."""
        quotes = [
            self.make_quote(10.0, 15.0),  # rel_spread = 5/12.5 = 0.4
            self.make_quote(10.0, 11.0),  # rel_spread = 1/10.5 ≈ 0.095
        ]
        filtered = filter_quotes(quotes, max_rel_spread=0.30)
        assert len(filtered) == 1
        assert filtered[0].ask == 11.0

    def test_filter_min_volume(self):
        """Test minimum volume filter."""
        quotes = [
            self.make_quote(9.5, 10.0, volume=10),  # Should be filtered (volume too low)
            self.make_quote(9.5, 10.0, volume=100),  # Should pass
            self.make_quote(9.5, 10.0, volume=None),  # Should be filtered (no volume)
        ]
        filtered = filter_quotes(quotes, min_bid=0.01, min_volume=50)
        assert len(filtered) == 1
        assert filtered[0].volume == 100

    def test_filter_min_oi(self):
        """Test minimum open interest filter."""
        quotes = [
            self.make_quote(9.5, 10.0, oi=10),  # Should be filtered (OI too low)
            self.make_quote(9.5, 10.0, oi=1000),  # Should pass
            self.make_quote(9.5, 10.0, oi=None),  # Should be filtered (no OI)
        ]
        filtered = filter_quotes(quotes, min_bid=0.01, min_oi=500)
        assert len(filtered) == 1
        assert filtered[0].open_interest == 1000


class TestComputeMids:
    """Test compute_mids utility."""

    def test_compute_mids_filters_invalid(self):
        """Test that compute_mids only returns quotes with valid mids."""
        quotes = [
            OptionQuote(
                ticker="SPY",
                expiry="2026-01-16",
                option_type="call",
                strike=450.0,
                bid=0.0,
                ask=10.0,
                last=None,
                iv_yahoo=None,
                open_interest=None,
                volume=None,
                underlying_spot=450.0,
                timestamp_utc="2025-12-31T12:00:00Z",
            ),
            OptionQuote(
                ticker="SPY",
                expiry="2026-01-16",
                option_type="call",
                strike=450.0,
                bid=5.0,
                ask=10.0,
                last=None,
                iv_yahoo=None,
                open_interest=None,
                volume=None,
                underlying_spot=450.0,
                timestamp_utc="2025-12-31T12:00:00Z",
            ),
        ]
        with_mids = compute_mids(quotes)
        assert len(with_mids) == 1
        assert with_mids[0].bid == 5.0


class TestQuoteToIV:
    """Test implied volatility computation from quotes."""

    def test_quote_to_iv_recovers_known_vol(self):
        """Test that IV solver recovers known volatility from BS price."""
        # Setup: generate BS price with known parameters
        S0 = 450.0
        K = 450.0
        r = 0.05
        T = 1.0
        sigma_true = 0.20

        # Compute BS price
        bs_call_price = bs_price(S0, K, r, T, sigma_true, "call")

        # Create quote with mid = BS price
        spread = 0.10
        quote = OptionQuote(
            ticker="SPY",
            expiry="2026-12-31",
            option_type="call",
            strike=K,
            bid=bs_call_price - spread / 2,
            ask=bs_call_price + spread / 2,
            last=None,
            iv_yahoo=None,
            open_interest=None,
            volume=None,
            underlying_spot=S0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )

        # Solve for IV
        iv = quote_to_iv(quote, r=r, T=T)

        assert iv is not None
        assert abs(iv - sigma_true) < 1e-6

    def test_quote_to_iv_call_below_arbitrage_bound(self):
        """Test that call price below intrinsic value returns None."""
        S0 = 450.0
        K = 400.0
        r = 0.05
        T = 1.0

        # Intrinsic value for call is max(S0 - K*exp(-rT), 0)
        intrinsic = S0 - K * np.exp(-r * T)

        # Set mid below intrinsic
        bad_price = intrinsic - 1.0

        quote = OptionQuote(
            ticker="SPY",
            expiry="2026-12-31",
            option_type="call",
            strike=K,
            bid=bad_price - 0.05,
            ask=bad_price + 0.05,
            last=None,
            iv_yahoo=None,
            open_interest=None,
            volume=None,
            underlying_spot=S0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )

        iv = quote_to_iv(quote, r=r, T=T)
        assert iv is None

    def test_quote_to_iv_call_above_arbitrage_bound(self):
        """Test that call price above spot returns None."""
        S0 = 450.0
        K = 400.0
        r = 0.05
        T = 1.0

        # Call cannot be worth more than spot
        bad_price = S0 + 1.0

        quote = OptionQuote(
            ticker="SPY",
            expiry="2026-12-31",
            option_type="call",
            strike=K,
            bid=bad_price - 0.05,
            ask=bad_price + 0.05,
            last=None,
            iv_yahoo=None,
            open_interest=None,
            volume=None,
            underlying_spot=S0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )

        iv = quote_to_iv(quote, r=r, T=T)
        assert iv is None

    def test_quote_to_iv_put_below_arbitrage_bound(self):
        """Test that put price below intrinsic value returns None."""
        S0 = 450.0
        K = 500.0
        r = 0.05
        T = 1.0

        # Intrinsic value for put is max(K*exp(-rT) - S0, 0)
        intrinsic = K * np.exp(-r * T) - S0

        # Set mid below intrinsic
        bad_price = intrinsic - 1.0

        quote = OptionQuote(
            ticker="SPY",
            expiry="2026-12-31",
            option_type="put",
            strike=K,
            bid=bad_price - 0.05,
            ask=bad_price + 0.05,
            last=None,
            iv_yahoo=None,
            open_interest=None,
            volume=None,
            underlying_spot=S0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )

        iv = quote_to_iv(quote, r=r, T=T)
        assert iv is None

    def test_quote_to_iv_put_above_arbitrage_bound(self):
        """Test that put price above discounted strike returns None."""
        S0 = 450.0
        K = 500.0
        r = 0.05
        T = 1.0

        # Put cannot be worth more than K*exp(-rT)
        max_put = K * np.exp(-r * T)
        bad_price = max_put + 1.0

        quote = OptionQuote(
            ticker="SPY",
            expiry="2026-12-31",
            option_type="put",
            strike=K,
            bid=bad_price - 0.05,
            ask=bad_price + 0.05,
            last=None,
            iv_yahoo=None,
            open_interest=None,
            volume=None,
            underlying_spot=S0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )

        iv = quote_to_iv(quote, r=r, T=T)
        assert iv is None

    def test_quote_to_iv_returns_none_when_no_mid(self):
        """Test that IV computation returns None when mid is None."""
        quote = OptionQuote(
            ticker="SPY",
            expiry="2026-12-31",
            option_type="call",
            strike=450.0,
            bid=0.0,
            ask=10.0,
            last=None,
            iv_yahoo=None,
            open_interest=None,
            volume=None,
            underlying_spot=450.0,
            timestamp_utc="2025-12-31T12:00:00Z",
        )

        iv = quote_to_iv(quote, r=0.05, T=1.0)
        assert iv is None


class TestExpiryToYears:
    """Test expiry date conversion."""

    def test_expiry_to_years_one_year(self):
        """Test conversion for approximately one year."""
        ref = datetime(2025, 12, 31, 12, 0, 0)
        T = expiry_to_years("2026-12-31", reference_date=ref)
        assert abs(T - 1.0) < 0.01  # 365 days / 365 = 1.0

    def test_expiry_to_years_half_year(self):
        """Test conversion for approximately half year."""
        ref = datetime(2025, 12, 31, 12, 0, 0)
        T = expiry_to_years("2026-06-30", reference_date=ref)
        # 181 days / 365 ≈ 0.496
        assert 0.48 < T < 0.52

    def test_expiry_to_years_invalid_format(self):
        """Test that invalid format raises ValueError."""
        ref = datetime(2025, 12, 31, 12, 0, 0)
        with pytest.raises(ValueError, match="Invalid expiry format"):
            expiry_to_years("31-12-2026", reference_date=ref)

    def test_expiry_to_years_past_expiry(self):
        """Test that past expiry raises ValueError."""
        ref = datetime(2025, 12, 31, 12, 0, 0)
        with pytest.raises(ValueError, match="in the past"):
            expiry_to_years("2025-01-01", reference_date=ref)


class TestYahooFetchMocked:
    """Test Yahoo Finance fetcher with mocked network calls."""

    def test_fetch_options_chain_mocked(self, monkeypatch):
        """Test fetch_options_chain with mocked yfinance."""
        # Check if pandas is available (needed for yfinance)
        try:
            import pandas as pd
        except ImportError:
            pytest.skip("pandas not installed (required for yfinance mock)")

        # Mock yfinance module
        mock_yf = MagicMock()

        calls_data = pd.DataFrame(
            {
                "strike": [440.0, 450.0, 460.0],
                "bid": [12.0, 8.0, 5.0],
                "ask": [12.5, 8.5, 5.5],
                "lastPrice": [12.2, 8.2, 5.2],
                "impliedVolatility": [0.21, 0.20, 0.22],
                "openInterest": [1000, 2000, 1500],
                "volume": [50, 100, 75],
            }
        )

        puts_data = pd.DataFrame(
            {
                "strike": [440.0, 450.0, 460.0],
                "bid": [5.0, 8.0, 12.0],
                "ask": [5.5, 8.5, 12.5],
                "lastPrice": [5.2, 8.2, 12.2],
                "impliedVolatility": [0.22, 0.20, 0.21],
                "openInterest": [1500, 2000, 1000],
                "volume": [75, 100, 50],
            }
        )

        # Mock option chain
        mock_chain = MagicMock()
        mock_chain.calls = calls_data
        mock_chain.puts = puts_data

        # Mock ticker
        mock_ticker = MagicMock()
        mock_ticker.option_chain.return_value = mock_chain
        mock_ticker.info = {"currentPrice": 450.0}
        mock_ticker.options = ["2026-01-16", "2026-02-20"]

        # Mock yfinance.Ticker
        mock_yf.Ticker.return_value = mock_ticker

        # Patch the yfinance import
        with patch.dict("sys.modules", {"yfinance": mock_yf}):
            from mc_pricer.data.yahoo_options import fetch_options_chain

            quotes = fetch_options_chain("SPY", "2026-01-16")

        # Verify results
        assert len(quotes) == 6  # 3 calls + 3 puts
        assert sum(1 for q in quotes if q.option_type == "call") == 3
        assert sum(1 for q in quotes if q.option_type == "put") == 3

        # Check a sample quote
        call_450 = [q for q in quotes if q.strike == 450.0 and q.option_type == "call"][0]
        assert call_450.bid == 8.0
        assert call_450.ask == 8.5
        assert call_450.underlying_spot == 450.0

    def test_fetch_options_chain_yfinance_not_installed(self, monkeypatch):
        """Test that ImportError is raised when yfinance is not installed."""
        # Remove yfinance from sys.modules if present
        import sys

        original_modules = sys.modules.copy()

        # Clear yfinance from modules
        if "yfinance" in sys.modules:
            del sys.modules["yfinance"]

        # Mock import to raise ImportError
        def mock_import(name, *args, **kwargs):
            if name == "yfinance":
                raise ImportError("No module named 'yfinance'")
            return original_modules.get(name)

        with patch("builtins.__import__", side_effect=mock_import):
            from mc_pricer.data.yahoo_options import fetch_options_chain

            with pytest.raises(ImportError, match="yfinance is required"):
                fetch_options_chain("SPY", "2026-01-16")
