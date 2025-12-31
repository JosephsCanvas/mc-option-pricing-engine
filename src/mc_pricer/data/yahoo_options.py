"""Yahoo Finance options chain data fetcher.

This module provides functionality to fetch real-time options chain data
from Yahoo Finance using the yfinance package. Requires yfinance to be
installed (optional dependency).

WARNING: This module makes network calls. Do not import automatically
in production code. Yahoo Finance data is unofficial and provided for
educational and research purposes only.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal

import numpy as np


@dataclass
class OptionQuote:
    """Single option quote from Yahoo Finance.

    Attributes
    ----------
    ticker : str
        Underlying ticker symbol (e.g., 'SPY', 'AAPL').
    expiry : str
        Expiration date in YYYY-MM-DD format.
    option_type : Literal["call", "put"]
        Type of option.
    strike : float
        Strike price.
    bid : float
        Bid price.
    ask : float
        Ask price.
    last : float | None
        Last traded price (may be None if not available).
    iv_yahoo : float | None
        Implied volatility from Yahoo (if available, as decimal e.g. 0.25 for 25%).
    open_interest : int | None
        Open interest.
    volume : int | None
        Trading volume.
    underlying_spot : float
        Spot price of underlying at fetch time.
    timestamp_utc : str
        ISO8601 timestamp when data was fetched (UTC).
    """

    ticker: str
    expiry: str
    option_type: Literal["call", "put"]
    strike: float
    bid: float
    ask: float
    last: float | None
    iv_yahoo: float | None
    open_interest: int | None
    volume: int | None
    underlying_spot: float
    timestamp_utc: str

    def mid(self) -> float | None:
        """Compute mid price as (bid + ask) / 2.

        Returns
        -------
        float | None
            Mid price if both bid and ask are positive, otherwise None.
        """
        if self.bid > 0 and self.ask > 0 and self.ask >= self.bid:
            return (self.bid + self.ask) / 2.0
        return None

    def rel_spread(self) -> float | None:
        """Compute relative bid-ask spread.

        Returns
        -------
        float | None
            (ask - bid) / max(mid, 1e-12) if mid is valid, else None.
        """
        mid = self.mid()
        if mid is None:
            return None
        return (self.ask - self.bid) / max(mid, 1e-12)


def fetch_options_chain(
    ticker: str,
    expiry: str,
) -> list[OptionQuote]:
    """Fetch options chain from Yahoo Finance for a given ticker and expiry.

    Parameters
    ----------
    ticker : str
        Ticker symbol (e.g., 'SPY', 'AAPL').
    expiry : str
        Expiration date in YYYY-MM-DD format. Must be a valid expiry
        available on Yahoo Finance for this ticker.

    Returns
    -------
    list[OptionQuote]
        List of option quotes (calls and puts combined).

    Raises
    ------
    ImportError
        If yfinance is not installed.
    ValueError
        If ticker is invalid or expiry is not available.

    Notes
    -----
    This function makes a network call to Yahoo Finance. Results may
    vary based on market hours and data availability. Yahoo Finance
    data is unofficial and should be used for educational/research
    purposes only.

    Examples
    --------
    >>> quotes = fetch_options_chain('SPY', '2026-01-16')  # doctest: +SKIP
    >>> print(f"Fetched {len(quotes)} quotes")  # doctest: +SKIP
    """
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError(
            "yfinance is required to fetch market data. Install with: pip install yfinance"
        ) from e

    # Fetch ticker data
    stock = yf.Ticker(ticker)

    # Get current spot price
    try:
        # Try to get current price from info
        info = stock.info
        spot = info.get("currentPrice") or info.get("regularMarketPrice")
        if spot is None:
            # Fallback: use history
            hist = stock.history(period="1d")
            if hist.empty:
                raise ValueError(f"Could not fetch spot price for ticker '{ticker}'")
            spot = float(hist["Close"].iloc[-1])
    except Exception as e:
        raise ValueError(f"Failed to fetch spot price for '{ticker}': {e}") from e

    # Fetch options chain for the given expiry
    try:
        chain = stock.option_chain(expiry)
    except Exception as e:
        # Check available expiries
        try:
            available = stock.options
            raise ValueError(
                f"Expiry '{expiry}' not available for '{ticker}'. "
                f"Available expiries: {list(available)}"
            ) from e
        except Exception:
            raise ValueError(
                f"Failed to fetch options chain for '{ticker}' expiry '{expiry}': {e}"
            ) from e

    # Get current timestamp
    timestamp = datetime.now(timezone.utc).isoformat()

    quotes: list[OptionQuote] = []

    # Process calls
    if hasattr(chain, "calls") and not chain.calls.empty:
        for _, row in chain.calls.iterrows():
            quote = OptionQuote(
                ticker=ticker,
                expiry=expiry,
                option_type="call",
                strike=float(row.get("strike", np.nan)),
                bid=float(row.get("bid", 0.0)),
                ask=float(row.get("ask", 0.0)),
                last=float(row["lastPrice"])
                if "lastPrice" in row and not np.isnan(row["lastPrice"])
                else None,
                iv_yahoo=float(row["impliedVolatility"])
                if "impliedVolatility" in row and not np.isnan(row["impliedVolatility"])
                else None,
                open_interest=int(row["openInterest"])
                if "openInterest" in row and not np.isnan(row["openInterest"])
                else None,
                volume=int(row["volume"])
                if "volume" in row and not np.isnan(row["volume"])
                else None,
                underlying_spot=float(spot),
                timestamp_utc=timestamp,
            )
            quotes.append(quote)

    # Process puts
    if hasattr(chain, "puts") and not chain.puts.empty:
        for _, row in chain.puts.iterrows():
            quote = OptionQuote(
                ticker=ticker,
                expiry=expiry,
                option_type="put",
                strike=float(row.get("strike", np.nan)),
                bid=float(row.get("bid", 0.0)),
                ask=float(row.get("ask", 0.0)),
                last=float(row["lastPrice"])
                if "lastPrice" in row and not np.isnan(row["lastPrice"])
                else None,
                iv_yahoo=float(row["impliedVolatility"])
                if "impliedVolatility" in row and not np.isnan(row["impliedVolatility"])
                else None,
                open_interest=int(row["openInterest"])
                if "openInterest" in row and not np.isnan(row["openInterest"])
                else None,
                volume=int(row["volume"])
                if "volume" in row and not np.isnan(row["volume"])
                else None,
                underlying_spot=float(spot),
                timestamp_utc=timestamp,
            )
            quotes.append(quote)

    return quotes


def filter_quotes(
    quotes: list[OptionQuote],
    *,
    min_bid: float = 0.01,
    max_rel_spread: float = 0.30,
    min_volume: int | None = None,
    min_oi: int | None = None,
) -> list[OptionQuote]:
    """Filter option quotes based on liquidity and quality criteria.

    Parameters
    ----------
    quotes : list[OptionQuote]
        List of option quotes to filter.
    min_bid : float, optional
        Minimum bid price (default: 0.01).
    max_rel_spread : float, optional
        Maximum relative bid-ask spread (default: 0.30 = 30%).
    min_volume : int | None, optional
        Minimum volume (if provided, filters out quotes below this).
    min_oi : int | None, optional
        Minimum open interest (if provided, filters out quotes below this).

    Returns
    -------
    list[OptionQuote]
        Filtered list of quotes.

    Notes
    -----
    Filtering criteria:
    - Bid must be > 0 and >= min_bid
    - Ask must be > 0
    - Ask must be >= bid
    - Relative spread must be <= max_rel_spread
    - If min_volume is set, volume must be >= min_volume
    - If min_oi is set, open_interest must be >= min_oi
    """
    filtered: list[OptionQuote] = []

    for quote in quotes:
        # Check bid/ask validity
        if quote.bid <= 0 or quote.ask <= 0:
            continue
        if quote.ask < quote.bid:
            continue
        if quote.bid < min_bid:
            continue

        # Check relative spread
        rel_spread = quote.rel_spread()
        if rel_spread is None or rel_spread > max_rel_spread:
            continue

        # Check volume if specified
        if min_volume is not None:
            if quote.volume is None or quote.volume < min_volume:
                continue

        # Check open interest if specified
        if min_oi is not None:
            if quote.open_interest is None or quote.open_interest < min_oi:
                continue

        filtered.append(quote)

    return filtered


def compute_mids(quotes: list[OptionQuote]) -> list[OptionQuote]:
    """Filter quotes to only those with valid mid prices.

    Parameters
    ----------
    quotes : list[OptionQuote]
        List of option quotes.

    Returns
    -------
    list[OptionQuote]
        List of quotes where mid() returns a valid value.
    """
    return [q for q in quotes if q.mid() is not None]
