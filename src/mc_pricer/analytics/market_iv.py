"""Implied volatility computation from market quotes.

This module provides utilities to compute implied volatilities from
market option quotes, with proper arbitrage bound checking.
"""

import math
from datetime import datetime

from mc_pricer.analytics.implied_vol import implied_vol
from mc_pricer.data.yahoo_options import OptionQuote


def quote_to_iv(
    quote: OptionQuote,
    r: float,
    T: float,  # noqa: N803
) -> float | None:
    """Compute implied volatility from an option quote.

    Parameters
    ----------
    quote : OptionQuote
        Market option quote with bid/ask prices.
    r : float
        Risk-free interest rate (annualized, as decimal e.g. 0.05 for 5%).
    T : float
        Time to maturity in years. Should be computed externally using
        the expiry date and current date with ACT/365 convention.

    Returns
    -------
    float | None
        Implied volatility (as decimal, e.g. 0.25 for 25%) if successful,
        None if:
        - Mid price is invalid (None)
        - Mid price violates arbitrage bounds
        - Implied vol solver fails

    Notes
    -----
    This function:
    1. Computes mid price from bid/ask
    2. Checks arbitrage bounds
    3. Solves for implied volatility using bisection method

    If the mid price is outside arbitrage-free bounds or if the solver
    fails, returns None rather than raising an exception. This is
    appropriate for market data which may contain stale or erroneous quotes.

    Examples
    --------
    >>> from mc_pricer.data.yahoo_options import OptionQuote
    >>> quote = OptionQuote(
    ...     ticker='SPY', expiry='2026-01-16', option_type='call',
    ...     strike=450.0, bid=10.0, ask=10.5, last=10.2,
    ...     iv_yahoo=0.20, open_interest=1000, volume=50,
    ...     underlying_spot=450.0, timestamp_utc='2025-12-31T12:00:00Z'
    ... )
    >>> iv = quote_to_iv(quote, r=0.05, T=1.0)
    >>> iv is not None
    True
    """
    # Get mid price
    mid = quote.mid()
    if mid is None:
        return None

    S0 = quote.underlying_spot  # noqa: N806
    K = quote.strike  # noqa: N806

    # Basic sanity checks
    if S0 <= 0 or K <= 0 or T <= 0:
        return None

    # Check arbitrage bounds
    discount = math.exp(-r * T)

    if quote.option_type == "call":
        # Call bounds: [max(S0 - K*discount, 0), S0]
        lower_bound = max(S0 - K * discount, 0.0)
        upper_bound = S0

        # Allow small tolerance for floating point
        if mid < lower_bound - 1e-8 or mid > upper_bound + 1e-8:
            return None
    elif quote.option_type == "put":
        # Put bounds: [max(K*discount - S0, 0), K*discount]
        lower_bound = max(K * discount - S0, 0.0)
        upper_bound = K * discount

        if mid < lower_bound - 1e-8 or mid > upper_bound + 1e-8:
            return None
    else:
        return None

    # Attempt to solve for implied volatility
    try:
        iv = implied_vol(
            price=mid,
            S0=S0,
            K=K,
            r=r,
            T=T,
            option_type=quote.option_type,
        )
        return iv
    except (ValueError, RuntimeError):
        # Solver failed or encountered invalid inputs
        return None


def expiry_to_years(expiry_str: str, reference_date: datetime | None = None) -> float:
    """Convert expiry date string to time in years using ACT/365.

    Parameters
    ----------
    expiry_str : str
        Expiry date in YYYY-MM-DD format.
    reference_date : datetime | None, optional
        Reference date for time calculation. If None, uses current UTC time.

    Returns
    -------
    float
        Time to expiry in years (ACT/365 convention).

    Raises
    ------
    ValueError
        If expiry_str cannot be parsed or if expiry is in the past.

    Examples
    --------
    >>> from datetime import datetime
    >>> ref = datetime(2025, 12, 31, 12, 0, 0)
    >>> T = expiry_to_years('2026-12-31', ref)
    >>> abs(T - 1.0) < 0.01  # Approximately 1 year
    True
    """
    if reference_date is None:
        reference_date = datetime.utcnow()

    try:
        expiry_date = datetime.strptime(expiry_str, "%Y-%m-%d")
    except ValueError as e:
        raise ValueError(f"Invalid expiry format '{expiry_str}'. Expected YYYY-MM-DD.") from e

    # Compute days using ACT/365
    days = (expiry_date - reference_date).days

    if days < 0:
        raise ValueError(
            f"Expiry '{expiry_str}' is in the past relative to {reference_date.date()}"
        )

    # Convert to years
    return days / 365.0
