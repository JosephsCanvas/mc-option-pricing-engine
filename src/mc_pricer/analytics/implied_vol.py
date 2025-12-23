"""
Implied volatility solver for European options.

Uses robust bisection method to solve for the volatility that
produces a given market price.
"""

import math

from mc_pricer.analytics.black_scholes import bs_price


def implied_vol(
    price: float,
    S0: float,
    K: float,
    r: float,
    T: float,
    option_type: str,
    *,
    tol: float = 1e-8,
    max_iter: int = 200,
    sigma_low: float = 1e-6,
    sigma_high: float = 5.0
) -> float:
    """
    Compute implied volatility using bisection method.

    Solves for σ such that BS(S0, K, r, T, σ, type) = market_price

    Parameters
    ----------
    price : float
        Observed market price of the option
    S0 : float
        Current spot price (must be > 0)
    K : float
        Strike price (must be > 0)
    r : float
        Risk-free interest rate (annualized)
    T : float
        Time to maturity in years (must be > 0)
    option_type : str
        'call' or 'put'
    tol : float, optional
        Convergence tolerance for price difference (default: 1e-8)
    max_iter : int, optional
        Maximum number of iterations (default: 200)
    sigma_low : float, optional
        Lower bound for volatility search (default: 1e-6)
    sigma_high : float, optional
        Upper bound for volatility search (default: 5.0)

    Returns
    -------
    float
        Implied volatility

    Raises
    ------
    ValueError
        If price is outside arbitrage bounds or if convergence fails

    Notes
    -----
    Arbitrage bounds for European options:
    - Call: max(0, S0 - K*exp(-rT)) <= price <= S0
    - Put: max(0, K*exp(-rT) - S0) <= price <= K*exp(-rT)

    Uses bisection which is robust but slower than Newton's method.
    Automatically expands search range if needed (up to sigma=10).
    """
    if S0 <= 0:
        raise ValueError("Spot price S0 must be positive")
    if K <= 0:
        raise ValueError("Strike K must be positive")
    if T <= 0:
        raise ValueError("Time to maturity T must be positive")
    if price < 0:
        raise ValueError("Price must be non-negative")
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")

    # Compute arbitrage bounds
    discount = math.exp(-r * T)

    if option_type == "call":
        # Call bounds: [max(S0 - K*discount, 0), S0]
        lower_bound = max(S0 - K * discount, 0.0)
        upper_bound = S0

        if price < lower_bound - tol:
            raise ValueError(
                f"Call price {price:.6f} is below arbitrage lower bound "
                f"{lower_bound:.6f} (intrinsic value)"
            )
        if price > upper_bound + tol:
            raise ValueError(
                f"Call price {price:.6f} exceeds arbitrage upper bound "
                f"{upper_bound:.6f} (spot price)"
            )
    else:  # put
        # Put bounds: [max(K*discount - S0, 0), K*discount]
        lower_bound = max(K * discount - S0, 0.0)
        upper_bound = K * discount

        if price < lower_bound - tol:
            raise ValueError(
                f"Put price {price:.6f} is below arbitrage lower bound "
                f"{lower_bound:.6f} (intrinsic value)"
            )
        if price > upper_bound + tol:
            raise ValueError(
                f"Put price {price:.6f} exceeds arbitrage upper bound "
                f"{upper_bound:.6f} (discounted strike)"
            )

    # Check if price is at boundary (very low vol or intrinsic)
    # For very small prices, use relative tolerance
    if price < 1e-6:
        # For near-zero prices, just check if price is effectively zero
        if price < 1e-10:
            return sigma_low
    elif abs(price - lower_bound) < tol:
        return sigma_low

    # Initialize bisection bounds
    sigma_l = sigma_low
    sigma_h = sigma_high

    # Ensure upper bound is high enough
    price_high = bs_price(S0, K, r, T, sigma_h, option_type)
    expand_attempts = 0
    max_expand = 5
    max_sigma = 10.0

    while price_high < price and expand_attempts < max_expand:
        sigma_h *= 2.0
        if sigma_h > max_sigma:
            sigma_h = max_sigma
            break
        price_high = bs_price(S0, K, r, T, sigma_h, option_type)
        expand_attempts += 1

    price_high = bs_price(S0, K, r, T, sigma_h, option_type)
    if price > price_high + tol:
        raise ValueError(
            f"Market price {price:.6f} is too high to be matched even with "
            f"σ={sigma_h:.2f} (BS price: {price_high:.6f}). "
            "This may indicate arbitrage or data error."
        )

    # Bisection loop
    price_low = bs_price(S0, K, r, T, sigma_l, option_type)

    for _ in range(max_iter):
        sigma_mid = 0.5 * (sigma_l + sigma_h)
        price_mid = bs_price(S0, K, r, T, sigma_mid, option_type)

        error = price_mid - price

        if abs(error) < tol:
            return sigma_mid

        # Update bounds
        if error > 0:
            # BS price too high, reduce sigma
            sigma_h = sigma_mid
            price_high = price_mid
        else:
            # BS price too low, increase sigma
            sigma_l = sigma_mid
            price_low = price_mid

        # Check for convergence in sigma space as well
        if abs(sigma_h - sigma_l) < tol:
            return sigma_mid

    # If we get here, convergence failed
    raise ValueError(
        f"Implied volatility did not converge after {max_iter} iterations. "
        f"Final bracket: [{sigma_l:.6f}, {sigma_h:.6f}], "
        f"prices: [{price_low:.6f}, {price_high:.6f}], target: {price:.6f}"
    )
