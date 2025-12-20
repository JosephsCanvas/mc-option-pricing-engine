"""
Black-Scholes analytical pricing formulas for validation.
"""

import numpy as np
from scipy.stats import norm


def black_scholes_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Compute European call option price using Black-Scholes formula.

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    T : float
        Time to maturity

    Returns
    -------
    float
        Call option price
    """
    if sigma == 0:
        # Handle zero volatility case
        return max(S0 - K * np.exp(-r * T), 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price


def black_scholes_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Compute European put option price using Black-Scholes formula.

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    T : float
        Time to maturity

    Returns
    -------
    float
        Put option price
    """
    if sigma == 0:
        # Handle zero volatility case
        return max(K * np.exp(-r * T) - S0, 0.0)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)
    return put_price
