"""
Black-Scholes analytical pricing formulas for validation.
"""

import math

import numpy as np


def norm_cdf(x: float) -> float:
    """
    Cumulative distribution function for standard normal distribution.

    Uses math.erf for calculation without scipy dependency.

    Parameters
    ----------
    x : float
        Input value

    Returns
    -------
    float
        CDF value at x
    """
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_pdf(x: float) -> float:
    """
    Probability density function for standard normal distribution.

    Parameters
    ----------
    x : float
        Input value

    Returns
    -------
    float
        PDF value at x
    """
    return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


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

    call_price = S0 * norm_cdf(d1) - K * np.exp(-r * T) * norm_cdf(d2)
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

    put_price = K * np.exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)
    return put_price


def black_scholes_delta_call(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Compute Delta for European call option using Black-Scholes formula.

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
        Call Delta
    """
    if sigma == 0:
        # Handle zero volatility case
        return 1.0 if S0 > K * np.exp(-r * T) else 0.0

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm_cdf(d1)


def black_scholes_delta_put(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Compute Delta for European put option using Black-Scholes formula.

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
        Put Delta
    """
    if sigma == 0:
        # Handle zero volatility case
        return -1.0 if S0 < K * np.exp(-r * T) else 0.0

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm_cdf(d1) - 1.0


def black_scholes_vega(S0: float, K: float, r: float, sigma: float, T: float) -> float:
    """
    Compute Vega for European option using Black-Scholes formula.

    Vega is the same for calls and puts.

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
        Vega (sensitivity to volatility)
    """
    if sigma == 0 or T == 0:
        return 0.0

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    vega = S0 * norm_pdf(d1) * np.sqrt(T)
    return vega
