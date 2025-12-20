"""
Pathwise derivative estimators for Greeks.
"""

import math

import numpy as np

from mc_pricer.greeks.types import GreekResult


def pathwise_delta_vega(
    S_T: np.ndarray,
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    option_type: str
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pathwise delta and vega samples.

    Parameters
    ----------
    S_T : np.ndarray
        Terminal asset prices from simulation
    S0 : float
        Initial spot price
    K : float
        Strike price
    r : float
        Risk-free rate
    sigma : float
        Volatility
    T : float
        Time to maturity
    option_type : str
        'call' or 'put'

    Returns
    -------
    delta_samples : np.ndarray
        Delta estimates for each path
    vega_samples : np.ndarray
        Vega estimates for each path
    """
    n = len(S_T)
    discount = math.exp(-r * T)

    # Indicator function
    if option_type.lower() == 'call':
        indicator = (S_T > K).astype(float)
        sign = 1.0
    elif option_type.lower() == 'put':
        indicator = (S_T < K).astype(float)
        sign = -1.0
    else:
        raise ValueError(f"Unknown option_type: {option_type}")

    # dS_T/dS0 = S_T / S0 (for GBM)
    dST_dS0 = S_T / S0

    # Delta samples
    delta_samples = discount * sign * indicator * dST_dS0

    # Vega samples
    if sigma > 0 and T > 0:
        # Recover Z from S_T
        sqrt_T = math.sqrt(T)
        Z = (np.log(S_T / S0) - (r - 0.5 * sigma**2) * T) / (sigma * sqrt_T)

        # dS_T/dsigma = S_T * (sqrt(T) * Z - sigma * T)
        dST_dsigma = S_T * (sqrt_T * Z - sigma * T)

        # Vega samples
        vega_samples = discount * sign * indicator * dST_dsigma
    else:
        # If sigma=0 or T=0, vega is zero
        vega_samples = np.zeros(n)

    return delta_samples, vega_samples


def summarize_samples(samples: np.ndarray) -> GreekResult:
    """
    Summarize Greek samples into a GreekResult.

    Parameters
    ----------
    samples : np.ndarray
        Individual Greek estimates from simulation paths

    Returns
    -------
    GreekResult
        Aggregated Greek estimate with statistics
    """
    n = len(samples)
    value = float(np.mean(samples))
    stderr = float(np.std(samples, ddof=1) / np.sqrt(n))

    # 95% confidence interval with z=1.96
    z_critical = 1.96
    ci_lower = value - z_critical * stderr
    ci_upper = value + z_critical * stderr

    return GreekResult(
        value=value,
        standard_error=stderr,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )
