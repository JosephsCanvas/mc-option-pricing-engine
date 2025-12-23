"""
Analytics module for Black-Scholes pricing and implied volatility.

Provides reference implementations of analytical pricing formulas
and volatility solving without scipy dependency.
"""

from mc_pricer.analytics.black_scholes import (
    bs_delta,
    bs_gamma,
    bs_price,
    bs_rho,
    bs_theta,
    bs_vega,
    norm_cdf,
    norm_pdf,
)
from mc_pricer.analytics.implied_vol import implied_vol

__all__ = [
    "bs_delta",
    "bs_gamma",
    "bs_price",
    "bs_rho",
    "bs_theta",
    "bs_vega",
    "implied_vol",
    "norm_cdf",
    "norm_pdf",
]
