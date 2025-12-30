"""
Monte Carlo Option Pricing Engine

A production-grade Monte Carlo simulation engine for pricing financial derivatives.
"""

from mc_pricer._version import __version__

# Core components
from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.models.heston import HestonModel
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine
from mc_pricer.pricers.lsm import AmericanPricingResult, price_american_lsm
from mc_pricer.pricers.monte_carlo import MonteCarloEngine, PricingResult

# Analytics
from mc_pricer.analytics.black_scholes import (
    bs_delta,
    bs_gamma,
    bs_price,
    bs_vega,
)
from mc_pricer.analytics.implied_vol import implied_vol

__all__ = [
    # Version
    "__version__",
    # Models
    # Version
    "__version__",
    # Models
    "GeometricBrownianMotion",
    "HestonModel",
    # Payoffs
    "HestonModel",
    # Payoffs
    "EuropeanCallPayoff",
    "EuropeanPutPayoff",
    # Pricers
    # Pricers
    "MonteCarloEngine",
    "HestonMonteCarloEngine",
    "PricingResult",
    "AmericanPricingResult",
    "price_american_lsm",
    # Analytics
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "implied_vol",
    "HestonMonteCarloEngine",
    "PricingResult",
    "AmericanPricingResult",
    "price_american_lsm",
    # Analytics
    "bs_price",
    "bs_delta",
    "bs_gamma",
    "bs_vega",
    "implied_vol",
]
