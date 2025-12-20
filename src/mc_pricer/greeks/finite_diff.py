"""
Greeks computation via finite differences.

Implements central finite difference estimators for Delta and Vega.
"""

from collections.abc import Callable

import numpy as np

from mc_pricer.greeks.types import GreekResult


def finite_diff_delta(
    engine_factory: Callable[[float, float, int], object],
    base_params: dict,
    h_rel: float = 1e-4,
    seed: int = 123
) -> GreekResult:
    """
    Compute Delta via central finite difference.

    Parameters
    ----------
    engine_factory : Callable
        Function that creates a pricing engine: engine_factory(S0, sigma, seed)
    base_params : dict
        Dictionary with 'S0', 'sigma', and other base parameters
    h_rel : float
        Relative step size for spot (h = h_rel * S0)
    seed : int
        Random seed for pricing runs

    Returns
    -------
    GreekResult
        Delta estimate with standard error and confidence interval
    """
    S0 = base_params['S0']
    sigma = base_params['sigma']

    # Compute step size
    h = h_rel * S0

    # Check if we can use central difference
    if S0 - h <= 0:
        # Use forward difference if central would give negative spot
        engine_up = engine_factory(S0 + h, sigma, seed)
        engine_base = engine_factory(S0, sigma, seed)
        V_up = engine_up.price().price
        V_base = engine_base.price().price
        delta = (V_up - V_base) / h
        is_central = False
    else:
        # Use central difference
        engine_up = engine_factory(S0 + h, sigma, seed)
        engine_down = engine_factory(S0 - h, sigma, seed)
        V_up = engine_up.price().price
        V_down = engine_down.price().price
        is_central = True

    # Estimate standard error using multiple seeds
    k_seeds = 10
    deltas = []

    for s in range(k_seeds):
        current_seed = seed + s * 1000
        if is_central:
            eng_up = engine_factory(S0 + h, sigma, current_seed)
            eng_down = engine_factory(S0 - h, sigma, current_seed)
            v_up = eng_up.price().price
            v_down = eng_down.price().price
            d = (v_up - v_down) / (2 * h)
        else:
            eng_up = engine_factory(S0 + h, sigma, current_seed)
            eng_base = engine_factory(S0, sigma, current_seed)
            v_up = eng_up.price().price
            v_base = eng_base.price().price
            d = (v_up - v_base) / h
        deltas.append(d)

    # Statistics
    mean_delta = float(np.mean(deltas))
    stderr = float(np.std(deltas, ddof=1) / np.sqrt(k_seeds))

    z_critical = 1.96
    ci_lower = mean_delta - z_critical * stderr
    ci_upper = mean_delta + z_critical * stderr

    return GreekResult(
        value=mean_delta,
        standard_error=stderr,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )


def finite_diff_vega(
    engine_factory: Callable[[float, float, int], object],
    base_params: dict,
    h_abs: float = 1e-4,
    seed: int = 123
) -> GreekResult:
    """
    Compute Vega via central finite difference.

    Parameters
    ----------
    engine_factory : Callable
        Function that creates a pricing engine: engine_factory(S0, sigma, seed)
    base_params : dict
        Dictionary with 'S0', 'sigma', and other base parameters
    h_abs : float
        Absolute step size for sigma
    seed : int
        Random seed for pricing runs

    Returns
    -------
    GreekResult
        Vega estimate with standard error and confidence interval
    """
    S0 = base_params['S0']
    sigma = base_params['sigma']

    # Compute step size
    h = h_abs

    # Check if we can use central difference
    if sigma - h < 1e-12:
        # Use forward difference if central would give too-small sigma
        engine_up = engine_factory(S0, sigma + h, seed)
        engine_base = engine_factory(S0, sigma, seed)
        V_up = engine_up.price().price
        V_base = engine_base.price().price
        vega = (V_up - V_base) / h
        is_central = False
    else:
        # Use central difference
        engine_up = engine_factory(S0, sigma + h, seed)
        engine_down = engine_factory(S0, sigma - h, seed)
        V_up = engine_up.price().price
        V_down = engine_down.price().price
        vega = (V_up - V_down) / (2 * h)
        is_central = True

    # Estimate standard error using multiple seeds
    k_seeds = 10
    vegas = []

    for s in range(k_seeds):
        current_seed = seed + s * 1000
        if is_central:
            eng_up = engine_factory(S0, sigma + h, current_seed)
            eng_down = engine_factory(S0, sigma - h, current_seed)
            v_up = eng_up.price().price
            v_down = eng_down.price().price
            v = (v_up - v_down) / (2 * h)
        else:
            eng_up = engine_factory(S0, sigma + h, current_seed)
            eng_base = engine_factory(S0, sigma, current_seed)
            v_up = eng_up.price().price
            v_base = eng_base.price().price
            v = (v_up - v_base) / h
        vegas.append(v)

    # Statistics
    mean_vega = float(np.mean(vegas))
    stderr = float(np.std(vegas, ddof=1) / np.sqrt(k_seeds))

    z_critical = 1.96
    ci_lower = mean_vega - z_critical * stderr
    ci_upper = mean_vega + z_critical * stderr

    return GreekResult(
        value=mean_vega,
        standard_error=stderr,
        ci_lower=ci_lower,
        ci_upper=ci_upper
    )

    def gamma(self) -> float:
        """Compute Gamma via central difference."""
        raise NotImplementedError("To be implemented")

    def vega(self) -> float:
        """Compute Vega via central difference."""
        raise NotImplementedError("To be implemented")
