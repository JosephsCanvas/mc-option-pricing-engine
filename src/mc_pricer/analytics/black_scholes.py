"""
Black-Scholes analytical pricing formulas for European options.

This module provides reference implementations of Black-Scholes pricing
and Greeks without scipy dependency.
"""

import math


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
        CDF value at x: P(Z <= x) where Z ~ N(0,1)
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
        PDF value at x: φ(x) = exp(-x²/2)/√(2π)
    """
    return math.exp(-0.5 * x**2) / math.sqrt(2.0 * math.pi)


def bs_price(S0: float, K: float, r: float, T: float, sigma: float, option_type: str) -> float:
    """
    Compute European option price using Black-Scholes formula.

    Parameters
    ----------
    S0 : float
        Initial spot price (must be > 0)
    K : float
        Strike price (must be > 0)
    r : float
        Risk-free interest rate (annualized)
    T : float
        Time to maturity in years (must be >= 0)
    sigma : float
        Volatility (annualized, must be >= 0)
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Option price

    Notes
    -----
    Edge cases:
    - T <= 0: returns intrinsic value max(S0 - K, 0) for call or max(K - S0, 0) for put
    - sigma <= 0: returns discounted intrinsic based on forward price
    """
    if S0 <= 0:
        raise ValueError("Spot price S0 must be positive")
    if K <= 0:
        raise ValueError("Strike K must be positive")
    if T < 0:
        raise ValueError("Time to maturity T must be non-negative")
    if sigma < 0:
        raise ValueError("Volatility sigma must be non-negative")
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")

    # Handle T = 0 edge case: return intrinsic value
    if T == 0:
        if option_type == "call":
            return max(S0 - K, 0.0)
        else:
            return max(K - S0, 0.0)

    # Handle sigma = 0 edge case: deterministic forward price
    if sigma == 0:
        forward = S0 * math.exp(r * T)
        discount = math.exp(-r * T)
        if option_type == "call":
            return max(forward - K, 0.0) * discount
        else:
            return max(K - forward, 0.0) * discount

    # Standard Black-Scholes calculation
    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        price = S0 * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:  # put
        price = K * math.exp(-r * T) * norm_cdf(-d2) - S0 * norm_cdf(-d1)

    return price


def bs_delta(S0: float, K: float, r: float, T: float, sigma: float, option_type: str) -> float:
    """
    Compute Delta for European option using Black-Scholes formula.

    Delta = ∂V/∂S

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    r : float
        Risk-free interest rate
    T : float
        Time to maturity in years
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Delta (sensitivity to spot price)

    Notes
    -----
    - Call delta: N(d1)
    - Put delta: N(d1) - 1
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T < 0 or sigma < 0:
        raise ValueError("T and sigma must be non-negative")
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")

    # Handle T = 0: digital payoff
    if T == 0:
        if option_type == "call":
            return 1.0 if S0 > K else 0.0
        else:
            return -1.0 if S0 < K else 0.0

    # Handle sigma = 0: digital payoff based on forward
    if sigma == 0:
        forward = S0 * math.exp(r * T)
        if option_type == "call":
            return 1.0 if forward > K else 0.0
        else:
            return -1.0 if forward < K else 0.0

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))

    if option_type == "call":
        return norm_cdf(d1)
    else:  # put
        return norm_cdf(d1) - 1.0


def bs_gamma(S0: float, K: float, r: float, T: float, sigma: float) -> float:
    """
    Compute Gamma for European option using Black-Scholes formula.

    Gamma = ∂²V/∂S² (same for calls and puts)

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    r : float
        Risk-free interest rate
    T : float
        Time to maturity in years
    sigma : float
        Volatility

    Returns
    -------
    float
        Gamma (sensitivity of delta to spot price)

    Notes
    -----
    Gamma = φ(d1) / (S * σ * √T)
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T < 0 or sigma < 0:
        raise ValueError("T and sigma must be non-negative")

    if T == 0 or sigma == 0:
        return 0.0

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    gamma = norm_pdf(d1) / (S0 * sigma * math.sqrt(T))
    return gamma


def bs_vega(S0: float, K: float, r: float, T: float, sigma: float) -> float:
    """
    Compute Vega for European option using Black-Scholes formula.

    Vega = ∂V/∂σ (same for calls and puts)

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    r : float
        Risk-free interest rate
    T : float
        Time to maturity in years
    sigma : float
        Volatility

    Returns
    -------
    float
        Vega (sensitivity to volatility)

    Notes
    -----
    Vega = S * φ(d1) * √T
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T < 0 or sigma < 0:
        raise ValueError("T and sigma must be non-negative")

    if T == 0 or sigma == 0:
        return 0.0

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    vega = S0 * norm_pdf(d1) * math.sqrt(T)
    return vega


def bs_theta(S0: float, K: float, r: float, T: float, sigma: float, option_type: str) -> float:
    """
    Compute Theta for European option using Black-Scholes formula.

    Theta = ∂V/∂t (negative of time derivative)

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    r : float
        Risk-free interest rate
    T : float
        Time to maturity in years
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Theta (sensitivity to time, typically negative)

    Notes
    -----
    For call: -[S*φ(d1)*σ/(2√T)] - r*K*exp(-rT)*N(d2)
    For put: -[S*φ(d1)*σ/(2√T)] + r*K*exp(-rT)*N(-d2)
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T < 0 or sigma < 0:
        raise ValueError("T and sigma must be non-negative")
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")

    if T == 0:
        return 0.0

    if sigma == 0:
        # For zero volatility, theta is essentially the interest carry
        forward = S0 * math.exp(r * T)
        discount = math.exp(-r * T)
        if option_type == "call":
            if forward > K:
                return r * K * discount
            else:
                return 0.0
        else:
            if forward < K:
                return -r * K * discount
            else:
                return 0.0

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    term1 = -(S0 * norm_pdf(d1) * sigma) / (2 * math.sqrt(T))

    if option_type == "call":
        term2 = -r * K * math.exp(-r * T) * norm_cdf(d2)
        theta = term1 + term2
    else:  # put
        term2 = r * K * math.exp(-r * T) * norm_cdf(-d2)
        theta = term1 + term2

    return theta


def bs_rho(S0: float, K: float, r: float, T: float, sigma: float, option_type: str) -> float:
    """
    Compute Rho for European option using Black-Scholes formula.

    Rho = ∂V/∂r

    Parameters
    ----------
    S0 : float
        Initial spot price
    K : float
        Strike price
    r : float
        Risk-free interest rate
    T : float
        Time to maturity in years
    sigma : float
        Volatility
    option_type : str
        'call' or 'put'

    Returns
    -------
    float
        Rho (sensitivity to interest rate)

    Notes
    -----
    For call: K*T*exp(-rT)*N(d2)
    For put: -K*T*exp(-rT)*N(-d2)
    """
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T < 0 or sigma < 0:
        raise ValueError("T and sigma must be non-negative")
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")

    if T == 0:
        return 0.0

    if sigma == 0:
        forward = S0 * math.exp(r * T)
        discount = math.exp(-r * T)
        if option_type == "call":
            if forward > K:
                return K * T * discount
            else:
                return 0.0
        else:
            if forward < K:
                return -K * T * discount
            else:
                return 0.0

    d1 = (math.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        rho = K * T * math.exp(-r * T) * norm_cdf(d2)
    else:  # put
        rho = -K * T * math.exp(-r * T) * norm_cdf(-d2)

    return rho
