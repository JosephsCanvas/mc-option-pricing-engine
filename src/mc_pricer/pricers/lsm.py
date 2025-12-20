"""
Longstaff-Schwartz (LSM) algorithm for American option pricing.
"""

from dataclasses import dataclass

import numpy as np

from mc_pricer.models.gbm import GeometricBrownianMotion


@dataclass
class AmericanPricingResult:
    """
    Container for American option pricing results using LSM.

    Attributes
    ----------
    price : float
        Estimated American option price
    stderr : float
        Standard error of the estimate
    ci_lower : float
        Lower bound of 95% confidence interval
    ci_upper : float
        Upper bound of 95% confidence interval
    n_paths : int
        Number of simulation paths used
    n_steps : int
        Number of time steps used in LSM
    basis : str
        Basis function type used ('poly2' or 'poly3')
    """
    price: float
    stderr: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    n_steps: int
    basis: str

    def __repr__(self) -> str:
        return (
            f"AmericanPricingResult(\n"
            f"  price={self.price:.6f},\n"
            f"  stderr={self.stderr:.6f},\n"
            f"  CI95=[{self.ci_lower:.6f}, {self.ci_upper:.6f}],\n"
            f"  n_paths={self.n_paths},\n"
            f"  n_steps={self.n_steps},\n"
            f"  basis='{self.basis}'\n"
            f")"
        )


def _basis_functions(spot: np.ndarray, basis: str) -> np.ndarray:
    """
    Compute basis functions for regression.

    Parameters
    ----------
    spot : np.ndarray
        Stock prices, shape (n,)
    basis : str
        Basis type: 'poly2' or 'poly3'

    Returns
    -------
    np.ndarray
        Basis matrix, shape (n, k) where k is number of basis functions
        poly2: [1, S, S^2]
        poly3: [1, S, S^2, S^3]
    """
    if basis == "poly2":
        return np.column_stack([np.ones_like(spot), spot, spot**2])
    elif basis == "poly3":
        return np.column_stack([np.ones_like(spot), spot, spot**2, spot**3])
    else:
        raise ValueError(f"Unknown basis type: {basis}. Use 'poly2' or 'poly3'.")


def price_american_lsm(
    model: GeometricBrownianMotion,
    strike: float,
    option_type: str,
    n_paths: int,
    n_steps: int = 50,
    basis: str = "poly2",
    seed: int | None = None
) -> AmericanPricingResult:
    """
    Price American options using Longstaff-Schwartz (LSM) algorithm.

    Parameters
    ----------
    model : GeometricBrownianMotion
        GBM model with S0, r, sigma, T parameters
    strike : float
        Strike price K
    option_type : str
        'call' or 'put'
    n_paths : int
        Number of Monte Carlo paths
    n_steps : int, optional
        Number of time steps for LSM (default: 50)
    basis : str, optional
        Basis function type: 'poly2' or 'poly3' (default: 'poly2')
    seed : int, optional
        Random seed (overrides model's seed if provided)

    Returns
    -------
    AmericanPricingResult
        Contains price, stderr, CI, and simulation parameters
    """
    if strike <= 0:
        raise ValueError("Strike must be positive")
    if option_type not in ["call", "put"]:
        raise ValueError("option_type must be 'call' or 'put'")
    if n_paths <= 0:
        raise ValueError("n_paths must be positive")
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if basis not in ["poly2", "poly3"]:
        raise ValueError("basis must be 'poly2' or 'poly3'")

    # Create a new model with the specified seed if provided
    if seed is not None:
        model = GeometricBrownianMotion(
            S0=model.S0,
            r=model.r,
            sigma=model.sigma,
            T=model.T,
            seed=seed
        )

    # Simulate full paths: shape (n_paths, n_steps+1) with S0 at index 0
    S_paths = model.simulate_paths(n_paths=n_paths, n_steps=n_steps, antithetic=False)

    # Time grid
    dt = model.T / n_steps
    df = np.exp(-model.r * dt)  # Discount factor per step

    # Intrinsic value function
    if option_type == "put":
        intrinsic = lambda S_t: np.maximum(strike - S_t, 0.0)
    else:  # call
        intrinsic = lambda S_t: np.maximum(S_t - strike, 0.0)

    # Initialize cashflow matrix: CF[i] = discounted cashflow at time t for path i
    # We maintain CF as the discounted value (PV at current time t)
    CF = intrinsic(S_paths[:, n_steps])  # Cashflow at maturity

    # Backward induction from t = n_steps-1 down to t = 1
    for t in range(n_steps - 1, 0, -1):
        S_t = S_paths[:, t]
        IV_t = intrinsic(S_t)

        # Discount continuation value from t+1 to t
        CF = CF * df

        # Identify in-the-money paths
        itm = IV_t > 0

        if np.sum(itm) == 0:
            # No ITM paths, continuation value stays as is
            continue

        # Extract ITM data
        S_itm = S_t[itm]
        IV_itm = IV_t[itm]
        CF_itm = CF[itm]  # This is already discounted to time t

        # Build basis matrix
        X = _basis_functions(S_itm, basis)

        # Check if we have enough data points for regression
        n_basis = X.shape[1]
        if len(S_itm) < n_basis:
            # Not enough data; fallback to no exercise at this step
            continue

        # Least-squares regression: regress continuation value against basis
        beta, _, _, _ = np.linalg.lstsq(X, CF_itm, rcond=None)
        C_hat = X @ beta

        # Exercise decision: exercise if immediate payoff > continuation value
        exercise = IV_itm > C_hat

        # Update cashflows for exercised paths
        # For exercised paths: replace continuation with immediate payoff
        itm_indices = np.where(itm)[0]
        exercised_indices = itm_indices[exercise]
        CF[exercised_indices] = IV_itm[exercise]

    # Final discounting: discount CF from t=1 to t=0
    CF = CF * df

    # Compute statistics
    price = np.mean(CF)
    stderr = np.std(CF, ddof=1) / np.sqrt(n_paths)
    z = 1.96  # 95% CI
    ci_lower = price - z * stderr
    ci_upper = price + z * stderr

    return AmericanPricingResult(
        price=price,
        stderr=stderr,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        n_paths=n_paths,
        n_steps=n_steps,
        basis=basis
    )
