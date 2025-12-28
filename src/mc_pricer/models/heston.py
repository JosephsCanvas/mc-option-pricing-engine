"""
Heston stochastic volatility model for asset price simulation.
"""

from typing import Literal

import numpy as np


def sample_variance_qe(
    v: np.ndarray,
    kappa: float,
    theta: float,
    xi: float,
    dt: float,
    z: np.ndarray,
    psi_c: float = 1.5
) -> np.ndarray:
    """
    Sample variance using Andersen's Quadratic-Exponential (QE) scheme.

    Implements the QE discretization for the CIR variance process in Heston:
        dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW_t

    The scheme handles two regimes based on psi = s^2 / m^2:
    - Quadratic regime (psi <= psi_c): Uses transformed normal approximation
    - Exponential regime (psi > psi_c): Uses exponential distribution approximation

    Parameters
    ----------
    v : np.ndarray
        Current variance values (shape: (n_paths,))
    kappa : float
        Mean reversion speed (must be > 0)
    theta : float
        Long-term variance level (must be > 0)
    xi : float
        Volatility of volatility (must be >= 0)
    dt : float
        Time step size (must be > 0)
    z : np.ndarray
        Standard normal random variables (shape: (n_paths,))
    psi_c : float, optional
        Critical threshold for regime switching (default: 1.5)

    Returns
    -------
    np.ndarray
        Next variance values (shape: (n_paths,))

    Notes
    -----
    Reference: Andersen, L. (2008). "Simple and efficient simulation of the
    Heston stochastic volatility model." Journal of Computational Finance.

    The QE scheme ensures non-negativity and has better convergence properties
    than Full Truncation Euler, especially for extreme parameter values.

    Special case: When xi=0, the variance is deterministic and follows
    the exact ODE solution.
    """
    # Handle deterministic case (xi = 0)
    if xi == 0:
        exp_minus_kappa_dt = np.exp(-kappa * dt)
        return theta + (v - theta) * exp_minus_kappa_dt

    # Precompute exponential terms
    exp_minus_kappa_dt = np.exp(-kappa * dt)
    one_minus_exp = 1.0 - exp_minus_kappa_dt

    # Compute conditional mean
    m = theta + (v - theta) * exp_minus_kappa_dt

    # Compute conditional variance
    # s^2 = v*xi^2*exp(-kappa*dt)*(1-exp(-kappa*dt))/kappa
    #     + theta*xi^2*(1-exp(-kappa*dt))^2/(2*kappa)
    s2 = (
        v * xi**2 * exp_minus_kappa_dt * one_minus_exp / kappa
        + theta * xi**2 * one_minus_exp**2 / (2.0 * kappa)
    )

    # Compute psi = s^2 / m^2
    # Handle division by zero: if m=0, set psi to large value (exponential regime)
    psi = np.divide(
        s2,
        m**2,
        out=np.full_like(s2, 10.0),  # Use exponential regime when m=0
        where=m > 1e-12
    )

    # Allocate output array
    v_next = np.zeros_like(v)

    # Quadratic regime (psi <= psi_c)
    quadratic_mask = psi <= psi_c
    if np.any(quadratic_mask):
        psi_quad = psi[quadratic_mask]
        m_quad = m[quadratic_mask]
        z_quad = z[quadratic_mask]

        # Compute b^2 = 2/psi - 1 + sqrt(2/psi) * sqrt(2/psi - 1)
        two_over_psi = 2.0 / psi_quad
        sqrt_term = np.sqrt(two_over_psi)
        b_squared = two_over_psi - 1.0 + sqrt_term * np.sqrt(
            np.maximum(two_over_psi - 1.0, 0.0)  # Ensure non-negative under sqrt
        )

        # Compute a = m / (1 + b^2)
        a = m_quad / (1.0 + b_squared)

        # v_next = a * (sqrt(b) + z)^2
        b = np.sqrt(b_squared)
        v_next[quadratic_mask] = np.maximum(a * (b + z_quad)**2, 0.0)

    # Exponential regime (psi > psi_c)
    exponential_mask = ~quadratic_mask
    if np.any(exponential_mask):
        psi_exp = psi[exponential_mask]
        m_exp = m[exponential_mask]

        # Compute p = (psi - 1) / (psi + 1)
        p = (psi_exp - 1.0) / (psi_exp + 1.0)

        # Compute beta = (1 - p) / m
        beta = (1.0 - p) / m_exp

        # Generate uniform random variables from standard normal z
        # u = Phi(z) where Phi is the standard normal CDF
        # Phi(z) = 0.5 * (1 + erf(z/sqrt(2)))
        # We use a high-precision approximation since scipy is not available
        z_exp = z[exponential_mask]
        # Approximation: erf(x) ≈ sign(x) * sqrt(1 - exp(-4*x^2/pi))
        # For normal CDF, use a more accurate approximation
        # Standard normal CDF approximation (Abramowitz & Stegun)
        x = z_exp / np.sqrt(2.0)
        t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
        d = 0.3989423 * np.exp(-x * x / 2.0)
        u = 1.0 - d * t * (
            0.3193815 + t * (
                -0.3565638 + t * (
                    1.781478 + t * (
                        -1.821256 + t * 1.330274
                    )
                )
            )
        )
        u = np.where(x >= 0, u, 1.0 - u)

        # v_next = 0 if u <= p, else -log((1-p)/(1-u))/beta
        # Ensure numerical stability: clip u to avoid log(0)
        u_clipped = np.clip(u, 1e-10, 1.0 - 1e-10)
        v_next[exponential_mask] = np.maximum(
            np.where(
                u_clipped <= p,
                0.0,
                -np.log((1.0 - p) / (1.0 - u_clipped)) / beta
            ),
            0.0  # Final safety net: ensure non-negative
        )

    return v_next


class HestonModel:
    """
    Heston stochastic volatility model for simulating asset price paths.

    The risk-neutral model follows:
        dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW1_t
        dv_t = kappa * (theta - v_t) * dt + xi * sqrt(v_t) * dW2_t

    where:
        S_t: asset price at time t
        v_t: instantaneous variance at time t
        r: risk-free interest rate
        kappa: mean reversion speed
        theta: long-term variance level
        xi: volatility of volatility
        rho: correlation between W1 and W2
        W1_t, W2_t: correlated Wiener processes

    Supports two variance discretization schemes:
    - Full Truncation Euler: Simple explicit scheme with max(v,0) truncation
    - Quadratic-Exponential (QE): Andersen's exact-distribution scheme
    """

    def __init__(
        self,
        S0: float,
        r: float,
        T: float,
        kappa: float,
        theta: float,
        xi: float,
        rho: float,
        v0: float,
        seed: int | None = None,
        scheme: Literal["full_truncation_euler", "qe"] = "full_truncation_euler"
    ):
        """
        Initialize Heston model parameters.

        Parameters
        ----------
        S0 : float
            Initial asset price (must be > 0)
        r : float
            Risk-free interest rate (annualized)
        T : float
            Time to maturity in years (must be > 0)
        kappa : float
            Mean reversion speed (must be > 0)
        theta : float
            Long-term variance level (must be > 0)
        xi : float
            Volatility of volatility (must be >= 0)
        rho : float
            Correlation between asset and variance Brownian motions (must be in [-1, 1])
        v0 : float
            Initial variance (must be >= 0)
        seed : int, optional
            Random seed for reproducibility
        scheme : Literal["full_truncation_euler", "qe"], optional
            Variance discretization scheme (default: "full_truncation_euler")
            - "full_truncation_euler": Simple explicit Euler with max(v,0) truncation
            - "qe": Andersen's Quadratic-Exponential scheme
        """
        if S0 <= 0:
            raise ValueError("Initial price S0 must be positive")
        if T <= 0:
            raise ValueError("Time to maturity T must be positive")
        if kappa <= 0:
            raise ValueError("Mean reversion speed kappa must be positive")
        if theta <= 0:
            raise ValueError("Long-term variance theta must be positive")
        if xi < 0:
            raise ValueError("Volatility of volatility xi must be non-negative")
        if not -1 <= rho <= 1:
            raise ValueError("Correlation rho must be in [-1, 1]")
        if v0 < 0:
            raise ValueError("Initial variance v0 must be non-negative")
        if scheme not in ("full_truncation_euler", "qe"):
            raise ValueError(
                f"Invalid scheme '{scheme}'. "
                "Must be 'full_truncation_euler' or 'qe'"
            )

        self.S0 = S0
        self.r = r
        self.T = T
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        self.seed = seed
        self.scheme = scheme

        # Initialize random number generator
        self._rng = np.random.default_rng(seed)

    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int,
        antithetic: bool = False
    ) -> np.ndarray:
        """
        Simulate asset price paths using the Heston model.

        Uses either Full Truncation Euler or QE scheme based on self.scheme:

        Full Truncation Euler:
            v_{t+dt} = max(v_t + kappa*(theta - max(v_t,0))*dt
                          + xi*sqrt(max(v_t,0))*sqrt(dt)*Z2, 0)
            S_{t+dt} = S_t * exp((r - 0.5*max(v_t,0))*dt
                          + sqrt(max(v_t,0))*sqrt(dt)*Z1)

        QE (Quadratic-Exponential):
            v_{t+dt} ~ conditional distribution via sample_variance_qe()
            S_{t+dt} = S_t * exp((r - 0.5*v_bar)*dt + sqrt(v_bar)*sqrt(dt)*Z1)
            where v_bar = max(v_t, 0) for asset price evolution

        Parameters
        ----------
        n_paths : int
            Number of paths to simulate
        n_steps : int
            Number of time steps per path
        antithetic : bool, optional
            Whether to use antithetic variates (default: False)
            If True, generates pairs of paths with (Z, -Z)

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, n_steps + 1) containing asset prices

        Notes
        -----
        When antithetic=True, n_paths must be even. The function will generate
        n_paths//2 base paths and n_paths//2 antithetic paths.
        """
        if antithetic and n_paths % 2 != 0:
            raise ValueError("n_paths must be even when using antithetic variates")

        dt = self.T / n_steps
        sqrt_dt = np.sqrt(dt)

        # Determine actual number of random draws needed
        n_base_paths = n_paths // 2 if antithetic else n_paths

        # Generate correlated random normals
        # z1 for asset price, z2 for variance
        z1 = self._rng.standard_normal((n_base_paths, n_steps))
        z2_indep = self._rng.standard_normal((n_base_paths, n_steps))

        # Correlate z2 with z1: z2 = rho * z1 + sqrt(1 - rho^2) * z2_indep
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * z2_indep

        # If antithetic, create paired normals
        if antithetic:
            z1 = np.vstack([z1, -z1])
            z2 = np.vstack([z2, -z2])

        # Initialize arrays
        s = np.zeros((n_paths, n_steps + 1))
        v = np.zeros((n_paths, n_steps + 1))

        s[:, 0] = self.S0
        v[:, 0] = self.v0

        if self.scheme == "full_truncation_euler":
            # Full Truncation Euler scheme
            for i in range(n_steps):
                # Truncate variance at zero for both drift and diffusion
                v_plus = np.maximum(v[:, i], 0.0)
                sqrt_v_plus = np.sqrt(v_plus)

                # Update variance (full truncation)
                v[:, i + 1] = np.maximum(
                    v[:, i] + self.kappa * (self.theta - v_plus) * dt
                    + self.xi * sqrt_v_plus * sqrt_dt * z2[:, i],
                    0.0
                )

                # Update asset price
                s[:, i + 1] = s[:, i] * np.exp(
                    (self.r - 0.5 * v_plus) * dt + sqrt_v_plus * sqrt_dt * z1[:, i]
                )

        else:  # scheme == "qe"
            # Quadratic-Exponential (QE) scheme
            for i in range(n_steps):
                # Update variance using QE scheme
                v[:, i + 1] = sample_variance_qe(
                    v=v[:, i],
                    kappa=self.kappa,
                    theta=self.theta,
                    xi=self.xi,
                    dt=dt,
                    z=z2[:, i]
                )

                # Use current variance for asset price evolution
                # (truncated at zero for numerical stability)
                v_plus = np.maximum(v[:, i], 0.0)
                sqrt_v_plus = np.sqrt(v_plus)

                # Update asset price
                s[:, i + 1] = s[:, i] * np.exp(
                    (self.r - 0.5 * v_plus) * dt + sqrt_v_plus * sqrt_dt * z1[:, i]
                )

        return s

    def simulate_terminal(
        self,
        n_paths: int,
        n_steps: int,
        antithetic: bool = False
    ) -> np.ndarray:
        """
        Simulate terminal asset prices using the Heston model.

        More efficient than simulate_paths when only terminal prices are needed.

        Parameters
        ----------
        n_paths : int
            Number of paths to simulate
        n_steps : int
            Number of time steps per path
        antithetic : bool, optional
            Whether to use antithetic variates (default: False)

        Returns
        -------
        np.ndarray
            Array of shape (n_paths,) containing terminal asset prices
        """
        # For now, use full path simulation and return terminal values
        # Could optimize later to avoid storing full paths
        paths = self.simulate_paths(n_paths, n_steps, antithetic)
        return paths[:, -1]
