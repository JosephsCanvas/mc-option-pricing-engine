"""
Heston stochastic volatility model for asset price simulation.
"""

import numpy as np


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

    Uses Full Truncation Euler scheme for variance simulation to ensure
    non-negativity.
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
        seed: int | None = None
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

        self.S0 = S0
        self.r = r
        self.T = T
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        self.v0 = v0
        self.seed = seed

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

        Uses Full Truncation Euler scheme:
            v_{t+dt} = max(v_t + kappa*(theta - max(v_t,0))*dt
                          + xi*sqrt(max(v_t,0))*sqrt(dt)*Z2, 0)
            S_{t+dt} = S_t * exp((r - 0.5*max(v_t,0))*dt
                          + sqrt(max(v_t,0))*sqrt(dt)*Z1)

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

        # Euler scheme with full truncation
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
