"""
Geometric Brownian Motion (GBM) model for asset price simulation.
"""


import numpy as np


class GeometricBrownianMotion:
    """
    Geometric Brownian Motion model for simulating asset price paths.

    The model follows:
        dS_t = μ * S_t * dt + σ * S_t * dW_t

    where:
        S_t: asset price at time t
        μ: drift (risk-free rate under risk-neutral measure)
        σ: volatility
        W_t: Wiener process (Brownian motion)
    """

    def __init__(
        self,
        S0: float,
        r: float,
        sigma: float,
        T: float,
        seed: int | None = None
    ):
        """
        Initialize GBM model parameters.

        Parameters
        ----------
        S0 : float
            Initial asset price (must be > 0)
        r : float
            Risk-free interest rate (annualized)
        sigma : float
            Volatility (annualized, must be >= 0)
        T : float
            Time to maturity in years (must be > 0)
        seed : int, optional
            Random seed for reproducibility
        """
        if S0 <= 0:
            raise ValueError("Initial price S0 must be positive")
        if sigma < 0:
            raise ValueError("Volatility sigma must be non-negative")
        if T <= 0:
            raise ValueError("Time to maturity T must be positive")

        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.rng = np.random.default_rng(seed)

    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int = 1,
        antithetic: bool = False
    ) -> np.ndarray:
        """
        Simulate asset price paths using GBM.

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths to simulate
        n_steps : int, optional
            Number of time steps per path (default: 1 for European options)
        antithetic : bool, optional
            If True, use antithetic variates for variance reduction

        Returns
        -------
        np.ndarray
            Array of shape (n_paths, n_steps + 1) containing simulated paths.
            Each row is a path, starting with S0.
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        dt = self.T / n_steps

        # Effective number of random paths to generate
        n_random = (n_paths + 1) // 2 if antithetic else n_paths

        # Generate random increments
        Z = self.rng.standard_normal((n_random, n_steps))

        if antithetic:
            # Create antithetic pairs
            Z = np.vstack([Z, -Z])[:n_paths]

        # Pre-compute drift and diffusion terms
        drift = (self.r - 0.5 * self.sigma**2) * dt
        diffusion = self.sigma * np.sqrt(dt)

        # Compute log returns
        log_returns = drift + diffusion * Z

        # Compute cumulative log returns and exponentiate
        log_S = np.log(self.S0) + np.cumsum(log_returns, axis=1)

        # Add initial price as first column
        S = np.column_stack([np.full(n_paths, self.S0), np.exp(log_S)])

        return S

    def simulate_terminal(
        self,
        n_paths: int,
        antithetic: bool = False
    ) -> np.ndarray:
        """
        Simulate terminal asset prices at maturity (optimized for European options).

        Parameters
        ----------
        n_paths : int
            Number of Monte Carlo paths to simulate
        antithetic : bool, optional
            If True, use antithetic variates for variance reduction

        Returns
        -------
        np.ndarray
            Array of shape (n_paths,) containing terminal prices S_T
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        # Effective number of random paths to generate
        n_random = (n_paths + 1) // 2 if antithetic else n_paths

        # Generate random normal variates
        Z = self.rng.standard_normal(n_random)

        if antithetic:
            # Create antithetic pairs
            Z = np.concatenate([Z, -Z])[:n_paths]

        # Compute terminal prices using closed-form solution
        # S_T = S_0 * exp((r - 0.5*σ²)*T + σ*√T*Z)
        log_ST = np.log(self.S0) + (self.r - 0.5 * self.sigma**2) * self.T + \
                 self.sigma * np.sqrt(self.T) * Z

        return np.exp(log_ST)
