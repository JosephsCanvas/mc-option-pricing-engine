"""
Multi-Asset Geometric Brownian Motion model with correlated dynamics.

Implements risk-neutral simulation of multiple assets under correlated GBM:
    dS_i/S_i = r dt + sigma_i dW_i
where W = (W_1, ..., W_d) is a correlated Brownian motion with correlation matrix rho.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from mc_pricer.rng.sobol import SobolGenerator


@dataclass
class MultiAssetGeometricBrownianMotion:
    """
    Multi-asset GBM model with correlated dynamics.

    Parameters
    ----------
    S0 : np.ndarray
        Initial asset prices, shape (d,) where d >= 2
    r : float
        Risk-free rate
    sigma : np.ndarray
        Volatilities for each asset, shape (d,)
    T : float
        Time to maturity
    corr : np.ndarray
        Correlation matrix, shape (d, d). Must be symmetric, positive semidefinite,
        with diagonal elements equal to 1.
    seed : Optional[int]
        Random seed for reproducibility

    Attributes
    ----------
    chol : np.ndarray
        Cholesky factor L of correlation matrix (computed at init)
    """

    S0: np.ndarray
    r: float
    sigma: np.ndarray
    T: float
    corr: np.ndarray
    seed: int | None = None

    def __post_init__(self) -> None:
        """Validate parameters and compute Cholesky decomposition."""
        self.S0 = np.asarray(self.S0, dtype=np.float64)
        self.sigma = np.asarray(self.sigma, dtype=np.float64)
        self.corr = np.asarray(self.corr, dtype=np.float64)

        # Validate dimensions
        if self.S0.ndim != 1 or len(self.S0) < 2:
            raise ValueError("S0 must be 1D array with at least 2 assets")

        d = len(self.S0)

        if self.sigma.shape != (d,):
            raise ValueError(f"sigma must have shape ({d},), got {self.sigma.shape}")

        if self.corr.shape != (d, d):
            raise ValueError(f"corr must have shape ({d}, {d}), got {self.corr.shape}")

        # Validate values
        if np.any(self.S0 <= 0):
            raise ValueError("All S0 must be positive")

        if np.any(self.sigma < 0):
            raise ValueError("All sigma must be non-negative")

        if self.T < 0:
            raise ValueError("T must be non-negative")

        # Validate correlation matrix
        if not np.allclose(self.corr, self.corr.T):
            raise ValueError("Correlation matrix must be symmetric")

        if not np.allclose(np.diag(self.corr), 1.0):
            raise ValueError("Correlation matrix diagonal must be 1")

        if np.any(self.corr < -1) or np.any(self.corr > 1):
            raise ValueError("All correlation entries must be in [-1, 1]")

        # Check positive semidefiniteness via Cholesky
        try:
            self.chol = np.linalg.cholesky(self.corr)
        except np.linalg.LinAlgError as e:
            raise ValueError("Correlation matrix must be positive semidefinite") from e

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return len(self.S0)

    def simulate_terminal(
        self,
        n_paths: int,
        antithetic: bool = False,
        rng_type: Literal["pseudo", "sobol"] = "pseudo",
        scramble: bool = False,
        qmc_dim_override: int | None = None,
    ) -> np.ndarray:
        """
        Simulate terminal asset prices S_T.

        Under risk-neutral dynamics:
            ln(S_T / S0) = (r - 0.5 * sigma^2) * T + sigma * sqrt(T) * Z

        where Z ~ N(0, corr) is generated via Z = U @ L.T with U ~ iid N(0,1)
        and L is the Cholesky factor of corr.

        Parameters
        ----------
        n_paths : int
            Number of paths
        antithetic : bool
            Use antithetic variates (requires even n_paths)
        rng_type : {"pseudo", "sobol"}
            RNG type
        scramble : bool
            Scramble Sobol sequence (only for sobol)
        qmc_dim_override : Optional[int]
            Override computed dimension for QMC (must be >= n_assets)

        Returns
        -------
        np.ndarray
            Terminal prices, shape (n_paths, n_assets)
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        if antithetic and n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True")

        d = self.n_assets
        required_dim = d

        # Handle QMC dimension
        if rng_type == "sobol":
            dim = qmc_dim_override if qmc_dim_override is not None else required_dim

            if dim < required_dim:
                raise ValueError(
                    f"qmc_dim_override ({dim}) must be >= required dimension ({required_dim})"
                )

            if dim > 21:
                raise ValueError(
                    f"Sobol dimension {dim} exceeds maximum of 21. "
                    f"Reduce number of assets or use rng_type='pseudo'"
                )
        else:
            dim = required_dim

        # Generate correlated normals
        Z = self._generate_correlated_normals(
            n_paths=n_paths,
            dim=dim,
            antithetic=antithetic,
            rng_type=rng_type,
            scramble=scramble,
        )

        # Truncate if dimension was overridden
        Z = Z[:, :d]

        # Risk-neutral drift
        drift = (self.r - 0.5 * self.sigma**2) * self.T
        diffusion = self.sigma * np.sqrt(self.T) * Z

        # Terminal prices
        log_ST = np.log(self.S0) + drift + diffusion
        S_T = np.exp(log_ST)

        return S_T

    def simulate_paths(
        self,
        n_paths: int,
        n_steps: int,
        antithetic: bool = False,
        rng_type: Literal["pseudo", "sobol"] = "pseudo",
        scramble: bool = False,
        qmc_dim_override: int | None = None,
    ) -> np.ndarray:
        """
        Simulate full price paths for all assets.

        Parameters
        ----------
        n_paths : int
            Number of paths
        n_steps : int
            Number of time steps
        antithetic : bool
            Use antithetic variates
        rng_type : {"pseudo", "sobol"}
            RNG type
        scramble : bool
            Scramble Sobol sequence
        qmc_dim_override : Optional[int]
            Override computed dimension for QMC (must be >= n_assets * n_steps)

        Returns
        -------
        np.ndarray
            Paths, shape (n_paths, n_steps + 1, n_assets), including t=0 as S0
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        if antithetic and n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True")

        d = self.n_assets
        required_dim = d * n_steps

        # Handle QMC dimension
        if rng_type == "sobol":
            dim = qmc_dim_override if qmc_dim_override is not None else required_dim

            if dim < required_dim:
                raise ValueError(
                    f"qmc_dim_override ({dim}) must be >= required dimension ({required_dim})"
                )

            if dim > 21:
                raise ValueError(
                    f"Sobol dimension {dim} (= {d} assets × {n_steps} steps) "
                    f"exceeds maximum of 21. "
                    f"Reduce n_steps, reduce n_assets, or use rng_type='pseudo'"
                )
        else:
            dim = required_dim

        # Generate correlated normals for all steps
        # Shape: (n_paths, dim) where dim >= d * n_steps
        Z = self._generate_correlated_normals(
            n_paths=n_paths,
            dim=dim,
            antithetic=antithetic,
            rng_type=rng_type,
            scramble=scramble,
        )

        # Reshape to (n_paths, n_steps, d)
        Z = Z[:, : d * n_steps].reshape(n_paths, n_steps, d)

        # Initialize paths: (n_paths, n_steps + 1, d)
        paths = np.zeros((n_paths, n_steps + 1, d))
        paths[:, 0, :] = self.S0

        dt = self.T / n_steps
        drift_dt = (self.r - 0.5 * self.sigma**2) * dt
        diffusion_dt = self.sigma * np.sqrt(dt)

        # Simulate each step
        for t in range(n_steps):
            dW = Z[:, t, :]  # (n_paths, d)
            log_increment = drift_dt + diffusion_dt * dW
            paths[:, t + 1, :] = paths[:, t, :] * np.exp(log_increment)

        return paths

    def _generate_correlated_normals(
        self,
        n_paths: int,
        dim: int,
        antithetic: bool,
        rng_type: Literal["pseudo", "sobol"],
        scramble: bool,
    ) -> np.ndarray:
        """
        Generate correlated standard normals Z ~ N(0, corr).

        Uses Z = U @ L.T where U ~ iid N(0,1) and L is Cholesky factor.

        For dimensions > n_assets, we generate iid blocks and correlate
        each d-block separately.

        Parameters
        ----------
        n_paths : int
            Number of paths
        dim : int
            Dimension for RNG (>= n_assets)
        antithetic : bool
            Use antithetic variates
        rng_type : {"pseudo", "sobol"}
            RNG type
        scramble : bool
            Scramble Sobol

        Returns
        -------
        np.ndarray
            Correlated normals, shape (n_paths, dim)
        """
        d = self.n_assets

        if antithetic:
            n_base = n_paths // 2
        else:
            n_base = n_paths

        # Generate iid normals U
        if rng_type == "pseudo":
            rng = np.random.default_rng(self.seed)
            U = rng.standard_normal((n_base, dim))
        else:  # sobol
            sobol_gen = SobolGenerator(
                dimension=dim, seed=self.seed if self.seed is not None else 0, scramble=scramble
            )
            unif = sobol_gen.generate(n_base)
            # Transform to normals via inverse CDF
            U = self._uniform_to_normal(unif)

        # Apply antithetic if needed
        if antithetic:
            U_anti = -U
            U = np.vstack([U, U_anti])

        # Apply correlation: Z = U @ L.T for each d-block
        # If dim > d, we treat each block of d dimensions as correlated
        Z = np.zeros_like(U)
        n_blocks = dim // d
        for i in range(n_blocks):
            start = i * d
            end = start + d
            Z[:, start:end] = U[:, start:end] @ self.chol.T

        # Handle remaining dimensions (if dim not divisible by d)
        remainder = dim % d
        if remainder > 0:
            start = n_blocks * d
            # For remainder, apply partial correlation (use top-left submatrix)
            L_sub = self.chol[:remainder, :remainder]
            Z[:, start:] = U[:, start:] @ L_sub.T

        return Z

    @staticmethod
    def _uniform_to_normal(u: np.ndarray) -> np.ndarray:
        """
        Transform uniform [0,1) to standard normal using inverse CDF.

        Uses probit approximation for numerical stability.
        """
        # Clip to avoid infinity at boundaries
        u = np.clip(u, 1e-10, 1 - 1e-10)

        # Use scipy.stats.norm.ppf equivalent via erf
        # For better accuracy, use rational approximation
        # Simple approach: use probit approximation
        from math import sqrt

        # Beasley-Springer-Moro algorithm (simplified)
        a = [
            -3.969683028665376e1,
            2.209460984245205e2,
            -2.759285104469687e2,
            1.383577518672690e2,
            -3.066479806614716e1,
            2.506628277459239e0,
        ]
        b = [
            -5.447609879822406e1,
            1.615858368580409e2,
            -1.556989798598866e2,
            6.680131188771972e1,
            -1.328068155288572e1,
        ]
        c = [
            -7.784894002430293e-3,
            -3.223964580411365e-1,
            -2.400758277161838e0,
            -2.549732539343734e0,
            4.374664141464968e0,
            2.938163982698783e0,
        ]
        d = [
            7.784695709041462e-3,
            3.224671290700398e-1,
            2.445134137142996e0,
            3.754408661907416e0,
        ]

        def _ppf_single(p: float) -> float:
            if p < 0.02425:
                q = sqrt(-2.0 * np.log(p))
                return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
                    (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
                )
            elif p > 0.97575:
                q = sqrt(-2.0 * np.log(1.0 - p))
                return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
                    (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
                )
            else:
                q = p - 0.5
                r = q * q
                return (
                    (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
                ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)

        return np.vectorize(_ppf_single)(u)
