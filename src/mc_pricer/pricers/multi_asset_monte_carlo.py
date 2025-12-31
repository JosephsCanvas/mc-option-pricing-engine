"""
Multi-asset Monte Carlo option pricing engine.

Implements pricing for multi-asset European derivatives under correlated GBM.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from mc_pricer.models.multi_gbm import MultiAssetGeometricBrownianMotion


@dataclass
class MultiAssetPricingResult:
    """
    Result from multi-asset Monte Carlo pricing.

    Attributes
    ----------
    price : float
        Estimated option price
    stderr : float
        Standard error of the estimate
    ci_lower : float
        Lower bound of 95% confidence interval
    ci_upper : float
        Upper bound of 95% confidence interval
    n_paths : int
        Number of simulation paths
    antithetic : bool
        Whether antithetic variates were used
    rng_type : str
        RNG type used ("pseudo" or "sobol")
    scramble : bool
        Whether Sobol scrambling was used
    n_assets : int
        Number of assets
    seed : Optional[int]
        Random seed (if provided)
    """

    price: float
    stderr: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    antithetic: bool
    rng_type: str
    scramble: bool
    n_assets: int
    seed: int | None = None

    def __repr__(self) -> str:
        return (
            f"MultiAssetPricingResult(\n"
            f"  price={self.price:.6f},\n"
            f"  stderr={self.stderr:.6f},\n"
            f"  CI=[{self.ci_lower:.6f}, {self.ci_upper:.6f}],\n"
            f"  n_paths={self.n_paths},\n"
            f"  n_assets={self.n_assets},\n"
            f"  rng='{self.rng_type}',\n"
            f"  antithetic={self.antithetic},\n"
            f"  scramble={self.scramble}\n"
            f")"
        )


class MultiAssetMonteCarloEngine:
    """
    Monte Carlo pricer for multi-asset European derivatives.

    Uses risk-neutral simulation under correlated GBM to price options
    with payoffs depending on multiple underlying assets.

    Note: Control variates are not supported for multi-asset options in this version.
    """

    def __init__(
        self,
        model: MultiAssetGeometricBrownianMotion,
        payoff: Callable[[np.ndarray], np.ndarray],
        n_paths: int,
        antithetic: bool = False,
        rng_type: Literal["pseudo", "sobol"] = "pseudo",
        scramble: bool = False,
        seed: int | None = None,
        qmc_dim_override: int | None = None,
    ) -> None:
        """
        Initialize multi-asset Monte Carlo pricer.

        Parameters
        ----------
        model : MultiAssetGeometricBrownianMotion
            Multi-asset GBM model
        payoff : Callable[[np.ndarray], np.ndarray]
            Payoff function that takes S_T (n_paths, n_assets) and returns
            payoff vector (n_paths,)
        n_paths : int
            Number of simulation paths
        antithetic : bool
            Use antithetic variates (requires even n_paths)
        rng_type : {"pseudo", "sobol"}
            Random number generator type
        scramble : bool
            Scramble Sobol sequence (only for sobol)
        seed : Optional[int]
            Random seed for reproducibility
        qmc_dim_override : Optional[int]
            Override computed dimension for QMC (advanced use)
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        if antithetic and n_paths % 2 != 0:
            raise ValueError("n_paths must be even when antithetic=True")

        self.model = model
        self.payoff = payoff
        self.n_paths = n_paths
        self.antithetic = antithetic
        self.rng_type = rng_type
        self.scramble = scramble
        self.seed = seed
        self.qmc_dim_override = qmc_dim_override

        # Update model seed if provided
        if self.seed is not None:
            self.model.seed = self.seed

    def price(self) -> MultiAssetPricingResult:
        """
        Compute option price via Monte Carlo simulation.

        Returns
        -------
        MultiAssetPricingResult
            Pricing result with price, standard error, and confidence interval
        """
        # Simulate terminal prices
        S_T = self.model.simulate_terminal(
            n_paths=self.n_paths,
            antithetic=self.antithetic,
            rng_type=self.rng_type,
            scramble=self.scramble,
            qmc_dim_override=self.qmc_dim_override,
        )

        # Compute payoffs
        payoffs = self.payoff(S_T)

        if payoffs.shape != (self.n_paths,):
            raise ValueError(
                f"Payoff function must return array of shape ({self.n_paths},), got {payoffs.shape}"
            )

        # Discount to present value
        discount_factor = np.exp(-self.model.r * self.model.T)
        discounted_payoffs = discount_factor * payoffs

        # Compute statistics
        price = float(np.mean(discounted_payoffs))
        stderr = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(self.n_paths))

        # 95% confidence interval (z = 1.96)
        z_critical = 1.96
        ci_lower = price - z_critical * stderr
        ci_upper = price + z_critical * stderr

        return MultiAssetPricingResult(
            price=price,
            stderr=stderr,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_paths=self.n_paths,
            antithetic=self.antithetic,
            rng_type=self.rng_type,
            scramble=self.scramble,
            n_assets=self.model.n_assets,
            seed=self.seed,
        )

    def __repr__(self) -> str:
        return (
            f"MultiAssetMonteCarloEngine(\n"
            f"  n_assets={self.model.n_assets},\n"
            f"  n_paths={self.n_paths},\n"
            f"  rng='{self.rng_type}',\n"
            f"  antithetic={self.antithetic}\n"
            f")"
        )
