"""
Monte Carlo pricing engine for European options.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mc_pricer.models.gbm import GeometricBrownianMotion


@dataclass
class PricingResult:
    """
    Container for Monte Carlo pricing results.

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
        Number of simulation paths used
    """
    price: float
    stderr: float
    ci_lower: float
    ci_upper: float
    n_paths: int

    def __repr__(self) -> str:
        return (
            f"PricingResult(\n"
            f"  price={self.price:.6f},\n"
            f"  stderr={self.stderr:.6f},\n"
            f"  CI95=[{self.ci_lower:.6f}, {self.ci_upper:.6f}],\n"
            f"  n_paths={self.n_paths}\n"
            f")"
        )


class MonteCarloEngine:
    """
    Monte Carlo pricing engine for European options.
    """

    def __init__(
        self,
        model: GeometricBrownianMotion,
        payoff: Callable[[np.ndarray], np.ndarray],
        n_paths: int,
        antithetic: bool = False,
        seed: int | None = None
    ):
        """
        Initialize Monte Carlo pricing engine.

        Parameters
        ----------
        model : GeometricBrownianMotion
            Asset price model for simulation
        payoff : Callable
            Payoff function that maps terminal prices to payoffs
        n_paths : int
            Number of Monte Carlo paths
        antithetic : bool, optional
            Use antithetic variates for variance reduction
        seed : int, optional
            Random seed (if not set, uses model's RNG)
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        self.model = model
        self.payoff = payoff
        self.n_paths = n_paths
        self.antithetic = antithetic

        # Override model's RNG if seed is provided
        if seed is not None:
            self.model.rng = np.random.default_rng(seed)

    def price(self) -> PricingResult:
        """
        Compute option price via Monte Carlo simulation.

        Returns
        -------
        PricingResult
            Pricing results including price, standard error, and confidence interval
        """
        # Simulate terminal asset prices
        S_T = self.model.simulate_terminal(
            n_paths=self.n_paths,
            antithetic=self.antithetic
        )

        # Compute payoffs
        payoffs = self.payoff(S_T)

        # Discount to present value
        discount_factor = np.exp(-self.model.r * self.model.T)
        discounted_payoffs = discount_factor * payoffs

        # Compute statistics
        price = np.mean(discounted_payoffs)
        stderr = np.std(discounted_payoffs, ddof=1) / np.sqrt(self.n_paths)

        # 95% confidence interval (use conventional 1.96 for 95% CI)
        z_critical = 1.96
        ci_lower = price - z_critical * stderr
        ci_upper = price + z_critical * stderr

        return PricingResult(
            price=price,
            stderr=stderr,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_paths=self.n_paths
        )

    def price_with_details(self) -> tuple[PricingResult, np.ndarray]:
        """
        Compute option price and return individual payoffs for analysis.

        Returns
        -------
        result : PricingResult
            Pricing results
        payoffs : np.ndarray
            Individual discounted payoffs
        """
        # Simulate terminal asset prices
        S_T = self.model.simulate_terminal(
            n_paths=self.n_paths,
            antithetic=self.antithetic
        )

        # Compute payoffs
        payoffs = self.payoff(S_T)

        # Discount to present value
        discount_factor = np.exp(-self.model.r * self.model.T)
        discounted_payoffs = discount_factor * payoffs

        # Compute statistics
        price = np.mean(discounted_payoffs)
        stderr = np.std(discounted_payoffs, ddof=1) / np.sqrt(self.n_paths)

        # 95% confidence interval (use conventional 1.96 for 95% CI)
        z_critical = 1.96
        ci_lower = price - z_critical * stderr
        ci_upper = price + z_critical * stderr

        result = PricingResult(
            price=price,
            stderr=stderr,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_paths=self.n_paths
        )

        return result, discounted_payoffs
