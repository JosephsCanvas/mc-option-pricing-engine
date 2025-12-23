"""
Monte Carlo pricing engine for European options under Heston model.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mc_pricer.models.heston import HestonModel


@dataclass
class HestonPricingResult:
    """
    Container for Heston Monte Carlo pricing results.

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
    n_steps : int
        Number of time steps per path
    """
    price: float
    stderr: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    n_steps: int

    def __repr__(self) -> str:
        return (
            f"HestonPricingResult(\n"
            f"  price={self.price:.6f},\n"
            f"  stderr={self.stderr:.6f},\n"
            f"  CI95=[{self.ci_lower:.6f}, {self.ci_upper:.6f}],\n"
            f"  n_paths={self.n_paths},\n"
            f"  n_steps={self.n_steps}\n"
            f")"
        )


class HestonMonteCarloEngine:
    """
    Monte Carlo pricing engine for European options under the Heston model.

    Simulates asset price paths using the Heston stochastic volatility model
    and computes the discounted expected payoff.
    """

    def __init__(
        self,
        model: HestonModel,
        payoff: Callable[[np.ndarray], np.ndarray],
        n_paths: int = 100000,
        n_steps: int = 200,
        antithetic: bool = False,
        seed: int | None = None
    ):
        """
        Initialize Heston Monte Carlo pricing engine.

        Parameters
        ----------
        model : HestonModel
            Heston model instance with calibrated parameters
        payoff : Callable[[np.ndarray], np.ndarray]
            Payoff function that takes terminal prices and returns payoffs
        n_paths : int, optional
            Number of simulation paths (default: 100000)
        n_steps : int, optional
            Number of time steps per path (default: 200)
        antithetic : bool, optional
            Whether to use antithetic variates (default: False)
        seed : int, optional
            Random seed for reproducibility (overrides model seed)
        """
        self.model = model
        self.payoff = payoff
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.antithetic = antithetic

        # Override model seed if provided
        if seed is not None:
            self.model._rng = np.random.default_rng(seed)
            self.model.seed = seed

    def price(self) -> HestonPricingResult:
        """
        Compute option price using Monte Carlo simulation.

        Returns
        -------
        HestonPricingResult
            Pricing results including price, standard error, and confidence interval

        Notes
        -----
        Uses 95% confidence interval with z = 1.96 for the normal distribution.
        The standard error is computed as std(discounted_payoffs) / sqrt(n_paths).
        """
        # Simulate terminal prices
        terminal_prices = self.model.simulate_terminal(
            self.n_paths, self.n_steps, self.antithetic
        )

        # Compute payoffs
        payoffs = self.payoff(terminal_prices)

        # Discount to present value
        discount_factor = np.exp(-self.model.r * self.model.T)
        discounted_payoffs = discount_factor * payoffs

        # Compute statistics
        price = float(np.mean(discounted_payoffs))
        stderr = float(np.std(discounted_payoffs, ddof=1) / np.sqrt(self.n_paths))

        # 95% confidence interval (z = 1.96)
        z = 1.96
        ci_lower = price - z * stderr
        ci_upper = price + z * stderr

        return HestonPricingResult(
            price=price,
            stderr=stderr,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            n_paths=self.n_paths,
            n_steps=self.n_steps
        )
