"""
Monte Carlo pricing engine for European options.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from mc_pricer.greeks.finite_diff import finite_diff_delta, finite_diff_vega
from mc_pricer.greeks.pathwise import pathwise_delta_vega, summarize_samples
from mc_pricer.greeks.types import GreeksResult
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
    control_variate_beta : float | None
        Beta coefficient used in control variate adjustment (None if not used)
    """
    price: float
    stderr: float
    ci_lower: float
    ci_upper: float
    n_paths: int
    control_variate_beta: float | None = None

    def __repr__(self) -> str:
        base = (
            f"PricingResult(\n"
            f"  price={self.price:.6f},\n"
            f"  stderr={self.stderr:.6f},\n"
            f"  CI95=[{self.ci_lower:.6f}, {self.ci_upper:.6f}],\n"
            f"  n_paths={self.n_paths}"
        )
        if self.control_variate_beta is not None:
            base += f",\n  control_variate_beta={self.control_variate_beta:.6f}"
        base += "\n)"
        return base


@dataclass
class PathDependentPricingResult(PricingResult):
    """
    Container for path-dependent Monte Carlo pricing results.

    Extends PricingResult with additional metadata specific to path-dependent options.
    """
    n_steps: int = 0
    rng_type: str = "pseudo"
    scramble: bool = False

    def __repr__(self) -> str:
        base = (
            f"PathDependentPricingResult(\n"
            f"  price={self.price:.6f},\n"
            f"  stderr={self.stderr:.6f},\n"
            f"  CI95=[{self.ci_lower:.6f}, {self.ci_upper:.6f}],\n"
            f"  n_paths={self.n_paths},\n"
            f"  n_steps={self.n_steps},\n"
            f"  rng_type='{self.rng_type}',\n"
            f"  scramble={self.scramble}"
        )
        if self.control_variate_beta is not None:
            base += f",\n  control_variate_beta={self.control_variate_beta:.6f}"
        base += "\n)"
        return base


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
        control_variate: bool = False,
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
        control_variate : bool, optional
            Use control variate variance reduction with S_T as control
        seed : int, optional
            Random seed (if not set, uses model's RNG)
        """
        if n_paths <= 0:
            raise ValueError("n_paths must be positive")

        self.model = model
        self.payoff = payoff
        self.n_paths = n_paths
        self.antithetic = antithetic
        self.control_variate = control_variate

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

        # Control variate adjustment
        beta = None
        if self.control_variate:
            # X = discounted payoffs, Y = discounted terminal prices
            X = discounted_payoffs
            Y = discount_factor * S_T
            # Known expectation: E[Y] = S0 (under risk-neutral measure)
            expected_Y = self.model.S0

            # Compute covariance and variance
            cov_XY = np.cov(X, Y, ddof=1)[0, 1]
            var_Y = np.var(Y, ddof=1)

            # Compute optimal beta coefficient
            if var_Y > 1e-14:
                beta = cov_XY / var_Y
                # Apply control variate adjustment
                discounted_payoffs = X - beta * (Y - expected_Y)

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
            n_paths=self.n_paths,
            control_variate_beta=beta
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

        # Control variate adjustment
        beta = None
        if self.control_variate:
            # X = discounted payoffs, Y = discounted terminal prices
            X = discounted_payoffs
            Y = discount_factor * S_T
            # Known expectation: E[Y] = S0 (under risk-neutral measure)
            expected_Y = self.model.S0

            # Compute covariance and variance
            cov_XY = np.cov(X, Y, ddof=1)[0, 1]
            var_Y = np.var(Y, ddof=1)

            # Compute optimal beta coefficient
            if var_Y > 1e-14:
                beta = cov_XY / var_Y
                # Apply control variate adjustment
                discounted_payoffs = X - beta * (Y - expected_Y)

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
            n_paths=self.n_paths,
            control_variate_beta=beta
        )

        return result, discounted_payoffs

    def price_path_dependent(
        self,
        n_steps: int,
        rng_type: str = "pseudo",
        scramble: bool = False,
        qmc_dim_override: int | None = None
    ) -> PathDependentPricingResult:
        """
        Price path-dependent options using full path simulation.

        This method simulates full price paths and computes payoffs based on
        the entire path history (e.g., Asian options, Barrier options).

        Parameters
        ----------
        n_steps : int
            Number of time steps in each path
        rng_type : str, optional
            Random number generator type: "pseudo" or "sobol" (default: "pseudo")
        scramble : bool, optional
            Whether to use digital shift scrambling for Sobol sequences (default: False)
        qmc_dim_override : int | None, optional
            Override dimension for QMC (advanced users only, default: None)

        Returns
        -------
        PathDependentPricingResult
            Pricing results with path-dependent metadata

        Notes
        -----
        Control variate variance reduction is NOT applied for path-dependent options
        as the terminal stock control variate is only valid for terminal payoffs.
        Antithetic variance reduction works with path simulation.

        For QMC, the dimension is n_steps for GBM (or 2*n_steps for Heston).
        Ensure n_steps <= 21 for Sobol unless using dimension override.
        """
        if n_steps <= 0:
            raise ValueError("n_steps must be positive")

        # Temporarily override model RNG settings if specified
        original_rng_type = self.model.rng_type
        original_scramble = self.model.scramble

        try:
            # Set RNG type for path simulation
            self.model.rng_type = rng_type
            self.model.scramble = scramble

            # Simulate full paths
            paths = self.model.simulate_paths(
                n_paths=self.n_paths,
                n_steps=n_steps,
                antithetic=self.antithetic
            )

            # Compute payoffs from paths
            payoffs = self.payoff(paths)

            # Discount to present value
            discount_factor = np.exp(-self.model.r * self.model.T)
            discounted_payoffs = discount_factor * payoffs

            # Note: Control variate is NOT applied for path-dependent options
            # The terminal stock CV only works for terminal payoffs

            # Compute statistics
            price = np.mean(discounted_payoffs)
            stderr = np.std(discounted_payoffs, ddof=1) / np.sqrt(self.n_paths)

            # 95% confidence interval
            z_critical = 1.96
            ci_lower = price - z_critical * stderr
            ci_upper = price + z_critical * stderr

            return PathDependentPricingResult(
                price=price,
                stderr=stderr,
                ci_lower=ci_lower,
                ci_upper=ci_upper,
                n_paths=self.n_paths,
                n_steps=n_steps,
                rng_type=rng_type,
                scramble=scramble,
                control_variate_beta=None
            )

        finally:
            # Restore original RNG settings
            self.model.rng_type = original_rng_type
            self.model.scramble = original_scramble

    def compute_greeks(
        self,
        option_type: str,
        method: str = "pw",
        fd_seeds: int = 10,
        fd_step_spot: float = 1e-4,
        fd_step_sigma: float = 1e-4
    ) -> GreeksResult:
        """
        Compute option Greeks (Delta and Vega).

        Parameters
        ----------
        option_type : str
            'call' or 'put'
        method : str, optional
            Method to use: 'pw' (pathwise), 'fd' (finite difference), or 'both'
        fd_seeds : int, optional
            Number of seeds for FD standard error estimation
        fd_step_spot : float, optional
            Relative step size for Delta FD
        fd_step_sigma : float, optional
            Absolute step size for Vega FD

        Returns
        -------
        GreeksResult
            Container with delta and vega estimates
        """
        if method not in ['pw', 'fd', 'both']:
            raise ValueError(f"method must be 'pw', 'fd', or 'both', got {method}")

        delta_result = None
        vega_result = None

        # Pathwise Greeks
        if method in ['pw', 'both']:
            # Simulate terminal prices
            S_T = self.model.simulate_terminal(
                n_paths=self.n_paths,
                antithetic=self.antithetic
            )

            # Compute pathwise delta and vega samples
            delta_samples, vega_samples = pathwise_delta_vega(
                S_T=S_T,
                S0=self.model.S0,
                K=self._get_strike_from_payoff(),
                r=self.model.r,
                sigma=self.model.sigma,
                T=self.model.T,
                option_type=option_type
            )

            # Summarize samples
            delta_pw = summarize_samples(delta_samples)
            vega_pw = summarize_samples(vega_samples)

            if method == 'pw':
                delta_result = delta_pw
                vega_result = vega_pw

        # Finite difference Greeks
        if method in ['fd', 'both']:
            # Create engine factory
            def engine_factory(S0_new, sigma_new, seed_new):
                from mc_pricer.models.gbm import GeometricBrownianMotion
                model_new = GeometricBrownianMotion(
                    S0=S0_new,
                    r=self.model.r,
                    sigma=sigma_new,
                    T=self.model.T,
                    seed=seed_new
                )
                return MonteCarloEngine(
                    model=model_new,
                    payoff=self.payoff,
                    n_paths=self.n_paths,
                    antithetic=self.antithetic,
                    control_variate=self.control_variate,
                    seed=seed_new
                )

            base_params = {
                'S0': self.model.S0,
                'sigma': self.model.sigma
            }

            # Compute FD Greeks
            delta_fd = finite_diff_delta(
                engine_factory=engine_factory,
                base_params=base_params,
                h_rel=fd_step_spot,
                seed=123
            )

            vega_fd = finite_diff_vega(
                engine_factory=engine_factory,
                base_params=base_params,
                h_abs=fd_step_sigma,
                seed=123
            )

            if method == 'fd':
                delta_result = delta_fd
                vega_result = vega_fd
            elif method == 'both':
                # For 'both', return PW in main fields and store FD separately
                # For simplicity, we'll return PW as the primary result
                delta_result = delta_pw
                vega_result = vega_pw

        return GreeksResult(
            delta=delta_result,
            vega=vega_result,
            method=method,
            fd_step_spot=fd_step_spot if method in ['fd', 'both'] else None,
            fd_step_sigma=fd_step_sigma if method in ['fd', 'both'] else None
        )

    def _get_strike_from_payoff(self) -> float:
        """
        Extract strike price from payoff function.

        This is a helper method that attempts to get the strike from common payoff types.
        """
        # Try to get strike from payoff object attributes
        if hasattr(self.payoff, 'strike'):
            return self.payoff.strike
        elif hasattr(self.payoff, 'K'):
            return self.payoff.K
        else:
            raise AttributeError(
                "Cannot extract strike from payoff. "
                "Payoff must have 'strike' or 'K' attribute."
            )

