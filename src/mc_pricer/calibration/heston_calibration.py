"""Heston model calibration to implied volatility surfaces.

This module provides research-grade calibration of Heston stochastic volatility
model parameters to match target implied volatilities across strikes and maturities.

Implementation uses numpy-only optimization (Nelder-Mead with random restarts)
for reproducibility and transparency. Includes Common Random Numbers (CRN) and
caching for improved performance while maintaining deterministic results.
"""

import hashlib
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mc_pricer.analytics.implied_vol import implied_vol
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine


@dataclass
class MarketQuote:
    """Single market option quote for calibration.

    Attributes
    ----------
    strike : float
        Option strike price.
    maturity : float
        Time to maturity in years.
    option_type : str
        'call' or 'put'.
    implied_vol : float
        Target implied volatility (e.g., from market).
    bid_ask_width : float, optional
        Bid-ask spread for weighting (narrower = higher weight).
        If None, uniform weighting is used.
    """

    strike: float
    maturity: float
    option_type: str
    implied_vol: float
    bid_ask_width: float | None = None

    def __post_init__(self):
        """Validate inputs."""
        if self.strike <= 0:
            raise ValueError("strike must be positive")
        if self.maturity <= 0:
            raise ValueError("maturity must be positive")
        if self.option_type not in ["call", "put"]:
            raise ValueError("option_type must be 'call' or 'put'")
        if self.implied_vol <= 0:
            raise ValueError("implied_vol must be positive")
        if self.bid_ask_width is not None and self.bid_ask_width < 0:
            raise ValueError("bid_ask_width must be non-negative")


@dataclass
class CalibrationConfig:
    """Configuration for Heston calibration.

    Attributes
    ----------
    n_paths : int
        Number of Monte Carlo paths per pricing.
    n_steps : int
        Number of time steps for Heston simulation.
    rng_type : str
        'pseudo' or 'sobol' for random number generation.
    scramble : bool
        Whether to use scrambling for Sobol sequences.
    seeds : list[int]
        Random seeds for reproducibility. One seed per restart.
    max_iter : int
        Maximum iterations for each optimization run.
    tol : float
        Tolerance for convergence.
    bounds : dict[str, tuple[float, float]]
        Parameter bounds: {'kappa': (min, max), 'theta': (min, max), ...}.
    heston_scheme : str
        Heston discretization scheme ('full_truncation_euler' or 'qe').
    regularization : float
        L2 penalty coefficient for parameter magnitude (default 0.0).
    use_crn : bool
        Use Common Random Numbers for variance reduction (default False).
    cache_size : int
        Maximum number of cached pricing results (default 1000).
    """

    n_paths: int = 10000
    n_steps: int = 50
    rng_type: str = "pseudo"
    scramble: bool = False
    seeds: list[int] = field(default_factory=lambda: [42])
    max_iter: int = 200
    tol: float = 1e-6
    bounds: dict[str, tuple[float, float]] = field(
        default_factory=lambda: {
            "kappa": (0.01, 10.0),
            "theta": (0.001, 1.0),
            "xi": (0.01, 2.0),
            "rho": (-0.999, 0.999),
            "v0": (0.001, 1.0),
        }
    )
    heston_scheme: str = "full_truncation_euler"
    regularization: float = 0.0
    use_crn: bool = False
    cache_size: int = 1000

    def __post_init__(self):
        """Validate configuration."""
        if self.n_paths <= 0:
            raise ValueError("n_paths must be positive")
        if self.n_steps <= 0:
            raise ValueError("n_steps must be positive")
        if self.rng_type not in ["pseudo", "sobol"]:
            raise ValueError("rng_type must be 'pseudo' or 'sobol'")
        if not self.seeds:
            raise ValueError("seeds list cannot be empty")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.tol <= 0:
            raise ValueError("tol must be positive")
        if self.regularization < 0:
            raise ValueError("regularization must be non-negative")
        if self.cache_size < 0:
            raise ValueError("cache_size must be non-negative")


@dataclass
class CalibrationResult:
    """Result of Heston calibration.

    Attributes
    ----------
    best_params : dict[str, float]
        Optimized Heston parameters.
    objective_value : float
        Final objective function value (RMSE).
    n_evals : int
        Total number of objective function evaluations.
    runtime_sec : float
        Total calibration runtime in seconds.
    diagnostics : dict[str, Any]
        Detailed diagnostics including convergence history.
    fitted_vols : list[float]
        Model implied volatilities at optimum.
    target_vols : list[float]
        Target implied volatilities from market quotes.
    residuals : list[float]
        Differences: fitted_vols - target_vols.
    cache_hits : int
        Number of cache hits (if caching enabled).
    cache_misses : int
        Number of cache misses (if caching enabled).
    """

    best_params: dict[str, float]
    objective_value: float
    n_evals: int
    runtime_sec: float
    diagnostics: dict[str, Any]
    fitted_vols: list[float]
    target_vols: list[float]
    residuals: list[float]
    cache_hits: int = 0
    cache_misses: int = 0


class HestonCalibrator:
    """Calibrate Heston model to implied volatility surface.

    Uses Nelder-Mead simplex algorithm with random restarts.
    All operations use numpy only (no scipy dependency).

    Supports Common Random Numbers (CRN) and result caching for
    improved performance while maintaining determinism.

    Parameters
    ----------
    S0 : float
        Current spot price.
    r : float
        Risk-free rate.
    quotes : list[MarketQuote]
        Market option quotes to fit.
    config : CalibrationConfig
        Calibration configuration.
    """

    def __init__(
        self,
        S0: float,  # noqa: N803
        r: float,
        quotes: list[MarketQuote],
        config: CalibrationConfig,
    ):
        """Initialize calibrator."""
        if S0 <= 0:
            raise ValueError("S0 must be positive")
        if not quotes:
            raise ValueError("quotes list cannot be empty")

        self.S0 = S0
        self.r = r
        self.quotes = quotes
        self.config = config
        self.n_evals = 0
        self.cache_hits = 0
        self.cache_misses = 0

        # LRU cache for pricing results: (seed, K, T, params_hash) -> (price, iv)
        self._cache: OrderedDict[tuple, tuple[float, float]] = OrderedDict()

        # Compute weights from bid-ask spreads
        self.weights = self._compute_weights()

        # Parameter names in order
        self.param_names = ["kappa", "theta", "xi", "rho", "v0"]

    def _compute_weights(self) -> np.ndarray:
        """Compute weights for each quote based on bid-ask width."""
        weights = []
        eps = 1e-8
        for quote in self.quotes:
            if quote.bid_ask_width is not None:
                # Weight inversely to squared bid-ask width
                w = 1.0 / (quote.bid_ask_width**2 + eps)
            else:
                w = 1.0
            weights.append(w)

        weights = np.array(weights)
        # Normalize weights to sum to number of quotes
        weights = weights * len(weights) / np.sum(weights)
        return weights

    def _params_to_dict(self, params: np.ndarray) -> dict[str, float]:
        """Convert parameter array to dictionary."""
        return {name: float(params[i]) for i, name in enumerate(self.param_names)}

    def _dict_to_params(self, params_dict: dict[str, float]) -> np.ndarray:
        """Convert parameter dictionary to array."""
        return np.array([params_dict[name] for name in self.param_names])

    def _apply_bounds(self, params: np.ndarray) -> np.ndarray:
        """Clip parameters to bounds."""
        bounded = params.copy()
        for i, name in enumerate(self.param_names):
            if name in self.config.bounds:
                lb, ub = self.config.bounds[name]
                bounded[i] = np.clip(bounded[i], lb, ub)
        return bounded

    def _params_hash(self, params_dict: dict[str, float]) -> str:
        """Compute hash of parameters for caching."""
        # Round to reasonable precision for hash stability
        param_str = ",".join(f"{name}={params_dict[name]:.8f}" for name in self.param_names)
        return hashlib.md5(param_str.encode()).hexdigest()[:16]

    def _get_cached_or_price(
        self,
        params_dict: dict[str, float],
        quote: MarketQuote,
        seed: int,
    ) -> tuple[float, float]:
        """Get cached result or price option.

        Returns
        -------
        tuple
            (price, implied_vol)
        """
        if not self.config.use_crn or self.config.cache_size == 0:
            # No caching
            price = self._price_option(params_dict, quote, seed)
            try:
                iv = implied_vol(
                    price=price,
                    S0=self.S0,
                    K=quote.strike,
                    r=self.r,
                    T=quote.maturity,
                    option_type=quote.option_type,
                )
                return price, iv
            except (ValueError, RuntimeError):
                return price, np.nan

        # Try cache
        params_hash = self._params_hash(params_dict)
        cache_key = (seed, quote.strike, quote.maturity, params_hash)

        if cache_key in self._cache:
            self.cache_hits += 1
            # Move to end (LRU)
            self._cache.move_to_end(cache_key)
            return self._cache[cache_key]

        # Cache miss - compute
        self.cache_misses += 1
        price = self._price_option(params_dict, quote, seed)

        try:
            iv = implied_vol(
                price=price,
                S0=self.S0,
                K=quote.strike,
                r=self.r,
                T=quote.maturity,
                option_type=quote.option_type,
            )
        except (ValueError, RuntimeError):
            iv = np.nan

        # Add to cache
        self._cache[cache_key] = (price, iv)

        # Enforce cache size limit
        if len(self._cache) > self.config.cache_size:
            self._cache.popitem(last=False)  # Remove oldest

        return price, iv

    def _price_option(
        self,
        params_dict: dict[str, float],
        quote: MarketQuote,
        seed: int,
    ) -> float:
        """Price single option with given Heston parameters."""
        # Create Heston model
        model = HestonModel(
            S0=self.S0,
            r=self.r,
            T=quote.maturity,
            kappa=params_dict["kappa"],
            theta=params_dict["theta"],
            xi=params_dict["xi"],
            rho=params_dict["rho"],
            v0=params_dict["v0"],
            seed=seed,
            scheme=self.config.heston_scheme,
            rng_type=self.config.rng_type,
            scramble=self.config.scramble,
        )

        # Create payoff
        if quote.option_type == "call":
            payoff = EuropeanCallPayoff(strike=quote.strike)
        else:
            payoff = EuropeanPutPayoff(strike=quote.strike)

        # Create engine and price
        engine = HestonMonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=self.config.n_paths,
            n_steps=self.config.n_steps,
            antithetic=True,  # Use antithetic for lower variance
            seed=seed,
        )

        result = engine.price()
        return result.price

    def _compute_fitted_vols(
        self,
        params_dict: dict[str, float],
        seed: int,
    ) -> list[float]:
        """Compute model implied vols for all quotes."""
        fitted_vols = []

        for quote in self.quotes:
            _, iv = self._get_cached_or_price(params_dict, quote, seed)
            fitted_vols.append(iv)

        return fitted_vols

    def objective(self, params: np.ndarray, seed: int) -> float:
        """Compute weighted RMSE between model and target implied vols.

        Parameters
        ----------
        params : np.ndarray
            Heston parameters [kappa, theta, xi, rho, v0].
        seed : int
            Random seed for pricing.

        Returns
        -------
        float
            Weighted RMSE plus penalty terms.
        """
        self.n_evals += 1

        # Apply bounds (soft clipping)
        params = self._apply_bounds(params)
        params_dict = self._params_to_dict(params)

        # Compute model implied vols
        fitted_vols = self._compute_fitted_vols(params_dict, seed)

        # Compute weighted squared errors
        target_vols = np.array([q.implied_vol for q in self.quotes])
        fitted_vols_array = np.array(fitted_vols)

        # Handle NaN values (pricing failures)
        valid_mask = ~np.isnan(fitted_vols_array)
        if not np.any(valid_mask):
            return 1e10  # Large penalty if all prices failed

        errors = fitted_vols_array[valid_mask] - target_vols[valid_mask]
        weights = self.weights[valid_mask]
        weighted_sq_errors = weights * errors**2

        # RMSE
        rmse = np.sqrt(np.mean(weighted_sq_errors))

        # Add regularization penalty
        if self.config.regularization > 0:
            # L2 penalty on normalized parameters
            param_scales = np.array(
                [
                    self.config.bounds[name][1] - self.config.bounds[name][0]
                    for name in self.param_names
                ]
            )
            normalized_params = params / param_scales
            l2_penalty = self.config.regularization * np.sum(normalized_params**2)
            rmse += l2_penalty

        return rmse

    def _nelder_mead(
        self,
        x0: np.ndarray,
        seed: int,
    ) -> tuple[np.ndarray, float, list[float]]:
        """Nelder-Mead simplex optimization.

        Parameters
        ----------
        x0 : np.ndarray
            Initial parameter guess.
        seed : int
            Random seed for objective evaluation.

        Returns
        -------
        tuple
            (best_params, best_value, history)
        """
        # Nelder-Mead parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5  # Contraction
        sigma = 0.5  # Shrink

        n = len(x0)

        # Initialize simplex with slight perturbations
        rng = np.random.default_rng(seed)
        simplex = np.zeros((n + 1, n))
        simplex[0] = x0

        for i in range(n):
            perturbed = x0.copy()
            # Perturb by 5% of range
            lb, ub = self.config.bounds[self.param_names[i]]
            perturbed[i] += 0.05 * (ub - lb) * (2 * rng.random() - 1)
            simplex[i + 1] = self._apply_bounds(perturbed)

        # Evaluate initial simplex
        f_values = np.array([self.objective(x, seed) for x in simplex])

        history = [float(np.min(f_values))]

        for _ in range(self.config.max_iter):
            # Sort simplex by objective value
            order = np.argsort(f_values)
            simplex = simplex[order]
            f_values = f_values[order]

            # Check convergence
            if np.max(np.abs(f_values[1:] - f_values[0])) < self.config.tol:
                break

            # Compute centroid of best n points
            centroid = np.mean(simplex[:-1], axis=0)

            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = self._apply_bounds(x_r)
            f_r = self.objective(x_r, seed)

            if f_values[0] <= f_r < f_values[-2]:
                # Accept reflection
                simplex[-1] = x_r
                f_values[-1] = f_r
            elif f_r < f_values[0]:
                # Try expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = self._apply_bounds(x_e)
                f_e = self.objective(x_e, seed)

                if f_e < f_r:
                    simplex[-1] = x_e
                    f_values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
            else:
                # Try contraction
                if f_r < f_values[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid + rho * (simplex[-1] - centroid)

                x_c = self._apply_bounds(x_c)
                f_c = self.objective(x_c, seed)

                if f_c < f_values[-1]:
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrink
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = self._apply_bounds(simplex[i])
                        f_values[i] = self.objective(simplex[i], seed)

            history.append(float(f_values[0]))

        return simplex[0], f_values[0], history

    def calibrate(self, initial_guess: dict[str, float] | None = None) -> CalibrationResult:
        """Run calibration with random restarts.

        Parameters
        ----------
        initial_guess : dict[str, float], optional
            Initial parameter guess. If None, uses midpoint of bounds.

        Returns
        -------
        CalibrationResult
            Calibration results with diagnostics.
        """
        start_time = time.time()
        self.n_evals = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self._cache.clear()

        # Set initial guess
        if initial_guess is None:
            # Use midpoint of bounds
            initial_guess = {
                name: (bounds[0] + bounds[1]) / 2 for name, bounds in self.config.bounds.items()
            }

        x0 = self._dict_to_params(initial_guess)

        # Run optimization with multiple restarts
        best_params = None
        best_value = np.inf
        all_histories = []
        restart_results = []

        for i, seed in enumerate(self.config.seeds):
            if i == 0:
                # First run: use initial guess
                x_init = x0
            else:
                # Subsequent runs: random restart within bounds
                rng = np.random.default_rng(seed)
                x_init = np.array(
                    [
                        rng.uniform(bounds[0], bounds[1])
                        for name, bounds in [(n, self.config.bounds[n]) for n in self.param_names]
                    ]
                )

            # Run optimization
            params, value, history = self._nelder_mead(x_init, seed)
            all_histories.append(history)
            restart_results.append(
                {
                    "seed": seed,
                    "final_value": float(value),
                    "n_iterations": len(history),
                    "params": self._params_to_dict(params),
                }
            )

            # Update best
            if value < best_value:
                best_value = value
                best_params = params

        runtime = time.time() - start_time

        # Compute final fitted vols and residuals
        best_params_dict = self._params_to_dict(best_params)
        fitted_vols = self._compute_fitted_vols(best_params_dict, self.config.seeds[0])
        target_vols = [q.implied_vol for q in self.quotes]
        residuals = [
            fv - tv if not np.isnan(fv) else np.nan for fv, tv in zip(fitted_vols, target_vols)
        ]

        # Build diagnostics
        diagnostics = {
            "n_restarts": len(self.config.seeds),
            "convergence_histories": all_histories,
            "restart_results": restart_results,
            "initial_guess": initial_guess,
            "bounds": self.config.bounds,
            "config": {
                "n_paths": self.config.n_paths,
                "n_steps": self.config.n_steps,
                "rng_type": self.config.rng_type,
                "scramble": self.config.scramble,
                "heston_scheme": self.config.heston_scheme,
                "max_iter": self.config.max_iter,
                "tol": self.config.tol,
                "use_crn": self.config.use_crn,
                "cache_size": self.config.cache_size,
            },
        }

        return CalibrationResult(
            best_params=best_params_dict,
            objective_value=float(best_value),
            n_evals=self.n_evals,
            runtime_sec=runtime,
            diagnostics=diagnostics,
            fitted_vols=fitted_vols,
            target_vols=target_vols,
            residuals=residuals,
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
        )
