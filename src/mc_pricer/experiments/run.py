"""
Experiment execution engine for reproducible simulations.
"""

import platform
import subprocess
import sys
import time
from datetime import datetime

import numpy as np

from mc_pricer.experiments.types import (
    ExperimentConfig,
    ExperimentMetadata,
    ExperimentResult,
    GreeksData,
)
from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine
from mc_pricer.pricers.lsm import price_american_lsm
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


def get_git_commit() -> str | None:
    """Get current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=1,
            check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def create_metadata(
    config: ExperimentConfig,
    seed: int,
    n_paths: int,
    n_steps: int | None
) -> ExperimentMetadata:
    """Create metadata for reproducibility."""
    return ExperimentMetadata(
        timestamp=datetime.now().isoformat(),
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        numpy_version=np.__version__,
        os_platform=platform.platform(),
        git_commit=get_git_commit(),
        seed=seed,
        model=config.model,
        option_type=config.option_type,
        style=config.style,
        n_paths=n_paths,
        n_steps=n_steps,
        antithetic=config.antithetic,
        control_variate=config.control_variate,
        compute_greeks=config.compute_greeks,
    )


def run_experiment(config: ExperimentConfig) -> list[ExperimentResult]:
    """
    Run experiment with given configuration.

    Executes pricing experiments across all combinations of n_paths and seeds
    specified in the configuration. Measures runtime and captures full metadata
    for reproducibility.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration

    Returns
    -------
    list[ExperimentResult]
        List of results, one per (n_paths, seed) combination

    Raises
    ------
    ValueError
        If configuration is invalid
    """
    results = []

    # Validate configuration
    if config.model == "gbm" and config.sigma is None:
        raise ValueError("GBM model requires sigma parameter")

    if config.model == "heston":
        missing = []
        if config.kappa is None:
            missing.append("kappa")
        if config.theta is None:
            missing.append("theta")
        if config.xi is None:
            missing.append("xi")
        if config.rho is None:
            missing.append("rho")
        if config.v0 is None:
            missing.append("v0")
        if missing:
            raise ValueError(f"Heston model requires: {', '.join(missing)}")

    # Create payoff
    if config.option_type == "call":
        payoff = EuropeanCallPayoff(strike=config.K)
    else:
        payoff = EuropeanPutPayoff(strike=config.K)

    # Run experiments
    for n_paths in config.n_paths_list:
        for seed in config.seeds:
            # Determine n_steps
            if config.model == "heston":
                n_steps = config.n_steps
            elif config.style == "american":
                n_steps = config.n_steps
            else:
                n_steps = None

            # Create metadata
            metadata = create_metadata(config, seed, n_paths, n_steps)

            # Build notes
            notes_parts = [config.model.upper()]
            if config.antithetic:
                notes_parts.append("antithetic")
            if config.control_variate and config.model == "gbm" and config.style == "european":
                notes_parts.append("control_variate")
            if config.style == "american":
                notes_parts.append(f"LSM_{config.lsm_basis}")
            notes = "+".join(notes_parts)

            # Start timing
            start_time = time.perf_counter()

            try:
                # Price based on style
                if config.style == "american":
                    # American option with LSM
                    if config.model != "gbm":
                        raise ValueError("American options only supported with GBM model")

                    model = GeometricBrownianMotion(
                        S0=config.S0,
                        r=config.r,
                        sigma=config.sigma,  # type: ignore
                        T=config.T,
                        seed=seed
                    )

                    result = price_american_lsm(
                        model=model,
                        strike=config.K,
                        option_type=config.option_type,
                        n_paths=n_paths,
                        n_steps=n_steps,  # type: ignore
                        basis=config.lsm_basis,
                        seed=seed
                    )

                    runtime = time.perf_counter() - start_time

                    exp_result = ExperimentResult(
                        config_name=config.name,
                        price=result.price,
                        stderr=result.stderr,
                        ci_lower=result.ci_lower,
                        ci_upper=result.ci_upper,
                        ci_width=result.ci_upper - result.ci_lower,
                        relative_error=result.stderr / result.price if result.price > 0 else 0,
                        n_paths=n_paths,
                        n_steps=n_steps,
                        runtime_seconds=runtime,
                        control_variate_beta=None,
                        greeks=None,
                        implied_vol=None,
                        metadata=metadata,
                        notes=notes
                    )

                elif config.model == "gbm":
                    # GBM European
                    model = GeometricBrownianMotion(
                        S0=config.S0,
                        r=config.r,
                        sigma=config.sigma,  # type: ignore
                        T=config.T,
                        seed=seed
                    )

                    engine = MonteCarloEngine(
                        model=model,
                        payoff=payoff,
                        n_paths=n_paths,
                        antithetic=config.antithetic,
                        control_variate=config.control_variate
                    )

                    result = engine.price()
                    runtime = time.perf_counter() - start_time

                    # Compute Greeks if requested
                    greeks_data = None
                    if config.compute_greeks:
                        greeks_result = engine.compute_greeks(
                            option_type=config.option_type,
                            method=config.greeks_method
                        )
                        delta_val = greeks_result.delta.value if greeks_result.delta else None
                        delta_se = (
                            greeks_result.delta.standard_error
                            if greeks_result.delta else None
                        )
                        vega_val = greeks_result.vega.value if greeks_result.vega else None
                        vega_se = (
                            greeks_result.vega.standard_error
                            if greeks_result.vega else None
                        )
                        greeks_data = GreeksData(
                            delta=delta_val,
                            delta_stderr=delta_se,
                            vega=vega_val,
                            vega_stderr=vega_se,
                            method=config.greeks_method
                        )

                    exp_result = ExperimentResult(
                        config_name=config.name,
                        price=result.price,
                        stderr=result.stderr,
                        ci_lower=result.ci_lower,
                        ci_upper=result.ci_upper,
                        ci_width=result.ci_upper - result.ci_lower,
                        relative_error=result.stderr / result.price if result.price > 0 else 0,
                        n_paths=n_paths,
                        n_steps=n_steps,
                        runtime_seconds=runtime,
                        control_variate_beta=result.control_variate_beta,
                        greeks=greeks_data,
                        implied_vol=None,
                        metadata=metadata,
                        notes=notes
                    )

                else:  # heston
                    model = HestonModel(
                        S0=config.S0,
                        r=config.r,
                        T=config.T,
                        kappa=config.kappa,  # type: ignore
                        theta=config.theta,  # type: ignore
                        xi=config.xi,  # type: ignore
                        rho=config.rho,  # type: ignore
                        v0=config.v0,  # type: ignore
                        seed=seed
                    )

                    engine = HestonMonteCarloEngine(
                        model=model,
                        payoff=payoff,
                        n_paths=n_paths,
                        n_steps=n_steps,  # type: ignore
                        antithetic=config.antithetic,
                        seed=seed
                    )

                    result = engine.price()
                    runtime = time.perf_counter() - start_time

                    exp_result = ExperimentResult(
                        config_name=config.name,
                        price=result.price,
                        stderr=result.stderr,
                        ci_lower=result.ci_lower,
                        ci_upper=result.ci_upper,
                        ci_width=result.ci_upper - result.ci_lower,
                        relative_error=result.stderr / result.price if result.price > 0 else 0,
                        n_paths=n_paths,
                        n_steps=n_steps,
                        runtime_seconds=runtime,
                        control_variate_beta=None,
                        greeks=None,
                        implied_vol=None,
                        metadata=metadata,
                        notes=notes
                    )

                results.append(exp_result)

            except Exception as e:
                # Log error but continue with other experiments
                print(f"Error in experiment {config.name} (n_paths={n_paths}, seed={seed}): {e}")
                continue

    return results
