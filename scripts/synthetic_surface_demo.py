"""Synthetic implied volatility surface calibration demo.

Demonstrates Heston model calibration to a generated surface with:
- Full mode: 15x3 grid, 50000 paths, 100 steps (~2-3 min)
- Fast mode: 5x2 grid, 10000 paths, 50 steps (<30 sec)

Outputs JSON artifact with git metadata for reproducible research.
"""

import argparse

import numpy as np

from mc_pricer.calibration import (
    CalibrationConfig,
    HestonCalibrator,
    MarketQuote,
)
from mc_pricer.experiments.artifacts import save_artifact


def generate_synthetic_surface(
    S0: float,  # noqa: N803
    r: float,
    strikes: np.ndarray,
    maturities: np.ndarray,
    true_params: dict[str, float],
    bid_ask_pct: float = 0.01,
) -> list[MarketQuote]:
    """Generate synthetic implied vol surface using true Heston parameters.

    Parameters
    ----------
    S0 : float
        Spot price.
    r : float
        Risk-free rate.
    strikes : np.ndarray
        Strike prices.
    maturities : np.ndarray
        Maturities in years.
    true_params : dict[str, float]
        True Heston parameters {'kappa': ..., 'theta': ..., ...}.
    bid_ask_pct : float
        Bid-ask spread as percentage of IV (default 1%).

    Returns
    -------
    list[MarketQuote]
        Synthetic market quotes.
    """
    from mc_pricer.analytics.implied_vol import implied_vol
    from mc_pricer.models.heston import HestonModel
    from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
    from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine

    quotes = []
    seed = 9999  # Fixed seed for synthetic data generation

    for T in maturities:
        for K in strikes:
            # Generate price with true parameters
            model = HestonModel(
                S0=S0,
                r=r,
                T=T,
                kappa=true_params["kappa"],
                theta=true_params["theta"],
                xi=true_params["xi"],
                rho=true_params["rho"],
                v0=true_params["v0"],
                seed=seed,
                scheme="full_truncation_euler",
            )

            payoff = EuropeanCallPayoff(strike=K)
            engine = HestonMonteCarloEngine(
                model=model,
                payoff=payoff,
                n_paths=50000,  # High accuracy for synthetic data
                n_steps=100,
                antithetic=True,
                seed=seed,
            )

            result = engine.price()
            price = result.price

            # Convert to implied vol
            try:
                iv = implied_vol(
                    price=price,
                    S0=S0,
                    K=K,
                    r=r,
                    T=T,
                    option_type="call",
                )

                # Add bid-ask spread
                ba_width = iv * bid_ask_pct

                quotes.append(
                    MarketQuote(
                        strike=K,
                        maturity=T,
                        option_type="call",
                        implied_vol=iv,
                        bid_ask_width=ba_width,
                    )
                )
            except (ValueError, RuntimeError):
                # Skip if IV calculation fails
                continue

    return quotes


def run_calibration(fast_mode: bool = False, output_path: str | None = None):
    """Run synthetic surface calibration.

    Parameters
    ----------
    fast_mode : bool
        If True, use reduced grid for <30s runtime.
    output_path : str, optional
        Path for JSON artifact output.
    """
    # Market parameters
    S0 = 100.0
    r = 0.05

    # True Heston parameters (to be recovered)
    true_params = {
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.3,
        "rho": -0.7,
        "v0": 0.04,
    }

    # Configure grid based on mode
    if fast_mode:
        print("Running in FAST mode (<30 seconds)...")
        strikes = S0 * np.array([0.9, 1.0, 1.1])
        maturities = np.array([0.25, 1.0])
        config = CalibrationConfig(
            n_paths=10000,
            n_steps=50,
            seeds=[42],  # Single restart
            max_iter=50,
            rng_type="pseudo",
            scramble=False,
            use_crn=True,  # Enable CRN for speed
            cache_size=1000,
        )
    else:
        print("Running in FULL mode (~2-3 minutes)...")
        strikes = S0 * np.linspace(0.8, 1.2, 5)
        maturities = np.array([0.25, 0.5, 1.0])
        config = CalibrationConfig(
            n_paths=50000,
            n_steps=100,
            seeds=[42, 123, 456],  # Three restarts
            max_iter=200,
            rng_type="pseudo",
            scramble=False,
            use_crn=True,  # Enable CRN
            cache_size=5000,
        )

    print(f"Grid: {len(strikes)} strikes x {len(maturities)} maturities")
    print(f"Paths: {config.n_paths}, Steps: {config.n_steps}")
    print(f"CRN: {config.use_crn}, Cache size: {config.cache_size}")
    print()

    # Generate synthetic surface
    print("Generating synthetic surface...")
    quotes = generate_synthetic_surface(S0, r, strikes, maturities, true_params)
    print(f"Generated {len(quotes)} market quotes")
    print()

    # Initial guess (intentionally poor)
    initial_guess = {
        "kappa": 1.0,
        "theta": 0.08,
        "xi": 0.5,
        "rho": -0.5,
        "v0": 0.08,
    }

    print("True parameters:")
    for name, value in true_params.items():
        print(f"  {name}: {value:.4f}")
    print()

    print("Initial guess:")
    for name, value in initial_guess.items():
        print(f"  {name}: {value:.4f}")
    print()

    # Run calibration
    print("Running calibration...")
    calibrator = HestonCalibrator(S0=S0, r=r, quotes=quotes, config=config)
    result = calibrator.calibrate(initial_guess=initial_guess)

    print("\nCalibration Results")
    print("=" * 60)
    print(f"Runtime: {result.runtime_sec:.2f} seconds")
    print(f"Objective value (RMSE): {result.objective_value:.6f}")
    print(f"Function evaluations: {result.n_evals}")
    print(f"Cache hits: {result.cache_hits}")
    print(f"Cache misses: {result.cache_misses}")
    if result.cache_hits + result.cache_misses > 0:
        hit_rate = result.cache_hits / (result.cache_hits + result.cache_misses)
        print(f"Cache hit rate: {hit_rate:.1%}")
    print()

    print("Fitted parameters:")
    for name, value in result.best_params.items():
        true_val = true_params[name]
        error_pct = 100 * abs(value - true_val) / abs(true_val)
        print(f"  {name}: {value:.4f} (true: {true_val:.4f}, error: {error_pct:.1f}%)")
    print()

    # Show fitted vs target IVs
    print("Fitted vs Target Implied Volatilities:")
    print(f"{'Strike':<10} {'Maturity':<10} {'Target':<12} {'Fitted':<12} {'Error'}")
    print("-" * 60)
    for i, quote in enumerate(quotes):
        target = result.target_vols[i]
        fitted = result.fitted_vols[i]
        error = result.residuals[i]
        if not np.isnan(fitted):
            print(
                f"{quote.strike:<10.2f} {quote.maturity:<10.2f} "
                f"{target:<12.4f} {fitted:<12.4f} {error:+.4f}"
            )

    # Save artifact if requested
    if output_path is not None:
        artifact_data = {
            "experiment": "synthetic_surface_calibration",
            "mode": "fast" if fast_mode else "full",
            "true_parameters": true_params,
            "fitted_parameters": result.best_params,
            "objective_value": result.objective_value,
            "runtime_sec": result.runtime_sec,
            "n_evals": result.n_evals,
            "cache_hits": result.cache_hits,
            "cache_misses": result.cache_misses,
            "config": {
                "n_paths": config.n_paths,
                "n_steps": config.n_steps,
                "n_restarts": len(config.seeds),
                "use_crn": config.use_crn,
                "cache_size": config.cache_size,
            },
            "grid": {
                "n_strikes": len(strikes),
                "n_maturities": len(maturities),
                "n_quotes": len(quotes),
            },
            "fitted_vols": result.fitted_vols,
            "target_vols": result.target_vols,
            "residuals": result.residuals,
        }

        save_artifact(artifact_data, output_path, include_metadata=True)
        print(f"\nArtifact saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate Heston model to synthetic surface"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast mode with reduced grid (<30 seconds)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for JSON artifact",
    )

    args = parser.parse_args()

    run_calibration(fast_mode=args.fast, output_path=args.out)
