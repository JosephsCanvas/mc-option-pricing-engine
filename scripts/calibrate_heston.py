"""Command-line tool for Heston model calibration.

Reads market quotes from CSV and calibrates Heston parameters.
Supports fast mode for quick iteration and full mode for production.
Outputs reproducible JSON artifacts with git metadata.

CSV format:
    strike,maturity,option_type,implied_vol,bid_ask_width
    95.0,0.25,call,0.25,0.005
    100.0,0.25,call,0.22,0.004
    ...

Example usage:
    # Fast mode (<30s) for iteration
    python scripts/calibrate_heston.py data/quotes.csv --fast --out results/calib_fast.json

    # Full mode for production
    python scripts/calibrate_heston.py data/quotes.csv --out results/calib_full.json
"""

import argparse
import csv
from pathlib import Path

from mc_pricer.calibration import (
    CalibrationConfig,
    HestonCalibrator,
    MarketQuote,
)
from mc_pricer.experiments.artifacts import save_artifact


def load_quotes_from_csv(csv_path: str) -> list[MarketQuote]:
    """Load market quotes from CSV file.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with columns: strike, maturity, option_type,
        implied_vol, bid_ask_width (optional).

    Returns
    -------
    list[MarketQuote]
        Parsed market quotes.
    """
    quotes = []

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            strike = float(row["strike"])
            maturity = float(row["maturity"])
            option_type = row["option_type"].lower()
            implied_vol = float(row["implied_vol"])

            # Bid-ask width is optional
            bid_ask_width = (
                float(row["bid_ask_width"]) if "bid_ask_width" in row else None
            )

            quotes.append(
                MarketQuote(
                    strike=strike,
                    maturity=maturity,
                    option_type=option_type,
                    implied_vol=implied_vol,
                    bid_ask_width=bid_ask_width,
                )
            )

    return quotes


def calibrate_from_csv(
    csv_path: str,
    S0: float,  # noqa: N803
    r: float,
    fast_mode: bool = False,
    output_path: str | None = None,
):
    """Calibrate Heston model from CSV quotes.

    Parameters
    ----------
    csv_path : str
        Path to CSV file with market quotes.
    S0 : float
        Current spot price.
    r : float
        Risk-free rate.
    fast_mode : bool
        If True, use fast configuration (<30s runtime).
    output_path : str, optional
        Path for JSON artifact output.
    """
    # Load quotes
    print(f"Loading quotes from: {csv_path}")
    quotes = load_quotes_from_csv(csv_path)
    print(f"Loaded {len(quotes)} market quotes")
    print()

    # Show quote summary
    strikes = sorted(set(q.strike for q in quotes))
    maturities = sorted(set(q.maturity for q in quotes))
    print(
        f"Unique strikes: {len(strikes)} "
        f"(range: {min(strikes):.2f} - {max(strikes):.2f})"
    )
    print(
        f"Unique maturities: {len(maturities)} "
        f"(range: {min(maturities):.2f} - {max(maturities):.2f})"
    )
    print()

    # Configure based on mode
    if fast_mode:
        print("Running in FAST mode (<30 seconds)...")
        config = CalibrationConfig(
            n_paths=10000,
            n_steps=50,
            seeds=[42],  # Single restart
            max_iter=50,
            rng_type="pseudo",
            scramble=False,
            use_crn=True,  # Enable CRN
            cache_size=1000,
            regularization=0.001,  # Small regularization
        )
    else:
        print("Running in FULL mode...")
        config = CalibrationConfig(
            n_paths=50000,
            n_steps=100,
            seeds=[42, 123, 456],  # Three restarts
            max_iter=200,
            rng_type="pseudo",
            scramble=False,
            use_crn=True,  # Enable CRN
            cache_size=5000,
            regularization=0.001,
        )

    print(f"Paths: {config.n_paths}, Steps: {config.n_steps}")
    print(f"Restarts: {len(config.seeds)}")
    print(f"CRN: {config.use_crn}, Cache size: {config.cache_size}")
    print()

    # Initial guess (midpoint of typical ranges)
    initial_guess = {
        "kappa": 2.0,
        "theta": 0.04,
        "xi": 0.5,
        "rho": -0.5,
        "v0": 0.04,
    }

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
        print(f"  {name}: {value:.4f}")
    print()

    # Show sample of fitted vs target
    print("Sample: Fitted vs Target Implied Volatilities:")
    print(f"{'Strike':<10} {'Maturity':<10} {'Target':<12} {'Fitted':<12} {'Error'}")
    print("-" * 60)
    # Show first 10 quotes
    for i in range(min(10, len(quotes))):
        quote = quotes[i]
        target = result.target_vols[i]
        fitted = result.fitted_vols[i]
        error = result.residuals[i]
        if not np.isnan(fitted):  # noqa: F821
            print(
                f"{quote.strike:<10.2f} {quote.maturity:<10.2f} "
                f"{target:<12.4f} {fitted:<12.4f} {error:+.4f}"
            )

    if len(quotes) > 10:
        print(f"... ({len(quotes) - 10} more quotes)")

    # Compute error statistics
    import numpy as np

    valid_residuals = [r for r in result.residuals if not np.isnan(r)]
    if valid_residuals:
        print()
        print("Error statistics:")
        print(f"  Mean absolute error: {np.mean(np.abs(valid_residuals)):.6f}")
        print(f"  Max absolute error: {np.max(np.abs(valid_residuals)):.6f}")
        print(f"  Std dev of errors: {np.std(valid_residuals):.6f}")

    # Save artifact if requested
    if output_path is not None:
        artifact_data = {
            "experiment": "heston_calibration_from_csv",
            "mode": "fast" if fast_mode else "full",
            "input_file": str(Path(csv_path).resolve()),
            "market_params": {
                "S0": S0,
                "r": r,
            },
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
                "regularization": config.regularization,
            },
            "data": {
                "n_quotes": len(quotes),
                "n_strikes": len(strikes),
                "n_maturities": len(maturities),
            },
            "fitted_vols": result.fitted_vols,
            "target_vols": result.target_vols,
            "residuals": result.residuals,
            "error_stats": {
                "mean_abs_error": float(np.mean(np.abs(valid_residuals)))
                if valid_residuals
                else None,
                "max_abs_error": float(np.max(np.abs(valid_residuals)))
                if valid_residuals
                else None,
                "std_error": float(np.std(valid_residuals)) if valid_residuals else None,
            },
        }

        save_artifact(artifact_data, output_path, include_metadata=True)
        print(f"\nArtifact saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate Heston model to market implied volatilities"
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to CSV file with market quotes",
    )
    parser.add_argument(
        "--S0",
        type=float,
        default=100.0,
        help="Current spot price (default: 100.0)",
    )
    parser.add_argument(
        "--r",
        type=float,
        default=0.05,
        help="Risk-free rate (default: 0.05)",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fast mode with reduced paths/steps (<30 seconds)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output path for JSON artifact",
    )

    args = parser.parse_args()

    calibrate_from_csv(
        csv_path=args.csv_path,
        S0=args.S0,
        r=args.r,
        fast_mode=args.fast,
        output_path=args.out,
    )
