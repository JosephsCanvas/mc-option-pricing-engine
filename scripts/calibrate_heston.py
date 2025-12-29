"""Command-line interface for Heston calibration.

Calibrate Heston model parameters to a volatility surface specified
via command-line arguments or input file.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.calibration import CalibrationConfig, HestonCalibrator, MarketQuote


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Calibrate Heston model to implied volatility surface"
    )

    # Market parameters
    parser.add_argument("--S0", type=float, required=True, help="Spot price")
    parser.add_argument("--r", type=float, required=True, help="Risk-free rate")

    # Surface specification
    parser.add_argument(
        "--quotes_file",
        type=str,
        default=None,
        help="JSON file with market quotes (overrides --strikes/--maturities/--vols)",
    )
    parser.add_argument(
        "--strikes",
        type=float,
        nargs="+",
        help="Strike prices (required if no --quotes_file)",
    )
    parser.add_argument(
        "--maturities",
        type=float,
        nargs="+",
        help="Maturities in years (required if no --quotes_file)",
    )
    parser.add_argument(
        "--vols",
        type=float,
        nargs="+",
        help="Implied vols (flattened grid: K1T1, K2T1, ..., required if no --quotes_file)",
    )

    # Calibration config
    parser.add_argument(
        "--n_paths", type=int, default=10000, help="Number of MC paths"
    )
    parser.add_argument(
        "--n_steps", type=int, default=50, help="Number of time steps"
    )
    parser.add_argument(
        "--rng",
        type=str,
        choices=["pseudo", "sobol"],
        default="pseudo",
        help="RNG type",
    )
    parser.add_argument("--scramble", action="store_true", help="Use Sobol scrambling")
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Random seeds for restarts",
    )
    parser.add_argument(
        "--max_iter", type=int, default=200, help="Max iterations per restart"
    )
    parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance")
    parser.add_argument(
        "--heston_scheme",
        type=str,
        choices=["full_truncation_euler", "qe"],
        default="qe",
        help="Heston discretization scheme",
    )

    # Initial guess
    parser.add_argument("--kappa_init", type=float, default=2.0, help="Initial kappa")
    parser.add_argument("--theta_init", type=float, default=0.04, help="Initial theta")
    parser.add_argument("--xi_init", type=float, default=0.3, help="Initial xi")
    parser.add_argument("--rho_init", type=float, default=-0.7, help="Initial rho")
    parser.add_argument("--v0_init", type=float, default=0.04, help="Initial v0")

    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="calibration_result.json",
        help="Output JSON file",
    )

    return parser.parse_args()


def load_quotes_from_file(filename: str) -> list[MarketQuote]:
    """Load market quotes from JSON file.

    Expected format:
    {
        "quotes": [
            {"strike": 100, "maturity": 1.0, "option_type": "call", "implied_vol": 0.2},
            ...
        ]
    }
    """
    with open(filename) as f:
        data = json.load(f)

    quotes = []
    for q in data["quotes"]:
        quotes.append(
            MarketQuote(
                strike=q["strike"],
                maturity=q["maturity"],
                option_type=q["option_type"],
                implied_vol=q["implied_vol"],
                bid_ask_width=q.get("bid_ask_width"),
            )
        )

    return quotes


def create_quotes_from_grid(
    strikes: list[float],
    maturities: list[float],
    vols: list[float],
) -> list[MarketQuote]:
    """Create quotes from strike/maturity/vol grid.

    Vols should be flattened in order: all strikes for T1, then all strikes for T2, etc.
    """
    if len(vols) != len(strikes) * len(maturities):
        raise ValueError(
            f"Expected {len(strikes) * len(maturities)} vols, got {len(vols)}"
        )

    quotes = []
    idx = 0
    for T in maturities:
        for K in strikes:
            quotes.append(
                MarketQuote(
                    strike=K,
                    maturity=T,
                    option_type="call",
                    implied_vol=vols[idx],
                )
            )
            idx += 1

    return quotes


def main():
    """Run calibration from command line."""
    args = parse_args()

    print("\n" + "=" * 80)
    print("Heston Calibration")
    print("=" * 80)

    # Load or create quotes
    if args.quotes_file:
        print(f"\nLoading quotes from {args.quotes_file}...")
        quotes = load_quotes_from_file(args.quotes_file)
    else:
        if not (args.strikes and args.maturities and args.vols):
            print("Error: Must provide either --quotes_file or --strikes/--maturities/--vols")
            sys.exit(1)

        print("\nCreating quotes from grid...")
        quotes = create_quotes_from_grid(args.strikes, args.maturities, args.vols)

    print(f"Loaded {len(quotes)} market quotes")

    # Print surface
    print("\nTarget Implied Volatility Surface:")
    maturities = sorted(set(q.maturity for q in quotes))
    strikes = sorted(set(q.strike for q in quotes))

    print(f"{'K':<10} ", end="")
    for T in maturities:
        print(f"T={T:<6.2f} ", end="")
    print()
    print("-" * (10 + 9 * len(maturities)))

    for K in strikes:
        print(f"{K:<10.2f} ", end="")
        for T in maturities:
            quote = next((q for q in quotes if q.strike == K and q.maturity == T), None)
            if quote:
                print(f"{quote.implied_vol:<9.4f}", end="")
            else:
                print(f"{'N/A':<9}", end="")
        print()

    # Create calibration config
    config = CalibrationConfig(
        n_paths=args.n_paths,
        n_steps=args.n_steps,
        rng_type=args.rng,
        scramble=args.scramble,
        seeds=args.seeds,
        max_iter=args.max_iter,
        tol=args.tol,
        heston_scheme=args.heston_scheme,
    )

    print("\nCalibration Configuration:")
    print(f"  Spot price (S0):     {args.S0:.2f}")
    print(f"  Risk-free rate (r):  {args.r:.4f}")
    print(f"  n_paths:             {config.n_paths}")
    print(f"  n_steps:             {config.n_steps}")
    print(f"  rng_type:            {config.rng_type}")
    print(f"  scramble:            {config.scramble}")
    print(f"  restarts:            {len(config.seeds)}")
    print(f"  max_iter:            {config.max_iter}")
    print(f"  scheme:              {config.heston_scheme}")

    # Initial guess
    initial_guess = {
        "kappa": args.kappa_init,
        "theta": args.theta_init,
        "xi": args.xi_init,
        "rho": args.rho_init,
        "v0": args.v0_init,
    }

    print("\nInitial Guess:")
    for name, value in initial_guess.items():
        print(f"  {name:8s}: {value:.6f}")

    # Create calibrator
    calibrator = HestonCalibrator(S0=args.S0, r=args.r, quotes=quotes, config=config)

    # Run calibration
    print("\n" + "=" * 80)
    print("Running Calibration...")
    print("=" * 80 + "\n")

    result = calibrator.calibrate(initial_guess=initial_guess)

    # Print results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    print(f"\nObjective Value (RMSE): {result.objective_value:.8f}")
    print(f"Function Evaluations:   {result.n_evals}")
    print(f"Runtime:                {result.runtime_sec:.2f} seconds")

    print("\nCalibrated Parameters:")
    for name, value in result.best_params.items():
        print(f"  {name:8s}: {value:.6f}")

    # Print restart results
    print(f"\nRestart Summary ({len(config.seeds)} restarts):")
    print(f"{'Seed':<10} {'Objective':<15} {'Iterations'}")
    print("-" * 40)
    for restart in result.diagnostics["restart_results"]:
        print(
            f"{restart['seed']:<10} {restart['final_value']:<15.8f} "
            f"{restart['n_iterations']}"
        )

    # Print fitted vs target
    print("\nFitted vs Target Implied Volatilities:")
    print(f"{'Strike':<8} {'Maturity':<10} {'Target':<10} {'Fitted':<10} {'Residual'}")
    print("-" * 60)
    for i, quote in enumerate(quotes):
        target = result.target_vols[i]
        fitted = result.fitted_vols[i]
        residual = result.residuals[i]
        print(
            f"{quote.strike:<8.1f} {quote.maturity:<10.2f} "
            f"{target:<10.6f} {fitted:<10.6f} {residual:>8.6f}"
        )

    # Residual statistics
    valid_residuals = [r for r in result.residuals if not np.isnan(r)]
    if valid_residuals:
        print("\nResidual Statistics:")
        print(f"  Mean:   {np.mean(valid_residuals):.8f}")
        print(f"  Std:    {np.std(valid_residuals):.8f}")
        print(f"  Max:    {np.max(np.abs(valid_residuals)):.8f}")
        print(f"  RMSE:   {np.sqrt(np.mean(np.array(valid_residuals)**2)):.8f}")

    # Save results
    output_data = {
        "market_params": {"S0": args.S0, "r": args.r},
        "calibrated_params": result.best_params,
        "objective_value": result.objective_value,
        "n_evals": result.n_evals,
        "runtime_sec": result.runtime_sec,
        "quotes": [
            {
                "strike": q.strike,
                "maturity": q.maturity,
                "option_type": q.option_type,
                "implied_vol": q.implied_vol,
            }
            for q in quotes
        ],
        "fitted_vols": result.fitted_vols,
        "residuals": [float(r) if not np.isnan(r) else None for r in result.residuals],
        "diagnostics": {
            "n_restarts": result.diagnostics["n_restarts"],
            "restart_results": result.diagnostics["restart_results"],
            "initial_guess": result.diagnostics["initial_guess"],
            "config": result.diagnostics["config"],
        },
    }

    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {args.output}")
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
