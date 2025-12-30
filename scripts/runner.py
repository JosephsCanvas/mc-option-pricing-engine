#!/usr/bin/env python
"""
Reproducible experiment runner.

Usage:
    python scripts/runner.py --experiment european_call_bench
    python scripts/runner.py --experiment heston_smile_bench
    python scripts/runner.py --experiment all
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.analytics.implied_vol import implied_vol
from mc_pricer.experiments import ExperimentConfig, run_experiment, save_results


def get_european_call_bench() -> ExperimentConfig:
    """
    Benchmark comparing variance reduction methods for European call.

    Compares:
    - Plain Monte Carlo
    - Antithetic variates
    - Control variate
    - Combined (antithetic + control variate)
    """
    return ExperimentConfig(
        name="european_call_variance_reduction",
        model="gbm",
        option_type="call",
        style="european",
        S0=100.0,
        K=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        n_paths_list=[10000, 50000, 100000],
        seeds=[42, 123, 456],
        antithetic=False,
        control_variate=False,
        compute_greeks=False,
    )


def get_heston_smile_bench() -> ExperimentConfig:
    """
    Heston volatility smile benchmark.

    Prices calls across strikes to demonstrate smile/skew.
    """
    return ExperimentConfig(
        name="heston_volatility_smile",
        model="heston",
        option_type="call",
        style="european",
        S0=100.0,
        K=100.0,  # Will be varied in post-processing
        r=0.05,
        T=1.0,
        kappa=2.0,
        theta=0.04,
        xi=0.3,
        rho=-0.7,
        v0=0.04,
        n_paths_list=[50000],
        n_steps=200,
        seeds=[42],
        antithetic=True,
    )


def run_european_call_benchmark(results_dir: Path) -> None:
    """Run European call variance reduction benchmark."""
    print("\n" + "=" * 80)
    print("BENCHMARK: European Call Variance Reduction")
    print("=" * 80)

    base_config = get_european_call_bench()
    all_results = []

    # 1. Plain Monte Carlo
    print("\n1. Plain Monte Carlo...")
    config1 = ExperimentConfig(**vars(base_config))
    config1.antithetic = False
    config1.control_variate = False
    results1 = run_experiment(config1)
    for r in results1:
        r.notes = "Plain MC"
    all_results.extend(results1)

    # 2. Antithetic variates only
    print("2. Antithetic variates...")
    config2 = ExperimentConfig(**vars(base_config))
    config2.antithetic = True
    config2.control_variate = False
    results2 = run_experiment(config2)
    for r in results2:
        r.notes = "Antithetic"
    all_results.extend(results2)

    # 3. Control variate only
    print("3. Control variate...")
    config3 = ExperimentConfig(**vars(base_config))
    config3.antithetic = False
    config3.control_variate = True
    results3 = run_experiment(config3)
    for r in results3:
        r.notes = "Control Variate"
    all_results.extend(results3)

    # 4. Combined
    print("4. Combined (antithetic + control variate)...")
    config4 = ExperimentConfig(**vars(base_config))
    config4.antithetic = True
    config4.control_variate = True
    results4 = run_experiment(config4)
    for r in results4:
        r.notes = "Combined"
    all_results.extend(results4)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / "european_call_bench" / timestamp
    save_results(all_results, out_dir, "European Call Variance Reduction")


def run_heston_smile_benchmark(results_dir: Path) -> None:
    """Run Heston volatility smile benchmark."""
    print("\n" + "=" * 80)
    print("BENCHMARK: Heston Volatility Smile")
    print("=" * 80)

    base_config = get_heston_smile_bench()
    strikes = [70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0]
    all_results = []

    print(f"\nPricing {len(strikes)} strikes...")
    for i, strike in enumerate(strikes, 1):
        print(f"  {i}/{len(strikes)}: K={strike:.1f}", end=" ", flush=True)

        config = ExperimentConfig(**vars(base_config))
        config.K = strike
        config.name = f"heston_smile_K{int(strike)}"

        results = run_experiment(config)

        # Compute implied volatility for each result
        for r in results:
            try:
                iv = implied_vol(
                    price=r.price,
                    S0=base_config.S0,
                    K=strike,
                    r=base_config.r,
                    T=base_config.T,
                    option_type="call",
                    tol=1e-6,
                )
                r.implied_vol = iv
                r.notes = f"K={strike:.0f}, IV={iv:.4f}"
                print(f"→ IV={iv:.4f}")
            except ValueError as e:
                print(f"→ IV computation failed: {e}")

        all_results.extend(results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = results_dir / "heston_smile_bench" / timestamp
    save_results(all_results, out_dir, "Heston Volatility Smile")

    # Print smile table
    print("\n" + "=" * 80)
    print("VOLATILITY SMILE")
    print("=" * 80)
    print(f"{'Strike':<10} {'Moneyness':<12} {'Price':<12} {'Implied Vol':<12}")
    print("-" * 80)
    for r in all_results:
        if r.implied_vol is not None:
            moneyness = r.metadata.n_paths  # Stored K in metadata temporarily
            actual_k = float(r.notes.split(",")[0].split("=")[1])
            moneyness = actual_k / base_config.S0
            print(f"{actual_k:<10.1f} {moneyness:<12.4f} {r.price:<12.6f} {r.implied_vol:<12.6f}")
    print("=" * 80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run reproducible experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["european_call_bench", "heston_smile_bench", "all"],
        help="Experiment to run",
    )

    parser.add_argument(
        "--results_dir", type=Path, default=Path("results"), help="Directory for results output"
    )

    args = parser.parse_args()

    # Create results directory
    args.results_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 80)
    print("REPRODUCIBLE EXPERIMENT RUNNER")
    print("=" * 80)
    print(f"Results directory: {args.results_dir.absolute()}")

    # Run experiments
    if args.experiment == "european_call_bench" or args.experiment == "all":
        run_european_call_benchmark(args.results_dir)

    if args.experiment == "heston_smile_bench" or args.experiment == "all":
        run_heston_smile_benchmark(args.results_dir)

    print("\n" + "=" * 80)
    print("✓ All experiments complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
