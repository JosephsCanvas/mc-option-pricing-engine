#!/usr/bin/env python
"""
Benchmark comparing Heston variance discretization schemes.

Compares Full Truncation Euler vs Quadratic-Exponential (QE) schemes
across different strikes, maturities, and path counts.

Example usage:
    python scripts/bench_heston_schemes.py
    python scripts/bench_heston_schemes.py --n_paths 50000 --n_seeds 5
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.analytics.implied_vol import implied_vol
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark Heston variance discretization schemes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--n_paths", type=int, default=50000, help="Number of Monte Carlo paths")
    parser.add_argument("--n_steps", type=int, default=200, help="Number of time steps")
    parser.add_argument(
        "--n_seeds", type=int, default=3, help="Number of random seeds for averaging"
    )
    parser.add_argument(
        "--output_dir", type=str, default="results", help="Directory to save benchmark results"
    )

    return parser.parse_args()


def compute_implied_vol_safe(s0, k, r, t, price, option_type="call"):  # noqa: N803
    """
    Compute implied volatility with error handling.

    Returns NaN if computation fails.
    """
    try:
        iv = implied_vol(
            market_price=price,
            S0=s0,
            K=k,
            r=r,
            T=t,
            option_type=option_type,
            initial_guess=0.3,
            tol=1e-6,
            max_iter=100,
        )
        return iv if iv is not None else np.nan
    except Exception:
        return np.nan


def run_benchmark(args):
    """Run the Heston scheme comparison benchmark."""
    print("=" * 80)
    print("Heston Variance Discretization Scheme Benchmark")
    print("=" * 80)
    print("\nParameters:")
    print(f"  n_paths:     {args.n_paths}")
    print(f"  n_steps:     {args.n_steps}")
    print(f"  n_seeds:     {args.n_seeds}")
    print(f"  output_dir:  {args.output_dir}")

    # Define Heston parameters (typical calibrated values)
    S0 = 100.0
    r = 0.05
    T = 1.0
    kappa = 2.0
    theta = 0.04
    xi = 0.3
    rho = -0.7
    v0 = 0.04

    print("\nHeston Model Parameters:")
    print(f"  S0 = {S0}, r = {r}, T = {T}")
    print(f"  κ = {kappa}, θ = {theta}, ξ = {xi}, ρ = {rho}, v₀ = {v0}")

    # Define strikes for volatility smile
    strikes = np.array([70, 80, 90, 100, 110, 120, 130])
    print(f"\nStrikes: {strikes}")

    # Storage for results
    results = {
        "parameters": {
            "S0": S0,
            "r": r,
            "T": T,
            "kappa": kappa,
            "theta": theta,
            "xi": xi,
            "rho": rho,
            "v0": v0,
            "n_paths": args.n_paths,
            "n_steps": args.n_steps,
            "n_seeds": args.n_seeds,
        },
        "strikes": strikes.tolist(),
        "schemes": {},
    }

    schemes = ["full_truncation_euler", "qe"]

    for scheme in schemes:
        print(f"\n{'=' * 80}")
        print(f"Running scheme: {scheme.upper()}")
        print(f"{'=' * 80}")

        scheme_results = {
            "prices": {str(K): [] for K in strikes},
            "stderrs": {str(K): [] for K in strikes},
            "impl_vols": {str(K): [] for K in strikes},
            "runtimes": {str(K): [] for K in strikes},
        }

        for seed_idx in range(args.n_seeds):
            seed = seed_idx * 100  # Use different seeds
            print(f"\n  Seed {seed_idx + 1}/{args.n_seeds} (seed={seed})")

            for K in strikes:
                # Create model
                model = HestonModel(
                    S0=S0,
                    r=r,
                    T=T,
                    kappa=kappa,
                    theta=theta,
                    xi=xi,
                    rho=rho,
                    v0=v0,
                    seed=seed,
                    scheme=scheme,
                )

                # Create payoff and engine
                payoff = EuropeanCallPayoff(K=K)
                engine = HestonMonteCarloEngine(
                    model=model, payoff=payoff, n_paths=args.n_paths, n_steps=args.n_steps
                )

                # Price option and time it
                start_time = time.perf_counter()
                result = engine.price()
                runtime = time.perf_counter() - start_time

                # Compute implied volatility
                impl_vol = compute_implied_vol_safe(S0, K, r, T, result.price)

                # Store results
                scheme_results["prices"][str(K)].append(result.price)
                scheme_results["stderrs"][str(K)].append(result.stderr)
                scheme_results["impl_vols"][str(K)].append(impl_vol)
                scheme_results["runtimes"][str(K)].append(runtime)

                print(
                    f"    K={K:>3.0f}: Price={result.price:8.5f} ± "
                    f"{result.stderr:.5f}, IV={impl_vol:.4f}, "
                    f"Time={runtime:.3f}s"
                )

        # Compute aggregated statistics
        scheme_results["mean_prices"] = {
            str(K): float(np.mean(scheme_results["prices"][str(K)])) for K in strikes
        }
        scheme_results["mean_stderrs"] = {
            str(K): float(np.mean(scheme_results["stderrs"][str(K)])) for K in strikes
        }
        scheme_results["mean_impl_vols"] = {
            str(K): float(np.nanmean(scheme_results["impl_vols"][str(K)])) for K in strikes
        }
        scheme_results["mean_runtimes"] = {
            str(K): float(np.mean(scheme_results["runtimes"][str(K)])) for K in strikes
        }
        scheme_results["total_runtime"] = float(
            np.sum([np.sum(scheme_results["runtimes"][str(K)]) for K in strikes])
        )

        # Compute smile width (max - min IV)
        valid_ivs = [
            iv for K in strikes for iv in scheme_results["impl_vols"][str(K)] if not np.isnan(iv)
        ]
        if valid_ivs:
            scheme_results["smile_width"] = float(np.max(valid_ivs) - np.min(valid_ivs))
        else:
            scheme_results["smile_width"] = np.nan

        results["schemes"][scheme] = scheme_results

    # Print summary comparison
    print(f"\n{'=' * 80}")
    print("SUMMARY COMPARISON")
    print(f"{'=' * 80}")

    print(f"\n{'Strike':<8} {'FT Euler IV':<15} {'QE IV':<15} {'IV Diff':<12} {'Price Diff':<12}")
    print("-" * 80)

    for K in strikes:
        ft_iv = results["schemes"]["full_truncation_euler"]["mean_impl_vols"][str(K)]
        qe_iv = results["schemes"]["qe"]["mean_impl_vols"][str(K)]
        iv_diff = qe_iv - ft_iv

        ft_price = results["schemes"]["full_truncation_euler"]["mean_prices"][str(K)]
        qe_price = results["schemes"]["qe"]["mean_prices"][str(K)]
        price_diff = qe_price - ft_price

        print(f"{K:<8.0f} {ft_iv:<15.6f} {qe_iv:<15.6f} {iv_diff:<12.6f} {price_diff:<12.6f}")

    print("\nSmile Width:")
    print(f"  FT Euler: {results['schemes']['full_truncation_euler']['smile_width']:.6f}")
    print(f"  QE:       {results['schemes']['qe']['smile_width']:.6f}")

    print("\nTotal Runtime:")
    print(f"  FT Euler: {results['schemes']['full_truncation_euler']['total_runtime']:.3f}s")
    print(f"  QE:       {results['schemes']['qe']['total_runtime']:.3f}s")

    # Save results to JSON
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "heston_scheme_benchmark.json"

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    # Also save a summary table
    summary_file = output_dir / "heston_scheme_summary.txt"
    with open(summary_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Heston Variance Discretization Scheme Benchmark\n")
        f.write("=" * 80 + "\n\n")

        f.write("Parameters:\n")
        f.write(f"  n_paths:     {args.n_paths}\n")
        f.write(f"  n_steps:     {args.n_steps}\n")
        f.write(f"  n_seeds:     {args.n_seeds}\n\n")

        f.write("Heston Model:\n")
        f.write(f"  S0={S0}, r={r}, T={T}\n")
        f.write(f"  κ={kappa}, θ={theta}, ξ={xi}, ρ={rho}, v₀={v0}\n\n")

        f.write(f"{'Strike':<8} {'FT Euler IV':<15} {'QE IV':<15} ")
        f.write(f"{'IV Diff':<12} {'FT Price':<12} {'QE Price':<12}\n")
        f.write("-" * 90 + "\n")

        for K in strikes:
            ft_iv = results["schemes"]["full_truncation_euler"]["mean_impl_vols"][str(K)]
            qe_iv = results["schemes"]["qe"]["mean_impl_vols"][str(K)]
            iv_diff = qe_iv - ft_iv
            ft_price = results["schemes"]["full_truncation_euler"]["mean_prices"][str(K)]
            qe_price = results["schemes"]["qe"]["mean_prices"][str(K)]

            f.write(
                f"{K:<8.0f} {ft_iv:<15.6f} {qe_iv:<15.6f} "
                f"{iv_diff:<12.6f} {ft_price:<12.6f} {qe_price:<12.6f}\n"
            )

        f.write("\nSmile Width:\n")
        f.write(f"  FT Euler: {results['schemes']['full_truncation_euler']['smile_width']:.6f}\n")
        f.write(f"  QE:       {results['schemes']['qe']['smile_width']:.6f}\n")

        f.write("\nTotal Runtime:\n")
        f.write(
            f"  FT Euler: {results['schemes']['full_truncation_euler']['total_runtime']:.3f}s\n"
        )
        f.write(f"  QE:       {results['schemes']['qe']['total_runtime']:.3f}s\n")

    print(f"✓ Summary saved to {summary_file}")


def main():
    """Main entry point."""
    args = parse_args()
    run_benchmark(args)


if __name__ == "__main__":
    main()
