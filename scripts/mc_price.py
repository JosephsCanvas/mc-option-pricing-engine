#!/usr/bin/env python
"""
Command-line interface for Monte Carlo option pricing.

Example usage:
    python scripts/mc_price.py --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 \\
        --n_paths 100000
    python scripts/mc_price.py --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 \\
        --n_paths 200000 --antithetic --option_type put
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.analytics.black_scholes import bs_delta, bs_gamma, bs_price, bs_vega
from mc_pricer.analytics.implied_vol import implied_vol
from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine
from mc_pricer.pricers.lsm import price_american_lsm
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo option pricing engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Market parameters
    parser.add_argument("--S0", type=float, required=True, help="Initial spot price")
    parser.add_argument("--K", type=float, required=True, help="Strike price")
    parser.add_argument("--r", type=float, required=True, help="Risk-free rate")
    parser.add_argument("--T", type=float, required=True, help="Time to maturity (years)")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        choices=["gbm", "heston"],
        default="gbm",
        help="Pricing model: gbm (Black-Scholes) or heston (stochastic volatility)"
    )

    # GBM parameters
    parser.add_argument("--sigma", type=float, help="Volatility (required for GBM)")

    # Heston parameters
    parser.add_argument("--kappa", type=float, help="Mean reversion speed (required for Heston)")
    parser.add_argument("--theta", type=float, help="Long-term variance (required for Heston)")
    parser.add_argument("--xi", type=float, help="Volatility of volatility (required for Heston)")
    parser.add_argument("--rho", type=float, help="Correlation (required for Heston)")
    parser.add_argument("--v0", type=float, help="Initial variance (required for Heston)")
    parser.add_argument("--n_steps", type=int, help="Number of time steps (required for Heston)")
    parser.add_argument(
        "--heston_scheme",
        type=str,
        choices=["full_truncation_euler", "qe"],
        default="full_truncation_euler",
        help="Heston variance discretization scheme"
    )

    # Simulation parameters
    parser.add_argument(
        "--n_paths",
        type=int,
        default=100000,
        help="Number of Monte Carlo paths"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--antithetic",
        action="store_true",
        help="Use antithetic variates for variance reduction"
    )
    parser.add_argument(
        "--control_variate",
        action="store_true",
        help="Use control variate variance reduction"
    )
    parser.add_argument(
        "--rng",
        type=str,
        choices=["pseudo", "sobol"],
        default="pseudo",
        help="RNG type: pseudo (pseudo-random) or sobol (Quasi-Monte Carlo)"
    )
    parser.add_argument(
        "--scramble",
        action="store_true",
        help="Use digital shift scrambling for Sobol sequences (QMC only)"
    )
    parser.add_argument(
        "--qmc_dim_override",
        type=int,
        default=None,
        help="Override dimension for QMC (advanced users only)"
    )

    # Option parameters
    parser.add_argument(
        "--option_type",
        type=str,
        choices=["call", "put"],
        default="call",
        help="Option type: call or put"
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["european", "american"],
        default="european",
        help="Option style: european or american"
    )

    # American option parameters
    parser.add_argument(
        "--lsm_steps",
        type=int,
        default=50,
        help="Number of time steps for LSM (American options only)"
    )
    parser.add_argument(
        "--lsm_basis",
        type=str,
        choices=["poly2", "poly3"],
        default="poly2",
        help="Basis functions for LSM: poly2 or poly3 (American options only)"
    )

    # Greeks parameters
    parser.add_argument(
        "--greeks",
        type=str,
        choices=["none", "delta", "vega", "all"],
        default="none",
        help="Greeks to compute: none, delta, vega, or all"
    )
    parser.add_argument(
        "--greeks_method",
        type=str,
        choices=["pw", "fd", "both"],
        default="pw",
        help="Method for Greeks computation: pw (pathwise), fd (finite difference), or both"
    )

    # Analytics parameters
    parser.add_argument(
        "--bs",
        action="store_true",
        help="Display Black-Scholes reference price and Greeks (European only)"
    )
    parser.add_argument(
        "--implied_vol",
        type=float,
        default=None,
        help="Compute implied volatility from given market price (European only)"
    )

    return parser.parse_args()


def main():
    """Main entry point for CLI."""
    args = parse_args()

    # Validate model-specific parameters
    if args.model == "gbm" and args.sigma is None:
        print("Error: --sigma is required for GBM model")
        sys.exit(1)

    if args.model == "heston":
        missing = []
        if args.kappa is None:
            missing.append("--kappa")
        if args.theta is None:
            missing.append("--theta")
        if args.xi is None:
            missing.append("--xi")
        if args.rho is None:
            missing.append("--rho")
        if args.v0 is None:
            missing.append("--v0")
        if args.n_steps is None:
            missing.append("--n_steps")
        if missing:
            print(f"Error: Heston model requires: {', '.join(missing)}")
            sys.exit(1)

    # Print input parameters
    print("=" * 70)
    print("Monte Carlo Option Pricing Engine")
    print("=" * 70)
    print("\nInput Parameters:")
    print(f"  Spot Price (S0):        {args.S0:,.2f}")
    print(f"  Strike Price (K):       {args.K:,.2f}")
    print(f"  Risk-free Rate (r):     {args.r:.4f}")
    if args.model == "gbm":
        print(f"  Volatility (σ):         {args.sigma:.4f}")
    else:
        print(f"  Kappa (κ):              {args.kappa:.4f}")
        print(f"  Theta (θ):              {args.theta:.4f}")
        print(f"  Xi (ξ):                 {args.xi:.4f}")
        print(f"  Rho (ρ):                {args.rho:.4f}")
        print(f"  Initial Variance (v0):  {args.v0:.4f}")
        print(f"  Scheme:                 {args.heston_scheme}")
    print(f"  Time to Maturity (T):   {args.T:.4f} years")
    print(f"  Option Type:            {args.option_type.upper()}")
    print(f"  Option Style:           {args.style.upper()}")
    print(f"  Model:                  {args.model.upper()}")
    print("\nSimulation Parameters:")
    print(f"  Number of Paths:        {args.n_paths:,}")
    if args.model == "heston":
        print(f"  Time Steps:             {args.n_steps}")
    if args.style == "american":
        print(f"  LSM Time Steps:         {args.lsm_steps}")
        print(f"  LSM Basis Functions:    {args.lsm_basis}")
    print(f"  Antithetic Variates:    {args.antithetic}")
    if args.style == "european" and args.model == "gbm":
        print(f"  Control Variate:        {args.control_variate}")
    print(f"  Random Seed:            {args.seed if args.seed is not None else 'None (random)'}")
    print(f"  RNG Type:               {args.rng.upper()}")
    if args.rng == "sobol":
        print(f"  Scrambling:             {args.scramble}")
        if args.qmc_dim_override:
            print(f"  QMC Dimension Override: {args.qmc_dim_override}")

    # Check for incompatible features with Heston
    if args.model == "heston":
        warnings = []
        if args.style == "american":
            warnings.append("American options not supported with Heston model")
        if args.control_variate:
            warnings.append("Control variate not supported with Heston model")
        if args.greeks != "none":
            warnings.append("Greeks not supported with Heston model")
        if warnings:
            print("\n⚠ Warnings:")
            for warning in warnings:
                print(f"  • {warning}")
            if args.style == "american":
                print("\nPlease use GBM model for American options.")
                sys.exit(1)

    # Create model
    if args.model == "gbm":
        model = GeometricBrownianMotion(
            S0=args.S0,
            r=args.r,
            sigma=args.sigma,
            T=args.T,
            seed=args.seed,
            rng_type=args.rng,
            scramble=args.scramble
        )
    else:  # heston
        model = HestonModel(
            S0=args.S0,
            r=args.r,
            T=args.T,
            kappa=args.kappa,
            theta=args.theta,
            xi=args.xi,
            rho=args.rho,
            v0=args.v0,
            seed=args.seed,
            scheme=args.heston_scheme,
            rng_type=args.rng,
            scramble=args.scramble
        )

    # Price the option
    print("\n" + "=" * 70)
    print("Pricing...")
    print("=" * 70)

    if args.style == "american":
        # American option pricing with LSM
        if args.control_variate or args.greeks != "none":
            print("\nNote: Control variate and Greeks not supported for American options (LSM)")

        if args.bs or args.implied_vol is not None:
            print("Note: Black-Scholes and implied volatility are European-only features")

        result = price_american_lsm(
            model=model,
            strike=args.K,
            option_type=args.option_type,
            n_paths=args.n_paths,
            n_steps=args.lsm_steps,
            basis=args.lsm_basis,
            seed=args.seed
        )

        # Print results
        print("\nResults (LSM American):")
        print(f"  Option Price:           {result.price:.6f}")
        print(f"  Standard Error:         {result.stderr:.6f}")
        print(f"  95% Confidence Interval: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
        print(f"  CI Width:               {result.ci_upper - result.ci_lower:.6f}")
        print(f"  Relative Error (σ/μ):   {result.stderr / result.price * 100:.4f}%")

        print("\n" + "=" * 70)
        return

    # European option pricing
    # Create payoff
    if args.option_type == "call":
        payoff = EuropeanCallPayoff(strike=args.K)
    else:
        payoff = EuropeanPutPayoff(strike=args.K)

    # Create pricing engine
    if args.model == "gbm":
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=args.n_paths,
            antithetic=args.antithetic,
            control_variate=args.control_variate
        )
    else:  # heston
        engine = HestonMonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=args.n_paths,
            n_steps=args.n_steps,
            antithetic=args.antithetic,
            seed=args.seed
        )

    # Price the option
    print("\n" + "=" * 70)
    print("Pricing...")
    print("=" * 70)
    result = engine.price()

    # Print results
    print("\nResults:")
    print(f"  Option Price:           {result.price:.6f}")
    print(f"  Standard Error:         {result.stderr:.6f}")
    print(f"  95% Confidence Interval: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
    print(f"  CI Width:               {result.ci_upper - result.ci_lower:.6f}")
    print(f"  Relative Error (σ/μ):   {result.stderr / result.price * 100:.4f}%")
    if hasattr(result, 'control_variate_beta') and result.control_variate_beta is not None:
        print(f"  Control Variate Beta:   {result.control_variate_beta:.6f}")
    if hasattr(result, 'n_steps'):
        print(f"  Time Steps Used:        {result.n_steps}")

    # Compute Greeks if requested (GBM only)
    if args.greeks != "none" and args.model == "gbm":
        print("\n" + "=" * 70)
        print("Computing Greeks...")
        print("=" * 70)

        if args.greeks_method == "both":
            # Compute both PW and FD
            greeks_pw = engine.compute_greeks(
                option_type=args.option_type,
                method='pw'
            )
            greeks_fd = engine.compute_greeks(
                option_type=args.option_type,
                method='fd'
            )

            print("\nGreeks (Pathwise):")
            if args.greeks in ["delta", "all"] and greeks_pw.delta is not None:
                print(
                    f"  Delta:  {greeks_pw.delta.value:.6f} "
                    f"± {greeks_pw.delta.standard_error:.6f}"
                )
                print(
                    f"    95% CI: [{greeks_pw.delta.ci_lower:.6f}, "
                    f"{greeks_pw.delta.ci_upper:.6f}]"
                )
            if args.greeks in ["vega", "all"] and greeks_pw.vega is not None:
                print(f"  Vega:   {greeks_pw.vega.value:.4f} ± {greeks_pw.vega.standard_error:.4f}")
                print(f"    95% CI: [{greeks_pw.vega.ci_lower:.4f}, {greeks_pw.vega.ci_upper:.4f}]")

            print("\nGreeks (Finite Difference):")
            if args.greeks in ["delta", "all"] and greeks_fd.delta is not None:
                print(
                    f"  Delta:  {greeks_fd.delta.value:.6f} "
                    f"± {greeks_fd.delta.standard_error:.6f}"
                )
                print(
                    f"    95% CI: [{greeks_fd.delta.ci_lower:.6f}, "
                    f"{greeks_fd.delta.ci_upper:.6f}]"
                )
            if args.greeks in ["vega", "all"] and greeks_fd.vega is not None:
                print(f"  Vega:   {greeks_fd.vega.value:.4f} ± {greeks_fd.vega.standard_error:.4f}")
                print(f"    95% CI: [{greeks_fd.vega.ci_lower:.4f}, {greeks_fd.vega.ci_upper:.4f}]")

        else:
            # Compute single method
            greeks = engine.compute_greeks(
                option_type=args.option_type,
                method=args.greeks_method
            )

            method_name = "Pathwise" if args.greeks_method == "pw" else "Finite Difference"
            print(f"\nGreeks ({method_name}):")
            if args.greeks in ["delta", "all"] and greeks.delta is not None:
                print(f"  Delta:  {greeks.delta.value:.6f} ± {greeks.delta.standard_error:.6f}")
                print(f"    95% CI: [{greeks.delta.ci_lower:.6f}, {greeks.delta.ci_upper:.6f}]")
            if args.greeks in ["vega", "all"] and greeks.vega is not None:
                print(f"  Vega:   {greeks.vega.value:.4f} ± {greeks.vega.standard_error:.4f}")
                print(f"    95% CI: [{greeks.vega.ci_lower:.4f}, {greeks.vega.ci_upper:.4f}]")

    # Black-Scholes analytics (European GBM only)
    if args.bs and args.model == "gbm":
        print("\n" + "=" * 70)
        print("Black-Scholes Analytics")
        print("=" * 70)

        bs_price_val = bs_price(args.S0, args.K, args.r, args.T, args.sigma, args.option_type)
        bs_delta_val = bs_delta(args.S0, args.K, args.r, args.T, args.sigma, args.option_type)
        bs_gamma_val = bs_gamma(args.S0, args.K, args.r, args.T, args.sigma)
        bs_vega_val = bs_vega(args.S0, args.K, args.r, args.T, args.sigma)

        print("\nBlack-Scholes Reference:")
        print(f"  Price:  {bs_price_val:.6f}")
        print(f"  Delta:  {bs_delta_val:.6f}")
        print(f"  Gamma:  {bs_gamma_val:.6f}")
        print(f"  Vega:   {bs_vega_val:.4f}")

        mc_error = result.price - bs_price_val
        print("\nMonte Carlo vs Black-Scholes:")
        print(f"  MC Price:    {result.price:.6f}")
        print(f"  BS Price:    {bs_price_val:.6f}")
        print(f"  Difference:  {mc_error:.6f}")
        print(f"  MC Stderr:   {result.stderr:.6f}")
        if result.stderr > 0:
            print(f"  Std Errors:  {abs(mc_error) / result.stderr:.2f}σ")

    # Implied volatility (European GBM only)
    if args.implied_vol is not None and args.model == "gbm":
        print("\n" + "=" * 70)
        print("Implied Volatility")
        print("=" * 70)

        try:
            iv = implied_vol(
                price=args.implied_vol,
                S0=args.S0,
                K=args.K,
                r=args.r,
                T=args.T,
                option_type=args.option_type
            )
            print(f"\nMarket Price:      {args.implied_vol:.6f}")
            print(f"Implied Vol:       {iv:.6f}")
            print(f"Input Vol:         {args.sigma:.6f}")
            print(f"Vol Difference:    {iv - args.sigma:.6f}")

            # Verify by computing BS price with implied vol
            verify_price = bs_price(args.S0, args.K, args.r, args.T, iv, args.option_type)
            print("\nVerification:")
            print(f"  BS(IV) Price:    {verify_price:.6f}")
            print(f"  Target Price:    {args.implied_vol:.6f}")
            print(f"  Price Error:     {abs(verify_price - args.implied_vol):.2e}")

        except ValueError as e:
            print(f"\nError computing implied volatility: {e}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
