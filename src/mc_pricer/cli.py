#!/usr/bin/env python
"""
Command-line interface for Monte Carlo option pricing.

This module provides the main CLI entrypoint for the mc-price command.

Example usage:
    mc-price --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 --n_paths 100000
    mc-price --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 --option_type put --antithetic
"""

import argparse
import sys

from mc_pricer.analytics.black_scholes import bs_delta, bs_gamma, bs_price, bs_vega
from mc_pricer.analytics.implied_vol import implied_vol
from mc_pricer.models.gbm import GeometricBrownianMotion
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.path_dependent import (
    AsianArithmeticCallPayoff,
    AsianArithmeticPutPayoff,
    DownAndOutPutPayoff,
    UpAndOutCallPayoff,
)
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine
from mc_pricer.pricers.lsm import price_american_lsm
from mc_pricer.pricers.monte_carlo import MonteCarloEngine


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    args : list[str] | None
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Monte Carlo option pricing engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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
        help="Pricing model: gbm (Black-Scholes) or heston (stochastic volatility)",
    )

    # GBM parameters
    parser.add_argument("--sigma", type=float, help="Volatility (required for GBM)")

    # Heston parameters
    parser.add_argument("--kappa", type=float, help="Mean reversion speed (required for Heston)")
    parser.add_argument("--theta", type=float, help="Long-term variance (required for Heston)")
    parser.add_argument("--xi", type=float, help="Volatility of volatility (required for Heston)")
    parser.add_argument("--rho", type=float, help="Correlation (required for Heston)")
    parser.add_argument("--v0", type=float, help="Initial variance (required for Heston)")
    parser.add_argument(
        "--n_steps",
        type=int,
        help="Number of time steps (required for Heston and path-dependent options)",
    )
    parser.add_argument(
        "--heston_scheme",
        type=str,
        choices=["full_truncation_euler", "qe"],
        default="full_truncation_euler",
        help="Heston variance discretization scheme",
    )

    # Simulation parameters
    parser.add_argument(
        "--n_paths",
        type=int,
        default=100000,
        help="Number of Monte Carlo paths",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--antithetic",
        action="store_true",
        help="Use antithetic variates for variance reduction",
    )
    parser.add_argument(
        "--control_variate",
        action="store_true",
        help="Use control variate variance reduction",
    )
    parser.add_argument(
        "--rng",
        type=str,
        choices=["pseudo", "sobol"],
        default="pseudo",
        help="RNG type: pseudo (pseudo-random) or sobol (Quasi-Monte Carlo)",
    )
    parser.add_argument(
        "--scramble",
        action="store_true",
        help="Use digital shift scrambling for Sobol sequences (QMC only)",
    )
    parser.add_argument(
        "--qmc_dim_override",
        type=int,
        default=None,
        help="Override dimension for QMC (advanced users only)",
    )

    # Option parameters
    parser.add_argument(
        "--product",
        type=str,
        choices=["european", "asian", "barrier"],
        default="european",
        help="Product type: european, asian, or barrier",
    )
    parser.add_argument(
        "--option_type",
        type=str,
        choices=["call", "put"],
        default="call",
        help="Option type: call or put",
    )
    parser.add_argument(
        "--style",
        type=str,
        choices=["european", "american"],
        default="european",
        help="Option style: european or american (applies to vanilla options)",
    )

    # Barrier option parameters
    parser.add_argument(
        "--barrier",
        type=float,
        default=None,
        help="Barrier level (required for barrier options)",
    )
    parser.add_argument(
        "--barrier_type",
        type=str,
        choices=["up_out_call", "down_out_put"],
        default="up_out_call",
        help="Barrier type: up_out_call or down_out_put",
    )

    # American option parameters
    parser.add_argument(
        "--lsm_steps",
        type=int,
        default=50,
        help="Number of time steps for LSM (American options only)",
    )
    parser.add_argument(
        "--lsm_basis",
        type=str,
        choices=["poly2", "poly3"],
        default="poly2",
        help="Basis functions for LSM: poly2 or poly3 (American options only)",
    )

    # Greeks parameters
    parser.add_argument(
        "--greeks",
        type=str,
        choices=["none", "delta", "vega", "all"],
        default="none",
        help="Greeks to compute: none, delta, vega, or all",
    )
    parser.add_argument(
        "--greeks_method",
        type=str,
        choices=["pw", "fd", "both"],
        default="pw",
        help="Method for Greeks computation: pw (pathwise), fd (finite difference), or both",
    )

    # Analytics parameters
    parser.add_argument(
        "--bs",
        action="store_true",
        help="Display Black-Scholes reference price and Greeks (European only)",
    )
    parser.add_argument(
        "--implied_vol",
        type=float,
        default=None,
        help="Compute implied volatility from given market price (European only)",
    )

    return parser.parse_args(args)


def main(args: list[str] | None = None) -> int:
    """Main entry point for CLI.

    Parameters
    ----------
    args : list[str] | None
        Command-line arguments. If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for errors).
    """
    parsed = parse_args(args)

    # Validate model-specific parameters
    if parsed.model == "gbm" and parsed.sigma is None:
        print("Error: --sigma is required for GBM model")
        return 1

    if parsed.model == "heston":
        missing = []
        if parsed.kappa is None:
            missing.append("--kappa")
        if parsed.theta is None:
            missing.append("--theta")
        if parsed.xi is None:
            missing.append("--xi")
        if parsed.rho is None:
            missing.append("--rho")
        if parsed.v0 is None:
            missing.append("--v0")
        if parsed.n_steps is None:
            missing.append("--n_steps")
        if missing:
            print(f"Error: Heston model requires: {', '.join(missing)}")
            return 1

    # Validate path-dependent option parameters
    if parsed.product in ["asian", "barrier"]:
        if parsed.n_steps is None:
            print(f"Error: --n_steps is required for {parsed.product} options")
            return 1
        if parsed.style == "american":
            print(f"Error: American style not supported for {parsed.product} options")
            return 1

    if parsed.product == "barrier":
        if parsed.barrier is None:
            print("Error: --barrier is required for barrier options")
            return 1

    # Print input parameters
    print("=" * 70)
    print("Monte Carlo Option Pricing Engine")
    print("=" * 70)
    print("\nInput Parameters:")
    print(f"  Spot Price (S0):        {parsed.S0:,.2f}")
    print(f"  Strike Price (K):       {parsed.K:,.2f}")
    print(f"  Risk-free Rate (r):     {parsed.r:.4f}")
    if parsed.model == "gbm":
        print(f"  Volatility (σ):         {parsed.sigma:.4f}")
    else:
        print(f"  Kappa (κ):              {parsed.kappa:.4f}")
        print(f"  Theta (θ):              {parsed.theta:.4f}")
        print(f"  Xi (ξ):                 {parsed.xi:.4f}")
        print(f"  Rho (ρ):                {parsed.rho:.4f}")
        print(f"  Initial Variance (v0):  {parsed.v0:.4f}")
        print(f"  Scheme:                 {parsed.heston_scheme}")
    print(f"  Time to Maturity (T):   {parsed.T:.4f} years")
    print(f"  Product Type:           {parsed.product.upper()}")
    print(f"  Option Type:            {parsed.option_type.upper()}")
    if parsed.product == "european":
        print(f"  Option Style:           {parsed.style.upper()}")
    if parsed.product == "barrier":
        print(f"  Barrier Level:          {parsed.barrier:,.2f}")
        print(f"  Barrier Type:           {parsed.barrier_type}")
    print(f"  Model:                  {parsed.model.upper()}")
    print("\nSimulation Parameters:")
    print(f"  Number of Paths:        {parsed.n_paths:,}")
    if parsed.product in ["asian", "barrier"] or parsed.model == "heston":
        print(f"  Time Steps:             {parsed.n_steps}")
    if parsed.style == "american":
        print(f"  LSM Time Steps:         {parsed.lsm_steps}")
        print(f"  LSM Basis Functions:    {parsed.lsm_basis}")
    print(f"  Antithetic Variates:    {parsed.antithetic}")
    if parsed.product == "european" and parsed.style == "european" and parsed.model == "gbm":
        print(f"  Control Variate:        {parsed.control_variate}")
    seed_str = parsed.seed if parsed.seed is not None else "None (random)"
    print(f"  Random Seed:            {seed_str}")
    print(f"  RNG Type:               {parsed.rng.upper()}")
    if parsed.rng == "sobol":
        print(f"  Scrambling:             {parsed.scramble}")
        if parsed.qmc_dim_override:
            print(f"  QMC Dimension Override: {parsed.qmc_dim_override}")

    # Check for incompatible features with Heston
    if parsed.model == "heston":
        warnings = []
        if parsed.style == "american":
            warnings.append("American options not supported with Heston model")
        if parsed.control_variate:
            warnings.append("Control variate not supported with Heston model")
        if parsed.greeks != "none":
            warnings.append("Greeks not supported with Heston model")
        if warnings:
            print("\n⚠ Warnings:")
            for warning in warnings:
                print(f"  • {warning}")
            if parsed.style == "american":
                print("\nPlease use GBM model for American options.")
                return 1

    # Check for incompatible features with path-dependent options
    if parsed.product in ["asian", "barrier"]:
        warnings = []
        if parsed.control_variate:
            warnings.append(f"Control variate not supported for {parsed.product} options")
        if parsed.greeks != "none":
            warnings.append(f"Greeks not supported for {parsed.product} options")
        if warnings:
            print("\n⚠ Warnings:")
            for warning in warnings:
                print(f"  • {warning}")

    # Create model
    if parsed.model == "gbm":
        model = GeometricBrownianMotion(
            S0=parsed.S0,
            r=parsed.r,
            sigma=parsed.sigma,
            T=parsed.T,
            seed=parsed.seed,
            rng_type=parsed.rng,
            scramble=parsed.scramble,
        )
    else:  # heston
        model = HestonModel(
            S0=parsed.S0,
            r=parsed.r,
            T=parsed.T,
            kappa=parsed.kappa,
            theta=parsed.theta,
            xi=parsed.xi,
            rho=parsed.rho,
            v0=parsed.v0,
            seed=parsed.seed,
            scheme=parsed.heston_scheme,
            rng_type=parsed.rng,
            scramble=parsed.scramble,
        )

    # Price the option
    print("\n" + "=" * 70)
    print("Pricing...")
    print("=" * 70)

    # Path-dependent option pricing (Asian or Barrier)
    if parsed.product in ["asian", "barrier"]:
        if parsed.bs or parsed.implied_vol is not None:
            print("Note: Black-Scholes and implied volatility are European-only features")

        # Create payoff
        if parsed.product == "asian":
            if parsed.option_type == "call":
                payoff = AsianArithmeticCallPayoff(strike=parsed.K)
            else:
                payoff = AsianArithmeticPutPayoff(strike=parsed.K)
        else:  # barrier
            if parsed.barrier_type == "up_out_call":
                payoff = UpAndOutCallPayoff(strike=parsed.K, barrier=parsed.barrier)
            else:  # down_out_put
                payoff = DownAndOutPutPayoff(strike=parsed.K, barrier=parsed.barrier)

        # Create pricing engine
        if parsed.model == "gbm":
            engine = MonteCarloEngine(
                model=model,
                payoff=payoff,
                n_paths=parsed.n_paths,
                antithetic=parsed.antithetic,
                control_variate=False,  # Never use CV for path-dependent
            )
        else:  # heston
            engine = HestonMonteCarloEngine(
                model=model,
                payoff=payoff,
                n_paths=parsed.n_paths,
                n_steps=parsed.n_steps,
                antithetic=parsed.antithetic,
                seed=parsed.seed,
            )

        # Price path-dependent option
        result = engine.price_path_dependent(
            n_steps=parsed.n_steps,
            rng_type=parsed.rng,
            scramble=parsed.scramble,
            qmc_dim_override=parsed.qmc_dim_override,
        )

        # Print results
        print(f"\nResults ({parsed.product.capitalize()}):")
        print(f"  Option Price:           {result.price:.6f}")
        print(f"  Standard Error:         {result.stderr:.6f}")
        print(f"  95% Confidence Interval: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
        print(f"  CI Width:               {result.ci_upper - result.ci_lower:.6f}")
        print(f"  Relative Error (σ/μ):   {result.stderr / result.price * 100:.4f}%")
        print(f"  Time Steps Used:        {result.n_steps}")
        print(f"  RNG Type:               {result.rng_type.upper()}")
        if result.rng_type == "sobol":
            print(f"  Scrambling:             {result.scramble}")

        print("\n" + "=" * 70)
        return 0

    if parsed.style == "american":
        # American option pricing with LSM
        if parsed.control_variate or parsed.greeks != "none":
            print("\nNote: Control variate and Greeks not supported for American options (LSM)")

        if parsed.bs or parsed.implied_vol is not None:
            print("Note: Black-Scholes and implied volatility are European-only features")

        result = price_american_lsm(
            model=model,
            strike=parsed.K,
            option_type=parsed.option_type,
            n_paths=parsed.n_paths,
            n_steps=parsed.lsm_steps,
            basis=parsed.lsm_basis,
            seed=parsed.seed,
        )

        # Print results
        print("\nResults (LSM American):")
        print(f"  Option Price:           {result.price:.6f}")
        print(f"  Standard Error:         {result.stderr:.6f}")
        print(f"  95% Confidence Interval: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
        print(f"  CI Width:               {result.ci_upper - result.ci_lower:.6f}")
        print(f"  Relative Error (σ/μ):   {result.stderr / result.price * 100:.4f}%")

        print("\n" + "=" * 70)
        return 0

    # European option pricing
    # Create payoff
    if parsed.option_type == "call":
        payoff = EuropeanCallPayoff(strike=parsed.K)
    else:
        payoff = EuropeanPutPayoff(strike=parsed.K)

    # Create pricing engine
    if parsed.model == "gbm":
        engine = MonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=parsed.n_paths,
            antithetic=parsed.antithetic,
            control_variate=parsed.control_variate,
        )
    else:  # heston
        engine = HestonMonteCarloEngine(
            model=model,
            payoff=payoff,
            n_paths=parsed.n_paths,
            n_steps=parsed.n_steps,
            antithetic=parsed.antithetic,
            seed=parsed.seed,
        )

    # Price the option
    result = engine.price()

    # Print results
    print("\nResults:")
    print(f"  Option Price:           {result.price:.6f}")
    print(f"  Standard Error:         {result.stderr:.6f}")
    print(f"  95% Confidence Interval: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
    print(f"  CI Width:               {result.ci_upper - result.ci_lower:.6f}")
    print(f"  Relative Error (σ/μ):   {result.stderr / result.price * 100:.4f}%")
    if hasattr(result, "control_variate_beta") and result.control_variate_beta is not None:
        print(f"  Control Variate Beta:   {result.control_variate_beta:.6f}")
    if hasattr(result, "n_steps"):
        print(f"  Time Steps Used:        {result.n_steps}")

    # Compute Greeks if requested (GBM only)
    if parsed.greeks != "none" and parsed.model == "gbm":
        print("\n" + "=" * 70)
        print("Computing Greeks...")
        print("=" * 70)

        if parsed.greeks_method == "both":
            # Compute both PW and FD
            greeks_pw = engine.compute_greeks(option_type=parsed.option_type, method="pw")
            greeks_fd = engine.compute_greeks(option_type=parsed.option_type, method="fd")

            print("\nGreeks (Pathwise):")
            if parsed.greeks in ["delta", "all"] and greeks_pw.delta is not None:
                print(
                    f"  Delta:  {greeks_pw.delta.value:.6f} ± {greeks_pw.delta.standard_error:.6f}"
                )
                print(
                    f"    95% CI: [{greeks_pw.delta.ci_lower:.6f}, {greeks_pw.delta.ci_upper:.6f}]"
                )
            if parsed.greeks in ["vega", "all"] and greeks_pw.vega is not None:
                print(f"  Vega:   {greeks_pw.vega.value:.4f} ± {greeks_pw.vega.standard_error:.4f}")
                print(f"    95% CI: [{greeks_pw.vega.ci_lower:.4f}, {greeks_pw.vega.ci_upper:.4f}]")

            print("\nGreeks (Finite Difference):")
            if parsed.greeks in ["delta", "all"] and greeks_fd.delta is not None:
                print(
                    f"  Delta:  {greeks_fd.delta.value:.6f} ± {greeks_fd.delta.standard_error:.6f}"
                )
                print(
                    f"    95% CI: [{greeks_fd.delta.ci_lower:.6f}, {greeks_fd.delta.ci_upper:.6f}]"
                )
            if parsed.greeks in ["vega", "all"] and greeks_fd.vega is not None:
                print(f"  Vega:   {greeks_fd.vega.value:.4f} ± {greeks_fd.vega.standard_error:.4f}")
                print(f"    95% CI: [{greeks_fd.vega.ci_lower:.4f}, {greeks_fd.vega.ci_upper:.4f}]")

        else:
            # Compute single method
            greeks = engine.compute_greeks(
                option_type=parsed.option_type, method=parsed.greeks_method
            )

            method_name = "Pathwise" if parsed.greeks_method == "pw" else "Finite Difference"
            print(f"\nGreeks ({method_name}):")
            if parsed.greeks in ["delta", "all"] and greeks.delta is not None:
                print(f"  Delta:  {greeks.delta.value:.6f} ± {greeks.delta.standard_error:.6f}")
                print(f"    95% CI: [{greeks.delta.ci_lower:.6f}, {greeks.delta.ci_upper:.6f}]")
            if parsed.greeks in ["vega", "all"] and greeks.vega is not None:
                print(f"  Vega:   {greeks.vega.value:.4f} ± {greeks.vega.standard_error:.4f}")
                print(f"    95% CI: [{greeks.vega.ci_lower:.4f}, {greeks.vega.ci_upper:.4f}]")

    # Black-Scholes analytics (European GBM only)
    if parsed.bs and parsed.model == "gbm":
        print("\n" + "=" * 70)
        print("Black-Scholes Analytics")
        print("=" * 70)

        bs_price_val = bs_price(
            parsed.S0, parsed.K, parsed.r, parsed.T, parsed.sigma, parsed.option_type
        )
        bs_delta_val = bs_delta(
            parsed.S0, parsed.K, parsed.r, parsed.T, parsed.sigma, parsed.option_type
        )
        bs_gamma_val = bs_gamma(parsed.S0, parsed.K, parsed.r, parsed.T, parsed.sigma)
        bs_vega_val = bs_vega(parsed.S0, parsed.K, parsed.r, parsed.T, parsed.sigma)

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
    if parsed.implied_vol is not None and parsed.model == "gbm":
        print("\n" + "=" * 70)
        print("Implied Volatility")
        print("=" * 70)

        try:
            iv = implied_vol(
                price=parsed.implied_vol,
                S0=parsed.S0,
                K=parsed.K,
                r=parsed.r,
                T=parsed.T,
                option_type=parsed.option_type,
            )
            print(f"\nMarket Price:      {parsed.implied_vol:.6f}")
            print(f"Implied Vol:       {iv:.6f}")
            print(f"Input Vol:         {parsed.sigma:.6f}")
            print(f"Vol Difference:    {iv - parsed.sigma:.6f}")

            # Verify by computing BS price with implied vol
            verify_price = bs_price(parsed.S0, parsed.K, parsed.r, parsed.T, iv, parsed.option_type)
            print("\nVerification:")
            print(f"  BS(IV) Price:    {verify_price:.6f}")
            print(f"  Target Price:    {parsed.implied_vol:.6f}")
            print(f"  Price Error:     {abs(verify_price - parsed.implied_vol):.2e}")

        except ValueError as e:
            print(f"\nError computing implied volatility: {e}")

    print("\n" + "=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
