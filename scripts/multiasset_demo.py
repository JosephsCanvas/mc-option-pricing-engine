"""
Multi-Asset Options Pricing Demo

Demonstrates basket and spread option pricing with varying correlations.
Shows the effect of correlation on option prices and the effectiveness of
variance reduction techniques (antithetic variates and QMC).
"""

import numpy as np

from mc_pricer.models.multi_gbm import MultiAssetGeometricBrownianMotion
from mc_pricer.payoffs.multi_asset import (
    BasketArithmeticCallPayoff,
    SpreadCallPayoff,
)
from mc_pricer.pricers.multi_asset_monte_carlo import MultiAssetMonteCarloEngine


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 80}")
    print(f"{title:^80}")
    print("=" * 80)


def basket_call_demo() -> None:
    """Demonstrate basket call pricing with varying correlations."""
    print_section("BASKET CALL OPTION: Effect of Correlation")

    # Parameters
    S0 = np.array([100.0, 100.0, 100.0])  # 3 assets
    r = 0.05
    sigma = np.array([0.2, 0.2, 0.2])
    T = 1.0
    strike = 100.0

    # Test different correlation levels
    rho_values = [-0.3, 0.0, 0.5]

    print("\nParameters:")
    print("  Assets: 3")
    print(f"  S0: {S0}")
    print(f"  r: {r}")
    print(f"  sigma: {sigma}")
    print(f"  T: {T} years")
    print(f"  Strike: {strike}")

    print(
        f"\n{'Correlation':<15} {'RNG Type':<12} {'N Paths':<12} {'Price':<12} {'Stderr':<12} {'95% CI':<25}"
    )
    print("-" * 90)

    for rho in rho_values:
        # Create correlation matrix with off-diagonal rho
        corr = np.array([[1.0, rho, rho], [rho, 1.0, rho], [rho, rho, 1.0]])

        model = MultiAssetGeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, corr=corr, seed=42)

        payoff = BasketArithmeticCallPayoff(strike=strike)

        # Pseudo RNG
        for n_paths in [5000, 20000]:
            engine = MultiAssetMonteCarloEngine(
                model=model,
                payoff=payoff,
                n_paths=n_paths,
                rng_type="pseudo",
                antithetic=False,
                seed=42,
            )

            result = engine.price()

            ci_str = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
            print(
                f"{rho:>6.1f}         {'pseudo':<12} {n_paths:<12,} {result.price:<12.4f} {result.stderr:<12.6f} {ci_str:<25}"
            )

        # Sobol QMC (single run with high paths)
        engine_sobol = MultiAssetMonteCarloEngine(
            model=model, payoff=payoff, n_paths=20000, rng_type="sobol", scramble=True, seed=42
        )

        result_sobol = engine_sobol.price()

        ci_str = f"[{result_sobol.ci_lower:.4f}, {result_sobol.ci_upper:.4f}]"
        print(
            f"{rho:>6.1f}         {'sobol':<12} {20000:<12,} {result_sobol.price:<12.4f} {result_sobol.stderr:<12.6f} {ci_str:<25}"
        )

    print("\nObservations:")
    print("  - Higher correlation -> Higher basket call price (assets move together)")
    print("  - Negative correlation -> Lower basket call price (diversification benefit)")
    print("  - QMC (Sobol) typically reduces stderr compared to pseudo-random")


def spread_call_demo() -> None:
    """Demonstrate spread call pricing with varying correlations."""
    print_section("SPREAD CALL OPTION: Effect of Correlation")

    # Parameters
    S0 = np.array([110.0, 100.0])  # 2 assets
    r = 0.05
    sigma = np.array([0.25, 0.25])
    T = 1.0
    strike = 5.0

    # Test different correlation levels
    rho_values = [-0.7, 0.0, 0.7]

    print("\nParameters:")
    print("  Assets: 2")
    print(f"  S0: {S0}")
    print(f"  r: {r}")
    print(f"  sigma: {sigma}")
    print(f"  T: {T} years")
    print(f"  Strike: {strike}")
    print("  Payoff: max(S1_T - S2_T - K, 0)")

    print(
        f"\n{'Correlation':<15} {'RNG Type':<12} {'N Paths':<12} {'Price':<12} {'Stderr':<12} {'95% CI':<25}"
    )
    print("-" * 90)

    for rho in rho_values:
        corr = np.array([[1.0, rho], [rho, 1.0]])

        model = MultiAssetGeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, corr=corr, seed=123)

        payoff = SpreadCallPayoff(strike=strike)

        # Pseudo RNG with antithetic
        for n_paths in [5000, 20000]:
            engine = MultiAssetMonteCarloEngine(
                model=model,
                payoff=payoff,
                n_paths=n_paths,
                rng_type="pseudo",
                antithetic=True,
                seed=123,
            )

            result = engine.price()

            ci_str = f"[{result.ci_lower:.4f}, {result.ci_upper:.4f}]"
            print(
                f"{rho:>6.1f}         {'pseudo+AV':<12} {n_paths:<12,} {result.price:<12.4f} {result.stderr:<12.6f} {ci_str:<25}"
            )

        # Sobol QMC
        engine_sobol = MultiAssetMonteCarloEngine(
            model=model, payoff=payoff, n_paths=20000, rng_type="sobol", scramble=True, seed=123
        )

        result_sobol = engine_sobol.price()

        ci_str = f"[{result_sobol.ci_lower:.4f}, {result_sobol.ci_upper:.4f}]"
        print(
            f"{rho:>6.1f}         {'sobol':<12} {20000:<12,} {result_sobol.price:<12.4f} {result_sobol.stderr:<12.6f} {ci_str:<25}"
        )

    print("\nObservations:")
    print("  - Higher positive correlation -> Lower spread call price (assets move together)")
    print("  - Negative correlation -> Higher spread call price (assets diverge)")
    print("  - Antithetic variates (AV) help reduce stderr")
    print("  - QMC particularly effective for low-dimensional problems")


def variance_reduction_comparison() -> None:
    """Compare variance reduction techniques for multi-asset pricing."""
    print_section("VARIANCE REDUCTION COMPARISON")

    # Simple 2-asset basket
    S0 = np.array([100.0, 100.0])
    r = 0.05
    sigma = np.array([0.2, 0.2])
    T = 1.0
    corr = np.array([[1.0, 0.3], [0.3, 1.0]])
    strike = 100.0

    model = MultiAssetGeometricBrownianMotion(S0=S0, r=r, sigma=sigma, T=T, corr=corr, seed=999)

    payoff = BasketArithmeticCallPayoff(strike=strike)

    print(f"\nBasket Call (2 assets, rho=0.3, K={strike})")
    print(f"{'Method':<25} {'N Paths':<12} {'Price':<12} {'Stderr':<12} {'Efficiency':<12}")
    print("-" * 75)

    n_paths = 50000

    # Standard pseudo
    engine_standard = MultiAssetMonteCarloEngine(
        model=model, payoff=payoff, n_paths=n_paths, rng_type="pseudo", antithetic=False, seed=999
    )
    result_standard = engine_standard.price()
    baseline_var = result_standard.stderr**2
    print(
        f"{'Pseudo (standard)':<25} {n_paths:<12,} {result_standard.price:<12.4f} {result_standard.stderr:<12.6f} {'1.00x':<12}"
    )

    # Antithetic
    engine_antithetic = MultiAssetMonteCarloEngine(
        model=model, payoff=payoff, n_paths=n_paths, rng_type="pseudo", antithetic=True, seed=999
    )
    result_antithetic = engine_antithetic.price()
    efficiency_av = baseline_var / (result_antithetic.stderr**2)
    print(
        f"{'Pseudo + Antithetic':<25} {n_paths:<12,} {result_antithetic.price:<12.4f} {result_antithetic.stderr:<12.6f} {efficiency_av:<12.2f}x"
    )

    # Sobol
    engine_sobol = MultiAssetMonteCarloEngine(
        model=model, payoff=payoff, n_paths=n_paths, rng_type="sobol", scramble=False, seed=999
    )
    result_sobol = engine_sobol.price()
    efficiency_sobol = baseline_var / (result_sobol.stderr**2)
    print(
        f"{'Sobol (unscrambled)':<25} {n_paths:<12,} {result_sobol.price:<12.4f} {result_sobol.stderr:<12.6f} {efficiency_sobol:<12.2f}x"
    )

    # Sobol + scramble
    engine_sobol_scramble = MultiAssetMonteCarloEngine(
        model=model, payoff=payoff, n_paths=n_paths, rng_type="sobol", scramble=True, seed=999
    )
    result_sobol_scramble = engine_sobol_scramble.price()
    efficiency_sobol_s = baseline_var / (result_sobol_scramble.stderr**2)
    print(
        f"{'Sobol (scrambled)':<25} {n_paths:<12,} {result_sobol_scramble.price:<12.4f} {result_sobol_scramble.stderr:<12.6f} {efficiency_sobol_s:<12.2f}x"
    )

    print("\nNote: Efficiency = Var(standard) / Var(method)")
    print("      Higher efficiency means fewer paths needed for same accuracy")


def main() -> None:
    """Run all multi-asset pricing demos."""
    print("\n" + "=" * 80)
    print(" " * 20 + "MULTI-ASSET OPTIONS PRICING DEMO")
    print("=" * 80)

    basket_call_demo()
    spread_call_demo()
    variance_reduction_comparison()

    print("\n" + "=" * 80)
    print("Demo complete! Multi-asset pricing capabilities verified.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
