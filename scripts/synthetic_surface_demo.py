"""Synthetic volatility surface demo for Heston calibration.

Generates a synthetic implied volatility surface from known Heston parameters,
optionally adds noise, then calibrates to recover the original parameters.
"""

import json
import sys
from pathlib import Path

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from mc_pricer.analytics.implied_vol import implied_vol
from mc_pricer.calibration import CalibrationConfig, HestonCalibrator, MarketQuote
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine


def generate_synthetic_surface(
    S0: float,  # noqa: N803
    r: float,
    true_params: dict[str, float],
    strikes: list[float],
    maturities: list[float],
    n_paths: int = 20000,
    n_steps: int = 100,
    seed: int = 42,
    noise_std: float = 0.0,
) -> list[MarketQuote]:
    """Generate synthetic implied vol surface from true Heston parameters.

    Parameters
    ----------
    S0 : float
        Spot price.
    r : float
        Risk-free rate.
    true_params : dict[str, float]
        True Heston parameters: kappa, theta, xi, rho, v0.
    strikes : list[float]
        Strike prices.
    maturities : list[float]
        Times to maturity (years).
    n_paths : int
        Number of MC paths for pricing.
    n_steps : int
        Number of time steps.
    seed : int
        Random seed.
    noise_std : float
        Standard deviation of Gaussian noise to add to implied vols.

    Returns
    -------
    list[MarketQuote]
        Synthetic market quotes.
    """
    quotes = []
    rng = np.random.default_rng(seed + 1000)  # Different seed for noise

    for T in maturities:
        # Create model for this maturity
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
            scheme="qe",
        )

        for K in strikes:
            # Price call option
            payoff = EuropeanCallPayoff(strike=K)
            engine = HestonMonteCarloEngine(
                model=model,
                payoff=payoff,
                n_paths=n_paths,
                n_steps=n_steps,
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

                # Add noise if requested
                if noise_std > 0:
                    iv += rng.normal(0, noise_std)
                    iv = max(iv, 0.01)  # Keep positive

                # Create quote with synthetic bid-ask spread
                moneyness = K / S0
                bid_ask = 0.01 * iv  # 1% of IV as bid-ask
                if abs(moneyness - 1.0) > 0.2:  # Wider for far OTM/ITM
                    bid_ask *= 2

                quotes.append(
                    MarketQuote(
                        strike=K,
                        maturity=T,
                        option_type="call",
                        implied_vol=iv,
                        bid_ask_width=bid_ask,
                    )
                )
            except (ValueError, RuntimeError) as e:
                print(f"Warning: Failed to compute IV for K={K}, T={T}: {e}")
                continue

    return quotes


def main():
    """Run synthetic surface calibration demo."""
    print("\n" + "=" * 80)
    print("Heston Calibration - Synthetic Surface Demo")
    print("=" * 80)

    # Market parameters
    S0 = 100.0
    r = 0.05

    # True Heston parameters (known ground truth)
    true_params = {
        "kappa": 2.0,
        "theta": 0.04,  # Long-term var ≈ σ² where σ=0.2
        "xi": 0.3,
        "rho": -0.7,
        "v0": 0.04,
    }

    print("\nTrue Heston Parameters:")
    for name, value in true_params.items():
        print(f"  {name:8s}: {value:.6f}")

    # Define strike and maturity grid
    strikes = [80.0, 90.0, 100.0, 110.0, 120.0]
    maturities = [0.25, 0.5, 1.0]

    print("\nSurface Grid:")
    print(f"  Strikes:    {strikes}")
    print(f"  Maturities: {maturities}")
    print(f"  Total quotes: {len(strikes) * len(maturities)}")

    # Generate synthetic surface
    print("\nGenerating synthetic surface...")
    quotes = generate_synthetic_surface(
        S0=S0,
        r=r,
        true_params=true_params,
        strikes=strikes,
        maturities=maturities,
        n_paths=20000,
        n_steps=100,
        seed=42,
        noise_std=0.005,  # Small noise: 0.5% vol points
    )

    print(f"Generated {len(quotes)} quotes")

    # Print surface
    print("\nTarget Implied Volatility Surface:")
    print(f"{'K/S0':<10} ", end="")
    for T in maturities:
        print(f"T={T:<6.2f} ", end="")
    print()
    print("-" * 50)

    for K in strikes:
        print(f"{K/S0:<10.2f} ", end="")
        for T in maturities:
            # Find quote
            quote = next((q for q in quotes if q.strike == K and q.maturity == T), None)
            if quote:
                print(f"{quote.implied_vol:<9.4f}", end="")
            else:
                print(f"{'N/A':<9}", end="")
        print()

    # Calibration configuration
    config = CalibrationConfig(
        n_paths=5000,  # Smaller for demo speed
        n_steps=50,
        rng_type="pseudo",
        scramble=False,
        seeds=[42, 123, 456],  # 3 restarts
        max_iter=100,
        tol=1e-6,
        heston_scheme="qe",
        regularization=0.0,
    )

    print("\nCalibration Config:")
    print(f"  n_paths:     {config.n_paths}")
    print(f"  n_steps:     {config.n_steps}")
    print(f"  rng_type:    {config.rng_type}")
    print(f"  restarts:    {len(config.seeds)}")
    print(f"  max_iter:    {config.max_iter}")
    print(f"  scheme:      {config.heston_scheme}")

    # Initial guess (deliberately offset from truth)
    initial_guess = {
        "kappa": 1.5,
        "theta": 0.06,
        "xi": 0.4,
        "rho": -0.5,
        "v0": 0.05,
    }

    print("\nInitial Guess:")
    for name, value in initial_guess.items():
        print(f"  {name:8s}: {value:.6f}")

    # Create calibrator
    calibrator = HestonCalibrator(S0=S0, r=r, quotes=quotes, config=config)

    # Run calibration
    print("\n" + "=" * 80)
    print("Running Calibration...")
    print("=" * 80)

    result = calibrator.calibrate(initial_guess=initial_guess)

    # Print results
    print("\n" + "=" * 80)
    print("Calibration Results")
    print("=" * 80)

    print(f"\nObjective Value (RMSE): {result.objective_value:.8f}")
    print(f"Function Evaluations:   {result.n_evals}")
    print(f"Runtime:                {result.runtime_sec:.2f} seconds")

    print("\nCalibrated Parameters:")
    print(f"{'Parameter':<10} {'True':<12} {'Calibrated':<12} {'Error':<12} {'Error %'}")
    print("-" * 70)
    for name in ["kappa", "theta", "xi", "rho", "v0"]:
        true_val = true_params[name]
        calib_val = result.best_params[name]
        error = calib_val - true_val
        error_pct = 100 * error / true_val if true_val != 0 else np.nan
        print(
            f"{name:<10} {true_val:<12.6f} {calib_val:<12.6f} "
            f"{error:<12.6f} {error_pct:>8.2f}%"
        )

    # Print fitted vs target vols
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

    # Save results to JSON
    output_file = "heston_calibration_demo.json"
    output_data = {
        "true_params": true_params,
        "initial_guess": initial_guess,
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
            "initial_guess": result.diagnostics["initial_guess"],
            "bounds": {k: list(v) for k, v in result.diagnostics["bounds"].items()},
            "config": result.diagnostics["config"],
        },
    }

    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✓ Results saved to {output_file}")

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Successfully calibrated Heston model to {len(quotes)} synthetic quotes")
    print("Parameter recovery accuracy:")
    for name in ["kappa", "theta", "xi", "rho", "v0"]:
        error_pct = abs(
            100 * (result.best_params[name] - true_params[name]) / true_params[name]
        )
        print(f"  {name:8s}: {error_pct:>6.2f}% error")


if __name__ == "__main__":
    main()
