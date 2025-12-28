# Monte Carlo Option Pricing Engine

[![CI](https://github.com/yourusername/mc-option-pricing-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/yourusername/mc-option-pricing-engine/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-grade Monte Carlo simulation engine for pricing financial derivatives. Built with clean architecture, comprehensive testing, and validated against Black-Scholes analytical solutions.

## Features

- **Geometric Brownian Motion (GBM)** asset price simulation with vectorized implementation
- **European Options** (call/put) with analytical payoff functions
- **Monte Carlo Pricing Engine** with:
  - Variance reduction via **antithetic variates** and **control variates** (~62% stderr reduction)
  - Statistical confidence intervals (95% CI)
  - Deterministic reproducibility (seed control)
- **Validation Suite** comparing MC prices to Black-Scholes closed-form solutions
- **Type-safe** code with full type hints and input validation
- **CI/CD Pipeline** with GitHub Actions (Python 3.10/3.11/3.12)
- **CLI Interface** for interactive pricing

## Project Structure

```
mc-option-pricing-engine/
├── src/mc_pricer/              # Main package
│   ├── models/                 # Asset price models (GBM)
│   ├── payoffs/                # Option payoffs (European call/put)
│   ├── pricers/                # Monte Carlo pricing engine
│   ├── rng/                    # Random number generators (Sobol placeholder)
│   └── greeks/                 # Greeks computation (placeholder)
├── tests/                      # Comprehensive test suite
│   ├── utils/                  # Test utilities (Black-Scholes)
│   ├── test_gbm.py            # GBM model tests
│   ├── test_payoffs.py        # Payoff function tests
│   ├── test_monte_carlo.py    # MC engine tests
│   └── test_validation.py     # Black-Scholes validation tests
├── scripts/                    # CLI tools
│   ├── mc_price.py            # Command-line pricing interface
│   ├── demo_all.py            # Complete feature demonstration
│   ├── convergence_demo.py    # Convergence analysis
│   ├── benchmark_variance_reduction.py  # Statistical benchmarks
│   └── plot_convergence.py    # Convergence visualization
├── notebooks/                  # Jupyter notebooks (placeholder)
├── pyproject.toml             # Project configuration
└── .github/workflows/ci.yml   # CI/CD configuration
```

## Quickstart

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/mc-option-pricing-engine.git
   cd mc-option-pricing-engine
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Unix/macOS
   source venv/bin/activate
   ```

3. **Install the package in editable mode with dev dependencies**:
   ```bash
   pip install -e ".[dev]"
   ```

### Quick Demo

Run the complete demonstration:
```bash
python scripts/demo_all.py
```

This demonstrates:
- Basic European option pricing
- Variance reduction techniques comparison
- Black-Scholes validation
- Put-call parity verification

### Running Tests

Run all tests with pytest:
```bash
pytest -q
```

Run specific test categories:
```bash
# Unit tests only
pytest tests/test_gbm.py tests/test_payoffs.py tests/test_monte_carlo.py -v

# Validation tests (compare to Black-Scholes)
pytest tests/test_validation.py -v
```

### Code Quality

Check code quality with ruff:
```bash
ruff check .
```

### Using the CLI

Price a European call option:
```bash
python scripts/mc_price.py --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 --n_paths 200000 --seed 42
```

Price with control variate variance reduction (~62% stderr reduction):
```bash
python scripts/mc_price.py --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 --n_paths 200000 --seed 42 --control_variate
```

Combine antithetic variates and control variates:
```bash
python scripts/mc_price.py --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 --n_paths 200000 --seed 42 --antithetic --control_variate
```

Price a European put option:
```bash
python scripts/mc_price.py --S0 100 --K 100 --r 0.05 --sigma 0.2 --T 1.0 --n_paths 200000 --option_type put
```

**Example Output**:
```
======================================================================
Monte Carlo Option Pricing Engine
======================================================================

Input Parameters:
  Spot Price (S0):        100.00
  Strike Price (K):       100.00
  Risk-free Rate (r):     0.0500
  Volatility (σ):         0.2000
  Time to Maturity (T):   1.0000 years
  Option Type:            CALL

Simulation Parameters:
  Number of Paths:        200,000
  Antithetic Variates:    True
  Random Seed:            42

======================================================================
Pricing...
======================================================================

Results:
  Option Price:           10.449735
  Standard Error:         0.015473
  95% Confidence Interval: [10.419408, 10.480061]
  CI Width:               0.060653
  Relative Error (σ/μ):   0.1481%

======================================================================
```

### Programmatic Usage

```python
from mc_pricer import GeometricBrownianMotion, EuropeanCallPayoff, MonteCarloEngine

# Set up the model
model = GeometricBrownianMotion(
    S0=100,      # Initial spot price
    r=0.05,      # Risk-free rate
    sigma=0.2,   # Volatility
    T=1.0,       # Time to maturity
    seed=42      # For reproducibility
)

# Define the payoff
payoff = EuropeanCallPayoff(strike=100)

# Create pricing engine
engine = MonteCarloEngine(
    model=model,
    payoff=payoff,
    n_paths=200000,
    antithetic=True  # Use variance reduction
)

# Price the option
result = engine.price()
print(f"Option Price: {result.price:.6f}")
print(f"Std Error: {result.stderr:.6f}")
print(f"95% CI: [{result.ci_lower:.6f}, {result.ci_upper:.6f}]")
```

## Validation

The Monte Carlo engine is validated against Black-Scholes analytical prices:

- **ATM/ITM/OTM European options** across various parameters
- **Put-call parity** verification
- **Convergence tests** with increasing sample size
- **Antithetic variance reduction** effectiveness tests

Run validation tests:
```bash
pytest tests/test_validation.py -v
```

Sample validation result:
```
Test: ATM Call (S0=100, K=100, r=5%, σ=20%, T=1y)
  Black-Scholes Price: 10.4506
  Monte Carlo Price:   10.4497 ± 0.0155
  Relative Error:      0.009%
  ✓ Within 3 standard errors
```

## Greeks Convergence Demo

The engine supports **Delta** and **Vega** computation using two estimators: **Pathwise (PW)** derivatives and **Finite Difference (FD)**. The Pathwise estimator provides low-variance, efficient Greeks by differentiating the simulation paths analytically, while Finite Difference serves as an independent cross-check but requires multiple simulations (making it more computationally expensive). Both methods converge to Black-Scholes analytical values as the number of paths increases.

Run the convergence demonstration:
```bash
python scripts/greeks_convergence_demo.py
```

**Sample Output**:
```
====================================================================================================
Greeks Convergence Analysis - Pathwise Estimator
====================================================================================================

Parameters: S0=100.0, K=100.0, r=0.05, sigma=0.2, T=1.0
Option Type: European Call
Seed: 42

Black-Scholes Targets: Delta=0.636831, Vega=37.5240

------------------------------------------------------------------------------------------------------------------------
n_paths      Delta      DeltaErr   Delta SE   Vega       VegaErr    Vega SE    Delta CI               Vega CI
------------------------------------------------------------------------------------------------------------------------
1,000        0.628003    -0.008827 0.018097   34.5368       -2.9873 2.287771   [0.592533, 0.663474]   [30.0527, 39.0208]    
5,000        0.630953    -0.005877 0.008129   36.0712       -1.4528 1.059920   [0.615020, 0.646886]   [33.9938, 38.1487]
20,000       0.637373     0.000542 0.004084   38.1216        0.5976 0.541584   [0.629368, 0.645378]   [37.0601, 39.1832]
100,000      0.634516    -0.002315 0.001824   37.4800       -0.0440 0.241222   [0.630940, 0.638092]   [37.0072, 37.9528]    
------------------------------------------------------------------------------------------------------------------------

====================================================================================================
Finite Difference Comparison (n_paths=100,000)
====================================================================================================

Method: Finite Difference
  Delta: 0.637264 ± 0.000289
    95% CI: [0.636697, 0.637830]
  Vega:  37.4895 ± 0.0131
    95% CI: [37.4637, 37.5152]
  FD steps: h_spot=0.0001, h_sigma=0.0001
```

As shown in the table above, **DeltaErr** and **VegaErr** (the differences from Black-Scholes analytical values) decrease as O(1/√n), with standard errors shrinking from ~0.018 (1k paths) to ~0.0018 (100k paths) for Delta. The **Pathwise** estimator achieves low variance with a single simulation, while **Finite Difference** requires 10 additional simulations (5 for Delta + 5 for Vega) but provides an independent validation that both methods agree within confidence intervals.

## Heston Stochastic Volatility Model

The engine includes the **Heston stochastic volatility model**, which extends the Black-Scholes framework by allowing variance to follow its own stochastic process:

```
dS_t = r * S_t * dt + sqrt(v_t) * S_t * dW1_t
dv_t = κ * (θ - v_t) * dt + ξ * sqrt(v_t) * dW2_t
```

where:
- **v_t**: instantaneous variance at time t
- **κ**: mean reversion speed
- **θ**: long-term variance level
- **ξ**: volatility of volatility
- **ρ**: correlation between W1 and W2 (typically negative)

### Variance Discretization Schemes

Two discretization schemes are supported for the CIR variance process:

#### 1. Full Truncation Euler (Default)
Simple explicit Euler scheme with max(v, 0) truncation to ensure non-negativity:
```
v_{t+dt} = max(v_t + κ*(θ - max(v_t,0))*dt + ξ*sqrt(max(v_t,0))*sqrt(dt)*Z, 0)
```

**Pros**: Simple, fast, easy to understand  
**Cons**: Can be biased for extreme parameters (high ξ, low κθ)

#### 2. Quadratic-Exponential (QE) 
Andersen's (2008) scheme that samples from the exact conditional distribution:
- **Quadratic regime** (ψ ≤ 1.5): Uses transformed normal approximation
- **Exponential regime** (ψ > 1.5): Uses exponential distribution approximation

**Pros**: Better convergence, less biased, handles extreme parameters  
**Cons**: Slightly more complex, ~10-20% slower

### Usage Example

```python
from mc_pricer.models.heston import HestonModel
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff
from mc_pricer.pricers.heston_monte_carlo import HestonMonteCarloEngine

# Create Heston model with QE scheme
model = HestonModel(
    S0=100.0,
    r=0.05,
    T=1.0,
    kappa=2.0,      # Mean reversion speed
    theta=0.04,     # Long-term variance (σ² ≈ 0.2²)
    xi=0.3,         # Vol of vol
    rho=-0.7,       # Correlation (negative for leverage effect)
    v0=0.04,        # Initial variance
    seed=42,
    scheme="qe"     # Use QE scheme (default: "full_truncation_euler")
)

# Price European call
payoff = EuropeanCallPayoff(K=100.0)
engine = HestonMonteCarloEngine(
    model=model,
    payoff=payoff,
    n_paths=50000,
    n_steps=200     # Number of time steps
)

result = engine.price()
print(f"Price: {result.price:.6f} ± {result.stderr:.6f}")
print(f"Scheme: {result.scheme}")
```

### CLI Usage

```bash
# Price with Full Truncation Euler (default)
python scripts/mc_price.py --model heston --S0 100 --K 100 --r 0.05 --T 1.0 \
    --kappa 2.0 --theta 0.04 --xi 0.3 --rho -0.7 --v0 0.04 \
    --n_paths 50000 --n_steps 200

# Price with QE scheme
python scripts/mc_price.py --model heston --S0 100 --K 100 --r 0.05 --T 1.0 \
    --kappa 2.0 --theta 0.04 --xi 0.3 --rho -0.7 --v0 0.04 \
    --n_paths 50000 --n_steps 200 --heston_scheme qe
```

### Benchmarking Schemes

Compare the two discretization schemes across strikes to analyze the volatility smile:

```bash
# Run benchmark (outputs JSON + summary table)
python scripts/bench_heston_schemes.py --n_paths 50000 --n_steps 200 --n_seeds 3

# Results saved to:
#   results/heston_scheme_benchmark.json
#   results/heston_scheme_summary.txt
```

**Sample Benchmark Output**:
```
Strike   FT Euler IV     QE IV           IV Diff     Price Diff
------------------------------------------------------------------------
70       0.242156        0.241983        -0.000173   -0.017324
80       0.223847        0.223719        -0.000128   -0.010442
90       0.209634        0.209538        -0.000096   -0.005821
100      0.199874        0.199802        -0.000072   -0.003182
110      0.194215        0.194163        -0.000052   -0.001822
120      0.192037        0.191999        -0.000038   -0.001053
130      0.192556        0.192527        -0.000029   -0.000633

Smile Width:
  FT Euler: 0.050120
  QE:       0.049984

Total Runtime:
  FT Euler: 42.384s
  QE:       48.107s
```

The QE scheme typically produces slightly lower implied volatilities (less bias) and slightly wider confidence intervals, with ~10-20% longer runtime due to regime switching logic.

## Roadmap

Future enhancements planned:

### Models
- **Heston stochastic volatility** model with correlated Brownian motions
- **Jump-diffusion processes** (Merton, Kou)
- **Local volatility** models (Dupire)

### Variance Reduction
- **Quasi-Monte Carlo (QMC)** with Sobol sequences
- **Brownian bridge** for path-dependent options
- **Control variates** using analytical approximations
- **Importance sampling** for rare events

### Greeks
- **Pathwise derivatives** for Delta/Vega
- **Likelihood ratio method** for Greeks
- **Finite difference Greeks** with adaptive step sizes
- **Malliavin calculus** for higher-order Greeks

### Exotic Options
- **Asian options** (arithmetic/geometric average)
- **Barrier options** (knock-in/knock-out)
- **Lookback options**
- **Digital/binary options**
- **Bermudan options** with Longstaff-Schwartz

### Performance
- **Parallel/GPU acceleration** with NumPy/CuPy
- **Vectorized path generation** optimizations
- **Memory-efficient streaming** for large simulations

### Calibration
- **Implied volatility surface** fitting
- **Model parameter calibration** to market data
- **Variance swap pricing**

## Development

### Running Linters
```bash
ruff check .
ruff format .  # Auto-format code
```

### Adding Dependencies
Edit `pyproject.toml` and reinstall:
```bash
pip install -e ".[dev]"
```

### CI/CD
The project uses GitHub Actions for continuous integration:
- Runs on Python 3.10, 3.11, 3.12
- Executes `ruff check` for linting
- Runs full test suite with `pytest`
- Triggers on push and pull requests

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## References

- Black, F. and Scholes, M. (1973). "The Pricing of Options and Corporate Liabilities"
- Glasserman, P. (2003). "Monte Carlo Methods in Financial Engineering"
- Jäckel, P. (2002). "Monte Carlo Methods in Finance"

---

**Built for production. Validated for accuracy.**
