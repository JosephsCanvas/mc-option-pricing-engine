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

**Built for production. Validated for accuracy. Ready for interviews.**
