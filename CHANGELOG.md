# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-12-30

### Added

#### Core Pricing Engine
- **Monte Carlo pricing** with configurable paths and reproducible seeds
- **European options** (call/put) with validated payoff functions
- **Geometric Brownian Motion** model with vectorized simulation
- **Confidence intervals** (95% CI) for all price estimates

#### Variance Reduction
- **Antithetic variates** for ~50% variance reduction
- **Control variates** using Black-Scholes analytical prices (~62% stderr reduction)

#### Greeks Computation
- **Pathwise (automatic differentiation style)** Delta and Vega
- **Finite difference (bump-and-revalue)** Delta and Vega
- Both methods validated against Black-Scholes analytical solutions

#### American Options
- **Least Squares Monte Carlo (LSM)** implementation
- Polynomial basis functions (poly2, poly3)
- Early exercise boundary detection

#### Analytics Module
- **Black-Scholes** closed-form pricing (price, delta, gamma, vega)
- **Implied volatility** solver using Brent's method
- Arbitrage bounds validation

#### Heston Stochastic Volatility
- **Full Truncation Euler** discretization scheme
- **Quadratic-Exponential (QE)** scheme (Andersen 2008)
- Correlated Brownian motions with configurable ρ
- Parameter validation (Feller condition checking)

#### Quasi-Monte Carlo
- **Sobol sequences** (up to 21 dimensions)
- **Digital shift scrambling** for randomized QMC
- Owen scrambling support

#### Path-Dependent Options
- **Asian options** (arithmetic average, call/put)
- **Barrier options** (up-and-out call, down-and-out put)

#### Heston Calibration
- **Weighted calibration** to implied volatility surfaces
- **Nelder-Mead optimizer** (numpy-only, no scipy)
- **Common Random Numbers (CRN)** for variance reduction
- **LRU price cache** for performance optimization
- Fast mode (<30 seconds) for rapid iteration
- JSON artifacts with git metadata for reproducibility

#### Experiments Framework
- Reproducible experiment runner with metadata capture
- JSON/pickle results serialization
- Summary table formatting for papers

#### CLI Interface
- `mc-price` command for interactive pricing
- Support for all models, options, and features
- Black-Scholes comparison and implied volatility

#### Quality & CI
- Full type hints (Python 3.10+)
- ruff linting (clean)
- pytest test suite (200+ tests)
- GitHub Actions CI (Python 3.10, 3.11, 3.12)
- MIT License

### Dependencies
- **numpy** (runtime, >=1.24.0)
- **pytest, ruff, build, twine** (dev)

[Unreleased]: https://github.com/JosephsCanvas/mc-option-pricing-engine/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/JosephsCanvas/mc-option-pricing-engine/releases/tag/v0.1.0
