"""
Types and dataclasses for reproducible experiments.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExperimentConfig:
    """
    Configuration for a reproducible experiment.

    Attributes
    ----------
    name : str
        Experiment identifier
    model : str
        Model type: 'gbm' or 'heston'
    option_type : str
        Option type: 'call' or 'put'
    style : str
        Option style: 'european' or 'american'
    S0 : float
        Initial spot price
    K : float
        Strike price
    r : float
        Risk-free rate
    T : float
        Time to maturity
    sigma : float | None
        Volatility (for GBM only)
    kappa : float | None
        Mean reversion speed (for Heston only)
    theta : float | None
        Long-term variance (for Heston only)
    xi : float | None
        Volatility of volatility (for Heston only)
    rho : float | None
        Correlation (for Heston only)
    v0 : float | None
        Initial variance (for Heston only)
    n_paths_list : list[int]
        List of path counts to test
    n_steps : int
        Number of time steps (for Heston or LSM)
    seeds : list[int]
        List of random seeds for reproducibility
    antithetic : bool
        Use antithetic variates
    control_variate : bool
        Use control variate (GBM European only)
    compute_greeks : bool
        Compute Greeks (GBM European only)
    greeks_method : str
        Method for Greeks: 'pw' or 'fd'
    lsm_basis : str
        LSM basis functions: 'poly2' or 'poly3' (American only)
    """

    name: str
    model: str
    option_type: str
    style: str
    S0: float
    K: float
    r: float
    T: float
    sigma: float | None = None
    kappa: float | None = None
    theta: float | None = None
    xi: float | None = None
    rho: float | None = None
    v0: float | None = None
    n_paths_list: list[int] = field(default_factory=lambda: [10000])
    n_steps: int = 50
    seeds: list[int] = field(default_factory=lambda: [42])
    antithetic: bool = False
    control_variate: bool = False
    compute_greeks: bool = False
    greeks_method: str = "pw"
    lsm_basis: str = "poly2"


@dataclass
class GreeksData:
    """Greeks computation results."""

    delta: float | None = None
    delta_stderr: float | None = None
    vega: float | None = None
    vega_stderr: float | None = None
    method: str | None = None


@dataclass
class ExperimentMetadata:
    """
    Metadata for reproducible experiments.

    Captures environment and configuration for full reproducibility.
    """

    timestamp: str
    python_version: str
    numpy_version: str
    os_platform: str
    git_commit: str | None
    seed: int
    model: str
    option_type: str
    style: str
    n_paths: int
    n_steps: int | None
    antithetic: bool
    control_variate: bool
    compute_greeks: bool


@dataclass
class ExperimentResult:
    """
    Results from a single experiment run.

    Attributes
    ----------
    config_name : str
        Name of the experiment configuration
    price : float
        Estimated option price
    stderr : float
        Standard error of the estimate
    ci_lower : float
        Lower bound of 95% confidence interval
    ci_upper : float
        Upper bound of 95% confidence interval
    ci_width : float
        Width of confidence interval
    relative_error : float
        Relative error (stderr/price)
    n_paths : int
        Number of simulation paths used
    n_steps : int | None
        Number of time steps (if applicable)
    runtime_seconds : float
        Wall-clock time for computation
    control_variate_beta : float | None
        Control variate coefficient (if used)
    greeks : GreeksData | None
        Greeks results (if computed)
    implied_vol : float | None
        Implied volatility (if computed)
    metadata : ExperimentMetadata
        Full metadata for reproducibility
    notes : str
        Additional notes or method description
    """

    config_name: str
    price: float
    stderr: float
    ci_lower: float
    ci_upper: float
    ci_width: float
    relative_error: float
    n_paths: int
    n_steps: int | None
    runtime_seconds: float
    control_variate_beta: float | None
    greeks: GreeksData | None
    implied_vol: float | None
    metadata: ExperimentMetadata
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert result to dictionary for JSON serialization."""
        result = {
            "config_name": self.config_name,
            "price": self.price,
            "stderr": self.stderr,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "ci_width": self.ci_width,
            "relative_error": self.relative_error,
            "n_paths": self.n_paths,
            "n_steps": self.n_steps,
            "runtime_seconds": self.runtime_seconds,
            "control_variate_beta": self.control_variate_beta,
            "implied_vol": self.implied_vol,
            "notes": self.notes,
            "metadata": {
                "timestamp": self.metadata.timestamp,
                "python_version": self.metadata.python_version,
                "numpy_version": self.metadata.numpy_version,
                "os_platform": self.metadata.os_platform,
                "git_commit": self.metadata.git_commit,
                "seed": self.metadata.seed,
                "model": self.metadata.model,
                "option_type": self.metadata.option_type,
                "style": self.metadata.style,
                "n_paths": self.metadata.n_paths,
                "n_steps": self.metadata.n_steps,
                "antithetic": self.metadata.antithetic,
                "control_variate": self.metadata.control_variate,
                "compute_greeks": self.metadata.compute_greeks,
            },
        }

        if self.greeks is not None:
            result["greeks"] = {
                "delta": self.greeks.delta,
                "delta_stderr": self.greeks.delta_stderr,
                "vega": self.greeks.vega,
                "vega_stderr": self.greeks.vega_stderr,
                "method": self.greeks.method,
            }

        return result
