"""
Greeks result types.
"""

from dataclasses import dataclass


@dataclass
class GreekResult:
    """
    Container for a single Greek estimate.

    Attributes
    ----------
    value : float
        Estimated Greek value
    standard_error : float
        Standard error of the estimate
    ci_lower : float
        Lower bound of 95% confidence interval
    ci_upper : float
        Upper bound of 95% confidence interval
    """

    value: float
    standard_error: float
    ci_lower: float
    ci_upper: float

    def __repr__(self) -> str:
        return (
            f"GreekResult(\n"
            f"  value={self.value:.6f},\n"
            f"  stderr={self.standard_error:.6f},\n"
            f"  CI95=[{self.ci_lower:.6f}, {self.ci_upper:.6f}]\n"
            f")"
        )


@dataclass
class GreeksResult:
    """
    Container for multiple Greeks estimates.

    Attributes
    ----------
    delta : GreekResult | None
        Delta estimate (if computed)
    vega : GreekResult | None
        Vega estimate (if computed)
    method : str
        Estimation method used ('pw', 'fd', or 'both')
    fd_step_spot : float | None
        Finite difference step size for spot (if FD used)
    fd_step_sigma : float | None
        Finite difference step size for sigma (if FD used)
    """

    delta: GreekResult | None
    vega: GreekResult | None
    method: str
    fd_step_spot: float | None = None
    fd_step_sigma: float | None = None

    def __repr__(self) -> str:
        lines = [f"GreeksResult(method='{self.method}')"]
        if self.delta is not None:
            lines.append(f"  Delta: {self.delta.value:.6f} ± {self.delta.standard_error:.6f}")
        if self.vega is not None:
            lines.append(f"  Vega:  {self.vega.value:.6f} ± {self.vega.standard_error:.6f}")
        if self.fd_step_spot is not None:
            lines.append(f"  FD step (spot): {self.fd_step_spot}")
        if self.fd_step_sigma is not None:
            lines.append(f"  FD step (sigma): {self.fd_step_sigma}")
        return "\n".join(lines)
