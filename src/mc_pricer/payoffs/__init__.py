"""
Payoffs package initialization.
"""

from mc_pricer.payoffs.multi_asset import (
    BasketArithmeticCallPayoff,
    BasketArithmeticPutPayoff,
    SpreadCallPayoff,
    SpreadPutPayoff,
)
from mc_pricer.payoffs.path_dependent import (
    AsianArithmeticCallPayoff,
    AsianArithmeticPutPayoff,
    DownAndOutPutPayoff,
    UpAndOutCallPayoff,
)
from mc_pricer.payoffs.plain_vanilla import EuropeanCallPayoff, EuropeanPutPayoff

__all__ = [
    "EuropeanCallPayoff",
    "EuropeanPutPayoff",
    "AsianArithmeticCallPayoff",
    "AsianArithmeticPutPayoff",
    "UpAndOutCallPayoff",
    "DownAndOutPutPayoff",
    "BasketArithmeticCallPayoff",
    "BasketArithmeticPutPayoff",
    "SpreadCallPayoff",
    "SpreadPutPayoff",
]
