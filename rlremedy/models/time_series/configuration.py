from dataclasses import dataclass
from typing import Optional


@dataclass
class SPParams:
    cycles: int = 5  # how many sine cycles
    resolution: int = 1000 # how many data points to generate
    sample_size: int = 1000 #


@dataclass
class OUParams:
    alpha: float = 0.07  # mean reversion parameter
    gamma: float = 0 # asymptotic mean
    beta: float = 0.001 # Brownian motion scale (standard deviation
    sample_size:  Optional[int] = 1000  # the sample size.
    seed:  Optional[int] = None  # random number generator seed
