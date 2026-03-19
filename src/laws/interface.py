"""
MetaLaw Discovery System v0.2-FROZEN — Law Interface

All gravity laws (static baselines and generated families) implement
GravityLaw. Rule generators produce GravityLaw instances from parameter
vectors.

Every implementation must include dimensional analysis comments on all
formulas. A formula without a dimensional comment is a bug.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class GravityLaw(ABC):
    """
    Abstract base for all gravity laws.

    Contract:
        - acceleration() returns gravitational acceleration in internal units
          [km^2 s^-2 kpc^-1]
        - All inputs are in internal units
        - Implementations must include dimensional analysis comments
    """

    @abstractmethod
    def acceleration(
        self,
        r: float,
        M_enclosed: float,
        rho: float,
    ) -> float:
        """
        Compute gravitational acceleration magnitude at radius r.

        Args:
            r: galactocentric radius [kpc]
            M_enclosed: total baryonic enclosed mass at r [M_sun]
            rho: local volume density estimate at r [M_sun kpc^-3]

        Returns:
            Gravitational acceleration [km^2 s^-2 kpc^-1], always >= 0
        """
        ...

    @abstractmethod
    def param_vector(self) -> np.ndarray:
        """Return current parameter values as array."""
        ...

    @abstractmethod
    def param_names(self) -> list[str]:
        """Return names corresponding to param_vector entries."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable family name."""
        ...

    @property
    def param_count(self) -> int:
        """Number of free parameters (for BIC)."""
        return len(self.param_vector())


class RuleGenerator(ABC):
    """
    A parameterized factory that produces GravityLaw instances.

    The generator defines a family of laws; each parameter vector
    instantiates a specific member of that family.
    """

    @abstractmethod
    def generate(self, params: np.ndarray) -> GravityLaw:
        """Produce a concrete GravityLaw from a parameter vector."""
        ...

    @abstractmethod
    def param_bounds(self) -> list[tuple[float, float]]:
        """
        Search bounds for each parameter dimension.
        Returns list of (lower, upper) tuples.
        """
        ...

    @abstractmethod
    def param_names(self) -> list[str]:
        """Names for each parameter in the generator's param vector."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable generator name."""
        ...
