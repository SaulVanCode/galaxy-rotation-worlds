"""
v0.3 Evolution Engine — Population and Family Registry

A LawFamily is a complete specification: name, fitting function,
structural complexity, parameter metadata, and lineage.

The population is the set of families competing in one generation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from src.scoring.multi_objective import FitnessVector
from src.types import FitResult, GalaxyData


@dataclass
class LawFamily:
    """Complete specification of a law family for the evolution engine."""

    name: str
    fit_fn: Callable[[GalaxyData, int], FitResult]  # (galaxy, seed) -> FitResult
    k_law: int              # number of law-level parameters
    k_pergalaxy: int        # number of per-galaxy fit parameters (typically 2: Υ_d, Υ_b)
    structural_complexity: float  # S score
    law_param_indices: list[int]  # indices into FitResult.params for law params
    lineage: list[str] = field(default_factory=list)
    generation: int = 0
    mutation_desc: str = ""
    fitness: FitnessVector | None = None

    @property
    def total_complexity(self) -> float:
        return self.k_law + self.k_pergalaxy + self.structural_complexity

    @property
    def f3(self) -> float:
        """Simplicity score."""
        t = self.total_complexity
        return 1.0 / t if t > 0 else 1.0

    def to_dict(self) -> dict:
        d = {
            "name": self.name,
            "k_law": self.k_law,
            "k_pergalaxy": self.k_pergalaxy,
            "structural_complexity": self.structural_complexity,
            "total_complexity": self.total_complexity,
            "lineage": self.lineage,
            "generation": self.generation,
            "mutation": self.mutation_desc,
        }
        if self.fitness:
            d["fitness"] = self.fitness.to_dict()
        return d


# Maximum structural complexity allowed
MAX_STRUCTURAL_COMPLEXITY = 15.0

# Maximum population size
MAX_POPULATION = 30
