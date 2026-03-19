"""
v0.3 Evolution Engine — Generation Runner

Evaluates a population of LawFamily objects on the train set,
computes 4D fitness, identifies Pareto front, generates mutants.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.data.sparc_adapter import load_sparc
from src.data.split import split_galaxies
from src.evolution.population import LawFamily, MAX_POPULATION
from src.scoring.multi_objective import (
    FitnessVector,
    compute_f1,
    compute_f2,
    compute_f4_rar,
    pareto_front,
)
from src.types import FitResult, GalaxyData


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent.parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sparc"


def evaluate_family(
    family: LawFamily,
    galaxies: list[GalaxyData],
    seed: int = 42,
) -> tuple[list[FitResult], FitnessVector]:
    """Evaluate one family on all galaxies, return results + fitness."""
    results = []
    for gal in galaxies:
        try:
            r = family.fit_fn(gal, seed)
            results.append(r)
        except Exception as e:
            # Skip galaxies where fitting fails
            continue

    if not results:
        fv = FitnessVector(family.name, 0.0, 0.0, family.f3, 0.0)
        return results, fv

    f1 = compute_f1(results)
    f2 = compute_f2(results, family.law_param_indices)
    f3 = family.f3
    f4 = compute_f4_rar(results, galaxies)

    fv = FitnessVector(family.name, f1, f2, f3, f4)
    family.fitness = fv
    return results, fv


def run_generation(
    population: list[LawFamily],
    galaxies: list[GalaxyData],
    generation_num: int,
    seed: int = 42,
) -> list[FitnessVector]:
    """
    Evaluate all families in the population and return fitness vectors.
    """
    print(f"\n{'='*70}")
    print(f"GENERATION {generation_num} — {len(population)} families")
    print(f"{'='*70}")

    all_vectors: list[FitnessVector] = []
    all_results: dict[str, list[FitResult]] = {}

    for i, family in enumerate(population):
        t0 = time.time()
        results, fv = evaluate_family(family, galaxies, seed)
        elapsed = time.time() - t0

        all_vectors.append(fv)
        all_results[family.name] = results

        n_ok = len(results)
        print(f"  [{i+1}/{len(population)}] {family.name:40s} "
              f"F1={fv.f1:.4f} F2={fv.f2:.4f} F3={fv.f3:.4f} F4={fv.f4:.4f} "
              f"({n_ok}/{len(galaxies)} gals, {elapsed:.1f}s)")

    # Pareto front
    front = pareto_front(all_vectors)
    front_names = {fv.family_name for fv in front}

    print(f"\nPareto front ({len(front)} families):")
    for fv in sorted(front, key=lambda x: -x.f1):
        print(f"  * {fv.family_name}")

    dominated = [fv for fv in all_vectors if fv.family_name not in front_names]
    if dominated:
        print(f"\nDominated ({len(dominated)}):")
        for fv in dominated:
            doms = [o.family_name for o in all_vectors if o.dominates(fv)]
            print(f"  x {fv.family_name} (by {', '.join(doms[:3])})")

    # Check for the gap: any family with F1 >= 0.35 AND F2 >= 0.7?
    gap_candidates = [fv for fv in all_vectors if fv.f1 >= 0.35 and fv.f2 >= 0.7]
    if gap_candidates:
        print(f"\n*** GAP CANDIDATES (F1>=0.35 & F2>=0.7): ***")
        for fv in gap_candidates:
            print(f"  ! {fv.family_name} F1={fv.f1:.4f} F2={fv.f2:.4f}")
    else:
        print(f"\n  No gap candidates yet (need F1>=0.35 & F2>=0.7)")

    # Save generation results
    gen_dir = ARTIFACTS_DIR / "evolution"
    gen_dir.mkdir(parents=True, exist_ok=True)

    output = {
        "generation": generation_num,
        "n_families": len(population),
        "fitness_vectors": [fv.to_dict() for fv in all_vectors],
        "pareto_front": [fv.family_name for fv in front],
        "gap_candidates": [fv.family_name for fv in gap_candidates],
        "families": [f.to_dict() for f in population],
    }

    with open(gen_dir / f"gen_{generation_num:03d}.json", "w") as f:
        json.dump(output, f, indent=2)

    return all_vectors
