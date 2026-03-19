"""
v0.3 Evolution Engine — Main Runner

Builds initial population (G0 = v0.2 families),
generates mutants, runs generation 1, analyzes Pareto front.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from src.data.sparc_adapter import load_sparc
from src.data.split import split_galaxies
from src.evolution.generation import run_generation
from src.evolution.mutations import (
    m2_add_correction,
    m3_compose,
    m4_freeze_pr_a0_alphas,
    m4_freeze_pr_alphas,
    m4_freeze_pr_w,
)
from src.evolution.population import LawFamily
from src.fitting.per_galaxy import (
    fit_mond,
    fit_newton_nfw,
    fit_newtonian,
    fit_piecewise_regime,
)
from src.fitting.universal import fit_piecewise_universal
from src.laws.mond_simple import SimpleMONDLaw

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sparc"


def build_g0() -> list[LawFamily]:
    """Build generation 0: the v0.2 families."""

    # Load universal params for PR-universal
    with open(ARTIFACTS_DIR / "aggregate" / "universal_params.json") as f:
        univ = json.load(f)
    univ_params = np.array([
        univ["params"]["log10_a0"],
        univ["params"]["alpha_inner"],
        univ["params"]["alpha_outer"],
        univ["params"]["w"],
    ])

    families = [
        LawFamily(
            name="Newton-baryons",
            fit_fn=fit_newtonian,
            k_law=0, k_pergalaxy=2,
            structural_complexity=0.0,
            law_param_indices=[],
            generation=0,
        ),
        LawFamily(
            name="Newton+NFW",
            fit_fn=fit_newton_nfw,
            k_law=0, k_pergalaxy=4,  # halo params per-galaxy
            structural_complexity=1.0,
            law_param_indices=[2, 3],
            generation=0,
        ),
        LawFamily(
            name="MOND-simple",
            fit_fn=fit_mond,
            k_law=0, k_pergalaxy=2,
            structural_complexity=2.0,
            law_param_indices=[],
            generation=0,
        ),
        LawFamily(
            name="PiecewiseRegime",
            fit_fn=fit_piecewise_regime,
            k_law=4, k_pergalaxy=2,
            structural_complexity=7.0,
            law_param_indices=[2, 3, 4, 5],
            generation=0,
        ),
        LawFamily(
            name="PiecewiseRegime-universal",
            fit_fn=lambda gal, seed=42: fit_piecewise_universal(gal, univ_params, seed),
            k_law=0, k_pergalaxy=2,
            structural_complexity=7.0,
            law_param_indices=[],
            generation=0,
        ),
    ]
    return families


def build_g1_mutants() -> list[LawFamily]:
    """Generate mutant families for generation 1."""

    with open(ARTIFACTS_DIR / "aggregate" / "universal_params.json") as f:
        univ = json.load(f)

    log_a0 = univ["params"]["log10_a0"]
    ai = univ["params"]["alpha_inner"]
    ao = univ["params"]["alpha_outer"]
    w = univ["params"]["w"]

    mutants = []

    # M4 variants: selective parameter freezing on PiecewiseRegime
    # Freeze w (most scattered param)
    mutants.append(m4_freeze_pr_w(w))

    # Freeze alphas (moderately universal)
    mutants.append(m4_freeze_pr_alphas(ai, ao))

    # Freeze a0 + alphas, keep only w free
    mutants.append(m4_freeze_pr_a0_alphas(log_a0, ai, ao))

    # M2: Add correction to MOND
    mond_family = LawFamily(
        name="MOND-simple", fit_fn=fit_mond,
        k_law=0, k_pergalaxy=2, structural_complexity=2.0,
        law_param_indices=[], generation=0,
    )

    # MOND + acceleration correction
    m2_accel = m2_add_correction(
        mond_family, "a_N/a0",
        base_law_factory=SimpleMONDLaw,
    )
    if m2_accel:
        mutants.append(m2_accel)

    # MOND + radius correction
    m2_radius = m2_add_correction(
        mond_family, "r/R0",
        base_law_factory=SimpleMONDLaw,
    )
    if m2_radius:
        mutants.append(m2_radius)

    # M3: Newton|MOND composition via acceleration
    newton_family = LawFamily(
        name="Newton-baryons", fit_fn=fit_newtonian,
        k_law=0, k_pergalaxy=2, structural_complexity=0.0,
        law_param_indices=[], generation=0,
    )

    m3_accel = m3_compose(newton_family, mond_family, "acceleration")
    if m3_accel:
        mutants.append(m3_accel)

    return [m for m in mutants if m is not None]


def main():
    max_galaxies = int(sys.argv[1]) if len(sys.argv) > 1 else None

    # Load data
    galaxies = load_sparc(DATA_DIR)
    split = split_galaxies(galaxies)
    train = split.train
    if max_galaxies:
        train = train[:max_galaxies]

    print(f"Train set: {len(train)} galaxies")

    # Build population
    g0 = build_g0()
    g1_mutants = build_g1_mutants()

    population = g0 + g1_mutants
    print(f"\nPopulation: {len(g0)} originals + {len(g1_mutants)} mutants = {len(population)}")
    for f in population:
        gen_tag = "G0" if f.generation == 0 else "G1"
        print(f"  [{gen_tag}] {f.name:45s} k_law={f.k_law} k_pg={f.k_pergalaxy} S={f.structural_complexity}")

    # Run evaluation
    vectors = run_generation(population, train, generation_num=1, seed=42)

    print("\nDone.")


if __name__ == "__main__":
    main()
