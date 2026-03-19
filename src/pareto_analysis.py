"""
MetaLaw Discovery System v0.3 — Pareto Front Analysis

Computes 4D fitness for all v0.2 families and identifies the Pareto front.
Uses train set results, saves full parameter vectors for F2 computation.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.data.sparc_adapter import load_sparc
from src.data.split import split_galaxies
from src.fitting.per_galaxy import fit_all_families
from src.fitting.universal import fit_piecewise_universal
from src.scoring.multi_objective import (
    compute_fitness,
    pareto_front,
    FitnessVector,
)
from src.types import FitResult, GalaxyData


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sparc"


def _sf(v):
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def run_pareto_analysis(max_galaxies: int | None = None):
    print("=" * 70)
    print("PARETO FRONT ANALYSIS (v0.3 Phase 1)")
    print("=" * 70)

    # Load and split
    galaxies = load_sparc(DATA_DIR)
    split = split_galaxies(galaxies)
    train = split.train
    if max_galaxies:
        train = train[:max_galaxies]

    print(f"\nTrain set: {len(train)} galaxies")

    # Load universal params
    with open(ARTIFACTS_DIR / "aggregate" / "universal_params.json") as f:
        univ = json.load(f)
    universal_params = np.array([
        univ["params"]["log10_a0"],
        univ["params"]["alpha_inner"],
        univ["params"]["alpha_outer"],
        univ["params"]["w"],
    ])

    # Fit all families, capturing full results
    print(f"\nFitting all families (capturing full param vectors)...")
    t0 = time.time()

    # Collect per-family results
    family_results: dict[str, list[FitResult]] = {
        "Newton-baryons": [],
        "Newton+NFW": [],
        "MOND-simple": [],
        "PiecewiseRegime": [],
        "PiecewiseRegime-universal": [],
    }

    for i, gal in enumerate(train):
        results = fit_all_families(gal, seed=42)
        for fam_name, r in results.items():
            family_results[fam_name].append(r)

        ur = fit_piecewise_universal(gal, universal_params, seed=42)
        family_results["PiecewiseRegime-universal"].append(ur)

        if (i + 1) % 20 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(train) - i - 1) / rate
            print(f"  [{i+1}/{len(train)}] ({elapsed:.0f}s, ~{eta:.0f}s remaining)")

    print(f"  Done in {time.time()-t0:.1f}s")

    # Law parameter indices for F2
    # These are the indices into FitResult.params that correspond to LAW parameters
    # (not upsilon values, which are always indices 0,1)
    law_param_indices = {
        "Newton-baryons": [],      # no law params
        "Newton+NFW": [2, 3],      # log10_M200, c (per-galaxy, but still law params)
        "MOND-simple": [],          # a0 is fixed
        "PiecewiseRegime": [2, 3, 4, 5],  # log10_a0, alpha_in, alpha_out, w
        "PiecewiseRegime-universal": [],   # all law params frozen
    }

    # Compute 4D fitness
    print("\nComputing 4D fitness vectors...")
    vectors: list[FitnessVector] = []

    for fam_name, results in family_results.items():
        fv = compute_fitness(
            family_name=fam_name,
            fit_results=results,
            galaxies=train,
            law_param_indices=law_param_indices[fam_name],
        )
        vectors.append(fv)
        print(f"  {fam_name:28s}  F1={fv.f1:.4f}  F2={fv.f2:.4f}  F3={fv.f3:.4f}  F4={fv.f4:.4f}")

    # Pareto front
    front = pareto_front(vectors)
    print(f"\nPareto front: {len(front)} families")
    for fv in front:
        print(f"  * {fv.family_name}")

    # Dominated families
    front_names = {fv.family_name for fv in front}
    for fv in vectors:
        if fv.family_name not in front_names:
            # Find who dominates it
            dominators = [o.family_name for o in vectors if o.dominates(fv)]
            print(f"  x {fv.family_name} (dominated by: {', '.join(dominators)})")

    # Per-parameter universality breakdown for PiecewiseRegime
    print("\n--- PiecewiseRegime per-parameter universality ---")
    pr_results = family_results["PiecewiseRegime"]
    param_names = ["upsilon_disk", "upsilon_bulge", "log10_a0", "alpha_inner", "alpha_outer", "w"]
    for j, pname in enumerate(param_names):
        values = np.array([r.params[j] for r in pr_results])
        med = np.median(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        cv = iqr / abs(med) if abs(med) > 1e-10 else float("inf")
        print(f"  {pname:15s}  median={med:8.4f}  IQR={iqr:8.4f}  CV={cv:.4f}")

    # NFW halo parameter distribution
    print("\n--- Newton+NFW per-parameter distribution ---")
    nfw_results = family_results["Newton+NFW"]
    nfw_param_names = ["upsilon_disk", "upsilon_bulge", "log10_M200", "c"]
    for j, pname in enumerate(nfw_param_names):
        values = np.array([r.params[j] for r in nfw_results])
        med = np.median(values)
        iqr = np.percentile(values, 75) - np.percentile(values, 25)
        cv = iqr / abs(med) if abs(med) > 1e-10 else float("inf")
        print(f"  {pname:15s}  median={med:8.4f}  IQR={iqr:8.4f}  CV={cv:.4f}")

    # Pairwise dominance matrix
    print("\n--- Pairwise dominance ---")
    names = [v.family_name for v in vectors]
    for i, vi in enumerate(vectors):
        for j, vj in enumerate(vectors):
            if i != j and vi.dominates(vj):
                print(f"  {vi.family_name} dominates {vj.family_name}")

    # Save results
    output = {
        "fitness_vectors": [fv.to_dict() for fv in vectors],
        "pareto_front": [fv.family_name for fv in front],
        "dominated": {
            fv.family_name: [o.family_name for o in vectors if o.dominates(fv)]
            for fv in vectors if fv.family_name not in front_names
        },
        "n_train_galaxies": len(train),
    }

    out_path = ARTIFACTS_DIR / "aggregate" / "pareto_analysis.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")

    # Print the geometry insight
    print("\n" + "=" * 70)
    print("PARETO GEOMETRY SUMMARY")
    print("=" * 70)
    print("\n  Axis meanings (all higher = better):")
    print("    F1: fit quality     — how well it matches v(r)")
    print("    F2: universality    — same law params across galaxies")
    print("    F3: simplicity      — fewer params + simpler formula")
    print("    F4: RAR transfer    — predicts acceleration relation")

    print("\n  Trade-off structure:")
    # Find which pairs of Pareto families trade off on which axes
    for i, a in enumerate(front):
        for j, b in enumerate(front):
            if i >= j:
                continue
            better_a = []
            better_b = []
            axes = [("F1", a.f1, b.f1), ("F2", a.f2, b.f2),
                    ("F3", a.f3, b.f3), ("F4", a.f4, b.f4)]
            for name, va, vb in axes:
                if va > vb + 0.01:
                    better_a.append(name)
                elif vb > va + 0.01:
                    better_b.append(name)
            if better_a and better_b:
                print(f"    {a.family_name} vs {b.family_name}:")
                print(f"      {a.family_name} wins on {', '.join(better_a)}")
                print(f"      {b.family_name} wins on {', '.join(better_b)}")


if __name__ == "__main__":
    import sys
    max_gals = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_pareto_analysis(max_gals)
