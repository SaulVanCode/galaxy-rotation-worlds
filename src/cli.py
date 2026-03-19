"""
MetaLaw Discovery System v0.2-FROZEN — CLI Entry Point

Runs the full pipeline:
1. Load SPARC data
2. Split train/val/test
3. Fit all families to all train galaxies
4. Compute universal PiecewiseRegime parameters
5. Re-fit with universal params (Level B)
6. Produce comparison tables
7. Save artifacts
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from src.data.sparc_adapter import load_sparc
from src.data.split import split_galaxies, log_split, DataSplit
from src.fitting.per_galaxy import fit_all_families, FAMILY_FITTERS
from src.fitting.universal import fit_piecewise_universal
from src.types import FitResult


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sparc"


def _safe_float(v):
    """Convert numpy types to Python float for JSON."""
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    return v


def run_pipeline(
    families: list[str] | None = None,
    max_galaxies: int | None = None,
    seed: int = 42,
):
    """Run the full v0.2 pipeline."""

    print("=" * 70)
    print("MetaLaw Discovery System v0.2-FROZEN")
    print("=" * 70)

    # --- Step 1: Load data ---
    print("\n[1/6] Loading SPARC data...")
    galaxies = load_sparc(DATA_DIR)
    print(f"  Loaded {len(galaxies)} galaxies (Q=1,2)")

    # --- Step 2: Split ---
    print("\n[2/6] Splitting train/val/test...")
    split = split_galaxies(galaxies)
    print(f"  {split.summary()}")

    # Save split record
    split_record = log_split(split)
    (ARTIFACTS_DIR / "aggregate").mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "aggregate" / "split.json", "w") as f:
        json.dump(split_record, f, indent=2)

    # --- Step 3: Fit all families to train galaxies ---
    train_galaxies = split.train
    if max_galaxies:
        train_galaxies = train_galaxies[:max_galaxies]

    if families is None:
        families = list(FAMILY_FITTERS.keys())

    print(f"\n[3/6] Fitting {len(families)} families to {len(train_galaxies)} train galaxies...")

    all_results: dict[str, dict[str, FitResult]] = {}  # galaxy -> family -> result
    t0 = time.time()

    for i, gal in enumerate(train_galaxies):
        results = fit_all_families(gal, seed=seed, families=families)
        all_results[gal.name] = results

        # Progress
        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(train_galaxies) - i - 1) / rate
            print(f"  [{i+1}/{len(train_galaxies)}] {gal.name:15s} "
                  f"({elapsed:.0f}s elapsed, ~{eta:.0f}s remaining)")

    elapsed_total = time.time() - t0
    print(f"  Done in {elapsed_total:.1f}s")

    # --- Step 4: Universal parameters for PiecewiseRegime ---
    print("\n[4/6] Computing universal PiecewiseRegime parameters...")
    pr_results = {name: res["PiecewiseRegime"] for name, res in all_results.items()
                  if "PiecewiseRegime" in res}

    # Extract law params (indices 2-5: log10_a0, alpha_in, alpha_out, w)
    law_params = np.array([r.params[2:6] for r in pr_results.values()])
    universal_params = np.median(law_params, axis=0)
    iqr_params = np.percentile(law_params, 75, axis=0) - np.percentile(law_params, 25, axis=0)

    param_names = ["log10_a0", "alpha_inner", "alpha_outer", "w"]
    print("  Universal (median) parameters:")
    for name, val, iqr in zip(param_names, universal_params, iqr_params):
        print(f"    {name:15s} = {val:.4f}  (IQR: {iqr:.4f})")

    # --- Step 5: Re-fit with universal params (Level B) ---
    print(f"\n[5/6] Re-fitting PiecewiseRegime with universal params on {len(train_galaxies)} galaxies...")

    universal_results: dict[str, FitResult] = {}
    for gal in train_galaxies:
        result = fit_piecewise_universal(gal, universal_params, seed=seed)
        universal_results[gal.name] = result

    # --- Step 6: Comparison tables ---
    print("\n[6/6] Building comparison tables...")

    # Per-galaxy table
    per_galaxy_dir = ARTIFACTS_DIR / "per_galaxy"
    per_galaxy_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for gal_name, family_results in all_results.items():
        for fam_name, r in family_results.items():
            rows.append({
                "galaxy": gal_name,
                "family": fam_name,
                "chi2_reduced": _safe_float(r.chi2_reduced),
                "bic": _safe_float(r.bic),
                "n_params": r.n_params,
                "upsilon_disk": _safe_float(r.params[0]),
                "upsilon_bulge": _safe_float(r.params[1]),
                "smoothness": _safe_float(r.smoothness),
                "converged": r.converged,
            })
        # Add universal PiecewiseRegime
        if gal_name in universal_results:
            ur = universal_results[gal_name]
            rows.append({
                "galaxy": gal_name,
                "family": "PiecewiseRegime-universal",
                "chi2_reduced": _safe_float(ur.chi2_reduced),
                "bic": _safe_float(ur.bic),
                "n_params": ur.n_params,
                "upsilon_disk": _safe_float(ur.params[0]),
                "upsilon_bulge": _safe_float(ur.params[1]),
                "smoothness": _safe_float(ur.smoothness),
                "converged": ur.converged,
            })

    with open(per_galaxy_dir / "all_fits.json", "w") as f:
        json.dump(rows, f, indent=2)

    # Aggregate table
    print("\n" + "=" * 90)
    print(f"{'Family':28s} {'med(chi2r)':>10s} {'med(BIC)':>10s} {'Params':>6s} {'Wins':>5s} {'med(Yd)':>8s}")
    print("-" * 90)

    family_names = families + ["PiecewiseRegime-universal"]
    aggregate = {}

    for fam in family_names:
        if fam == "PiecewiseRegime-universal":
            fam_results = list(universal_results.values())
        else:
            fam_results = [all_results[g][fam] for g in all_results if fam in all_results[g]]

        if not fam_results:
            continue

        chi2s = [r.chi2_reduced for r in fam_results]
        bics = [r.bic for r in fam_results]
        yds = [r.params[0] for r in fam_results]

        aggregate[fam] = {
            "median_chi2_reduced": _safe_float(np.median(chi2s)),
            "median_bic": _safe_float(np.median(bics)),
            "iqr_bic": _safe_float(np.percentile(bics, 75) - np.percentile(bics, 25)),
            "n_galaxies": len(fam_results),
            "n_params": fam_results[0].n_params,
            "median_upsilon_disk": _safe_float(np.median(yds)),
        }

    # Count wins (lowest BIC per galaxy)
    for gal_name in all_results:
        best_bic = float("inf")
        best_fam = ""
        for fam in family_names:
            if fam == "PiecewiseRegime-universal" and gal_name in universal_results:
                bic = universal_results[gal_name].bic
            elif fam in all_results[gal_name]:
                bic = all_results[gal_name][fam].bic
            else:
                continue
            if bic < best_bic:
                best_bic = bic
                best_fam = fam
        if best_fam in aggregate:
            aggregate[best_fam]["wins"] = aggregate[best_fam].get("wins", 0) + 1

    for fam in family_names:
        if fam not in aggregate:
            continue
        a = aggregate[fam]
        print(f"{fam:28s} {a['median_chi2_reduced']:10.2f} {a['median_bic']:10.1f} "
              f"{a['n_params']:6d} {a.get('wins', 0):5d} {a['median_upsilon_disk']:8.3f}")

    print("=" * 90)

    # Generalization metric
    if "PiecewiseRegime" in aggregate and "PiecewiseRegime-universal" in aggregate:
        delta = aggregate["PiecewiseRegime-universal"]["median_bic"] - aggregate["PiecewiseRegime"]["median_bic"]
        print(f"\nGeneralization gap (universal - free): delta_BIC = {delta:.1f}")
        print(f"  (Small = parameters are universal; Large = family is over-flexible)")

    # Save universal params
    universal_record = {
        "params": {n: _safe_float(v) for n, v in zip(param_names, universal_params)},
        "iqr": {n: _safe_float(v) for n, v in zip(param_names, iqr_params)},
        "n_galaxies_used": len(pr_results),
    }

    # Save aggregate
    agg_dir = ARTIFACTS_DIR / "aggregate"
    with open(agg_dir / "aggregate_results.json", "w") as f:
        json.dump(aggregate, f, indent=2)
    with open(agg_dir / "universal_params.json", "w") as f:
        json.dump(universal_record, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    print("Done.")


if __name__ == "__main__":
    import sys

    max_gals = None
    if len(sys.argv) > 1:
        max_gals = int(sys.argv[1])

    run_pipeline(max_galaxies=max_gals)
