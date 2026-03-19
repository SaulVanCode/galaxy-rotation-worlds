"""
MetaLaw Discovery System v0.2-FROZEN — Final Test Set Evaluation

This script runs ONCE on the held-out test set (30 galaxies).
Results are the final reported numbers for v0.2.

Protocol:
1. Load universal PiecewiseRegime params from train results
2. Fit all 5 variants (4 families + universal) to each test galaxy
3. Produce comparison table
4. Save final report artifact
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np

from src.data.sparc_adapter import load_sparc
from src.data.split import split_galaxies
from src.fitting.per_galaxy import fit_all_families
from src.fitting.universal import fit_piecewise_universal
from src.types import FitResult


ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "sparc"


def _sf(v):
    """Safe float for JSON."""
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return str(v)
    return v


def run_test_evaluation():
    print("=" * 70)
    print("FINAL TEST SET EVALUATION (one-time, no re-runs)")
    print("=" * 70)

    # Load data and split (deterministic — same split as training)
    galaxies = load_sparc(DATA_DIR)
    split = split_galaxies(galaxies)
    test_galaxies = split.test

    print(f"\nTest set: {len(test_galaxies)} galaxies")
    print(f"Names: {', '.join(g.name for g in test_galaxies[:5])}...")

    # Load universal params from training
    univ_path = ARTIFACTS_DIR / "aggregate" / "universal_params.json"
    assert univ_path.exists(), f"Run training pipeline first! Missing: {univ_path}"
    with open(univ_path) as f:
        univ_record = json.load(f)
    universal_params = np.array([
        univ_record["params"]["log10_a0"],
        univ_record["params"]["alpha_inner"],
        univ_record["params"]["alpha_outer"],
        univ_record["params"]["w"],
    ])
    print(f"\nUniversal PiecewiseRegime params (from training):")
    for k, v in univ_record["params"].items():
        print(f"  {k:15s} = {v:.4f}")

    # Fit all families to test set
    print(f"\nFitting 5 variants to {len(test_galaxies)} test galaxies...")
    seed = 42

    all_results: dict[str, dict[str, FitResult]] = {}
    universal_results: dict[str, FitResult] = {}

    t0 = time.time()
    for i, gal in enumerate(test_galaxies):
        results = fit_all_families(gal, seed=seed)
        all_results[gal.name] = results
        universal_results[gal.name] = fit_piecewise_universal(gal, universal_params, seed=seed)

        if (i + 1) % 5 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f"  [{i+1}/{len(test_galaxies)}] {gal.name}")

    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    # Build per-galaxy comparison table
    families = ["Newton-baryons", "Newton+NFW", "MOND-simple", "PiecewiseRegime", "PiecewiseRegime-universal"]

    print("\n" + "=" * 100)
    print(f"{'Galaxy':15s} {'Newton':>8s} {'NFW':>8s} {'MOND':>8s} {'PR-free':>8s} {'PR-univ':>8s} {'Best':>18s}")
    print("-" * 100)

    per_galaxy_rows = []
    wins = {f: 0 for f in families}

    for gal in test_galaxies:
        bics = {}
        for fam in families:
            if fam == "PiecewiseRegime-universal":
                r = universal_results[gal.name]
            else:
                r = all_results[gal.name][fam]
            bics[fam] = r.bic

        best_fam = min(bics, key=bics.get)
        wins[best_fam] += 1

        print(f"{gal.name:15s} "
              f"{bics['Newton-baryons']:8.1f} "
              f"{bics['Newton+NFW']:8.1f} "
              f"{bics['MOND-simple']:8.1f} "
              f"{bics['PiecewiseRegime']:8.1f} "
              f"{bics['PiecewiseRegime-universal']:8.1f} "
              f"{'<-- ' + best_fam:>18s}")

        # Collect row data
        row = {"galaxy": gal.name, "n_points": gal.n_points}
        for fam in families:
            if fam == "PiecewiseRegime-universal":
                r = universal_results[gal.name]
            else:
                r = all_results[gal.name][fam]
            row[f"{fam}_bic"] = _sf(r.bic)
            row[f"{fam}_chi2r"] = _sf(r.chi2_reduced)
            row[f"{fam}_Yd"] = _sf(r.params[0])
            row[f"{fam}_Yb"] = _sf(r.params[1])
        row["best"] = best_fam
        per_galaxy_rows.append(row)

    # Aggregate
    print("=" * 100)
    print(f"\n{'AGGREGATE':15s} {'med(BIC)':>10s} {'med(chi2r)':>10s} {'Wins':>5s} {'Params':>6s} {'med(Yd)':>8s}")
    print("-" * 65)

    aggregate = {}
    for fam in families:
        if fam == "PiecewiseRegime-universal":
            results_list = list(universal_results.values())
        else:
            results_list = [all_results[g.name][fam] for g in test_galaxies]

        bics = [r.bic for r in results_list]
        chi2s = [r.chi2_reduced for r in results_list]
        yds = [r.params[0] for r in results_list]

        agg = {
            "median_bic": _sf(np.median(bics)),
            "iqr_bic": _sf(np.percentile(bics, 75) - np.percentile(bics, 25)),
            "median_chi2_reduced": _sf(np.median(chi2s)),
            "n_params": results_list[0].n_params,
            "wins": wins[fam],
            "median_upsilon_disk": _sf(np.median(yds)),
            "n_galaxies": len(results_list),
        }
        aggregate[fam] = agg

        print(f"{fam:28s} {agg['median_bic']:10.1f} {agg['median_chi2_reduced']:10.2f} "
              f"{agg['wins']:5d} {agg['n_params']:6d} {agg['median_upsilon_disk']:8.3f}")

    print("=" * 65)

    # Generalization gap
    if "PiecewiseRegime" in aggregate and "PiecewiseRegime-universal" in aggregate:
        delta = aggregate["PiecewiseRegime-universal"]["median_bic"] - aggregate["PiecewiseRegime"]["median_bic"]
        print(f"\nGeneralization gap: delta_BIC = {delta:.1f}")

    # "Interesting" threshold check
    pr_univ_bic = aggregate["PiecewiseRegime-universal"]["median_bic"]
    nfw_bic = aggregate["Newton+NFW"]["median_bic"]
    nfw_iqr = aggregate["Newton+NFW"]["iqr_bic"]
    n = len(test_galaxies)
    threshold = 2 * nfw_iqr / math.sqrt(n)

    print(f"\n--- PRE-REGISTERED THRESHOLD CHECK ---")
    print(f"Condition 1: PiecewiseRegime beats ALL baselines in median BIC?")
    pr_free_bic = aggregate["PiecewiseRegime"]["median_bic"]
    beats_all = (
        pr_free_bic < aggregate["Newton-baryons"]["median_bic"]
        and pr_free_bic < nfw_bic
        and pr_free_bic < aggregate["MOND-simple"]["median_bic"]
    )
    print(f"  PR-free median BIC = {pr_free_bic:.1f}")
    print(f"  Newton = {aggregate['Newton-baryons']['median_bic']:.1f}, "
          f"NFW = {nfw_bic:.1f}, MOND = {aggregate['MOND-simple']['median_bic']:.1f}")
    print(f"  Result: {'YES' if beats_all else 'NO'}")

    print(f"\nCondition 2: PR-universal comparable to NFW?")
    print(f"  |PR-univ - NFW| = {abs(pr_univ_bic - nfw_bic):.1f}")
    print(f"  Threshold (2*IQR_NFW/sqrt(N)) = {threshold:.1f}")
    comparable = abs(pr_univ_bic - nfw_bic) < threshold
    print(f"  Result: {'YES (comparable)' if comparable else 'NO (not comparable)'}")

    if beats_all:
        conclusion = "INTERESTING: PiecewiseRegime outperforms all baselines."
    elif comparable:
        conclusion = "INTERESTING: PR-universal matches NFW with fewer per-galaxy parameters."
    else:
        conclusion = "NOT INTERESTING: Generated family does not outperform baselines on this dataset."

    print(f"\n{'='*70}")
    print(f"CONCLUSION: {conclusion}")
    print(f"{'='*70}")

    # Save final report
    report = {
        "version": "v0.2-FROZEN",
        "dataset": "SPARC",
        "n_test_galaxies": len(test_galaxies),
        "test_galaxy_names": [g.name for g in test_galaxies],
        "universal_params": univ_record,
        "aggregate": aggregate,
        "per_galaxy": per_galaxy_rows,
        "threshold_check": {
            "condition_1_beats_all": beats_all,
            "condition_2_comparable": comparable,
            "nfw_iqr": _sf(nfw_iqr),
            "threshold": _sf(threshold),
            "pr_univ_minus_nfw": _sf(abs(pr_univ_bic - nfw_bic)),
        },
        "conclusion": conclusion,
        "seed": seed,
    }

    report_path = ARTIFACTS_DIR / "v02_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nFinal report: {report_path}")


if __name__ == "__main__":
    run_test_evaluation()
