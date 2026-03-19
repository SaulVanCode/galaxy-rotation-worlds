# Galaxy Rotation Worlds

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19104088.svg)](https://doi.org/10.5281/zenodo.19104088)

**An evolution engine that breeds physical laws and tests them against real galaxy data.**

Built on the [SPARC dataset](http://astroweb.cwru.edu/SPARC/) (159 galaxies). Evaluates law families not by a single metric, but by a 4D Pareto front: fit quality, parameter universality, structural simplicity, and predictive transfer.

## What it does

Starts with three known explanations for galaxy rotation curves:
- **Newtonian gravity + dark matter halo (NFW)**
- **MOND** (Modified Newtonian Dynamics)
- **Newtonian baryons only** (null hypothesis)

Then introduces a **parameterized rule-generator** (PiecewiseRegime) that smoothly transitions between gravity regimes, and evolves mutations of all families across generations — freezing parameters, adding correction terms, composing regimes — to explore the hypothesis space.

## Key finding

A mutant family (**PR-only-a0-free**) fills a gap in the Pareto front that no baseline covers:

```
                F1(fit)  F2(universality)  Per-galaxy params
Newton+NFW       0.520       0.479              4
PR-only-a0-free  0.389       0.915              3          ← gap candidate
MOND             0.261       1.000              2
```

The law:

```
a(r) = a₀ × [σ(log(a_N/a₀)/0.615) × (a_N/a₀)^1.05 + (1-σ) × (a_N/a₀)^0.52]
```

The exponents (1.05, 0.52) and transition width (0.615) are **universal across all 159 galaxies**. Only the transition scale (a₀) varies per galaxy. This achieves nearly double the universality of dark matter halos with one fewer parameter.

Correlation analysis of a₀ against galaxy properties (mass, size, velocity, surface density) found **no strong driver** (best r=0.36 with V_max). The variation appears intrinsic.

## What this is NOT

- Not a discovery of new physics
- Not a proof that dark matter doesn't exist
- Not a replacement for MOND or ΛCDM

It's a demonstration that **an automated hypothesis evolution system** can rediscover known phenomenology from raw data and identify trade-offs that fixed baselines leave uncovered.

## Results timeline

| Version | Result |
|---------|--------|
| v0.2 | NFW wins on BIC. Honest null result. PiecewiseRegime independently finds MOND-like exponents. |
| v0.3 Gen 1 | 6 mutants generated. PR-M4(w frozen) approaches the Pareto gap (F1=0.42, F2=0.69). |
| v0.3 Gen 2 | **PR-only-a0-free enters the gap** (F1=0.37, F2=0.90). Confirmed on held-out test set. |
| a₀ analysis | No strong correlation with galaxy properties. a₀ behaves as a mildly varying constant. |

## Architecture

```
src/
├── constants.py           # G, a₀ in kpc/(km/s)²/M☉ units (single source)
├── laws/                  # Gravity law implementations
│   ├── newtonian.py       # Baseline: G*M/r²
│   ├── newtonian_nfw.py   # Baseline: baryons + NFW dark matter halo
│   ├── mond_simple.py     # Baseline: MOND with fixed a₀
│   └── piecewise_regime.py # Generated: smooth two-regime transition
├── mass_model.py          # Exponential disk + Hernquist bulge
├── simulator.py           # v(r) = √(r·a(r)) with smoothness filter
├── data/
│   ├── sparc_adapter.py   # Loads SPARC MRT files, validates units
│   └── split.py           # Deterministic 60/20/20 stratified split
├── fitting/
│   ├── per_galaxy.py      # Level A: differential_evolution per galaxy
│   └── universal.py       # Level B: fix law params, fit only Υ
├── scoring/
│   ├── bic.py             # BIC = χ² + k·ln(n)
│   └── multi_objective.py # F1(fit), F2(universality), F3(simplicity), F4(RAR)
├── evolution/
│   ├── population.py      # LawFamily dataclass + registry
│   ├── mutations.py       # M2(term addition), M3(composition), M4(param freezing)
│   └── generation.py      # Evaluate population, compute Pareto front
├── cli.py                 # v0.2 full pipeline
├── evolve.py              # v0.3 generation 1
├── evolve_g2.py           # v0.3 generation 2
├── test_gap_candidate.py  # Test set evaluation
└── a0_correlation.py      # a₀ vs galaxy properties
```

## Running

```bash
pip install -e ".[dev]"

# Run tests (78 passing)
pytest tests/ -v

# v0.2 pipeline (all 4 baselines on 99 train galaxies)
python -m src.cli

# v0.3 evolution (generation 1)
python -m src.evolve

# v0.3 generation 2 (produces gap candidate)
python -m src.evolve_g2

# Test set evaluation
python -m src.test_gap_candidate

# a₀ correlation analysis
python -m src.a0_correlation
```

## Data

Uses [SPARC](http://astroweb.cwru.edu/SPARC/) (Lelli, McGaugh & Schombert 2016). 175 galaxies with Spitzer 3.6μm photometry and accurate rotation curves. After quality filtering (Q=1,2): 159 galaxies.

## Unit system

All internal computation uses: kpc, km/s, M☉. G = 4.3009×10⁻⁶ kpc·(km/s)²/M☉.

A 1000× error in G was caught during development (pc vs kpc confusion) — only detected when predictions were compared against real SPARC data, not by unit tests. This is documented in the commit history as a lesson in why closed-loop validation against reality matters.

## Built with

Claude (Anthropic) as research architect. All code, analysis, and scientific discipline decisions made in conversation. The full design→implement→test→analyze loop is preserved in the commit history.
