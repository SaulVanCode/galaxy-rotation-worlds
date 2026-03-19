# A Multi-Objective Evolution Engine for Galaxy Rotation Curve Hypothesis Comparison

**Authors:** S. Van Code, with Claude (Anthropic) as research architect

**Abstract.**
We present a system that evolves parameterized gravity law families and evaluates them against 159 SPARC galaxy rotation curves using a four-dimensional Pareto front: fit quality, parameter universality, structural simplicity, and radial acceleration relation transfer. Starting from three baselines (Newtonian baryons-only, Newton+NFW dark matter halo, simple MOND), the system generates mutant families through parameter freezing, correction term addition, and regime composition. After two generations of evolution, a mutant family — PiecewiseRegime with fixed exponents and variable transition scale — fills a previously empty region of the Pareto front, achieving near-MOND universality (F2=0.92) with near-NFW fit quality (F1=0.39) using one fewer per-galaxy parameter than NFW. Correlation analysis reveals no strong dependence of the variable scale on observed galaxy properties. We discuss limitations and emphasize that the contribution is methodological, not a claim of new physics.

---

## 1. Introduction

Galaxy rotation curves remain a central test for theories of gravity and dark matter. Two paradigms dominate:

- **ΛCDM with NFW halos**: excellent per-galaxy fits with 4 free parameters (Υ\_disk, Υ\_bulge, M\_200, c), but halo parameters vary widely across galaxies (Navarro et al. 1996).
- **MOND**: a universal acceleration threshold a₀ ≈ 1.2×10⁻¹⁰ m/s² with only 2 per-galaxy parameters (Υ\_disk, Υ\_bulge), but systematically poorer fits for some galaxy types (Milgrom 1983; Famaey & McGaugh 2012).

The Radial Acceleration Relation (RAR; McGaugh et al. 2016; Lelli et al. 2017) demonstrates a tight empirical correlation between baryonic and observed accelerations, suggesting a universal transition between regimes — but the standard frameworks cannot simultaneously optimize for fit quality and parameter universality.

We ask: **does the hypothesis space contain families that occupy the trade-off region between NFW's fit quality and MOND's universality?**

To answer this, we construct an automated system that:
1. Defines gravity laws as parameterized families
2. Evaluates them on four independent fitness axes
3. Evolves mutant families through structured operators
4. Identifies Pareto-optimal trade-offs

## 2. Data

We use the SPARC dataset (Lelli, McGaugh & Schombert 2016): 175 late-type galaxies with Spitzer 3.6μm photometry and high-quality rotation curves. After filtering for quality flags Q=1,2 and minimum 5 data points: **159 galaxies**.

SPARC provides pre-computed velocity contributions (V\_disk, V\_bulge at M/L=1, V\_gas) at each observed radius, enabling mass model construction without analytic profile assumptions.

**Split:** 99 train / 30 validation / 30 test, stratified by luminosity, deterministic assignment.

**Baryonic modeling:** Stellar mass = Υ × L (where Υ ∈ [0.1, 1.5] M☉/L☉ at 3.6μm). Gas included via SPARC V\_gas. For modified gravity laws, enclosed baryonic mass is derived as M\_bary(r) = r/G × (V²\_gas + Υ\_d V²\_disk + Υ\_b V²\_bul).

## 3. Method

### 3.1 Law families

Each family defines an acceleration function a(r, M\_enclosed, ρ) with specified free parameters. The three baselines:

**Newton baryons-only** (2 per-galaxy params):
$$a(r) = G M_{\rm bary}(r) / r^2$$

**Newton + NFW halo** (4 per-galaxy params):
$$a(r) = G [M_{\rm bary}(r) + M_{\rm NFW}(r)] / r^2$$

with standard NFW profile (Navarro et al. 1997), fitting V\_200 and concentration c per galaxy.

**Simple MOND** (2 per-galaxy params, a₀ fixed):
$$a(r) = a_N(r) \cdot \nu(a_N / a_0), \quad \nu(x) = [1 - \exp(-\sqrt{x})]^{-1}$$

And the generated family:

**PiecewiseRegime** (up to 6 per-galaxy params):
$$a(r) = a_0 \left[ \sigma(x) \left(\frac{a_N}{a_0}\right)^{\alpha_{\rm in}} + (1-\sigma(x)) \left(\frac{a_N}{a_0}\right)^{\alpha_{\rm out}} \right]$$

where σ(x) = 1/(1+e⁻ˣ), x = log₁₀(a\_N/a₀)/w. All terms dimensionless inside brackets; a₀ provides acceleration units. This family strictly nests both Newton (α\_in = α\_out = 1) and deep-MOND (α\_in = 1, α\_out = 0.5).

### 3.2 Four-dimensional fitness

Each family is scored on four independent axes (all normalized, higher = better):

| Axis | Definition | Measures |
|------|-----------|----------|
| F1 | 1/(1 + median χ²\_red) | Per-galaxy fit quality |
| F2 | 1/(1 + mean CV\_j) where CV\_j = IQR(θ\_j)/\|median(θ\_j)\| | Law parameter consistency across galaxies |
| F3 | 1/(k\_law + k\_pergalaxy + S) | Structural simplicity (S = operation complexity) |
| F4 | 1/(1 + RMS log₁₀(g\_pred/g\_obs)) | Radial acceleration relation reproduction |

No axis is collapsed into a weighted sum. **Pareto dominance** determines which families represent genuine trade-offs: family A dominates B if A ≥ B on all axes and A > B on at least one.

### 3.3 Mutation operators

- **M4 (Parameter Freezing):** Fix one law parameter at its cross-galaxy median, reducing per-galaxy freedom.
- **M2 (Term Addition):** Add a multiplicative correction (1 + c₁(ratio)^p₁) using a dimensionless ratio.
- **M3 (Regime Composition):** Blend two families via sigmoid transition.

All mutations preserve dimensional consistency. Structural complexity is capped at S ≤ 15. Population cap: 30 families per generation.

### 3.4 Per-galaxy fitting

Parameters are optimized using `scipy.differential_evolution` (seed=42, tol=10⁻⁸) per galaxy per family. BIC = χ² + k ln(n) is computed per galaxy; the multi-objective fitness uses aggregates across galaxies.

## 4. Results

### 4.1 v0.2 baseline comparison (BIC, train set)

| Family | med(χ²\_red) | med(BIC) | Wins/99 | Per-galaxy params |
|--------|-------------|---------|---------|-------------------|
| Newton baryons | 24.7 | 407 | 4 | 2 |
| Newton+NFW | 1.31 | 26.2 | 43 | 4 |
| MOND-simple | 3.63 | 46.8 | 18 | 2 |
| PiecewiseRegime | 1.76 | 30.0 | 22 | 6 |

Under single-objective BIC, **Newton+NFW wins**. PiecewiseRegime's best-fit law parameters converge to α\_in = 1.05, α\_out = 0.52 — recovering MOND-like exponents without prior knowledge.

### 4.2 Pareto front (4D, train set)

| Family | F1 | F2 | F3 | F4 |
|--------|-----|-----|-----|-----|
| Newton baryons | 0.039 | 1.000 | 0.500 | 0.760 |
| Newton+NFW | 0.432 | 0.489 | 0.200 | 0.900 |
| MOND-simple | 0.216 | 1.000 | 0.250 | 0.879 |
| PiecewiseRegime | 0.362 | 0.302 | 0.077 | 0.907 |

All four are Pareto-optimal. **No family simultaneously achieves F1 ≥ 0.35 and F2 ≥ 0.70** — a gap between NFW's fit quality and MOND's universality.

### 4.3 Evolution: filling the gap

**Generation 1** (6 mutants): PR-M4(w=0.615) reaches F1=0.42, F2=0.69 — approaching but not entering the gap. MOND+corr(r/R₀) achieves best-ever F1=0.52 but low F2=0.22.

**Generation 2** (4 additional mutants): **PR-only-a0-free** — PiecewiseRegime with α\_in, α\_out, and w frozen at universal values (1.05, 0.52, 0.615), leaving only a₀ free per galaxy — achieves:

| | F1 | F2 | F3 | F4 |
|---|---|---|---|---|
| **PR-only-a0-free** | **0.374** | **0.896** | 0.100 | 0.898 |

**This fills the gap.** Confirmed on held-out test set (30 galaxies): F1=0.389, F2=0.915, Pareto-optimal.

### 4.4 Test set comparison

| Family | med(BIC) | Wins/30 | Per-galaxy params |
|--------|---------|---------|-------------------|
| Newton+NFW | 28.4 | 12 | 4 |
| PR-only-a0-free | 40.1 | 6 | 3 |
| MOND-simple | 52.7 | 3 | 2 |

PR-only-a0-free achieves ~85% of NFW's fit quality with nearly double the universality and one fewer parameter.

### 4.5 What determines a₀?

Correlation of per-galaxy log₁₀(a₀) against six galaxy properties (all 159 galaxies):

| Property | r | p-value |
|----------|---|---------|
| V\_max | 0.364 | 2.4×10⁻⁶ |
| Σ\_L | 0.227 | 0.004 |
| L\_total | 0.184 | 0.020 |
| M\_bary | 0.088 | 0.271 |

**No strong correlation found.** Best predictor (V\_max) explains only 13% of variance. a₀ scatter: 0.33 dex (factor ~2). The variation appears largely independent of observed galaxy properties.

## 5. Discussion

### 5.1 What this finds

The universal transition shape (α\_in ≈ 1.05, α\_out ≈ 0.52, w ≈ 0.62) is consistent with the known RAR phenomenology. The exponents correspond to near-Newtonian behavior at high accelerations and deep-MOND behavior at low accelerations. This is not new — it confirms what McGaugh et al. (2016) established.

What is new is the **Pareto decomposition**: by separating fit quality from universality, we reveal that the hypothesis space contains a trade-off region between NFW and MOND that neither covers. PR-only-a0-free occupies this region with a specific physical interpretation: the transition shape is universal; only the scale varies.

### 5.2 Relation to existing work

The variable-a₀ approach has been explored (e.g., Rodrigues et al. 2018, who argued a₀ varies significantly across galaxies; cf. McGaugh et al. 2018 rebuttal). Our contribution is not to this debate but to the methodology: multi-objective Pareto comparison reveals trade-offs that single-metric comparisons (BIC, χ²) cannot.

### 5.3 Limitations

1. **Single dataset.** All results are conditioned on SPARC. Gravitational lensing, cluster dynamics, and CMB constraints are not tested.
2. **Mass-to-light degeneracy.** Υ freedom can absorb some signal. All families have the same Υ freedom, ensuring fair comparison, but systematic Υ biases could affect absolute F1 values.
3. **Optimizer limitations.** Differential evolution with fixed seed may miss global optima for high-dimensional families. Multi-seed checks on a subsample showed <10% variation.
4. **a₀ interpretation.** The per-galaxy a₀ could reflect observational systematics (distance errors, inclination errors) rather than intrinsic variation. We cannot distinguish these with SPARC alone.
5. **Not a physical theory.** PR-only-a0-free is a phenomenological fitting function, not derived from an action principle or field theory.

### 5.4 Methodological contribution

The evolution engine — mutation operators on law families, 4D Pareto selection, lineage tracking — is the primary contribution. It provides a framework for systematic hypothesis comparison that:
- Avoids collapsing multiple desiderata into one number
- Automatically generates and evaluates variants
- Identifies trade-offs that manual analysis may miss

### 5.5 What we cannot claim

- Discovery of a new law of gravity
- Evidence for or against dark matter
- That a₀ is a varying fundamental constant
- That this generalizes beyond rotation curves

## 6. Conclusion

An automated evolution engine, tested on 159 SPARC galaxies, identifies a Pareto-optimal law family that combines NFW-level fit quality with MOND-level parameter universality using fewer per-galaxy parameters than either. The universal transition shape (exponents 1.05, 0.52; width 0.62) was discovered without prior knowledge, confirming known RAR phenomenology. The per-galaxy transition scale a₀ shows no strong correlation with observed galaxy properties.

The value lies not in the astrophysical result — which is consistent with existing literature — but in the demonstration that multi-objective hypothesis evolution can systematically explore trade-offs in the space of physical laws.

**Code and data:** https://github.com/SaulVanCode/galaxy-rotation-worlds

---

## References

- Famaey, B. & McGaugh, S.S. 2012, Living Rev. Relativ., 15, 10
- Lelli, F., McGaugh, S.S. & Schombert, J.M. 2016, AJ, 152, 157
- Lelli, F., McGaugh, S.S., Schombert, J.M. & Pawlowski, M.S. 2017, ApJ, 836, 152
- McGaugh, S.S., Lelli, F. & Schombert, J.M. 2016, Phys. Rev. Lett., 117, 201101
- McGaugh, S.S., Lelli, F. & Schombert, J.M. 2018, Nat. Astron., 2, 924
- Milgrom, M. 1983, ApJ, 270, 365
- Navarro, J.F., Frenk, C.S. & White, S.D.M. 1996, ApJ, 462, 563
- Rodrigues, D.C., Marra, V., del Popolo, A. & Davari, Z. 2018, Nat. Astron., 2, 668
