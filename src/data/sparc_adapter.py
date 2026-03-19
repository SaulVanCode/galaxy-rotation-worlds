"""
MetaLaw Discovery System v0.2-FROZEN — SPARC Data Adapter

Single point of entry for all external data. Reads SPARC fixed-width files,
validates units, and outputs GalaxyData in internal units.

SPARC data files:
    SPARC_Lelli2016c.mrt — Galaxy properties (175 galaxies)
    MassModels_Lelli2016c.mrt — Rotation curves + mass model contributions

Units in SPARC (already match internal units for key quantities):
    Radii: kpc ✓
    Velocities: km/s ✓
    Luminosities: 10^9 L_sun (must multiply by 1e9 for L_sun)
    Distance: Mpc
    Inclination: degrees

IMPORTANT: SPARC provides Vdisk and Vbul at M/L = 1 M_sun/L_sun.
For a given mass-to-light ratio Υ:
    V_component = sqrt(Υ) * V_at_ML1

For modified gravity laws, we derive enclosed baryonic mass from
velocity contributions:
    M_bary(r) = r/G * (V²_gas + Υ_d * V²_disk + Υ_b * V²_bul)

This is more accurate than analytic profiles because SPARC already
accounts for non-spherical geometry in the mass decomposition.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.types import GalaxyData


# Default data directory
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "sparc"

# Header lines to skip in each file
_PROPERTIES_HEADER_LINES = 98   # Lines before data in SPARC_Lelli2016c.mrt
_MASSMODELS_HEADER_LINES = 25   # Lines before data in MassModels_Lelli2016c.mrt


def _parse_galaxy_properties(filepath: Path) -> dict[str, dict]:
    """
    Parse SPARC_Lelli2016c.mrt (galaxy properties table).

    Whitespace-delimited format (19 columns per line):
        0:  Galaxy name
        1:  Hubble type
        2:  Distance [Mpc]
        3:  e_D [Mpc]
        4:  Distance method
        5:  Inclination [deg]
        6:  e_Inc [deg]
        7:  L[3.6] [10^9 L_sun]
        8:  e_L[3.6]
        9:  Reff [kpc]
        10: SBeff [L_sun/pc^2]
        11: Rdisk [kpc]
        12: SBdisk [L_sun/pc^2]
        13: MHI [10^9 M_sun]
        14: RHI [kpc]
        15: Vflat [km/s]
        16: e_Vflat [km/s]
        17: Quality flag (1=high, 2=medium, 3=low)
        18: References
    """
    galaxies: dict[str, dict] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[_PROPERTIES_HEADER_LINES:]:
        line = line.strip()
        if not line or line.startswith("-"):
            continue

        parts = line.split()
        if len(parts) < 18:
            continue

        try:
            name = parts[0]
            props = {
                "name": name,
                "hubble_type": int(parts[1]),
                "distance": float(parts[2]),              # Mpc
                "e_distance": float(parts[3]),
                "inclination": float(parts[5]),            # deg
                "e_inclination": float(parts[6]),
                # Luminosities: file gives 10^9 L_sun, convert to L_sun
                "luminosity_total": float(parts[7]) * 1e9,  # L_sun
                "e_luminosity": float(parts[8]) * 1e9,
                "R_eff": float(parts[9]),                  # kpc
                "SB_eff": float(parts[10]),                # L_sun/pc^2
                "R_disk": float(parts[11]),                # kpc
                "SB_disk": float(parts[12]),               # L_sun/pc^2
                "M_HI": float(parts[13]) * 1e9,           # M_sun
                "R_HI": float(parts[14]),                  # kpc
                "V_flat": float(parts[15]),                # km/s
                "e_Vflat": float(parts[16]),
                "quality": int(parts[17]),
            }
            galaxies[name] = props
        except (ValueError, IndexError):
            continue  # skip malformed lines

    return galaxies


def _parse_rotation_curves(filepath: Path) -> dict[str, dict[str, list]]:
    """
    Parse MassModels_Lelli2016c.mrt (rotation curves + mass models).

    Fixed-width format (0-indexed byte positions):
        0:11   Galaxy ID
        12:18  Distance [Mpc]
        19:25  Radius [kpc]
        26:32  Vobs [km/s]
        33:38  e_Vobs [km/s]
        39:45  Vgas [km/s]
        46:52  Vdisk [km/s] (at M/L=1)
        53:59  Vbul [km/s] (at M/L=1)
        60:67  SBdisk [L_sun/pc^2]
        68:76  SBbul [L_sun/pc^2]
    """
    curves: dict[str, dict[str, list]] = {}

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[_MASSMODELS_HEADER_LINES:]:
        if not line.strip() or line.startswith("-"):
            continue

        name = line[0:11].strip()
        if not name:
            continue

        try:
            r = float(line[19:25].strip())
            v_obs = float(line[26:32].strip())
            e_v = float(line[33:38].strip())
            v_gas = float(line[39:45].strip())
            v_disk = float(line[46:52].strip())
            v_bul = float(line[53:59].strip())
        except (ValueError, IndexError):
            continue

        if name not in curves:
            curves[name] = {
                "r": [], "v_obs": [], "e_v": [],
                "v_gas": [], "v_disk": [], "v_bul": [],
            }

        curves[name]["r"].append(r)
        curves[name]["v_obs"].append(v_obs)
        curves[name]["e_v"].append(e_v)
        curves[name]["v_gas"].append(v_gas)
        curves[name]["v_disk"].append(v_disk)
        curves[name]["v_bul"].append(v_bul)

    return curves


def _estimate_bulge_scale_length(galaxy_props: dict) -> float:
    """
    Estimate bulge scale length from effective radius.

    SPARC doesn't provide a separate bulge scale length.
    For Hernquist profile: R_eff ≈ 1.815 * a (scale radius).
    We use R_eff / 4 as a rough bulge scale (bulge is compact).

    Returns 0 if galaxy has no significant bulge.
    """
    # Check if galaxy has bulge: Hubble type <= 4 (Sa-Sbc) typically have bulges
    # But more reliable: check if Vbul is non-zero in the data
    # We'll set this to R_disk/4 as a rough estimate for galaxies with bulges
    # This is only used for the analytic mass model (tests); real fitting uses
    # SPARC's pre-computed Vbul directly.
    r_disk = galaxy_props.get("R_disk", 1.0)
    return r_disk / 4.0


def load_sparc(
    data_dir: Path | str | None = None,
    quality_filter: tuple[int, ...] = (1, 2),
    min_points: int = 5,
) -> list[GalaxyData]:
    """
    Load SPARC dataset and return list of GalaxyData objects.

    Args:
        data_dir: path to directory containing SPARC .mrt files
        quality_filter: only include galaxies with these quality flags
        min_points: minimum number of rotation curve points

    Returns:
        List of GalaxyData sorted by galaxy name (for deterministic ordering)
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    props_file = data_dir / "SPARC_Lelli2016c.mrt"
    curves_file = data_dir / "MassModels_Lelli2016c.mrt"

    assert props_file.exists(), f"Galaxy properties file not found: {props_file}"
    assert curves_file.exists(), f"Mass models file not found: {curves_file}"

    # Parse both files
    galaxy_props = _parse_galaxy_properties(props_file)
    rotation_curves = _parse_rotation_curves(curves_file)

    galaxies: list[GalaxyData] = []

    for name in sorted(galaxy_props.keys()):
        props = galaxy_props[name]

        # Quality filter
        if props["quality"] not in quality_filter:
            continue

        # Must have rotation curve data
        if name not in rotation_curves:
            continue

        curve = rotation_curves[name]

        # Minimum points filter
        if len(curve["r"]) < min_points:
            continue

        # Convert to numpy arrays
        radii = np.array(curve["r"])          # [kpc] ✓
        v_obs = np.array(curve["v_obs"])      # [km/s] ✓
        v_err = np.array(curve["e_v"])        # [km/s] ✓
        v_gas = np.array(curve["v_gas"])      # [km/s] ✓
        v_disk = np.array(curve["v_disk"])    # [km/s at M/L=1] ✓
        v_bul = np.array(curve["v_bul"])      # [km/s at M/L=1] ✓

        # Validate: no NaN, no negative radii, errors > 0
        if np.any(np.isnan(radii)) or np.any(radii <= 0):
            continue
        if np.any(np.isnan(v_obs)):
            continue
        # Ensure errors are positive (replace zeros with minimum)
        v_err = np.maximum(v_err, 1.0)  # floor at 1 km/s

        # Determine if galaxy has a bulge
        has_bulge = np.any(np.abs(v_bul) > 0.1)

        # Luminosity split: SPARC gives total luminosity.
        # For galaxies with bulges, we estimate bulge fraction from Hubble type.
        # This is approximate but only affects the Υ prior interpretation.
        L_total = props["luminosity_total"]  # L_sun
        if has_bulge and props["hubble_type"] <= 4:
            # Early types: ~20% bulge
            bulge_fraction = 0.2
        else:
            bulge_fraction = 0.0

        L_disk = L_total * (1.0 - bulge_fraction)
        L_bulge = L_total * bulge_fraction

        galaxy = GalaxyData(
            name=name,
            radii=radii,
            v_obs=v_obs,
            v_err=v_err,
            v_gas=v_gas,
            luminosity_disk=L_disk,
            luminosity_bulge=L_bulge,
            scale_length_disk=props["R_disk"],
            scale_length_bulge=_estimate_bulge_scale_length(props) if has_bulge else 0.0,
            distance=props["distance"],
            inclination=props["inclination"],
            quality=props["quality"],
        )

        # Attach raw SPARC velocity contributions as extra attributes
        # These are used by the fitting module instead of the analytic mass model
        object.__setattr__(galaxy, "v_disk_ml1", v_disk)
        object.__setattr__(galaxy, "v_bul_ml1", v_bul)

        galaxies.append(galaxy)

    return galaxies


def get_sparc_velocity_contributions(galaxy: GalaxyData) -> tuple[np.ndarray, np.ndarray]:
    """
    Get SPARC pre-computed disk and bulge velocity contributions at M/L=1.

    Returns:
        (v_disk_ml1, v_bul_ml1) both in km/s
    """
    return galaxy.v_disk_ml1, galaxy.v_bul_ml1  # type: ignore[attr-defined]
