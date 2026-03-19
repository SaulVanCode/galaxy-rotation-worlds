"""
MetaLaw Discovery System v0.2-FROZEN — Physical Constants

Internal unit system:
    Distance:      kpc
    Velocity:      km/s
    Mass:          M_sun
    Acceleration:  km^2 s^-2 kpc^-1
    Density:       M_sun kpc^-3

All modules import constants from here. No other module may define G or a0.
"""

# Gravitational constant in internal units
# G = 4.3009e-3 kpc (km/s)^2 / M_sun
# Source: standard value, G = 6.674e-11 m^3 kg^-1 s^-2 converted
G_INTERNAL: float = 4.3009e-3  # kpc km^2 s^-2 M_sun^-1

# MOND critical acceleration in internal units
# a0 = 1.2e-10 m/s^2
# Conversion: 1 km^2/(s^2 kpc) = 3.241e-14 m/s^2
# Therefore: a0 = 1.2e-10 / 3.241e-14 = 3.703e3 km^2 s^-2 kpc^-1
A0_INTERNAL: float = 3.703e3  # km^2 s^-2 kpc^-1

# Conversion factor: m/s^2 → internal acceleration units
# 1 m/s^2 = 1 / 3.241e-14 km^2 s^-2 kpc^-1
MS2_TO_INTERNAL: float = 1.0 / 3.241e-14

# Softening radius squared to prevent division by zero at r=0
# Physical motivation: no observation probes below 10 pc
SOFTENING_KPC2: float = 0.01**2  # (0.01 kpc)^2

# Floor for log arguments to prevent -inf
LOG_FLOOR: float = 1e-20
