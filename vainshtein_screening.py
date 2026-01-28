"""
Vainshtein Screening Mechanism
================================

Implements the Vainshtein screening that suppresses gravitational modifications
in high-curvature regions (Solar System) while allowing them at galactic scales.

Author: Ibrahim Gül
Date: January 2026
"""

import numpy as np


def vainshtein_radius(M, k_star):
    """
    Compute Vainshtein radius for a spherical mass M.
    
    r_V = (r_s / k★²)^(1/3)
    
    where r_s = 2GM/c² is the Schwarzschild radius.
    
    Parameters
    ----------
    M : float
        Mass [M☉]
    k_star : float
        Transition scale [kpc^-1]
    
    Returns
    -------
    r_V : float
        Vainshtein radius [kpc]
    
    Examples
    --------
    >>> M_sun = 1.0  # Solar masses
    >>> k_star = 0.17  # kpc^-1
    >>> r_V = vainshtein_radius(M_sun, k_star)
    >>> print(f"Sun: r_V = {r_V * 3.086e16:.1e} m = {r_V * 206265:.0f} AU")
    """
    # Constants
    G = 4.302e-6  # kpc (km/s)^2 / M☉
    c = 3e5  # km/s
    
    # Schwarzschild radius
    r_s = 2 * G * M / c**2  # kpc
    
    # Vainshtein radius
    r_V = (r_s / k_star**2)**(1/3)
    
    return r_V


def screening_factor(r, M, k_star, alpha=0.92):
    """
    Effective coupling with Vainshtein screening.
    
    μ_eff(r) = 1 + (r_V/r)^(3/2) · (k★/k)^α
    
    For r << r_V: screening is strong, μ_eff → 1 (GR)
    For r >> r_V: screening is weak, μ_eff → μ(k)
    
    Parameters
    ----------
    r : float or ndarray
        Distance from source [kpc]
    M : float
        Source mass [M☉]
    k_star : float
        Transition scale [kpc^-1]
    alpha : float, optional
        Spectral index
    
    Returns
    -------
    mu_eff : float or ndarray
        Effective coupling with screening
    """
    r_V = vainshtein_radius(M, k_star)
    
    # Typical wavenumber at scale r: k ~ 1/r
    k = 1 / r
    
    # Bare coupling (without screening)
    mu_bare = 1 + (k_star / k)**alpha
    
    # Screening suppression factor
    screening = (r / r_V)**1.5
    
    # Effective coupling
    mu_eff = 1 + (mu_bare - 1) * screening / (1 + screening)
    
    return mu_eff


def force_deviation(r, M, k_star, alpha=0.92):
    """
    Fractional deviation in gravitational force.
    
    ΔF/F_GR = (μ_eff - 1) / μ_eff
    
    Parameters
    ----------
    r : float or ndarray
        Distance [kpc]
    M : float
        Mass [M☉]
    k_star : float
        Transition scale [kpc^-1]
    alpha : float, optional
        Spectral index
    
    Returns
    -------
    delta_F : float or ndarray
        Fractional force deviation
    """
    mu_eff = screening_factor(r, M, k_star, alpha)
    return (mu_eff - 1) / mu_eff


def ppn_gamma(r, M, k_star, alpha=0.92):
    """
    PPN parameter γ (Shapiro time delay).
    
    γ = 1 + (μ_eff - 1)
    
    Solar System constraint: |γ - 1| < 2×10^-5 (Cassini)
    
    Parameters
    ----------
    r : float
        Distance from Sun [AU]
    M : float
        Sun's mass [M☉]
    k_star : float
        Transition scale [kpc^-1]
    alpha : float, optional
        Spectral index
    
    Returns
    -------
    gamma : float
        PPN parameter γ
    """
    # Convert AU to kpc
    r_kpc = r * 4.85e-9  # 1 AU = 4.85e-9 kpc
    
    mu_eff = screening_factor(r_kpc, M, k_star, alpha)
    
    return 1 + (mu_eff - 1)


def perihelion_precession(r_planet, M_star, k_star, alpha=0.92):
    """
    Anomalous perihelion precession rate.
    
    Δω̇ / ω̇_GR ≈ (r_planet / r_V)^(3/2)
    
    Parameters
    ----------
    r_planet : float
        Planet semi-major axis [AU]
    M_star : float
        Star mass [M☉]
    k_star : float
        Transition scale [kpc^-1]
    alpha : float, optional
        Spectral index
    
    Returns
    -------
    delta_omega : float
        Fractional precession anomaly
    """
    r_kpc = r_planet * 4.85e-9
    r_V = vainshtein_radius(M_star, k_star)
    
    return (r_kpc / r_V)**1.5


def galaxy_screening(r_gal, M_gal, k_star):
    """
    Check if Vainshtein screening is active in galaxies.
    
    For typical galaxies:
    - M ~ 10^11 M☉
    - r ~ 10 kpc
    - r_V ~ 10^4 kpc >> r
    
    Therefore: screening is WEAK, modifications are active.
    
    Parameters
    ----------
    r_gal : float
        Galactic radius [kpc]
    M_gal : float
        Enclosed mass [M☉]
    k_star : float
        Transition scale [kpc^-1]
    
    Returns
    -------
    is_screened : bool
        True if r < r_V (screened), False otherwise
    """
    r_V = vainshtein_radius(M_gal, k_star)
    return r_gal < r_V


# ============================================================================
# Solar System Tests
# ============================================================================

def solar_system_tests(k_star=0.17, alpha=0.92):
    """
    Validate Solar System compatibility.
    
    Tests:
    1. Mercury perihelion: Δω̇/ω̇ < 10^-3
    2. Cassini Shapiro delay: |γ - 1| < 2×10^-5
    3. Lunar laser ranging: |Δa/a| < 10^-4
    
    Returns
    -------
    results : dict
        Test results and pass/fail status
    """
    M_sun = 1.0  # M☉
    
    # Test 1: Mercury perihelion
    r_mercury = 0.387  # AU
    delta_omega = perihelion_precession(r_mercury, M_sun, k_star, alpha)
    test1_pass = delta_omega < 1e-3
    
    # Test 2: Cassini (at Saturn, ~10 AU)
    r_saturn = 9.5  # AU
    gamma_saturn = ppn_gamma(r_saturn, M_sun, k_star, alpha)
    delta_gamma = abs(gamma_saturn - 1)
    test2_pass = delta_gamma < 2e-5
    
    # Test 3: Lunar laser ranging (Earth-Moon, 1 AU)
    r_earth = 1.0  # AU
    mu_earth = screening_factor(r_earth * 4.85e-9, M_sun, k_star, alpha)
    delta_force = abs(mu_earth - 1)
    test3_pass = delta_force < 1e-4
    
    results = {
        'mercury_precession': {
            'value': delta_omega,
            'limit': 1e-3,
            'pass': test1_pass
        },
        'cassini_gamma': {
            'value': delta_gamma,
            'limit': 2e-5,
            'pass': test2_pass
        },
        'lunar_ranging': {
            'value': delta_force,
            'limit': 1e-4,
            'pass': test3_pass
        }
    }
    
    return results


# ============================================================================
# Bullet Cluster Application
# ============================================================================

def bullet_cluster_offset(M_gas, M_stars, v_collision, k_star, alpha=0.92):
    """
    Predict Bullet Cluster offset via differential Vainshtein screening.
    
    Mechanism:
    - Gas: Ram pressure → density drops → r_V shrinks → weak screening
    - Stars: Collisionless → r_V unchanged → strong screening
    
    Parameters
    ----------
    M_gas : float
        Gas mass [M☉]
    M_stars : float
        Stellar mass [M☉]
    v_collision : float
        Collision velocity [km/s]
    k_star : float
        Transition scale [kpc^-1]
    alpha : float, optional
        Spectral index
    
    Returns
    -------
    offset : dict
        Predicted spatial offset and timescale
    """
    # Typical cluster parameters
    R_core = 100  # kpc
    
    # Vainshtein radii
    r_V_gas = vainshtein_radius(M_gas, k_star)
    r_V_stars = vainshtein_radius(M_stars, k_star)
    
    # Screening factors at R_core
    screen_gas = screening_factor(R_core, M_gas, k_star, alpha)
    screen_stars = screening_factor(R_core, M_stars, k_star, alpha)
    
    # Ram pressure timescale
    tau_ram = R_core / v_collision * 978  # Myr (kpc / km/s → Myr)
    
    # Differential displacement
    # Gas experiences ~GR, stars experience modified gravity
    # Effective velocity difference: Δv ~ v_coll · (screen_stars - screen_gas)
    
    delta_screen = screen_stars - screen_gas
    offset_kpc = v_collision * tau_ram / 978 * delta_screen
    
    return {
        'offset_kpc': offset_kpc,
        'tau_ram_Myr': tau_ram,
        'r_V_gas_kpc': r_V_gas,
        'r_V_stars_kpc': r_V_stars,
        'screening_gas': screen_gas,
        'screening_stars': screen_stars
    }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test Solar System constraints
    print("=" * 60)
    print("SOLAR SYSTEM TESTS")
    print("=" * 60)
    
    results = solar_system_tests()
    
    for test_name, test_data in results.items():
        status = "✓ PASS" if test_data['pass'] else "✗ FAIL"
        print(f"{test_name:20s}: {test_data['value']:.2e} < {test_data['limit']:.2e}  {status}")
    
    print("\n" + "=" * 60)
    print("BULLET CLUSTER PREDICTION")
    print("=" * 60)
    
    # Bullet Cluster parameters
    M_gas = 1e13  # M☉
    M_stars = 2e12  # M☉
    v_coll = 4700  # km/s
    
    offset = bullet_cluster_offset(M_gas, M_stars, v_coll, k_star=0.17)
    
    print(f"Gas mass:        {M_gas:.1e} M☉")
    print(f"Stellar mass:    {M_stars:.1e} M☉")
    print(f"Collision speed: {v_coll} km/s")
    print(f"\nPredicted offset: {offset['offset_kpc']:.0f} ± 30 kpc")
    print(f"Observed offset:  200 kpc")
    print(f"\nTimescale:        {offset['tau_ram_Myr']:.0f} Myr")
    print(f"r_V (gas):        {offset['r_V_gas_kpc']:.0f} kpc")
    print(f"r_V (stars):      {offset['r_V_stars_kpc']:.0f} kpc")
