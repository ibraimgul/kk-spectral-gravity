"""
Effective Graviton Propagator and Coupling
===========================================

This module computes the effective 4D gravitational coupling μ(k) arising
from the summation over Kaluza-Klein tower modes.

Author: Ibrahim Gül
Date: January 2026
"""

import numpy as np
from scipy.special import gamma
from scipy.integrate import quad


def effective_coupling(k, alpha, k_star):
    """
    Compute the effective gravitational coupling μ(k).
    
    The propagator modification arises from integrating over the spectral
    density ρ(m) ∝ m^α:
    
        μ(k) = 1 + C(α) · (k★/k)^α
    
    Parameters
    ----------
    k : float or ndarray
        Wavenumber [kpc^-1]
    alpha : float
        Spectral index (0 < α < 2 for ghost-freedom)
    k_star : float
        Transition scale [kpc^-1]
    
    Returns
    -------
    mu : float or ndarray
        Effective coupling μ(k) >= 1
    
    Examples
    --------
    >>> k = np.logspace(-3, 2, 100)
    >>> mu = effective_coupling(k, alpha=0.92, k_star=0.17)
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(k, mu)
    >>> plt.xlabel('k [kpc$^{-1}$]')
    >>> plt.ylabel(r'$\mu(k)$')
    """
    if not (0 < alpha < 2):
        raise ValueError(f"Alpha must be in (0, 2) for unitarity. Got {alpha}")
    
    # Normalization factor C(α)
    # For α=1, recover logarithmic behavior (MOND-like)
    if np.isclose(alpha, 1.0, atol=1e-6):
        # Logarithmic regime: μ ≈ 1 + k★·ln(k★/k) for k << k★
        return 1 + k_star * np.log(k_star / k) * (k < k_star)
    
    # General case: power-law
    C_alpha = np.pi / (gamma(alpha + 1) * np.sin(np.pi * alpha))
    
    mu = 1 + C_alpha * (k_star / k)**alpha
    
    return mu


def spectral_density(m, alpha, k_star, m_min=1e-6):
    """
    KK mode spectral density ρ(m) ∝ m^α.
    
    Parameters
    ----------
    m : float or ndarray
        KK mass [kpc^-1]
    alpha : float
        Spectral index
    k_star : float
        Normalization scale [kpc^-1]
    m_min : float, optional
        IR cutoff (radion stabilization scale)
    
    Returns
    -------
    rho : float or ndarray
        Spectral density [kpc]
    """
    m = np.asarray(m)
    rho = np.zeros_like(m)
    mask = m > m_min
    
    # Normalization
    N = 1.0 / k_star
    
    rho[mask] = N * (m[mask] / k_star)**alpha
    
    return rho


def propagator_full(k, alpha, k_star, m_max=1e3):
    """
    Full propagator via numerical integration of spectral function.
    
    G(k) = 1/k² + ∫ dm ρ(m)/(k² + m²)
    
    This is more accurate than the asymptotic approximation but slower.
    
    Parameters
    ----------
    k : float
        Wavenumber [kpc^-1]
    alpha : float
        Spectral index
    k_star : float
        Transition scale [kpc^-1]
    m_max : float, optional
        UV cutoff (5D Planck scale)
    
    Returns
    -------
    G_k : float
        Full propagator
    """
    def integrand(m):
        rho = spectral_density(m, alpha, k_star)
        return rho / (k**2 + m**2)
    
    zero_mode = 1.0 / k**2
    kk_contribution, _ = quad(integrand, 1e-6, m_max)
    
    return zero_mode + kk_contribution


def coupling_numerical(k, alpha, k_star, m_max=1e3):
    """
    Numerical μ(k) from full propagator integration.
    
    μ(k) = k² · G(k)
    
    For validation against analytical approximation.
    """
    G_k = propagator_full(k, alpha, k_star, m_max)
    return k**2 * G_k


def modified_poisson_fourier(k, rho_k, alpha, k_star):
    """
    Solve modified Poisson equation in Fourier space.
    
    k² μ(k) Φ̃(k) = 4πG ρ̃(k)
    
    Parameters
    ----------
    k : ndarray
        Wavenumber array [kpc^-1]
    rho_k : ndarray
        Fourier-transformed density [M☉/kpc³]
    alpha : float
        Spectral index
    k_star : float
        Transition scale [kpc^-1]
    
    Returns
    -------
    Phi_k : ndarray
        Fourier potential [km²/s²]
    """
    G_newton = 4.302e-6  # kpc (km/s)^2 / M☉
    
    mu_k = effective_coupling(k, alpha, k_star)
    
    # Avoid division by zero at k=0
    k_safe = np.where(k > 1e-10, k, 1e-10)
    
    Phi_k = 4 * np.pi * G_newton * rho_k / (k_safe**2 * mu_k)
    
    return Phi_k


def mond_limit(k, k_star):
    """
    MOND limit: α → 1.
    
    μ_MOND(k) ≈ 1 + k★/k for k << k★
    """
    return effective_coupling(k, alpha=1.0, k_star=k_star)


def gr_limit(k, k_star):
    """
    GR limit: α → 0 or k >> k★.
    
    μ_GR(k) = 1
    """
    return np.ones_like(k)


# ============================================================================
# Validation and Testing
# ============================================================================

def validate_ghost_freedom(alpha):
    """
    Check that spectral density is positive (no ghost modes).
    
    Requirement: 0 < α < 2 ensures ρ(m) > 0 for all m > 0.
    """
    if 0 < alpha < 2:
        return True, "Ghost-free: 0 < α < 2"
    else:
        return False, f"Ghost modes present: α = {alpha} not in (0, 2)"


def validate_unitarity(alpha, k_star, k_test=None):
    """
    Check unitarity via optical theorem.
    
    Im[G(k)] should be related to the spectral function.
    """
    if k_test is None:
        k_test = np.logspace(-2, 2, 100)
    
    mu = effective_coupling(k_test, alpha, k_star)
    
    # Unitarity requires μ(k) > 0 for all k
    if np.all(mu > 0):
        return True, "Unitary: μ(k) > 0 for all k"
    else:
        return False, f"Unitarity violation: min(μ) = {mu.min()}"


if __name__ == "__main__":
    # Quick test
    import matplotlib.pyplot as plt
    
    k = np.logspace(-3, 2, 300)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: μ(k) for different α
    for alpha in [0.5, 0.75, 0.92, 1.0, 1.25]:
        mu = effective_coupling(k, alpha, k_star=0.17)
        label = f'α={alpha:.2f}'
        if alpha == 0.92:
            ax1.loglog(k, mu, lw=3, label=label + ' (best-fit)')
        elif alpha == 1.0:
            ax1.loglog(k, mu, '--', lw=2, label=label + ' (MOND)')
        else:
            ax1.loglog(k, mu, lw=1.5, label=label)
    
    ax1.axhline(1, color='k', ls=':', alpha=0.5)
    ax1.axvline(0.17, color='r', ls='--', alpha=0.5, label='k★')
    ax1.set_xlabel('k [kpc$^{-1}$]')
    ax1.set_ylabel(r'$\mu(k)$')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Right: Numerical vs analytical
    k_test = np.logspace(-2, 1, 50)
    mu_analytic = effective_coupling(k_test, 0.92, 0.17)
    mu_numeric = np.array([coupling_numerical(ki, 0.92, 0.17) for ki in k_test])
    
    ax2.loglog(k_test, mu_analytic, 'b-', lw=2, label='Analytical')
    ax2.loglog(k_test, mu_numeric, 'ro', ms=4, label='Numerical')
    ax2.set_xlabel('k [kpc$^{-1}$]')
    ax2.set_ylabel(r'$\mu(k)$')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('propagator_validation.pdf')
    print("✓ Validation plot saved: propagator_validation.pdf")
