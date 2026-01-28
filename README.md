# Non-Local Gravitational Coupling from Kaluza-Klein Resonances

> **A Phenomenological Framework with Vainshtein Screening**

**Author:** Ibrahim Gül  
**Affiliation:** Independent Researcher, Ankara, Türkiye  
**Contact:** gulpoetika@gmail.com

---

## 📄 Abstract

We develop a phenomenological model of scale-dependent gravity motivated by Kaluza-Klein (KK) tower resonances in a compact extra dimension. Unlike warped scenarios, we employ a toroidal compactification with orbifolding ($S^1/\mathbb{Z}_2$), yielding a discrete mass spectrum with spectral density $\rho(m) \propto m^{\alpha}$. 

**Key Results:**
- SPARC rotation curves: $\alpha = 0.92 \pm 0.11$ (consistent with MOND $\alpha=1$)
- Reduced chi-squared: $\chi^2_\nu = 0.94$
- $S_8$ tension alleviation: $S_8 = 0.79 \pm 0.04$
- Bullet Cluster offset explained via differential Vainshtein screening
- Solar System safe: $|\gamma - 1| < 10^{-5}$

---

## 📂 Repository Structure

```
.
├── README.md                          # This file
├── data/
│   ├── sparc_q1_i40.csv              # SPARC galaxy sample (Q=1, i>40°)
│   └── data_description.txt           # Data format documentation
├── analysis/
│   ├── rotation_curve_analysis.py     # SPARC rotation curve fitting
│   ├── bayesian_inference.py          # MultiNest parameter estimation
│   └── model_comparison.py            # Evidence calculation
├── cosmology/
│   ├── growth_factor.py               # Modified growth equation solver
│   ├── halo_mass_function.py          # Sheth-Tormen HMF with μ(k)
│   └── power_spectrum.py              # Matter power spectrum computation
├── theory/
│   ├── propagator.py                  # Effective propagator μ(k)
│   ├── vainshtein_screening.py        # Screening mechanism
│   └── energy_conservation.py         # Brane-bulk exchange
├── results/
│   ├── chains/                        # MCMC posterior chains
│   ├── corner_plot.pdf                # Figure 1: Posteriors
│   ├── rar_fit.pdf                    # Figure 2: RAR
│   └── ...                            # Other figures
├── notebooks/
│   ├── 01_data_exploration.ipynb      # SPARC sample analysis
│   ├── 02_rotation_curves.ipynb       # Velocity modeling
│   ├── 03_bayesian_fit.ipynb          # Parameter inference
│   └── 04_cosmology.ipynb             # S8 predictions
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
└── CITATION.cff                       # Citation metadata
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/ibraimgul/KK-Leakage-SPARC.git
cd KK-Leakage-SPARC

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np
from theory.propagator import effective_coupling
from analysis.rotation_curve_analysis import fit_galaxy

# Compute effective coupling μ(k)
k = np.logspace(-3, 2, 100)  # kpc^-1
alpha = 0.92
k_star = 0.17  # kpc^-1
mu_k = effective_coupling(k, alpha, k_star)

# Fit a single galaxy
galaxy_data = load_sparc_galaxy('NGC2403')
results = fit_galaxy(galaxy_data, alpha=0.92, k_star=0.17)
print(f"Chi-squared: {results['chi2']:.2f}")
```

---

## 📊 Data

### SPARC Sample

We analyze **118 galaxies** from the SPARC database with:
- Quality flag `Q = 1` (highest reliability)
- Inclination `i > 40°` (minimize deprojection errors)
- Distance uncertainty `σ_D/D < 0.25`

**Data columns:**
- `Rad`: Radius [kpc]
- `Vobs`: Observed circular velocity [km/s]
- `e_Vobs`: Velocity uncertainty [km/s]
- `Vgas`: Gas contribution [km/s]
- `Vdisk`: Stellar disk contribution [km/s]
- `Vbul`: Bulge contribution [km/s]

**Source:** [Lelli et al. 2016, AJ 152, 157](https://ui.adsabs.harvard.edu/abs/2016AJ....152..157L)

---

## 🧮 Methodology

### 1. Modified Poisson Equation

```python
# Fourier-space solution
def modified_poisson(k, rho_k, alpha, k_star):
    mu_k = 1 + (k_star / k)**alpha
    Phi_k = 4 * np.pi * G * rho_k / (k**2 * mu_k)
    return Phi_k
```

### 2. Rotation Curve Prediction

```python
# Hankel transform (FFTLog)
def compute_velocity(R, Sigma, alpha, k_star):
    k, Sigma_k = hankel_transform(R, Sigma)
    mu_k = effective_coupling(k, alpha, k_star)
    g_R = inverse_hankel_transform(k, k * Sigma_k / mu_k)
    v_circ = np.sqrt(R * g_R)
    return v_circ
```

### 3. Bayesian Inference

```python
# MultiNest sampling
def log_likelihood(theta):
    alpha, log_k_star, Upsilon_star, D = theta
    v_pred = predict_velocity(alpha, k_star, Upsilon_star, D)
    chi2 = np.sum((v_obs - v_pred)**2 / sigma_v**2)
    return -0.5 * chi2

# Run nested sampling
sampler = dynesty.NestedSampler(log_likelihood, prior_transform, ndim=4)
sampler.run_nested()
results = sampler.results
```

---

## 📈 Key Results

### Parameter Constraints

| Parameter | Median | 68% C.I. |
|-----------|--------|----------|
| Spectral index α | 0.92 | ±0.11 |
| Leakage scale log₁₀(k★/kpc⁻¹) | -0.78 | +0.16/-0.19 |
| Reduced χ²ᵥ | 0.94 | ±0.03 |

### Model Comparison

| Model | ln Z | Δln Z | Significance |
|-------|------|-------|--------------|
| α free | -1690.4 | — | — |
| α=1 (MOND) | -1691.1 | 0.7±0.3 | 0.9σ (inconclusive) |

**Interpretation:** Current data cannot distinguish α=1 vs. α≈0.9 at high confidence, but scale-dependent gravity fits well.

### Cosmological Predictions

- **S₈ suppression:** 0.79 ± 0.04 (cf. Planck: 0.832±0.013, KiDS: 0.766±0.020)
- **CMB lensing:** ~4% suppression at ℓ>100
- **Cluster dynamics:** Velocity dispersion enhanced by ~10% at r₂₀₀

---

## 🔬 Physical Mechanisms

### Vainshtein Screening

Ensures Solar System compatibility via nonlinear kinetic terms:

$$r_V = \left( \frac{r_s}{k_*^2} \right)^{1/3}$$

For the Sun: $r_V \sim 2000$ AU → Mercury deviations $< 10^{-5}$

### Bullet Cluster Offset

**Mechanism:** Differential screening, not propagation delay
- **Gas:** Ram pressure → density drops → $r_V$ shrinks → weak screening
- **Stars:** Collisionless → $r_V$ intact → strong screening → "phantom DM"
- **Result:** ~200 kpc offset without violating Lorentz invariance

---

## 📚 Citation

If you use this code or results, please cite:

```bibtex
@article{Gul2026,
  author  = {G\"ul, Ibrahim},
  title   = {Non-Local Gravitational Coupling from Kaluza-Klein Resonances: A Phenomenological Framework with Vainshtein Screening},
  journal = {arXiv preprint arXiv:2501.xxxxx},
  year    = {2026},
  eprint  = {2501.xxxxx},
  archivePrefix = {arXiv},
  primaryClass = {gr-qc}
}
```

---

## 📖 Documentation

Detailed documentation available in `docs/`:
- [Theory Overview](docs/theory.md)
- [Data Processing](docs/data.md)
- [Analysis Pipeline](docs/analysis.md)
- [API Reference](docs/api.md)

---

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

---

## 🔗 References

**Key Papers:**
1. Lelli et al. 2016, AJ 152, 157 - [SPARC database](https://doi.org/10.3847/0004-6256/152/6/157)
2. McGaugh et al. 2016, PRL 117, 201101 - [RAR](https://doi.org/10.1103/PhysRevLett.117.201101)
3. Skordis & Zlosnik 2021, PRL 127, 161302 - [RMOND](https://doi.org/10.1103/PhysRevLett.127.161302)
4. Vainshtein 1972, PLB 39, 393 - [Screening mechanism](https://doi.org/10.1016/0370-2693(72)90147-5)

**Cosmology:**
- Planck 2020, A&A 641, A6 - [CMB power spectra](https://doi.org/10.1051/0004-6361/201833910)
- Heymans et al. 2021, A&A 646, A140 - [KiDS S₈](https://doi.org/10.1051/0004-6361/202039063)

---

## 📧 Contact

**Ibrahim Gül**  
Independent Researcher  
📧 gulpoetika@gmail.com  
🌐 [GitHub](https://github.com/ibraimgul)

---

## ⭐ Acknowledgments

- SPARC collaboration for public data
- MultiNest/dynesty developers
- CLASS/CAMB teams for Boltzmann codes
- Anthropic Claude for manuscript preparation assistance

---

**Last updated:** January 2026
