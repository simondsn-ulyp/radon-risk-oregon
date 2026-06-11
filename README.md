# radon-risk-oregon
Hierarchical Bayesian ordinal regression on Oregon geologic data using JAX/blackjax
Hierarchical Bayesian Radon Risk Modeling — Oregon
Bayesian hierarchical ordinal regression model for radon potential in Oregon, fit using Hamiltonian Monte Carlo with the No-U-Turn Sampler (NUTS) implemented from scratch in JAX via blackjax.
Overview
Radon is a known indoor health hazard whose potential varies with underlying geology. This project fits a multilevel ordered logistic model to ~120,000 geologic polygons across Oregon, grouped into 66 geological terranes (source: DOGAMI Open-File Report O-18-01). The goal is to quantify terrane-level uncertainty in radon risk classification rather than producing deterministic point estimates.

Key methodological choices:

Hierarchical structure: terrane-level random intercepts with partial pooling, stabilizing estimates for sparsely observed terranes
Non-centered parameterization: avoids funnel geometry and improves NUTS mixing in the hierarchical prior
Ordered logistic likelihood: respects the ordinal structure of the 3-level radon rank outcome
NUTS in JAX: log-joint density written in jax.numpy for automatic differentiation; step size and diagonal mass matrix tuned via dual averaging during warmup
Model
For polygon $i$ in terrane $j(i)$, the ordered logistic model is:

$$\Pr(y_i \leq c \mid \alpha_{j(i)}) = \sigma(\kappa_c - \alpha_{j(i)}), \quad c = 1, 2$$

Terrane effects use a non-centered parameterization:

$$\alpha_j = \mu_\alpha + \sigma_\alpha \alpha_j^{\text{raw}}, \quad \alpha_j^{\text{raw}} \sim \mathcal{N}(0, 1)$$

with priors $\mu_\alpha \sim \mathcal{N}(0, 25)$, $\log \sigma_\alpha \sim \mathcal{N}(0, 1)$, and an exponentiated gap parameterization to enforce $\kappa_1 < \kappa_2$.
Data
The DOGAMI dataset consists of N = 120,049 geologic polygons with:

TERRANE_GR: geological terrane group (J = 66 levels)
FIN_RN_RNK: final radon potential rank in {1, 2, 3}

The outcome is highly imbalanced: ~78% rank 1, ~16% rank 2, ~6% rank 3.
Results
NUTS was run with 2,000 warmup iterations (adaptive) and 4,000 post-warmup iterations. Key diagnostics:

Parameter
Mean
SD
ESS
R-hat
μ_α
−1.38
3.36
~476
1.004
σ_α
2.28
0.24
~101
1.016
κ₁
0.98
3.37
~478
1.005
κ₂
2.89
3.37
~478
1.006


Adapted step size: ε ≈ 0.0017, mean acceptance rate ≈ 0.92
Posterior predictive check: model closely reproduces the marginal rank distribution (rank 1: observed 0.784, predicted 0.784)
High radon risk (rank 3) is concentrated in a small subset of terranes; a few approach P(rank 3) ≈ 0.8
Setup
pip install jax jaxlib blackjax pandas numpy matplotlib

Data: download the DOGAMI radon potential polygon feature class from the Oregon Spatial Data Library.
Usage
# Load and preprocess data

import pandas as pd

df = pd.read_csv("radon_data.csv")

df = df.dropna(subset=["TERRANE_GR", "FIN_RN_RNK"])

# Run model (see radon_model.py)

python radon_model.py
References
Hoffman & Gelman (2014). The No-U-Turn Sampler. JMLR, 15, 1593–1623.
Neal (2011). MCMC using Hamiltonian dynamics. Handbook of MCMC.
Betancourt (2017). A conceptual introduction to HMC. arXiv:1701.02434.
Gelman et al. (2013). Bayesian Data Analysis, 3rd ed.
DOGAMI (2018). Radon potential of Oregon. Open-File Report O-18-01.

