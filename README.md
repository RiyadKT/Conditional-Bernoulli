# Conditional Bernoulli Sampling – Monte Carlo Project

This repository explores and compares several algorithms for sampling from a conditional Bernoulli distribution.  
Given a vector of independent Bernoulli variables \\( X_i \\sim \\text{Bern}(p_i) \\), we aim to sample vectors  
\\( X \\in \\{0,1\\}^N \\) such that \\( \\sum X_i = k \\).

This project is based on the paper:  
"A Simple Markov Chain for Independent Bernoulli Variables Conditioned on Their Sum"  
by Heng, Jacob, and Ju (2020) — [arXiv:2012.03103](https://arxiv.org/pdf/2012.03103)

---

## Project Structure

```bash
├── main.py                   # Core implementation  
├── main.ipynb                # Annotated notebook for analysis  
├── ReportMC.pdf              # Formal report  
├── requirements.txt          # Dependencies  
├── README.md  
└── plots/  
    ├── Comp_dyn_rej.png  
    ├── Distance_Rej_MCMC.png  
    ├── Distance_Rej_MCMC_2.png  
    ├── MCMC_distr.png  
    ├── Rejection_k.png  
    ├── Rejection_var.png  
    ├── RQMC.png  
    ├── RQMC_log_log.png  
    └── MCMC_plots/  
        ├── MCMC_distr_00111.png  
        ├── MCMC_distr_01011.png  
        └── ... (other per-initial-state plots)

```

---

## Methods Compared

1. **Rejection Sampling**  
   Naive method: generate i.i.d. samples and reject if the constraint is not met.  
   Becomes impractical as $N \to \infty$ or $k$ is rare.

2. **Dynamic Programming (Exact)**  
   Recursive table $q(i,n)$ enables exact sampling of constrained Bernoulli vectors.  
   Complexity: $\mathcal{O}(N^2)$.

3. **MCMC Sampler**  
   Markov Chain with constrained state space.  
   Swap 0 and 1 coordinates, accept moves with Metropolis-Hastings ratio.  
   Invariant and irreducible under mild assumptions.

4. **RQMC (Randomized Quasi-Monte Carlo)**  
   Uses Sobol sequences to replace uniform draws in the dynamic algorithm.  
   Improves convergence: $\text{MSE} = \mathcal{O}(1/n^{2 - \varepsilon})$.

---

## How to Run

Install dependencies:
