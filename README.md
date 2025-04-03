# MC

## Introduction

Sampling from independent Bernoulli random variables conditioned on their sum is a classical problem in statistics and combinatorics. Formally, given a vector of success probabilities $$p = (p_1, \dots, p_N)$$ and a fixed integer $$k$$, the goal is to sample binary vectors $$x \in \{0,1\}^N$$ such that:

$$
\sum_{i=1}^N x_i = k
$$

and the unconditioned distribution of each component is $$x_i \sim \text{Bernoulli}(p_i)$$.

Naively applying rejection sampling becomes inefficient for small or large values of $$k$$, as valid samples may be rare. To address this, an **exact dynamic sampling algorithm** based on conditional probabilities and dynamic programming was introduced in the paper:

> *"A Simple Markov Chain for Independent Bernoulli Variables Conditioned on Their Sum"*  
> (Hermans & Lelis, 2020)

This project builds on that exact algorithm and introduces randomized **Quasi-Monte Carlo (RQMC)** version, replacing i.i.d. uniform draws with low-discrepancy sequences to improve convergence when estimating the conditional distribution.
