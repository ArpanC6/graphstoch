# GraphStoch: Mathematical Theory

This document develops the theoretical foundation behind GraphStoch's core
model. The empirical results in `README.md` (e.g. the dt stability threshold
observed on the Karate Club graph, the divergence of Euler-Maruyama at large
step sizes) are consequences of the results proved here, not independent
observations - this document exists to make that connection explicit and
rigorous.

## 1. Setup and Notation

Let `G = (V, E)` be a connected, undirected graph with `n = |V|` nodes,
adjacency matrix `A`, degree matrix `D`, and graph Laplacian `L = D - A`.

**Fact 1 (spectral properties of L).** `L` is symmetric and positive
semi-definite, so it admits a real eigendecomposition

    L = U Λ U^T,   Λ = diag(λ_1, ..., λ_n),   0 = λ_1 ≤ λ_2 ≤ ... ≤ λ_n,

where `U` is orthogonal (`U^T U = I`). If `G` is connected, `λ_1 = 0` is a
**simple** eigenvalue (algebraic multiplicity 1) with eigenvector
`u_1 = 1/√n · (1, 1, ..., 1)^T` - the constant vector. This is a standard
result in spectral graph theory (Fiedler, 1973); we take it as given.

`λ_2` is the **algebraic connectivity** (Fiedler value), and `λ_n` is the
largest eigenvalue. Both play a direct role below.

GraphStoch models node states as the solution to the linear SDE

    dX_t = -L X_t dt + σ dW_t,     X_0 given,                        (1)

where `X_t ∈ ℝⁿ` is the vector of node states, and `W_t` is a standard
n-dimensional Wiener process (independent noise per node). Because the drift
is linear in `X_t` and the diffusion coefficient `σ` is constant (does not
depend on `X_t`), (1) is a **linear SDE with additive noise** - a
multivariate Ornstein-Uhlenbeck (OU) process with mean-reversion matrix `L`.
This single fact is what makes the results below possible: linear
additive-noise SDEs are one of the few classes with a closed-form solution.

## 2. Exact Solution

**Theorem 1 (closed-form solution).** The solution to (1) is

    X_t = e^{-Lt} X_0  +  σ ∫_0^t e^{-L(t-s)} dW_s.                   (2)

*Proof sketch.* Apply Itô's formula to `Y_t = e^{Lt} X_t`:

    dY_t = L e^{Lt} X_t dt + e^{Lt} dX_t
         = L e^{Lt} X_t dt + e^{Lt}(-L X_t dt + σ dW_t)
         = σ e^{Lt} dW_t.

Integrating, `Y_t = Y_0 + σ ∫_0^t e^{Ls} dW_s`, and substituting back
`X_t = e^{-Lt} Y_t` gives (2). ∎

This is exactly the standard OU-process solution, specialized to the graph
Laplacian as the mean-reversion matrix. It says every simulation GraphStoch
runs has a known exact answer - `sde_solve` (Euler-Maruyama) and `sra3_solve`
(adaptive Runge-Kutta) are both numerical approximations of (2), and (2)
itself is available as an exact validation baseline (see Section 5).

### 2.1 Mean behavior

Taking expectations in (2) (the stochastic integral has zero mean):

    E[X_t] = e^{-Lt} X_0.

Diagonalizing `L = U Λ U^T`, this is `E[X_t] = U e^{-Λt} U^T X_0`. Because
`λ_1 = 0`, the component of `X_0` along `u_1` (the constant/consensus
direction) is **left unchanged** by the mean dynamics, while every other
component decays as `e^{-λ_i t} → 0`. In plain terms: on average, node
values converge to the graph-wide consensus value (the projection of `X_0`
onto the all-ones direction), and they do so at a rate governed by `λ_2`,
the slowest-decaying non-trivial mode. This is the formal version of "the
Laplacian's structural pull dominates" claimed informally in the README.

### 2.2 Covariance and a non-obvious instability

    Cov(X_t) = σ² ∫_0^t e^{-2Ls} ds = σ² U ( ∫_0^t e^{-2Λs} ds ) U^T
             = σ² U · diag( (1 - e^{-2λ_i t}) / (2λ_i) )_{i=1}^n · U^T,

with the convention that the `i = 1` term (where `λ_1 = 0`) is interpreted
via L'Hôpital's rule:

    lim_{λ→0} (1 - e^{-2λt}) / (2λ) = t.

**This is the key structural fact**: the variance of `X_t` along the
constant-vector direction `u_1` grows **linearly and without bound** in
`t` (it is `σ²t`, a pure Brownian motion / random walk along that
direction), while variance along every other eigendirection converges to a
finite stationary value `σ²/(2λ_i)` as `t → ∞`.

**Consequence.** The process (1) is *not* asymptotically stationary as a
whole - only the component orthogonal to the graph's consensus direction is.
This matters practically: any long-run simulation of "true signal" via
`sde_solve` (as done to generate ground truth in the synthetic benchmarks)
will have an ever-growing, unbounded overall offset if run for too long,
superimposed on a well-behaved, bounded fluctuation in every other mode.
This was not previously identified as a distinct phenomenon in the
project's benchmarks; the "signal" generation step should be re-examined
in light of it (see Section 6, open item).

## 3. Stability of the Euler-Maruyama Discretization

The from-scratch solver implements

    X_{k+1} = X_k - L X_k Δt + σ√Δt Z_k,     Z_k ~ N(0, I).           (3)

**Theorem 2 (stability threshold).** The mean recursion
`E[X_{k+1}] = (I - LΔt) E[X_k]` is stable (mean magnitude does not diverge)
if and only if

    Δt < 2 / λ_max(L) = 2 / λ_n.

*Proof.* Diagonalize as before: in the eigenbasis, mode `i` evolves as

    x_{k+1}^{(i)} = (1 - λ_i Δt) x_k^{(i)}.

This is a stable linear recursion iff the amplification factor satisfies
`|1 - λ_i Δt| ≤ 1`, i.e. `0 ≤ λ_i Δt ≤ 2`, i.e. `Δt ≤ 2/λ_i` (for `λ_i > 0`;
the `λ_1 = 0` mode is always neutrally stable, amplification factor exactly
1, consistent with Section 2.1). This must hold for **every** mode
simultaneously, so the binding constraint is the largest eigenvalue:

    Δt < 2 / λ_n.  ∎

This is exactly the empirical rule the project adopted ad hoc in Section 13
of the project log ("always derive dt from the graph's own Laplacian
eigenvalues") - Theorem 2 is the proof that rule was correct, and pins down
*why* high-degree hub nodes tighten the bound: `λ_n` grows with the maximum
node degree (by Gershgorin's circle theorem, `λ_n ≤ 2 · max_i D_ii`), so
denser or hub-heavy graphs force a smaller stable `Δt`, matching what was
observed empirically on the Karate Club graph (`λ_max` implied by
`stable_dt ≈ 0.0384`) versus the 4-node chain (`stable_dt ≈ 0.586`).

## 4. Convergence Order of Euler-Maruyama

For a general SDE `dX_t = a(X_t) dt + b(X_t) dW_t`, Euler-Maruyama has
strong convergence order 0.5 and weak order 1.0 (Kloeden & Platen, 1992).
However, GraphStoch's SDE (1) has **additive noise**: the diffusion
coefficient is the constant `σ`, not a function of `X_t`. For additive-noise
SDEs, the Milstein scheme's correction term (which involves the derivative
of `b`) vanishes identically, so **Euler-Maruyama coincides exactly with the
Milstein scheme**, which is known to achieve strong order 1.0 - a full order
better than the general case (Kloeden & Platen, 1992, Ch. 10).

This is worth stating explicitly because it means the from-scratch EM
solver is not merely "the simple baseline" relative to SRA3 - for this
specific additive-noise setting, EM already attains the best strong-order
guarantee a fixed-step scheme can offer without adaptive stepping. SRA3's
practical advantage (demonstrated in the README's solver comparison) is
**numerical stability at large step sizes via adaptive stepping**, not a
higher convergence order at a matched, sufficiently small step size.

## 5. A Free, Exact Validation Baseline

Because (2) is closed-form, it can be evaluated numerically (via
`exp(-L*t)` and a discretized stochastic integral, or via the exact
Gaussian transition law implied by Section 2.2) to check both `sde_solve`
and `sra3_solve` against ground truth at any `t`, independent of step-size
choice. This is a stronger validation than comparing solvers to each other
alone, since both `sde_solve` and `sra3_solve` could in principle share a
common bias while both being wrong relative to (2). This is listed as an
open implementation item in Section 6.

## 6. Open Items / Next Steps

This document establishes the theory; the following are not yet done and
are natural next steps before any preprint:

1. **Implement the closed-form solution (2)** in `GraphStoch.jl` as
   `exact_solve(L, X0, sigma, t)`, using `exp(-L*t)` (matrix exponential)
   and the exact Gaussian covariance from Section 2.2, to use as a ground
   truth for validating both `sde_solve` and `sra3_solve` at matched `t`.
2. **Re-examine "true signal" generation** in the synthetic and Karate Club
   benchmarks in light of Section 2.2 - since long diffusion runs have
   unbounded variance in the consensus direction, the choice of diffusion
   time used to generate "true signal" should be justified rather than
   arbitrary, or the consensus-direction drift should be removed/centered
   before use as ground truth.
3. **State Theorem 2 in the README** alongside the empirical stability
   table, replacing "we found that dt is critical" with the proved bound
   and the Gershgorin-based intuition for why hub nodes tighten it.
4. **Empirically verify strong order 1.0** (Section 4) by running EM at a
   sequence of halving step sizes against the exact solution (2) and
   checking that error halves at rate ~2^1 rather than ~2^0.5 - this would
   be a concrete, checkable claim to include in any writeup.

## References

- Fiedler, M. (1973). Algebraic connectivity of graphs. *Czechoslovak
  Mathematical Journal*.
- Kloeden, P. E., & Platen, E. (1992). *Numerical Solution of Stochastic
  Differential Equations*. Springer.
- Øksendal, B. (2003). *Stochastic Differential Equations: An Introduction
  with Applications*. Springer. (Standard reference for the OU process
  closed-form solution used in Section 2.)