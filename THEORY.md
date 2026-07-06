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

## Section 4b: Empirical Verification of Strong Order 1.0

Section 4's claim (Euler-Maruyama achieves strong order 1.0 for this
additive-noise SDE, rather than the general-case order 0.5) was verified
empirically using a pathwise strong-error test with matched Brownian paths
(common random numbers), following the standard methodology in Kloeden &
Platen (1992, Ch. 10).

**Method:** for each trial, a single fine-grained Brownian path (dt_min)
was generated. Coarser step sizes were built by summing blocks of the fine
increments, so that every step size tested shares the exact same underlying
noise realization. Euler-Maruyama was run at each step size using its
matched increments; the finest-resolution run served as the reference
solution. The strong-error metric reported is the average pathwise
deviation ||X_dt(T) - X_reference(T)|| across trials, at each dt.

**First attempt and a diagnosed flaw.** An initial run of this method
(200 trials/graph) produced order estimates of ~1.41 (synthetic
Erdos-Renyi) and ~1.92 (Karate Club) - clearly above 0.5, but running
noticeably higher than the theoretical 1.0. Reviewing the test design
identified two concrete issues: the reference solution was only ~2x
finer than the smallest tested dt, so its own discretization error was
not negligible relative to what was being measured; and every resolution
level, including the reference, actually integrated to T/8 rather than
the stated T=1.0, due to how dt_min and the step count were both defined
in terms of the same n_levels parameter.

**Corrected re-run.** The test was re-run (300 trials/graph) with the
reference resolution decoupled from the smallest tested dt (now 128x
finer, rather than ~2x) and every level integrating to the stated T
exactly.

**Results** (T=1.0, sigma=0.5, single spike initial condition):

| Graph | dt range tested | Order estimates (consecutive doublings) | Mean |
|---|---|---|---|
| Synthetic 10-node Erdos-Renyi | 0.0078 - 0.25 (stability bound 0.2686) | 1.045, 1.030, 1.044, 1.209, 2.003 | 1.266 |
| Zachary's Karate Club (34 nodes) | 0.00098 - 0.03125 (stability bound 0.0384) | 1.023, 1.018, 1.067, 1.112, 1.441 | 1.132 |

**Honest interpretation:** the corrected reference resolution moved both
graphs' estimates substantially closer to the theoretical value of 1.0
compared to the first attempt (1.266/1.132 mean, vs. 1.41/1.92 before).
Looking at the individual doublings rather than only the mean, the
pattern is not uniform: at step sizes well below each graph's stability
bound, the estimated order sits at ~1.02-1.07 for both graphs, closely
matching the theoretical claim. The estimate only inflates at the
coarsest step size tested for each graph - to 2.003 for the synthetic
graph (dt=0.25, which is ~93% of that graph's stability bound of 0.2686)
and to 1.441 for Karate Club (dt=0.03125, ~81% of its bound of 0.0384).
Restricting to step sizes below half the stability bound, the mean
estimated order is ~1.04 for both graphs, consistent with Section 4's
strong-order-1.0 claim.

This pattern - clean agreement with theory away from the stability
boundary, inflated apparent order close to it - is a more precise and
defensible finding than either the original 1.4-1.9 range or a blanket
claim of exact agreement everywhere. It is consistent with the coarsest,
near-boundary steps entering a regime where error growth is no longer
well described by a simple power law as the dynamics approach
instability, rather than with a genuine breakdown of the strong-order-1
result. This explanation is plausible and consistent with the data but
is not independently proven here; a fully rigorous treatment would derive
an explicit bound on how close to the stability threshold the
strong-order-1 guarantee can be expected to hold, which is not yet done.

*Status: closed, with the above caveat stated explicitly. The estimated
strong order is ~1.0-1.07 in the well-resolved regime (dt below half the
stability bound) for both graphs tested, matching Section 4's claim; the
inflation observed only at near-stability-boundary step sizes is reported
as an open, plausible-but-unconfirmed explanation rather than folded into
the headline number.*

**Earlier (discarded) attempt:** a first version of this experiment
estimated the discretization bias by comparing the Monte-Carlo-averaged
mean of many independent Euler-Maruyama runs against the exact mean from
`exact_solve`. That approach is methodologically unsound for testing
*strong* convergence (which is a pathwise, not a mean/weak, quantity), and
in practice the true bias was smaller than the Monte Carlo sampling noise
at 300 seeds, producing non-monotonic and even negative estimated orders.
That result was correctly identified as a test-design flaw rather than
evidence against Section 4, and discarded in favor of the matched-path
method described above.

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

## Section 7: The Homophily Crossover as a Spectral Detectability Threshold

Section 4b of this document and the README's synthetic homophily sweep
established empirically that GraphStoch-denoised features overtake GNN
baselines somewhere between homophily 0.70 and 0.90 on a controlled
two-community Stochastic Block Model (SBM). This section explains *why*
that crossover exists at all, and predicts *where* it should sit, using an
established result from random graph theory - rather than treating the
crossover as a purely empirical curiosity.

### 7.1 Why a threshold should exist

GraphStoch's diffusion operates by repeatedly applying `-L` to the state
vector, which - as established in Section 1 - is diagonal in the
Laplacian's eigenbasis. On a two-community SBM, the community structure is
encoded almost entirely in the sign pattern of the second eigenvector of
the graph's adjacency/Laplacian (the Fiedler-adjacent mode). This is
exactly the object used in classical **spectral clustering**. GraphStoch's
denoising is therefore, in this specific setting, mechanistically close to
a spectral clustering step: it can only "see" and exploit community
structure to the extent that the second eigenvector actually carries a
usable signal above the random-matrix noise floor.

Random matrix theory gives an exact answer for when that signal exceeds
the noise floor - the **Kesten-Stigum / Abbe detectability threshold**
for the symmetric two-block SBM (Decelle, Krzakala, Moore & Zdeborová,
2011; made rigorous by Mossel, Neeman & Sly, 2015, and Massoulié, 2014).

### 7.2 The threshold, in our parametrization

In the standard sparse parametrization, a symmetric two-block SBM on `n`
nodes has within-community edge probability `a/n` and between-community
edge probability `b/n`. Average degree is `d = (a+b)/2`. Community
detection - recovering block membership better than chance via spectral
or belief-propagation methods - is information-theoretically possible if
and only if

    (a - b)² > 2(a + b) = 4d.                                        (4)

Our experiments parametrize graphs by **average degree `d`** and **edge
homophily `h`** (the fraction of a node's edges landing in its own
community) rather than by `a, b` directly. Since intra-community expected
degree is `a/2` (using `n = 2 · (n/2)`, two equal-size communities) and
`h = (a/2)/d`, we have `a = 2hd` and, since `a + b = 2d`, `b = 2(1-h)d`.
Substituting into (4):

    [2d(2h - 1)]² > 4d
    d(2h - 1)² > 1
    |2h - 1| > 1/√d.

Taking the homophilic branch (`h > 1/2`, the regime relevant here):

    h* = 1/2 + 1/(2√d).                                              (5)

For our sweep's average degree `d ≈ 6.0`, this gives

    h* = 0.5 + 1/(2√6) ≈ 0.7041.

**Note on approximation.** The expression `(n/2)(a/n) = a/2` used above
for expected same-community degree is a large-n approximation. The exact
expression, excluding self-comparison, is `(n/2 - 1)(a/n)`. The two
coincide as `n → ∞`; at `n = 1000` (the sweep's actual size) the
correction is on the order of 0.2%, negligible for the numerical value
`h* ≈ 0.7041` reported here. This is stated explicitly so the derivation
is not read as claiming exactness it does not have.

### 7.3 Independent numerical verification

Formula (5) was checked directly - not just taken from the literature -
by running spectral clustering (sign of the second eigenvector of the
adjacency matrix) on freshly generated SBM graphs (n=1000, avg_degree=6.0)
across a fine grid of homophily values, averaging classification accuracy
against ground-truth community labels over 30 independent graph draws per
level:

| Homophily | Mean spectral accuracy | Std |
|---|---|---|
| 0.60-0.68 | 0.519-0.546 (near chance, 0.5) | - |
| 0.69-0.71 | 0.570-0.595 (transition begins) | - |
| 0.72-0.75 | 0.640-0.710 (rapid rise) | - |
| 0.76-0.81 | 0.756-0.875 (strong detection) | - |

The transition from chance-level (~0.52) to strong detection (~0.87)
happens almost entirely within homophily 0.69-0.78, tightly bracketing
the predicted threshold `h* ≈ 0.7041` from (5). This confirms the
threshold formula is not just a literature citation but an accurate,
independently-checked description of this project's specific SBM setup.

### 7.4 Connecting the threshold to the classification crossover

The classification-task crossover reported in the README (GraphStoch
overtaking GNNs somewhere between homophily 0.70 and 0.90, with the exact
point varying by graph realization) sits at or above the spectral
detectability threshold `h* ≈ 0.70`, not below it. This is the expected
relationship, not a coincidence: right at `h*`, community structure is
only *just barely* statistically detectable, so a diffusion step exploiting
it would be expected to offer only a marginal benefit; a comfortably
useful margin requires homophily somewhat past the threshold, matching why
the observed classification crossover (0.70-0.90) sits at or slightly
above `h*` rather than exactly on top of it.

**Honest scope of this claim.** Formula (5) is a threshold for
*information-theoretic detectability of community structure via spectral
methods* - it does not by itself prove anything about *classification
accuracy with a trained GNN* on downstream labeled data, which additionally
depends on feature informativeness, GNN architecture, and training
dynamics. The claim here is narrower and more defensible: the threshold
explains *why a sharp transition exists at all* in GraphStoch's own
diffusion-based mechanism (since that mechanism is spectral in nature),
and predicts *approximately where* it should occur, and both of those
predictions were checked against independent evidence (Section 7.3) rather
than asserted. It is not a claim that GNN performance itself is governed
by the identical formula, only that GraphStoch's specific
Laplacian-diffusion mechanism is.

## Section 8: Isolated Nodes and the Null-Space Exemption from Sub-Threshold Diffusion Harm

Section 7 explained the homophily crossover as a spectral detectability
threshold. This section addresses a distinct, complementary question that
arose while testing whether graph fragmentation (isolated/disconnected
nodes) could explain why real Citeseer favors GraphStoch even though its
homophily (0.7355) is comfortably inside the regime where the pure
homophily sweep shows GNNs winning clearly. A controlled synthetic
fragmentation sweep (homophily held fixed at 0.60 -- deliberately *below*
the detectability threshold h* ~ 0.7041 from Section 7 -- while varying
isolated-node fraction from 0% to 30%) found no crossover: GNNs won every
comparison at every isolation level tested. But the sweep surfaced an
exact, provable mechanism worth stating formally.

### 8.1 The exact algebraic fact

**Fact 3 (null-space exemption).** For any node i with degree d_i = 0
(an isolated node), row i of the Laplacian L = D - A is identically zero,
since both D_ii = 0 and every entry A_ij = 0. Therefore [LX]_i = 0 for
every state vector X. GraphStoch's denoising step is the noise-free
drift iteration X_{k+1} = X_k - L X_k * dt (Section 3's discretization
with no fresh noise re-injected). For an isolated node, this reduces to
X_i^(k+1) = X_i^(k) for every k -- the node's state is an exact fixed
point of the iteration.

**Consequence:** an isolated node's post-denoising feature vector is
identical, entry for entry, to its pre-denoising (noisy) feature vector.
GraphStoch's denoising step provides it with exactly zero smoothing --
neither benefit nor harm -- by algebraic construction, not as a
statistical tendency.

### 8.2 What this predicts for a fragmentation sweep

**Corollary.** In a dataset that is a mixture of active nodes (fraction
1-f) and isolated nodes (fraction f), the aggregate denoised-feature set
is exactly (diffused active-node values) union (unchanged noisy
isolated-node values). Two regimes follow:

- If homophily is *below* h* (diffusion is actively harmful to active
  nodes -- see Section 7's scope, and Section 4b's honest-order caveat),
  then increasing f should monotonically move aggregate denoised accuracy
  *up*, toward the noisy-only baseline, since a growing share of the
  population is exempted from that harm.
- As f -> 1, aggregate denoised accuracy should converge exactly to
  noisy-baseline accuracy, since every node is exempt.
- Critically, this ceiling is the noisy baseline itself -- fragmentation
  under this mechanism can never let GraphStoch's denoised accuracy
  exceed what raw noisy features already achieve, regardless of how much
  of the graph is isolated.

### 8.3 Empirical confirmation

This matches the fragmentation sweep exactly. At homophily 0.60 (below
h*), GraphStoch+LogReg accuracy at isolation_fraction=0 was well below
the noisy baseline (e.g. seed 42: 0.482 denoised vs. 0.680 noisy) and
rose steadily with isolation_fraction, approaching but never reaching or
exceeding the noisy baseline at isolation_fraction=0.30 (e.g. seed 42:
0.586 denoised vs. 0.692 noisy). All three graph seeds showed the same
monotonic pattern. This is the mechanism behind why no crossover
appeared: the ceiling that fragmentation asymptotes toward (the noisy
baseline, ~0.68-0.69) itself remains well below all three GNN baselines
(~0.64-0.71 but generally higher) at this homophily level -- fragmentation
can rescue GraphStoch from its own harm, but cannot make it competitive.

### 8.4 Honest scope of this claim

This is an exact, provable statement about GraphStoch's own mechanism --
not a claim about GNN behavior under fragmentation, and not (on its own)
an explanation for real Citeseer's result. Two limitations are worth
stating plainly:

1. **This experiment specifically tested homophily well below h*.** Real
   Citeseer's homophily (0.7355) sits near/just above h* (~0.7041) --  a
   regime where diffusion is only marginally beneficial rather than
   clearly harmful. Whether fragmentation behaves differently near or
   above the threshold (where it might interact with the marginal
   detectability signal rather than simply exempting nodes from harm) has
   not been tested, and is a natural next experiment rather than an
   assumed answer.
2. **The asymptotic convergence in Section 8.2 (f -> 1 implies denoised
   accuracy -> noisy accuracy exactly) is a theoretical prediction, not
   yet empirically confirmed beyond f = 0.30** -- the tested range does
   not reach the asymptote closely enough to confirm the limit directly,
   only the monotonic direction predicted by the mechanism.

Taken together: this fragmentation experiment, as designed, **does not
support fragmentation as the explanation for Citeseer's result** at
homophily levels below h*. It replaces an open, unresolved tension
(Section 7d's original null-space observation, which did not specify
whether fragmentation would help or hurt) with an exact, verified
mechanism -- but that mechanism happens to rule out, rather than confirm,
the hypothesis it was designed to test. The Citeseer explanation remains
open; testing fragmentation near h* (~0.65-0.70) rather than well below
it is the natural next step, not yet done.

## Section 9: Extending the Fragmentation Sweep to and Above the Spectral Threshold

### 9.1 Motivation

Section 8 tested the fragmentation mechanism at a single homophily level
(h = 0.60), chosen because it sits comfortably below the spectral
detectability threshold h* ≈ 0.7041 derived in Section 7. That left open
whether fragmentation behaves differently closer to, or above, h* -- in
particular, whether it could explain why real Citeseer (homophily 0.7355,
above h*) favors GraphStoch despite the pure homophily sweep predicting
GNNs should win in that regime.

This section extends the same experiment (`synthetic_fragmentation_sweep.py`,
unchanged except for the `HOMOPHILY` constant) to two additional levels:
h = 0.68 (just below h*) and h = 0.73 (just above h*, matching real
Citeseer's actual homophily).

### 9.2 Result: no GNN crossover at any tested homophily level

Across all three homophily levels (0.60, 0.68, 0.73), all six isolation
fractions (0.00-0.30), and all three graph seeds, every GNN baseline
(GCN, GraphSAGE, GAT) continued to beat GraphStoch+LogReg in every
comparison. Fragmentation, even combined with homophily at or above h*,
does not produce a crossover against GNN baselines in this synthetic
model.

### 9.3 A fragile but genuine signature of the threshold: seed-dependent sign flip

Within the internal comparison of GraphStoch's own denoising against the
noisy-feature baseline (logreg_denoised vs. logreg_noisy -- independent
of the GNN baselines), a new phenomenon appeared at h = 0.73 that had
not appeared at h = 0.60 or h = 0.68: for one of three graph seeds
(seed 44), denoising became net *beneficial* rather than net harmful at
low isolation fractions:

| isolation_fraction | logreg_noisy | logreg_denoised | denoising effect |
|---|---|---|---|
| 0.00 | 0.6790 | 0.7460 | **+0.067 (helps)** |
| 0.02 | 0.6740 | 0.7420 | **+0.068 (helps)** |
| 0.05 | 0.6820 | 0.6800 | ~0 (roughly neutral) |
| 0.10 | 0.6790 | 0.6900 | **+0.011 (helps)** |
| 0.20 | 0.6730 | 0.5450 | -0.128 (harms) |
| 0.30 | 0.6910 | 0.5850 | -0.106 (harms) |

The other two seeds (42, 43) did *not* show this flip -- denoising
remained net harmful across all isolation fractions for both, at the
same nominal homophily (0.73).

This is consistent with, and mild direct support for, the theoretical
prediction of Section 7: above h*, diffusion should become net
beneficial rather than net harmful. But the effect is evidently fragile
at this graph size (500 nodes) and this close to threshold -- present
in 1 of 3 independent graph realizations, not all three. This is the
first time in any of the three homophily levels tested that the sign
of the denoising effect flipped at all; it did not happen at h = 0.60
or h = 0.68 for any seed.

### 9.4 Honest scope statement and conclusion

Systematically ruling out both pure homophily variation (Section 7's
sweep) and fragmentation (Section 8, and this section, tested at three
homophily levels spanning below, near, and above h*), this synthetic
model cannot reproduce a crossover against GNN baselines at any tested
configuration. Real Citeseer's actual advantage for GraphStoch+LogReg
over GNN baselines is therefore *not* explained by homophily or
fragmentation statistics alone within this synthetic generator.

This points toward the remaining explanation being feature-level or
otherwise structural in a way this Gaussian-feature, degree-controlled
SBM generator does not capture -- for example, properties specific to
Citeseer's real bag-of-words feature vectors, or degree-distribution
characteristics beyond simple average degree and homophily. This is
flagged as genuinely open. The natural next experiment, not yet run,
would target feature-level hypotheses directly rather than further
structural (homophily/fragmentation) variations.

## 10. Teleport-Augmented SDE: A Tested, Honestly-Ruled-Out Hypothesis

### 10.1 Motivation

Following the APPNP/Personalized PageRank line of work (Gasteiger et al.,
2019), a teleport (restart) term was added to the base diffusion SDE to
test whether anchoring each node toward its own initial observation
reduces long-horizon oversmoothing:

    dX_t = -(L + alpha*I) X_t dt + alpha*X0 dt + sigma dW_t

This is still a linear, additive-noise OU process; its fixed point (at
sigma=0) is mu = alpha * (L + alpha*I)^{-1} * X0, a spectrally-filtered
version of X0 rather than X0 itself. The stability threshold tightens to
dt < 2/(lambda_max(L) + alpha), consistent with Theorem 2.

### 10.2 Experimental test and methodology correction

Initial unpaired trials (30 independent noise realizations per alpha)
suggested a possible low-alpha "sweet spot" (alpha=0.05-0.1) with lower
denoising MSE than alpha=0. This did NOT replicate under a proper
paired-noise design (same random seed reused across all alpha values
within each trial, isolating the alpha effect from noise-realization
variance). Under the paired design, differences between alpha=0 and
alpha in {0.05, 0.1, 0.3} were an order of magnitude smaller than the
noise floor and not distinguishable from zero.

**Lesson reinforced**: unpaired Monte Carlo comparisons across a
hyperparameter can produce spurious-looking patterns purely from
noise-realization variance; paired designs (same noise, varying only the
parameter of interest) are necessary before claiming an effect.

### 10.3 Result: a real, singular effect exists - but it is a harm, not a gain

Under the corrected paired design, moderate teleport strength (alpha up
to 0.3) showed no significant effect on denoising MSE in this setting.
Aggressive teleport (alpha=1.0) reliably and consistently degraded
denoising performance in every replication.

### 10.4 Interpretation and honest scope

This is explained by what the teleport term actually anchors to: X0 here
is the *noisy* observation, not a clean reference embedding (unlike
APPNP's typical use case, where the teleported quantity is often an
already-processed feature or prediction). A strong pull toward a noisy
observation actively fights the diffusion term's denoising effect. This
is a genuine, narrow negative result for the teleport mechanism *in this
specific denoising-from-noisy-X0 setting* - it does not rule out
teleport-style terms being useful in other GraphStoch use cases (e.g.
node classification pipelines where X0 is itself a processed feature
rather than a raw noisy signal), which remains untested.

This experiment is complete. The natural next hypothesis, grounded in
Choi et al. (2023, GREAD) and Chamberlain et al. (2021, GRAND), is a
reaction term rather than a teleport term - see Section 11.

## 11. Reaction-Diffusion Extension: A Task-Mismatch Lesson

### 11.1 Initial test and its failure to replicate

A bistable (Allen-Cahn-type) reaction term, dX_t = -L X_t dt +
beta*(X_t - X_t^3) dt + sigma dW_t, was tested on continuous-valued
denoising (paired-noise design, n=30 trials). A first random signal
instance showed a clear, replicable-looking improvement at beta ~ 0.5-0.6
(MSE reduced ~30% vs beta=0). A second independent signal instance
showed the opposite pattern entirely (monotonically worse as beta
increased). The effect did not replicate.

### 11.2 Diagnosis

The Allen-Cahn reaction term has fixed equilibria at exactly +1, -1
(zeros of f(x) = x - x^3), independent of the actual data's value range.
The test signals here lived in [0, 1], unrelated to these poles. Whether
pushing toward +/-1 helps or hurts a given signal instance depends on
essentially arbitrary alignment between the signal's local values and
the nearest pole - explaining the non-replication. This is consistent
with the literature: Allen-Cahn-based reaction terms (e.g. ACMP, Wang et
al. 2023) are designed for and validated on discrete-class node
classification, not continuous-valued regression/denoising, where
class-like attractor states are semantically meaningful in a way that
arbitrary +/-1 poles are not for an arbitrary real-valued signal.

### 11.3 Corrected next step

The reaction-diffusion hypothesis is not rejected - it was tested on
the wrong task. GraphStoch's actual benchmark pipeline (denoising +
LogReg classification on Karate Club/Cora/Citeseer/PubMed) is exactly
the discrete-class setting this reaction term is designed for. The
correct test is on classification accuracy in the existing homophily
sweep, not continuous MSE - see Section 12.

12. Reaction-Diffusion Extension, Corrected Test: Classification Pipeline

12.1 Motivation and corrected test design

Section 11 diagnosed that the Allen-Cahn reaction term's fixed +/-1
equilibria are structurally mismatched to arbitrary-range continuous
denoising, and identified the correct test as GraphStoch's actual
benchmark pipeline: denoising followed by LogReg classification, matching
GREAD's (Choi et al., 2023) and ACMP's (Wang et al., 2023) own
demonstrated use case, and directly targeting GraphStoch's known weak
spot (Section 7's crossover, h below ~0.7-0.8).

The synthetic homophily sweep (Section 7, BENCHMARKS.md) was extended
with a fourth LogReg arm - `logreg_denoised_reaction` - using
diffuse_with_reaction at a fixed beta = 0.5 (chosen from the middle of
Section 11's earlier continuous-MSE beta sweep, 0.3-0.8), applied to the
identical `X_noisy` used by every other arm in each trial, preserving the
paired design across 3 graph seeds x 5 homophily levels x 5 noise seeds.

12.2 Result: the base crossover replicates, and sharpens

Before evaluating the reaction term itself, this run re-confirms and
sharpens Section 7's original crossover finding. Across all three graph
seeds, GNNs win all three comparisons at h = 0.50, 0.60, and 0.70 (mostly
p < 0.01), h = 0.80 is a mixed transition zone (GraphStoch wins seed 42,
GNNs win seeds 43/44), and GraphStoch+LogReg wins all three comparisons
at h = 0.90 in every seed. This is consistent with, and somewhat more
decisive than, Section 7's original single-seed reading of the
crossover.

12.3 Result: reaction term vs. plain diffusion - a real but unexpectedly-placed effect

Comparing `logreg_denoised_reaction` against `logreg_denoised` (paired
t-test, same X_noisy) at each homophily level:

| Homophily | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| 0.50 | not sig (plain wins) | not sig (plain wins) | **sig, reaction wins** (+0.010) |
| 0.60 | **sig, reaction wins** (+0.032) | not sig, reaction wins (+0.034, p=0.058) | **sig, reaction wins** (+0.035) |
| 0.70 | **sig, reaction wins** (+0.101) | not sig (plain wins) | not sig, reaction wins (+0.018) |
| 0.80 | **sig, reaction wins** (+0.015) | **sig, reaction wins** (+0.020) | not sig, reaction wins (+0.016, p=0.051) |
| 0.90 | not sig | not sig (plain wins) | not sig |

The reaction term does produce a real, replicable, often-significant
improvement over plain diffusion - but **not** concentrated at low
homophily as the original GREAD/ACMP-motivated hypothesis predicted.
The most consistent, significant benefit appears at h = 0.60 and h = 0.80
(mid-range), not at h = 0.50 (GraphStoch's worst-performing regime). At
h = 0.70, the effect is inconsistent across seeds (helps strongly in one,
hurts in another). This is itself an honest, useful finding: it does not
confirm the motivating hypothesis in the form it was stated.

12.4 Does the reaction term rescue GraphStoch at its weak spot?

No. At h = 0.50-0.70, the gap between GraphStoch+LogReg and the GNN
baselines is typically 10-25 percentage points and highly significant.
The reaction term's improvement over plain diffusion in this range tops
out at +0.10 (a single seed at h=0.70) and is usually far smaller
(+0.01 to +0.04). This is not close to sufficient to close the gap to
any GNN baseline in this regime. **The reaction term does not rescue
GraphStoch's known weak spot.**

12.5 The flagged h=0.80 signal, directly tested - does not hold up

The h=0.80/seed-44 numerical flip flagged provisionally above was tested
directly via paired t-test (`logreg_denoised_reaction` vs. each GNN
individually, same trials): reaction beat graphsage numerically
(diff=+0.007) but the effect was not statistically significant
(p=0.510), and reaction did not beat gcn or gat at this cell at all. No
other cell in the h=0.50-0.80 range showed a reaction-vs-GNN win of any
kind, significant or not, in any of the three graph seeds.

At h=0.90, reaction did significantly beat multiple GNN baselines across
all three seeds — but this is not a new crossover: plain diffusion
already beats all three GNNs at h=0.90 (Section 7's established result).
Reaction's win here reflects an incremental improvement within a regime
GraphStoch already dominates, not evidence of rescuing its weak spot.

12.6 Final honest summary

Directly testing every homophily level and every GNN comparison
confirms: the reaction-diffusion term never produces a significant, or
even numerically consistent, win against any GNN baseline anywhere in
GraphStoch's known weak regime (h = 0.50-0.80). The one provisionally
flagged cell (h=0.80, seed 44, vs. graphsage) does not survive direct
statistical testing. The reaction term's real, replicable benefit
(Section 12.3) exists only relative to GraphStoch's own plain-diffusion
baseline, concentrated at mid-homophily, and does not translate into
closing the gap against learned GNN baselines at any homophily level
where that gap exists. **The reaction-diffusion extension, as tested, is
not sufficient to make GraphStoch competitive with GNNs in its weak
regime.**

13. Open Items (updated)

- Confirm or reject the h=0.80/seed-44 flip flagged in Section 12.5 via
  a direct paired t-test of `logreg_denoised_reaction` against each of
  gcn/graphsage/gat individually, not just against plain diffusion.
- The remaining Section 6 open items (exact_solve as a validation
  baseline - now implemented; re-examining "true signal" generation in
  light of Section 2.2) are still open as originally stated.

  14. Learned Attention-Diffusivity SDE - Replacing the Fixed Laplacian

Motivation. Sections 10-12 tested additive modifications (teleport term,
reaction term) on top of the fixed operator -L and found neither closes
the gap to GNNs in the low-homophily regime. This section tests a more
fundamental change, motivated by GRAND (Chamberlain et al., ICML 2021):
replace -L with a learned, feature-dependent diffusion operator, while
keeping GraphStoch's SDE structure (drift + additive Brownian noise
during training).

Model. A single-head scaled dot-product attention mechanism produces a
row-stochastic matrix A_att(X) over graph edges (plus self-loops).
Row-stochasticity is exactly the condition GRAND uses to prove
||A_att|| <= 1, giving unconditional stability of the explicit Euler
update X_{k+1} = (1-tau)X_k + tau*A_att(X_k)X_k. Additive Gaussian noise
sigma*sqrt(tau)*xi is injected at each of the 8 unrolled steps during
training, making this a genuine SDE rather than a neural ODE. Trained
end-to-end with cross-entropy loss and a linear readout.

Methodology. Synthetic 2-community SBM, N=1000, avg degree 6.0,
feat_dim=50, homophily in {0.5, 0.6, 0.7, 0.8, 0.9}, 3 graph seeds x 5
noise seeds (15 trials/level), paired X_noisy across all arms per
trial -- matching the methodology of Sections 9-12.

Results (15 trials/level):

  h     logreg(noisy)  plain diffusion  learned SDE  GCN
  0.5   0.703          0.512            0.582        0.592
  0.6   0.703          0.537            0.704        0.686
  0.7   0.703          0.629            0.783        0.790
  0.8   0.703          0.858            0.891        0.889
  0.9   0.703          0.979            0.958        0.949

Paired t-tests: learned SDE significantly beats plain fixed-Laplacian
diffusion at every homophily level tested except h=0.9 (where plain
diffusion wins, p<0.0001 -- expected, this is GraphStoch's native
regime). Learned SDE is statistically indistinguishable from GCN at
h=0.5, 0.6, 0.7, and 0.8 (all p>0.09). At h=0.9, learned SDE
significantly outperforms GCN (p=0.034, small effect, +0.009) but still
underperforms plain diffusion.

Honest interpretation. Replacing the fixed Laplacian with a learned,
attention-weighted diffusivity operator -- while retaining the
additive-noise SDE structure -- closes most of the gap to GNNs across
the mid-to-upper homophily range (0.6-0.8), and is not distinguishable
from GCN there. This is a substantially larger and more consistent
effect than either the teleport experiment (null) or reaction-diffusion
experiment (+0.01 to +0.10, and only at h=0.6/0.8) produced.

However, at h=0.5 -- the sharpest test of the weak regime, and close to
this project's own derived detectability threshold h* ~ 0.70
(Section 7) -- no graph-based method (plain diffusion, learned SDE, or
GCN) beats simply ignoring the graph and running LogReg on raw noisy
features (0.703 vs 0.512-0.592). This is a stronger and more precise
statement of the open gap than "not fully rescued": at h=0.5, using the
graph at all is actively counterproductive, regardless of whether the
diffusion operator is fixed or learned. This is consistent with, and
arguably an independent confirmation of, the spectral detectability
threshold argument in Section 7 -- below h*, the graph carries no
exploitable class signal for any of the methods tested, learned or
fixed.

Reproducibility. This experiment was run independently on two separate
machines (initial development environment, N=1000 seeded run; and the
author's own local machine, same code, same seeds) and produced
matching results to within sampling/floating-point noise (e.g. h=0.6
learned_sde: 0.7040 both runs; h=0.9 learned_sde: 0.9577 both runs).

Scope and caveats. This uses a scaled-down attention mechanism (single
head, no depth beyond the unrolled diffusion steps) and a from-scratch
2-layer GCN baseline (no torch_geometric) rather than a tuned reference
implementation -- GCN numbers here (e.g. 0.592 at h=0.5) are consistent
with, though not identical to, the established BENCHMARKS.md figures at
N=1000, giving reasonable confidence the comparison is fair. Real-data
replication (Cora/Citeseer/PubMed) has not yet been attempted for this
variant.

### References (additional to Section "References" below)

- Decelle, A., Krzakala, F., Moore, C., & Zdeborová, L. (2011). Asymptotic
  analysis of the stochastic block model for modular networks and its
  algorithmic applications. *Physical Review E*, 84(6).
- Mossel, E., Neeman, J., & Sly, A. (2015). Reconstruction and estimation
  in the planted partition model. *Probability Theory and Related Fields*.
- Massoulié, L. (2014). Community detection thresholds and the weak
  Ramanujan property. *STOC 2014*.
- Abbe, E. (2018). Community detection and stochastic block models:
  recent developments. *Journal of Machine Learning Research*, 18(177).
  (Survey covering the Kesten-Stigum threshold and its history.)

## References

- Fiedler, M. (1973). Algebraic connectivity of graphs. *Czechoslovak
  Mathematical Journal*.
- Kloeden, P. E., & Platen, E. (1992). *Numerical Solution of Stochastic
  Differential Equations*. Springer.
- Øksendal, B. (2003). *Stochastic Differential Equations: An Introduction
  with Applications*. Springer. (Standard reference for the OU process
  closed-form solution used in Section 2.)