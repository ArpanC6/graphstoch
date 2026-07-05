# GraphStoch

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21208912.svg)](https://doi.org/10.5281/zenodo.21208912)

Stochastic Differential Equation solver on graph topology for noise robust node state prediction. Julia core engine + Python API.

## Why GraphStoch?

Real world graph data - social networks, crypto transaction graphs, brain connectivity networks - looks powerful, but is riddled with random noise. Standard Graph Neural Networks (GNNs) don't explicitly model this noise, so they end up learning it as if it were signal.

Three domains, one shared problem:

- **Social networks**: users post erratically due to hype, mood, or randomness that doesn't reflect their long term behavior. GNNs treat this one-off noise as signal, degrading recommendations and community detection.
- **Crypto transaction networks**: wallets generate a mix of regular activity, bot traffic, and pure noise (test transfers, airdrops, wash trading). Without separating noise from signal, real threats get diluted and harmless wallets get over-flagged.
- **Brain / neuron networks**: neuron activation data is full of sensor noise and biological jitter. If a model can't separate this from the stable underlying pattern, clinically important structure gets lost in the noise.

The core issue: every node's behavior is a mix of a slow, stable pattern (signal) and fast, random fluctuation (noise). Standard GNNs have no explicit mathematical model for this separation. GraphStoch's approach: treat each node's state as a stochastic process, explicitly simulating both graph diffusion (via the Laplacian) and random noise (via Brownian motion), so the model can distinguish long-term structure from momentary fluctuation instead of conflating the two.

## The Math (short version)

Node states evolve according to

    dX_t = -L X_t dt + sigma dW_t

where `L = D - A` is the graph Laplacian, `-L X_t dt` is a deterministic diffusion term that pulls each node's value toward its neighbors' average, and `sigma dW_t` is explicit Brownian noise. We discretize this via Euler-Maruyama (from scratch, no third-party SDE library, to keep the mechanics transparent) and also provide an adaptive-step SRA3 solver (via `StochasticDiffEq.jl`) for numerical stability on graphs with high-degree hub nodes.

This is a linear, additive-noise SDE - a multivariate Ornstein-Uhlenbeck process with the graph Laplacian as its mean-reversion matrix - which is why it has a closed-form solution, a provable stability threshold, and a provable convergence order. See **`THEORY.md`** for the full derivations; every empirical stability/convergence observation below is a consequence of a proved theorem there, not an independent finding.

## Installation

```bash
git clone https://github.com/ArpanC6/graphstoch.git
cd graphstoch

# Python side (creates/activates a venv under python/venv is recommended)
cd python
pip install -r requirements.txt   # torch, torch_geometric, scipy, scikit-learn, networkx, juliacall

# Julia core is managed automatically by juliacall on first import - no
# separate Julia installation step is required beyond having Julia itself
# on your system (or letting juliacall provision its own isolated Julia).
```

> Note: import `graphstoch` (or anything from `juliacall`) **before** importing `torch` in any script - this ordering avoids a known juliacall/torch initialization conflict.

## Quick Example

```python
from graphstoch import GraphSDE
import numpy as np

# adjacency matrix of your graph (n x n, symmetric, 0/1)
A = np.array(...)

model = GraphSDE(A, noise_level=0.5, dt=0.01)

# safe step size for this specific graph's Laplacian spectrum
safe_dt = model.stable_dt() * 0.5
model = GraphSDE(A, noise_level=0.5, dt=safe_dt)

noisy_features = ...          # (n_nodes, n_features) array
denoised = model.denoise(noisy_features, time_steps=50)
```

## Headline Results

- **Zachary's Karate Club** (real 34-node social network): GraphStoch denoising achieves **~54x lower MSE** than naive neighbor-averaging when recovering a noisy signal (0.0712 vs 3.3835).
- **Cora / Citeseer / PubMed** (denoising-as-GNN-preprocessing): result is dataset-dependent - GNNs win clearly on Cora and PubMed, but GraphStoch+LogReg holds a modest, statistically significant edge on Citeseer.
- **Controlled synthetic homophily sweep**: isolates edge homophily as a *causal* driver of this flip - GNNs win below homophily ~0.7-0.8, GraphStoch wins above it, replicated across 3 independent graph realizations.
- **Spectral detectability threshold**: this crossover is not just empirical - it matches a predicted threshold `h* = 1/2 + 1/(2*sqrt(d)) ~ 0.7041` from random graph theory (Kesten-Stigum/Abbe), independently verified via spectral clustering simulation.
- **Controlled synthetic fragmentation sweep**: tested whether graph fragmentation (isolated nodes) explains why real Citeseer favors GraphStoch despite its homophily being in the "GNNs should win" range. Result: fragmentation does **not** produce a crossover at homophily below h* - instead it revealed a provable mechanism (isolated nodes are an exact fixed point of GraphStoch's diffusion, so they're exempt from - but never benefit beyond - the noisy baseline).

Full methodology, tables, and per-seed statistics for all of the above are in **`BENCHMARKS.md`**.

## Theory Summary

GraphStoch's SDE has a closed-form Ornstein-Uhlenbeck solution, a provable Euler-Maruyama stability threshold (`dt < 2/lambda_max(L)`), and provably achieves strong convergence order 1.0 (a full order better than the general SDE case) because its noise is additive. The homophily crossover in the benchmarks above is explained by an exact spectral detectability threshold from random graph theory, and the fragmentation sweep's null result is explained by an exact algebraic fact about the Laplacian's null space for isolated nodes. Full derivations, proofs, and honest scope statements for all of these are in **`THEORY.md`**.

## Status

- [x] Phase 1: Graph Laplacian construction (from scratch)
- [x] Phase 2: Euler-Maruyama SDE solver (from scratch)
- [x] Phase 3: Python wrapper (juliacall) - `GraphSDE` class with `.simulate()`, `.denoise()`, `.stable_dt()`
- [x] Phase 4: Benchmark vs standard GNN on noisy synthetic data
- [x] Phase 5: SRA3 adaptive solver integration (`StochasticDiffEq.jl`) for improved numerical stability
- [x] Phase 6: Real dataset benchmark (Zachary's Karate Club) with dynamically-derived stable dt
- [x] Phase 7: Cora & Citeseer benchmarks vs GCN / GraphSAGE / GAT
- [x] Phase 8: PubMed benchmark - disentangles structural vs feature-level explanations for the Cora/Citeseer flip
- [x] Phase 9: Controlled synthetic homophily sweep - establishes homophily as a causal driver, replicated across 3 seeds
- [x] Phase 10: Controlled synthetic fragmentation sweep - rules out fragmentation as the Citeseer explanation below h*; identifies the null-space exemption mechanism
- [ ] Phase 11: Near-threshold fragmentation sweep (homophily ~0.65-0.70) - open question, not yet run
- [x] Phase 12a: Zenodo DOI set up for citable, versioned releases (10.5281/zenodo.21208912)
- [ ] Phase 12b: README/BENCHMARKS/THEORY cross-linking pass, arXiv preprint

## Tech Stack

- **Julia**: core simulation engine
- **StochasticDiffEq.jl**: adaptive SRA3 solver for additive-noise SDEs
- **Python**: wrapper API, via juliacall (`GraphSDE` class)
- **NetworkX**: real-world and synthetic graph generation
- **PyTorch Geometric**: Cora/Citeseer/PubMed dataset loading and GNN baselines (GCN, GraphSAGE, GAT)
- **scikit-learn**: Logistic Regression baseline for the denoising-as-preprocessing comparison

## Future Work

- Test fragmentation's effect near the spectral detectability threshold (homophily ~0.65-0.70) rather than well below it - the current fragmentation result rules out one regime, not all of them.
- Target an arXiv preprint once the above is settled.

## Citation

If you use GraphStoch in your own work, please cite this repository:

```
Chakraborty, A. (2026). GraphStoch: Stochastic Differential Equation
modeling for noise-robust node state prediction on graphs (v1.0.0).
Zenodo. https://doi.org/10.5281/zenodo.21208912
```

This DOI always resolves to the latest version. Individual versioned DOIs will be added for future releases.

## License

MIT