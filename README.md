# GraphStoch

Stochastic Differential Equation solver on graph topology for 
noise robust node state prediction. Julia core engine + Python API.

## Why GraphStoch?

Real world graph data - social networks, crypto transaction graphs, 
brain connectivity networks looks powerful, but is riddled with 
random noise. Standard Graph Neural Networks (GNNs) don't explicitly 
model this noise, so they end up learning it as if it were signal.

**Three domains, one shared problem:**

 **Social networks**: users post erratically due to hype, mood, or 
  randomness that doesn't reflect their long term behavior. GNNs 
  treat this one off noise as signal, degrading recommendations and 
  community detection.
- **Crypto transaction networks**: wallets generate a mix of regular 
  activity, bot traffic, and pure noise (test transfers, airdrops, 
  wash trading). Without separating noise from signal, real threats 
  get diluted and harmless wallets get over-flagged.
- **Brain / neuron networks**: neuron activation data is full of 
  sensor noise and biological jitter. If a model can't separate this 
  from the stable underlying pattern, clinically important structure 
  gets lost in the noise.

**The core issue**: every node's behavior is a mix of a slow, stable 
pattern (signal) and fast, random fluctuation (noise). Standard GNNs 
have no explicit mathematical model for this separation they 
assume observed data is roughly clean, so noise leaks directly into 
the learned weights.

**GraphStoch's approach**: treat each node's state as a stochastic 
process. Explicitly simulate both graph diffusion (via the Laplacian) 
and random noise (via Brownian motion), so the model can distinguish 
long term structure from momentary fluctuation instead of 
conflating the two.

## The Math

### Graph Laplacian (L = D - A)

The Laplacian measures how different each node is from its 
neighbors, and provides the mechanism to smooth that difference out 
over the graph.

- **Adjacency matrix (A)**: encodes which nodes are connected.
- **Degree matrix (D)**: diagonal matrix where D[i,i] = number of 
  connections node i has.
- **Laplacian**: L = D - A

Given a state vector x over the graph, Lx measures how far each 
node's value is from its neighbors' values. Nodes close to their 
neighbors have small Laplacian values; outliers have large ones.

This makes -Lx a diffusion operator: repeatedly applying it pulls 
every node's value toward the average of its neighbors, smoothing 
out noisy, disconnected values into a stable pattern.

### The Stochastic Differential Equation

We model each node's state evolution as:

    dX_t = -L X_t dt + σ dW_t

This says node states change over time due to two combined effects:

**1. Deterministic part: -L X_t dt**

- X_t is the vector of all node values at time t.
- L is the graph Laplacian, capturing how different each node is 
  from its neighbors.
- -L X_t dt pulls every node's value a little closer to its 
  neighbors' average at each instant.

This part is fully predictable: given X_t and L, you know exactly 
which direction the system moves next. This is the diffusion / 
smoothing behavior.

**2. Noise part: σ dW_t**

- W_t is a Wiener process (Brownian motion) continuous time, pure 
  random noise.
- dW_t is a tiny random increment of that noise.
- σ controls the noise strength larger σ means bigger random jitter.

This term tells the model: "real data has randomness, so we add a 
stochastic term explicitly instead of ignoring it."

**Put together**: node states drift toward their neighborhood's 
average, but the path isn't a straight line it's a random walk 
around that trend, and we model both parts mathematically instead 
of averaging them away.

### Euler-Maruyama: from continuous math to discrete code

Computers can't simulate continuous time they work in discrete 
steps. Euler Maruyama approximates the SDE by breaking it into small 
time steps:

    X_{t+Δt} = X_t - L X_t · Δt + σ√Δt · Z

Where:
- Δt is a small time step (e.g. 0.01, 0.1)
- Z is a standard normal random vector, sampled fresh at each step
- √Δt correctly scales the noise to match Brownian motion's 
  statistical properties

**Deterministic part in code**: at each step, compute -L X_t, scale 
by Δt, and add to X_t. This is pure matrix multiplication - fully 
predictable.

**Noise part in code**: at each step, sample a random vector Z 
(`randn` in Julia), scale it by σ√Δt, and add it to X_t. This 
injects small random jumps that approximate continuous Brownian 
motion.

We implement this from scratch (no third-party SDE libraries) to 
show the mechanics explicitly rather than treating the solver as a 
black box.

## Results So Far

Simulating a 4 node chain graph with an initial spike at node 1, 
comparing clean diffusion vs. the stochastic (noisy) version:

![Clean vs Noisy Diffusion](notebooks/all_nodes_diffusion.png)

Even with random noise injected at every step, the noisy trajectory 
(red) tracks the same overall trend as the clean deterministic 
diffusion (blue) demonstrating that the Laplacian's structural pull 
dominates over random fluctuation.

## Benchmark: GraphStoch vs Naive Neighbor Averaging

We compared GraphStoch's diffusion process against a naive GNN-style 
baseline (iterative neighbor averaging) on a 30-node random graph 
with heavy noise (σ=2.0 relative to signal scale).

**Key finding**: naive averaging converges quickly but plateaus at a 
suboptimal error (~0.237 MSE) because repeated unweighted averaging 
over-smooths the graph nodes lose their individual structure and 
collapse toward a single average value. GraphStoch continues to 
improve with more steps (0.36 -> 0.20 -> 0.17 MSE) since the 
Laplacian based dynamics preserve more of the underlying graph 
structure.

We also found that step size (dt) is critical for the Euler-Maruyama 
solver: dt=0.1 gives stable convergence, but dt≥0.2 causes the 
solution to diverge numerically - a known stability constraint of 
Euler-based SDE solvers confirming the theory behind why small step 
sizes are required.

| Iterations | Naive MSE | GraphStoch MSE (dt=0.1) |
|---|---|---|
| 3  | 0.473 | 1.118 |
| 5  | 0.285 | 0.717 |
| 10 | 0.233 | 0.363 |
| 20 | 0.237 | 0.201 |
| 50 | 0.238 | **0.173** |

GraphStoch requires more iterations to converge but avoids the 
over smoothing plateau naive averaging suffers from at the cost of 
slower initial convergence.

## Status

- Phase 1: Graph Laplacian construction (from scratch)
- Phase 2: Euler-Maruyama SDE solver (from scratch)
- Phase 3: Python wrapper (juliacall)
- Phase 4: Benchmark vs standard GNN on noisy data

## Tech Stack

- **Julia**: core simulation engine
- **Python**: wrapper API (planned, via juliacall)
- **Plots.jl**: visualization

## License

MIT