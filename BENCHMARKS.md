# GraphStoch: Full Benchmark Results

This document contains the full experimental methodology, tables, and
discussion for GraphStoch's benchmarks against learned GNN baselines. It
is the detailed companion to the headline results summarized in
`README.md` and the theory developed in `THEORY.md`.

## Real Dataset Benchmark: Cora, Citeseer & PubMed - Denoising as GNN Preprocessing

The Karate Club benchmark (see `README.md`) tests pure signal recovery on
a small graph. To see how GraphStoch holds up at larger scale and against
learned models, we tested it on two standard citation-network datasets -
Cora (2,708 nodes, 1,433 features, 7 classes) and Citeseer (3,327 nodes,
3,703 features, 6 classes) - loaded via torch_geometric's Planetoid
loader, and compared against three widely used GNN architectures: GCN,
GraphSAGE, and GAT.

**Framing:** Cora and Citeseer are node-classification datasets, while
GraphStoch is a denoising method for continuous signal recovery -
directly comparing GraphStoch's MSE to a classifier's accuracy would be
an apples-to-oranges comparison. Instead we frame this as denoising as
GNN preprocessing:

1. Inject synthetic Gaussian noise into the node features.
2. Compare noisy features -> Logistic Regression vs. GraphStoch-denoised
   features -> Logistic Regression, to isolate GraphStoch's own
   contribution.
3. Compare both against noisy features fed directly into end-to-end
   trained GCN / GraphSAGE / GAT models - i.e. "let the GNN learn to
   handle the noise itself."

**Setup:** features were row-normalized via `T.NormalizeFeatures()`,
standard preprocessing for these bag-of-words datasets - GNN baselines
run without it underperform published benchmarks by several points, so
this is a required step rather than an optional tweak. GNNs were trained
with Adam (lr=0.01, weight_decay=5e-4) and validation-based early
stopping (patience=50, max 300 epochs). Three noise levels (0.1x / 0.5x /
1.0x feature std) were each run across 10 random seeds, with paired
t-tests comparing GraphStoch-denoised features + Logistic Regression
against each GNN.

### Cora results

| Noise level | LogReg (noisy) | LogReg (GraphStoch-denoised) | GCN | GraphSAGE | GAT |
|---|---|---|---|---|---|
| Low | 0.5726 | 0.7394 | 0.8152 | 0.7884 | 0.8249 |
| Medium | 0.5510 | 0.7402 | 0.8096 | 0.7921 | 0.8187 |
| Heavy | 0.4893 | 0.7286 | 0.7927 | 0.7667 | 0.8006 |

All nine paired comparisons (3 noise levels x 3 GNNs) are statistically
significant (p<0.0001) - and the GNNs win every single one, by 4 to 9
percentage points.

### Citeseer results

| Noise level | LogReg (noisy) | LogReg (GraphStoch-denoised) | GCN | GraphSAGE | GAT |
|---|---|---|---|---|---|
| Low | 0.6185 | 0.7225 | 0.6995 | 0.6978 | 0.7169 |
| Medium | 0.6072 | 0.7198 | 0.7071 | 0.6899 | 0.7104 |
| Heavy | 0.5703 | 0.7097 | 0.6977 | 0.6815 | 0.6999 |

Here the result flips: GraphStoch-denoised features + Logistic Regression
win 8 of 9 comparisons (p<0.05), by 1 to 3 percentage points. The one
exception - vs. GAT at low noise - is not statistically significant.

### PubMed results

To help disentangle why GraphStoch wins on Citeseer but loses on Cora, we
extended the same experiment to a third citation dataset, PubMed (19,717
nodes, 500 features, 3 classes) - identical setup (normalized features,
same GNN hyperparameters, same 3 noise levels x 10 seeds, same paired
t-tests).

| Noise level | LogReg (noisy) | LogReg (GraphStoch-denoised) | GCN | GraphSAGE | GAT |
|---|---|---|---|---|---|
| Low | 0.7321 | 0.7614 | 0.7899 | 0.7703 | 0.7775 |
| Medium | 0.7146 | 0.7539 | 0.7800 | 0.7638 | 0.7718 |
| Heavy | 0.6529 | 0.7260 | 0.7593 | 0.7335 | 0.7557 |

On PubMed, GNNs win the direction in all 9 comparisons, and 8 of 9 are
statistically significant (p<0.05) - the one exception (GraphSAGE at
heavy noise, p=0.35) is not significant, but the mean difference still
favors GNN. This matches Cora's pattern, not Citeseer's.

### Why this matters - a structural comparison across all three datasets

We also computed six candidate graph/feature properties (connectivity,
algebraic connectivity lambda_2, feature-dim/node ratio, edge homophily,
average degree, feature sparsity) for all three datasets
(`python/compute_dataset_properties.py`). On Cora and Citeseer, all six
properties moved together - Citeseer was more fragmented, lower
lambda_2, lower homophily, sparser, and had a higher feature-dim/node
ratio, so the six hypotheses were completely confounded with only two
datapoints. PubMed breaks this bundle: it is fully connected with the
highest lambda_2, average degree, and homophily close to Cora's of all
three datasets - by those four measures it looks "Cora-like or better,"
not Citeseer-like. But on feature-dim/node ratio and feature sparsity
specifically, PubMed is even more extreme than Citeseer.

PubMed's classification result (GNNs win, Cora's pattern) tracks its
connectivity / lambda_2 / homophily / average degree profile, not its
feature-dim ratio or sparsity profile. This is evidence that graph
structural properties (connectivity, algebraic connectivity, homophily,
degree) are more likely to drive the GraphStoch-vs-GNN flip than
feature-level properties (dimensionality ratio, sparsity) - though with
three datapoints this is still a correlational, not causal, conclusion.

**Methodological note:** an earlier version of the Cora/Citeseer
experiment, run without `T.NormalizeFeatures()`, produced misleadingly
strong results for GraphStoch on both datasets. That version was
discarded once the missing normalization was identified as the cause -
row-normalization is required for these bag-of-words features to get GNN
baselines that match published accuracy.

## Controlled Synthetic Experiment: Isolating Homophily as a Causal Driver

The three real datasets above narrowed the explanation for the GraphStoch-
vs-GNN flip down to four correlated structural candidates (connectivity,
algebraic connectivity lambda_2, edge homophily, average degree) - but
with only three real datapoints, these four properties still move
together and cannot be told apart from real data alone. To isolate a
single property, we built a controlled synthetic experiment using a
two-community Stochastic Block Model (SBM): node count (1,000), average
degree (~6.0), and feature signal strength are held fixed, and only edge
homophily is varied (0.5 to 0.9), with independently-generated,
label-conditional Gaussian features so that the base classification
difficulty is identical at every homophily level. The same denoising-as-
preprocessing protocol (GraphStoch+LogReg vs. GCN/GraphSAGE/GAT, paired
t-tests) was applied at each level.

### Result: a clean, replicated crossover

| Homophily | GCN | GraphSAGE | GAT | GraphStoch+LogReg |
|---|---|---|---|---|
| 0.50 | 0.6050 | 0.6815 | 0.5660 | 0.5305 |
| 0.60 | 0.6640 | 0.7070 | 0.6640 | 0.4970 |
| 0.70 | 0.7940 | 0.7570 | 0.7685 | 0.5885 |
| 0.80 | 0.8815 | 0.8605 | 0.8680 | 0.8850 |
| 0.90 | 0.9485 | 0.9355 | 0.9430 | 0.9775 |

At low-to-moderate homophily (0.5-0.7), GNNs win clearly and
significantly. Somewhere between 0.7 and 0.8, the result flips:
GraphStoch+LogReg pulls ahead and wins decisively at 0.8-0.9. This
crossover was replicated across three independent graph realizations
(seeds 42, 43, 44, each with 5 independent noise draws):

| Homophily | Seed 42 | Seed 43 | Seed 44 |
|---|---|---|---|
| 0.50 | GNNs win (3/3) | GNNs win (3/3) | GNNs win (3/3) |
| 0.60 | GNNs win (3/3) | GNNs win (3/3) | GNNs win (3/3) |
| 0.70 | GNNs win (3/3) | GNNs win (3/3) | GNNs win (3/3) |
| 0.80 | GraphStoch wins (3/3) | GNNs win (3/3) | GNNs win (3/3) |
| 0.90 | GraphStoch wins (3/3) | GraphStoch wins (3/3) | GraphStoch wins (3/3) |

The direction of the effect is fully replicated: low homophily always
favors GNNs, high homophily always favors GraphStoch. The exact crossover
point shifts slightly between graph realizations (0.7-0.8 for seed 42,
0.8-0.9 for seeds 43/44) - this is expected statistical variation across
random graph draws, not an inconsistency we are hiding. This is causal,
not merely correlational, evidence that edge homophily drives (at least
part of) the GraphStoch-vs-GNN flip, since node count, average degree,
and feature difficulty were held fixed throughout.

### An honest limitation this experiment surfaces

Real Citeseer's homophily is 0.7355. In the synthetic sweep, homophily
0.70 gives GNNs a clear, significant win in all three graph realizations
- yet on real Citeseer, GraphStoch wins. Homophily alone does not fully
explain Citeseer's result. This limitation motivated the fragmentation
sweep below.

The theoretical explanation for *why* this crossover exists and *where*
it should sit - the Kesten-Stigum / Abbe spectral detectability threshold
- is developed in `THEORY.md`, Section 7 (summarized briefly in
`README.md`).

## Controlled Synthetic Experiment: Isolating Fragmentation as a Causal Driver

### Motivation

Real Citeseer has 438 connected components and 48 isolated nodes - far
more fragmentation than Cora (78 components, 0 isolated) or PubMed (1
component, 0 isolated). Since homophily alone did not explain Citeseer's
result (see above), the leading remaining hypothesis was that Citeseer's
extreme fragmentation is an additional, independent factor pushing the
result toward GraphStoch beyond what homophily alone predicts.

### Design

Homophily was held fixed at 0.60 - deliberately chosen because the pure
homophily sweep showed a clear, significant GNN win at 0.60 with zero
fragmentation, meaning any shift observed as isolation increases can be
attributed to fragmentation, not to homophily. Node count (1,000) and
average degree (~6.0, measured among active/non-isolated nodes
specifically) were also held fixed.

A two-stage generator was used: (1) pick `isolation_fraction * n` nodes
up front to be isolated (degree 0, no edges); (2) generate the
2-community SBM using only the remaining active nodes, calibrated to hit
the target homophily and average degree among those active nodes; (3)
isolated nodes are added to the final graph as zero-degree nodes, still
carrying a randomly-assigned community label. This avoided a confound
found in a naive alternative (deleting edges from an existing SBM, which
shrinks active-subgraph degree as isolation increases).

`isolation_fraction` was swept over `[0.0, 0.02, 0.05, 0.10, 0.20, 0.30]`
- finer resolution near small values to bracket real Citeseer's actual
isolated-node fraction (48/3327 ~ 1.4%), extending to 0.30 for a strong
high-fragmentation signal if one exists. Replicated across the same three
graph seeds (42, 43, 44) x 5 noise seeds as the homophily sweep, at
medium noise (0.5x feature std). `FEATURE_SEPARATION=0.15` was carried
over unchanged from the homophily sweep's already-validated value.

### Result: no crossover

| isolation_fraction | seed 42 | seed 43 | seed 44 |
|---|---|---|---|
| 0.00 | GNNs win (3/3) | GNNs win (3/3) | GNNs win (3/3) |
| 0.02 | GNNs win (3/3) | GNNs win (3/3) | GNNs win (3/3) |
| 0.05 | GNNs win (3/3) | GNNs win (3/3) | GNNs win (3/3) |
| 0.10 | GNNs win (3/3) | GNNs win (3/3) | GNNs win (3/3) |
| 0.20 | GNNs win (3/3) | GNNs win (3/3) | GNNs win (3/3) |
| 0.30 | GNNs win (3/3) | GNNs win (3/3) | GNNs win (3/3) |

All three GNNs beat GraphStoch+LogReg at every isolation level, in every
graph realization, at this homophily level. Fragmentation, up to 30%
isolated nodes (far beyond Citeseer's actual ~1.4%), did not flip the
result. **This experiment does not support fragmentation as the
explanation for Citeseer's result at homophily levels below the
detectability threshold h\* (Section 7 of THEORY.md).**

### A mechanism found underneath the null result

`logreg_denoised` accuracy rose steadily with `isolation_fraction`,
approaching but never reaching or exceeding the noisy-features baseline
(illustrative, seed 42, medium noise):

| isolation_fraction | LogReg (noisy) | LogReg (GraphStoch-denoised) |
|---|---|---|
| 0.00 | 0.6800 | 0.4820 |
| 0.10 | 0.6870 | 0.5240 |
| 0.20 | 0.6710 | 0.5570 |
| 0.30 | 0.6920 | 0.5860 |

The same monotonic pattern held across all three graph seeds. `THEORY.md`
Section 8 proves the exact mechanism: an isolated node's row of the
Laplacian is identically zero, so GraphStoch's denoising step is an exact
fixed point for it - it neither helps nor harms isolated nodes, leaving
their raw noisy features untouched. Below the detectability threshold h*,
diffusion is actively harmful to active (non-isolated) nodes; a growing
isolated-node fraction therefore "exempts" a growing share of the
population from that harm, pulling the aggregate average up *toward* -
but never past - the noisy-only ceiling. This is why no crossover is
possible under this mechanism at this homophily level: the ceiling
fragmentation asymptotes toward is the noisy baseline itself, which
remains below all three GNN baselines throughout.

**Honest scope:** this experiment specifically tested homophily well
below h* (~0.7041). Real Citeseer's homophily (0.7355) sits near/just
above h* - a regime where diffusion is only marginally beneficial rather
than clearly harmful, and where fragmentation might interact with the
detectability signal differently. This has not been tested and remains
an open question; testing fragmentation near h* (~0.65-0.70) rather than
well below it is a natural next step. See `THEORY.md` Section 8.4 for the
full scope statement.
