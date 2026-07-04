import numpy as np
import networkx as nx
from graphstoch import GraphSDE
from graphstoch.core import _GraphStoch

# Load a real-world graph (Zachary's Karate Club)
# 34 real people, real friendships, collected in the 1970s.
G = nx.karate_club_graph()
A = nx.to_numpy_array(G)
n = A.shape[0]
print(f"Loaded Karate Club graph: {n} nodes, {G.number_of_edges()} edges")

# Build the model and check the stability limit
# First build with a placeholder dt just to compute the Laplacian
model = GraphSDE(A, noise_level=0.5, dt=0.01)
safe_dt = model.stable_dt() * 0.5  # 50% safety margin below the stability limit
print(f"Stable dt limit for this graph: {model.stable_dt():.4f}")
print(f"Using dt = {safe_dt:.4f}")

model = GraphSDE(A, noise_level=0.5, dt=safe_dt)

# Create a realistic "ground truth" signal 
# Start from a random seed and diffuse it smooth, simulating some
# real underlying node property (e.g. an influence/activity score).
np.random.seed(42)
X_seed = np.random.randn(n) * 5
true_signal = model.denoise(X_seed, time_steps=200)

# Add heavy observation noise
noise_level = 2.0
np.random.seed(1)
X_noisy = true_signal + noise_level * np.random.randn(n)

# Denoise with GraphStoch (Laplacian diffusion)
X_denoised_graphstoch = model.denoise(X_noisy, time_steps=50)

# Naive GNN-style baseline (neighbor averaging)
X_denoised_naive = np.array(
    _GraphStoch.naive_gnn_multi(A, X_noisy, 50)
)

# Compare errors
def mse(a, b):
    return np.mean((a - b) ** 2)

mse_noisy = mse(true_signal, X_noisy)
mse_naive = mse(true_signal, X_denoised_naive)
mse_graphstoch = mse(true_signal, X_denoised_graphstoch)

print("\nResults on Zachary's Karate Club (real dataset)")
print(f"MSE (raw noisy data):       {mse_noisy:.4f}")
print(f"MSE (naive GNN averaging):  {mse_naive:.4f}")
print(f"MSE (GraphStoch denoising): {mse_graphstoch:.4f}")

print("\nConvergence across iterations (Karate Club)")
for iters in [3, 5, 10, 20, 50, 100]:
    naive_result = np.array(_GraphStoch.naive_gnn_multi(A, X_noisy, iters))
    graphstoch_result = model.denoise(X_noisy, time_steps=iters)

    mse_n = mse(true_signal, naive_result)
    mse_g = mse(true_signal, graphstoch_result)

    print(f"Iterations={iters:4d} | Naive MSE: {mse_n:.4f} | GraphStoch MSE: {mse_g:.4f}")