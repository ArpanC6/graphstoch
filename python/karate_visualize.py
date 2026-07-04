import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from graphstoch import GraphSDE
from graphstoch.core import _GraphStoch

# Exact same setup as karate_benchmark.py, for reproducibility 
G = nx.karate_club_graph()
A = nx.to_numpy_array(G)
n = A.shape[0]

model = GraphSDE(A, noise_level=0.5, dt=0.01)
safe_dt = model.stable_dt() * 0.5
model = GraphSDE(A, noise_level=0.5, dt=safe_dt)

np.random.seed(42)
X_seed = np.random.randn(n) * 5
true_signal = model.denoise(X_seed, time_steps=200)

noise_level = 2.0
np.random.seed(1)
X_noisy = true_signal + noise_level * np.random.randn(n)

X_denoised_graphstoch = model.denoise(X_noisy, time_steps=50)

def mse(a, b):
    return np.mean((a - b) ** 2)

# Plot 1: Convergence curve (log-scale y) 
iters_list = [3, 5, 10, 20, 50, 100]
naive_mses, graphstoch_mses = [], []
for iters in iters_list:
    naive_result = np.array(_GraphStoch.naive_gnn_multi(A, X_noisy, iters))
    graphstoch_result = model.denoise(X_noisy, time_steps=iters)
    naive_mses.append(mse(true_signal, naive_result))
    graphstoch_mses.append(mse(true_signal, graphstoch_result))

plt.figure(figsize=(7, 5))
plt.plot(iters_list, naive_mses, marker='o', label='Naive GNN averaging')
plt.plot(iters_list, graphstoch_mses, marker='o', label='GraphStoch (Laplacian diffusion)')
plt.yscale('log')
plt.xlabel('Iterations')
plt.ylabel('MSE (log scale)')
plt.title("Karate Club: Denoising Convergence\n(Naive plateaus early; GraphStoch keeps improving)")
plt.legend()
plt.grid(True, which='both', alpha=0.3)
plt.tight_layout()
plt.savefig('../notebooks/karate_club_convergence.png', dpi=150)
plt.close()
print("Saved ../notebooks/karate_club_convergence.png")

# Plot 2: Graph with true / noisy / denoised signal
pos = nx.spring_layout(G, seed=42)
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

signals = [
    ("True Signal", true_signal),
    (f"Noisy (MSE={mse(true_signal, X_noisy):.3f})", X_noisy),
    (f"GraphStoch Denoised (MSE={mse(true_signal, X_denoised_graphstoch):.3f})", X_denoised_graphstoch),
]

vmin = min(true_signal.min(), X_noisy.min(), X_denoised_graphstoch.min())
vmax = max(true_signal.max(), X_noisy.max(), X_denoised_graphstoch.max())

for ax, (title, values) in zip(axes, signals):
    nodes = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color=values, cmap='coolwarm',
        vmin=vmin, vmax=vmax, node_size=200
    )
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.3)
    ax.set_title(title)
    ax.axis('off')

fig.suptitle("Zachary's Karate Club: Signal Recovery", fontsize=14)
fig.colorbar(nodes, ax=axes, shrink=0.6, label='Signal value')
plt.savefig('../notebooks/karate_club_denoising.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved ../notebooks/karate_club_denoising.png")
