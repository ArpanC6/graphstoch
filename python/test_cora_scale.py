import time
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from graphstoch import GraphSDE

dataset = Planetoid(root="../data/Cora", name="Cora")
data = dataset[0]

A = to_dense_adj(data.edge_index)[0].numpy()
print(f"Adjacency shape: {A.shape}")

t0 = time.time()
model = GraphSDE(A, noise_level=0.5, dt=0.01)
stable_dt = model.stable_dt()
t1 = time.time()

print(f"stable_dt(): {stable_dt:.6f}")
print(f"Time to compute stable_dt: {t1-t0:.2f}s")

X = data.x.numpy()
X_small = X[:, :10]
t2 = time.time()
denoised = model.denoise(X_small, time_steps=50)
t3 = time.time()
denoised_arr = np.array(denoised)
print(f"Time to denoise 10 feature columns, 50 steps: {t3-t2:.2f}s")
print(f"Denoised shape: {denoised_arr.shape}")
