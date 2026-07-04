import time
import numpy as np
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_dense_adj
from graphstoch import GraphSDE

dataset = Planetoid(root="../data/Cora", name="Cora")
data = dataset[0]
A = to_dense_adj(data.edge_index)[0].numpy()
model = GraphSDE(A, noise_level=0.5, dt=model_dt if False else 0.005)

X = data.x.numpy()
t0 = time.time()
denoised = model.denoise(X, time_steps=50)
t1 = time.time()
print(f"Full 1433-feature denoise, 50 steps: {t1-t0:.2f}s")
