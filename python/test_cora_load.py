import torch
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root="../data/Cora", name="Cora")
data = dataset[0]

print(f"Dataset: {dataset}")
print(f"Number of graphs: {len(dataset)}")
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Number of features per node: {dataset.num_features}")
print(f"Number of classes: {dataset.num_classes}")
print(f"Train nodes: {data.train_mask.sum().item()}")
print(f"Val nodes: {data.val_mask.sum().item()}")
print(f"Test nodes: {data.test_mask.sum().item()}")
