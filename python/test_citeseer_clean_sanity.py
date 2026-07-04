import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.linear_model import LogisticRegression
import numpy as np

dataset = Planetoid(root="../data/Citeseer", name="Citeseer")
data = dataset[0]


class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index)


class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        return self.conv2(x, edge_index)


def train_and_eval(model_class, name, epochs=200, lr=0.01, weight_decay=5e-4):
    model = model_class(dataset.num_features, 16, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        pred = model(data.x, data.edge_index).argmax(dim=1)
        test_acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    print(f"{name}: test accuracy = {test_acc:.4f}")
    return test_acc


# Also check clean LogReg for comparison
X = data.x.numpy()
y = data.y.numpy()
clf = LogisticRegression(max_iter=1000)
clf.fit(X[data.train_mask.numpy()], y[data.train_mask.numpy()])
preds = clf.predict(X[data.test_mask.numpy()])
acc_logreg_clean = (preds == y[data.test_mask.numpy()]).mean()
print(f"LogReg (clean, no noise): test accuracy = {acc_logreg_clean:.4f}")

torch.manual_seed(42)
train_and_eval(GCN, "GCN (clean)")
torch.manual_seed(42)
train_and_eval(GraphSAGE, "GraphSAGE (clean)")
torch.manual_seed(42)
train_and_eval(GAT, "GAT (clean)")
