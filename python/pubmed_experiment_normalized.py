from graphstoch import GraphSDE  # noqa: E402
from graphstoch.core import _GraphStoch  # noqa: E402

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_dense_adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.linear_model import LogisticRegression
from scipy import stats
import json
import copy

dataset = Planetoid(root="../data/PubMed", name="PubMed", transform=T.NormalizeFeatures())
data = dataset[0]
A = to_dense_adj(data.edge_index)[0].numpy()
X_clean = data.x.numpy()
y = data.y.numpy()
train_mask = data.train_mask.numpy()
val_mask = data.val_mask.numpy()
test_mask = data.test_mask.numpy()

feature_std = X_clean.std()
NOISE_LEVELS = {"low": 0.1 * feature_std, "medium": 0.5 * feature_std, "heavy": 1.0 * feature_std}
SEEDS = list(range(10))

gs_model = GraphSDE(A, noise_level=0.5, dt=0.01)
safe_dt = gs_model.stable_dt() * 0.5
gs_model = GraphSDE(A, noise_level=0.5, dt=safe_dt)
print(f"stable_dt: {gs_model.stable_dt():.6f}, using dt={safe_dt:.6f}")
print(f"feature_std (normalized): {feature_std:.6f}")


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


def train_gnn_and_eval(model_class, x_noisy_tensor, edge_index, epochs=300, lr=0.01, weight_decay=5e-4, patience=50):
    model = model_class(dataset.num_features, 16, dataset.num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_tensor = torch.tensor(y, dtype=torch.long)
    train_mask_t = torch.tensor(train_mask)
    val_mask_t = torch.tensor(val_mask)
    test_mask_t = torch.tensor(test_mask)

    best_val_acc = 0
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_noisy_tensor, edge_index)
        loss = F.cross_entropy(out[train_mask_t], y_tensor[train_mask_t])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x_noisy_tensor, edge_index).argmax(dim=1)
            val_acc = (pred[val_mask_t] == y_tensor[val_mask_t]).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x_noisy_tensor, edge_index).argmax(dim=1)
        acc = (pred[test_mask_t] == y_tensor[test_mask_t]).float().mean().item()
    return acc


def eval_logreg(X_features):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_features[train_mask], y[train_mask])
    preds = clf.predict(X_features[test_mask])
    return (preds == y[test_mask]).mean()


results = {level: {"logreg_noisy": [], "logreg_denoised": [], "gcn": [], "graphsage": [], "gat": []}
           for level in NOISE_LEVELS}

for level_name, noise_std in NOISE_LEVELS.items():
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        X_noisy = X_clean + rng.normal(0, noise_std, size=X_clean.shape)
        X_denoised = np.array(gs_model.denoise(X_noisy, time_steps=50))

        acc_noisy = eval_logreg(X_noisy)
        acc_denoised = eval_logreg(X_denoised)

        x_noisy_tensor = torch.tensor(X_noisy, dtype=torch.float32)
        torch.manual_seed(seed)
        acc_gcn = train_gnn_and_eval(GCN, x_noisy_tensor, data.edge_index)
        torch.manual_seed(seed)
        acc_sage = train_gnn_and_eval(GraphSAGE, x_noisy_tensor, data.edge_index)
        torch.manual_seed(seed)
        acc_gat = train_gnn_and_eval(GAT, x_noisy_tensor, data.edge_index)

        results[level_name]["logreg_noisy"].append(acc_noisy)
        results[level_name]["logreg_denoised"].append(acc_denoised)
        results[level_name]["gcn"].append(acc_gcn)
        results[level_name]["graphsage"].append(acc_sage)
        results[level_name]["gat"].append(acc_gat)

    print(f"Noise level {level_name} done.")

print("\nSUMMARY (PubMed, normalized features)")
for level_name in NOISE_LEVELS:
    print(f"\nNoise level: {level_name}")
    for method, accs in results[level_name].items():
        arr = np.array(accs)
        print(f"  {method:16s}: {arr.mean():.4f} +/- {arr.std():.4f}")

print("\nPAIRED T-TEST: GraphStoch(denoised)+LogReg vs each GNN baseline")
for level_name in NOISE_LEVELS:
    denoised = np.array(results[level_name]["logreg_denoised"])
    print(f"\nNoise level: {level_name}")
    for method in ["gcn", "graphsage", "gat"]:
        other = np.array(results[level_name][method])
        t_stat, p_val = stats.ttest_rel(denoised, other)
        mean_diff = denoised.mean() - other.mean()
        direction = "GraphStoch+LogReg WINS" if mean_diff > 0 else f"{method.upper()} wins"
        sig = "SIGNIFICANT (p<0.05)" if p_val < 0.05 else "not significant"
        print(f"  vs {method:10s}: mean_diff={mean_diff:+.4f} | t={t_stat:.3f} | p={p_val:.4f} | {sig} | {direction}")

with open("pubmed_experiment_results_normalized.json", "w") as f:
    json.dump({lvl: {k: list(v) for k, v in results[lvl].items()} for lvl in results}, f, indent=2)
print("\nSaved raw results to pubmed_experiment_results_normalized.json")
