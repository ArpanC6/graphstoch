from graphstoch import GraphSDE  # noqa: E402

import numpy as np
import pandas as pd
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats
import json
import copy

# US-county-fb election regression experiment (Jia & Benson 2020 dataset).
#
# Goal: test GraphStoch's denoising on a structurally different,
# real-world dataset framed as node REGRESSION rather than
# classification, per Bastian Epping's suggestion. Node features are
# county-level demographic covariates plus Facebook Social Connectedness
# fractions, and the target is the 2016 presidential election margin
# (gop votes - dem votes) / (gop votes + dem votes), already normalized
# to zero mean and unit std per the dataset's README.
#
# Downstream models: Ridge regression on noisy and denoised features
# (the regression analog of the logreg_noisy/logreg_denoised arms in
# the synthetic sweep), a capacity-matched MLP on denoised features
# (Section 15's architecture, hidden_channels=16, adapted for scalar
# regression), and a GCN on noisy features as the full-GNN comparison.
# This checks whether Section 15's finding (MLP-on-denoised competes
# with full-GNN-on-noisy) also holds here, not just on synthetic SBM
# data.

DATA_DIR = "../data/US-county-fb"
GRAPH_FILE = f"{DATA_DIR}/US-county-fb-graph.txt"
FEATS_FILE = f"{DATA_DIR}/US-county-fb-2016-feats.csv"

FEATURE_COLS = ["sh050m", "sh100m", "sh500m", "income", "migration",
                "birth", "death", "education", "unemployment"]
TARGET_COL = "election"

NOISE_FRACTION = 0.5
NOISE_SEEDS = list(range(5))
SPLIT_SEEDS = [0, 1, 2]  # multiple train/val/test splits, to check whether the
                         # gcn_noisy vs mlp_denoised gap is stable across splits
                         # rather than an artifact of one particular partition
RIDGE_ALPHA = 1.0


def load_county_graph():
    """
    Load the US-county-fb dataset.

    Node IDs in the edge list are 1-indexed row positions into the
    feature CSV, confirmed empirically: the max node id in the edge
    list (3106) equals the row count of the feature file. This is not
    the same as the FIPS code despite the README's wording, so nodes
    are remapped to 0-indexed positions matching the CSV row order.
    """
    feats = pd.read_csv(FEATS_FILE)
    n_nodes = len(feats)

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))
    with open(GRAPH_FILE) as f:
        for line in f:
            u, v = line.split()
            u, v = int(u) - 1, int(v) - 1
            G.add_edge(u, v)

    X = feats[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = feats[TARGET_COL].to_numpy(dtype=np.float32)
    return G, X, y


def graph_to_edge_index(G):
    edges = list(G.edges())
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    return torch.tensor([src, dst], dtype=torch.long)


def make_masks(n_nodes, seed, train_frac=0.6, val_frac=0.2):
    rng = np.random.default_rng(seed)
    idx = np.arange(n_nodes)
    rng.shuffle(idx)
    n_train = int(train_frac * n_nodes)
    n_val = int(val_frac * n_nodes)
    train_mask = np.zeros(n_nodes, dtype=bool)
    val_mask = np.zeros(n_nodes, dtype=bool)
    test_mask = np.zeros(n_nodes, dtype=bool)
    train_mask[idx[:n_train]] = True
    val_mask[idx[n_train:n_train + n_val]] = True
    test_mask[idx[n_train + n_val:]] = True
    return train_mask, val_mask, test_mask


class GCNReg(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, 1)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.conv2(x, edge_index).squeeze(-1)


class MLPReg(torch.nn.Module):
    """
    Plain 2-layer MLP, capacity-matched to the GCN (hidden_channels=16),
    adapted for scalar regression output. Tests whether Section 15's
    finding (GraphStoch's explicit smoothing makes a full GNN's implicit
    smoothing redundant) carries over from the synthetic SBM setting to
    this real, structurally different dataset.
    """
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index=None):
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        return self.lin2(x).squeeze(-1)


def train_and_eval(model, x_tensor, edge_index, y, train_mask, val_mask, test_mask,
                    epochs=300, lr=0.01, weight_decay=5e-4, patience=50):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    train_mask_t = torch.tensor(train_mask)
    val_mask_t = torch.tensor(val_mask)
    test_mask_t = torch.tensor(test_mask)

    best_val_mse = float("inf")
    best_state = None
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x_tensor, edge_index)
        loss = F.mse_loss(out[train_mask_t], y_tensor[train_mask_t])
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x_tensor, edge_index)
            val_mse = F.mse_loss(pred[val_mask_t], y_tensor[val_mask_t]).item()

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor, edge_index).numpy()
    mse = mean_squared_error(y[test_mask], pred[test_mask])
    r2 = r2_score(y[test_mask], pred[test_mask])
    return mse, r2


def eval_ridge(X_features, y, train_mask, test_mask):
    reg = Ridge(alpha=RIDGE_ALPHA)
    reg.fit(X_features[train_mask], y[train_mask])
    preds = reg.predict(X_features[test_mask])
    mse = mean_squared_error(y[test_mask], preds)
    r2 = r2_score(y[test_mask], preds)
    return mse, r2


print("Loading US-county-fb dataset...")
G, X_clean, y = load_county_graph()
n_nodes = G.number_of_nodes()
print(f"Loaded {n_nodes} nodes, {G.number_of_edges()} edges, {X_clean.shape[1]} features")

giant = max(nx.connected_components(G), key=len)
print(f"Giant component covers {len(giant)}/{n_nodes} nodes ({len(giant) / n_nodes:.4f})")

A = nx.to_numpy_array(G)
edge_index = graph_to_edge_index(G)

feature_std = X_clean.std()
noise_std = NOISE_FRACTION * feature_std

print("Building GraphSDE model and computing stable dt (may take a moment for a graph this size)...")
gs_model = GraphSDE(A, noise_level=0.5, dt=0.01)
safe_dt = gs_model.stable_dt() * 0.5
gs_model = GraphSDE(A, noise_level=0.5, dt=safe_dt)
print(f"Using dt={safe_dt:.6f}")

results = {sp: {"ridge_noisy": [], "ridge_denoised": [], "mlp_denoised": [], "gcn_noisy": []}
           for sp in SPLIT_SEEDS}

for split_seed in SPLIT_SEEDS:
    print(f"\nSplit seed {split_seed}")
    train_mask, val_mask, test_mask = make_masks(n_nodes, seed=split_seed)

    for seed in NOISE_SEEDS:
        print(f"  Noise seed {seed}...")
        rng = np.random.default_rng(seed)
        X_noisy = X_clean + rng.normal(0, noise_std, size=X_clean.shape).astype(np.float32)
        X_denoised = np.array(gs_model.denoise(X_noisy, time_steps=50))

        mse_noisy, r2_noisy = eval_ridge(X_noisy, y, train_mask, test_mask)
        mse_denoised, r2_denoised = eval_ridge(X_denoised, y, train_mask, test_mask)

        x_noisy_tensor = torch.tensor(X_noisy, dtype=torch.float32)
        x_denoised_tensor = torch.tensor(X_denoised, dtype=torch.float32)

        torch.manual_seed(seed)
        mlp_model = MLPReg(X_clean.shape[1], 16)
        mse_mlp, r2_mlp = train_and_eval(mlp_model, x_denoised_tensor, edge_index, y,
                                          train_mask, val_mask, test_mask)

        torch.manual_seed(seed)
        gcn_model = GCNReg(X_clean.shape[1], 16)
        mse_gcn, r2_gcn = train_and_eval(gcn_model, x_noisy_tensor, edge_index, y,
                                          train_mask, val_mask, test_mask)

        results[split_seed]["ridge_noisy"].append((mse_noisy, r2_noisy))
        results[split_seed]["ridge_denoised"].append((mse_denoised, r2_denoised))
        results[split_seed]["mlp_denoised"].append((mse_mlp, r2_mlp))
        results[split_seed]["gcn_noisy"].append((mse_gcn, r2_gcn))

        print(f"    ridge_noisy    : mse={mse_noisy:.4f} r2={r2_noisy:.4f}")
        print(f"    ridge_denoised : mse={mse_denoised:.4f} r2={r2_denoised:.4f}")
        print(f"    mlp_denoised   : mse={mse_mlp:.4f} r2={r2_mlp:.4f}")
        print(f"    gcn_noisy      : mse={mse_gcn:.4f} r2={r2_gcn:.4f}")

print("\nSummary per split seed (mean +/- std across noise seeds)")
for split_seed in SPLIT_SEEDS:
    print(f"\n  Split seed {split_seed}")
    for method, vals in results[split_seed].items():
        mse_arr = np.array([v[0] for v in vals])
        r2_arr = np.array([v[1] for v in vals])
        print(f"    {method:14s}: mse={mse_arr.mean():.4f}+/-{mse_arr.std():.4f}  "
              f"r2={r2_arr.mean():.4f}+/-{r2_arr.std():.4f}")

print("\nPaired t-tests per split seed")
for split_seed in SPLIT_SEEDS:
    print(f"\n  Split seed {split_seed}")
    ridge_denoised_r2 = np.array([v[1] for v in results[split_seed]["ridge_denoised"]])
    ridge_noisy_r2 = np.array([v[1] for v in results[split_seed]["ridge_noisy"]])
    mlp_r2 = np.array([v[1] for v in results[split_seed]["mlp_denoised"]])
    gcn_r2 = np.array([v[1] for v in results[split_seed]["gcn_noisy"]])

    t, p = stats.ttest_rel(ridge_denoised_r2, ridge_noisy_r2)
    print(f"    denoising helps (ridge_denoised vs ridge_noisy), R2: "
          f"mean_diff={ridge_denoised_r2.mean() - ridge_noisy_r2.mean():+.4f} t={t:.3f} p={p:.4f}")

    t, p = stats.ttest_rel(mlp_r2, gcn_r2)
    print(f"    MLP-on-denoised vs full-GCN-on-noisy, R2: "
          f"mean_diff={mlp_r2.mean() - gcn_r2.mean():+.4f} t={t:.3f} p={p:.4f}")

print("\nCross-split check: does gcn_noisy beat mlp_denoised in every split seed?")
for split_seed in SPLIT_SEEDS:
    mlp_r2 = np.array([v[1] for v in results[split_seed]["mlp_denoised"]])
    gcn_r2 = np.array([v[1] for v in results[split_seed]["gcn_noisy"]])
    winner = "gcn_noisy" if gcn_r2.mean() > mlp_r2.mean() else "mlp_denoised"
    print(f"  split {split_seed}: mlp_denoised r2={mlp_r2.mean():.4f}, "
          f"gcn_noisy r2={gcn_r2.mean():.4f}, winner={winner}")

with open("election_regression_results.json", "w") as f:
    json.dump({
        "n_nodes": n_nodes,
        "n_edges": G.number_of_edges(),
        "giant_component_frac": len(giant) / n_nodes,
        "results": {str(sp): {k: [{"mse": v[0], "r2": v[1]} for v in vals] for k, vals in results[sp].items()}
                    for sp in SPLIT_SEEDS},
    }, f, indent=2)
print("\nSaved results to election_regression_results.json")