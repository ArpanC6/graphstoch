from graphstoch import GraphSDE  # noqa: E402
from graphstoch.core import _GraphStoch  # noqa: E402

import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from sklearn.linear_model import LogisticRegression
from scipy import stats
import json
import copy

# SYNTHETIC GRAPH GENERATION (Stochastic Block Model)
#
# Goal: hold node count, average degree, and feature signal
# strength FIXED, and vary ONLY edge homophily, to test whether
# homophily alone drives the GraphStoch-vs-GNN flip seen across
# Cora / Citeseer / PubMed.


N_NODES = 1000
AVG_DEGREE = 6.0
FEAT_DIM = 50
FEATURE_SEPARATION = 0.15  # controls base classification difficulty, fixed across all levels
                           # (tuned so noisy-LogReg accuracy lands ~0.63-0.69, a realistic non-trivial
                           # range similar to the real Cora/Citeseer/PubMed experiments, leaving room
                           # for denoising and GNNs to differentiate rather than saturating at ~1.0)
GRAPH_SEED = 42            # graph structure is fixed per homophily level (mirrors real-dataset scripts,
                           # where the graph is fixed and only injected noise varies across seeds)
FEATURE_SEED = 123         # feature generation seed, fixed so features are comparable across levels

HOMOPHILY_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]
SEEDS = list(range(10))


def make_sbm_graph(n_nodes, avg_degree, homophily_target, seed):
    """2-community SBM, calibrated so avg_degree and homophily hit their targets directly."""
    sizes = [n_nodes // 2, n_nodes - n_nodes // 2]
    s = sizes[0]
    intra_deg = homophily_target * avg_degree
    inter_deg = avg_degree - intra_deg
    p_intra = np.clip(intra_deg / (s - 1), 0, 1)
    p_inter = np.clip(inter_deg / s, 0, 1)
    probs = [[p_intra, p_inter], [p_inter, p_intra]]
    G = nx.stochastic_block_model(sizes, probs, seed=seed)
    labels = np.array([G.nodes[i]["block"] for i in range(n_nodes)])
    return G, labels


def make_features(labels, n_nodes, feat_dim, sep, seed):
    """Label-conditional Gaussian features, independent of graph structure."""
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(labels))
    means = rng.normal(0, 1, size=(n_classes, feat_dim)) * sep
    X = means[labels] + rng.normal(0, 1, size=(n_nodes, feat_dim))
    return X.astype(np.float32)


def graph_to_edge_index(G, n_nodes):
    edges = list(G.edges())
    src = [u for u, v in edges] + [v for u, v in edges]
    dst = [v for u, v in edges] + [u for u, v in edges]
    return torch.tensor([src, dst], dtype=torch.long)


def make_masks(n_nodes, labels, seed, train_frac=0.6, val_frac=0.2):
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



# MODELS (identical to cora/citeseer/pubmed scripts)


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


def train_gnn_and_eval(model_class, x_noisy_tensor, edge_index, y, train_mask, val_mask, test_mask,
                        num_features, num_classes, epochs=300, lr=0.01, weight_decay=5e-4, patience=50):
    model = model_class(num_features, 16, num_classes)
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


def eval_logreg(X_features, y, train_mask, test_mask):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_features[train_mask], y[train_mask])
    preds = clf.predict(X_features[test_mask])
    return (preds == y[test_mask]).mean()


# MAIN SWEEP
#
# Noise level is fixed at "medium" (0.5x feature std) across all
# homophily levels, to keep total runtime tractable while still
# varying the one axis of interest. (5 homophily levels x 10 seeds
# is already comparable in cost to one full noise-level sweep on a
# real dataset, despite the synthetic graph being much smaller than
# PubMed.)


NOISE_FRACTION = 0.5

results = {h: {"logreg_noisy": [], "logreg_denoised": [], "gcn": [], "graphsage": [], "gat": []}
           for h in HOMOPHILY_LEVELS}

property_log = {}

for h in HOMOPHILY_LEVELS:
    G, labels = make_sbm_graph(N_NODES, AVG_DEGREE, h, GRAPH_SEED)
    X_clean = make_features(labels, N_NODES, FEAT_DIM, FEATURE_SEPARATION, FEATURE_SEED)
    A = nx.to_numpy_array(G)
    edge_index = graph_to_edge_index(G, N_NODES)
    train_mask, val_mask, test_mask = make_masks(N_NODES, labels, seed=0)

    # log actual realized properties for honesty/reporting
    same = sum(1 for u, v in G.edges() if labels[u] == labels[v])
    total = G.number_of_edges()
    actual_h = same / total
    avg_deg = np.mean([d for _, d in G.degree()])
    giant = max(nx.connected_components(G), key=len)
    Gg = G.subgraph(giant).copy()
    lam2 = np.linalg.eigvalsh(nx.laplacian_matrix(Gg).astype(float).toarray())[1]
    property_log[h] = {"actual_homophily": actual_h, "avg_degree": avg_deg,
                        "lambda2_giant": lam2, "giant_frac": len(giant) / N_NODES}

    feature_std = X_clean.std()
    noise_std = NOISE_FRACTION * feature_std

    gs_model = GraphSDE(A, noise_level=0.5, dt=0.01)
    safe_dt = gs_model.stable_dt() * 0.5
    gs_model = GraphSDE(A, noise_level=0.5, dt=safe_dt)

    print(f"\nhomophily target={h:.2f} (actual={actual_h:.4f}), avg_deg={avg_deg:.4f}, "
          f"lambda2={lam2:.5f}")
    print(f"stable_dt: {gs_model.stable_dt():.6f}, using dt={safe_dt:.6f}")

    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        X_noisy = X_clean + rng.normal(0, noise_std, size=X_clean.shape).astype(np.float32)
        X_denoised = np.array(gs_model.denoise(X_noisy, time_steps=50))

        acc_noisy = eval_logreg(X_noisy, labels, train_mask, test_mask)
        acc_denoised = eval_logreg(X_denoised, labels, train_mask, test_mask)

        x_noisy_tensor = torch.tensor(X_noisy, dtype=torch.float32)
        torch.manual_seed(seed)
        acc_gcn = train_gnn_and_eval(GCN, x_noisy_tensor, edge_index, labels, train_mask, val_mask,
                                      test_mask, FEAT_DIM, 2)
        torch.manual_seed(seed)
        acc_sage = train_gnn_and_eval(GraphSAGE, x_noisy_tensor, edge_index, labels, train_mask, val_mask,
                                       test_mask, FEAT_DIM, 2)
        torch.manual_seed(seed)
        acc_gat = train_gnn_and_eval(GAT, x_noisy_tensor, edge_index, labels, train_mask, val_mask,
                                      test_mask, FEAT_DIM, 2)

        results[h]["logreg_noisy"].append(acc_noisy)
        results[h]["logreg_denoised"].append(acc_denoised)
        results[h]["gcn"].append(acc_gcn)
        results[h]["graphsage"].append(acc_sage)
        results[h]["gat"].append(acc_gat)

    print(f"homophily={h:.2f} done.")

print("\nSUMMARY (Synthetic SBM homophily sweep, noise level = medium/0.5x)")
for h in HOMOPHILY_LEVELS:
    print(f"\nHomophily target: {h:.2f} (actual={property_log[h]['actual_homophily']:.4f})")
    for method, accs in results[h].items():
        arr = np.array(accs)
        print(f"  {method:16s}: {arr.mean():.4f} +/- {arr.std():.4f}")

print("\nPAIRED T-TEST: GraphStoch(denoised)+LogReg vs each GNN baseline")
for h in HOMOPHILY_LEVELS:
    denoised = np.array(results[h]["logreg_denoised"])
    print(f"\nHomophily target: {h:.2f}")
    for method in ["gcn", "graphsage", "gat"]:
        other = np.array(results[h][method])
        t_stat, p_val = stats.ttest_rel(denoised, other)
        mean_diff = denoised.mean() - other.mean()
        direction = "GraphStoch+LogReg WINS" if mean_diff > 0 else f"{method.upper()} wins"
        sig = "SIGNIFICANT (p<0.05)" if p_val < 0.05 else "not significant"
        print(f"  vs {method:10s}: mean_diff={mean_diff:+.4f} | t={t_stat:.3f} | p={p_val:.4f} | {sig} | {direction}")

with open("synthetic_homophily_sweep_results.json", "w") as f:
    json.dump({
        "properties": {str(h): property_log[h] for h in HOMOPHILY_LEVELS},
        "results": {str(h): {k: list(v) for k, v in results[h].items()} for h in HOMOPHILY_LEVELS}
    }, f, indent=2)
print("\nSaved raw results to synthetic_homophily_sweep_results.json")
