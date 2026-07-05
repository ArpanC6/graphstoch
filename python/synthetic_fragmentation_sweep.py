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

# Synthetic graph generation (Stochastic Block Model + isolated nodes).
#
# Goal: this is the fragmentation-sweep companion to
# synthetic_homophily_sweep_v2.py. There, homophily was varied while node
# count / average degree / feature signal strength were held fixed, and a
# clean crossover was found around homophily 0.70-0.90. Real Citeseer sits
# at homophily 0.7355 (where the homophily sweep says GNNs should still win
# clearly) yet GraphStoch actually wins on real Citeseer -- so homophily
# alone does not fully explain that result. The leading remaining
# hypothesis is Citeseer's extreme fragmentation (438 connected components,
# 48 isolated nodes -- far more than Cora or PubMed). This script holds
# homophily FIXED at 0.60 (chosen because the pure homophily sweep showed a
# clear, significant GNN win at 0.60 with zero fragmentation, so any
# crossover seen here as fragmentation increases can be attributed to
# fragmentation, not to homophily) and varies ONLY the isolated-node
# fraction.
#
# Generator design: a naive approach (deleting random nodes' edges directly
# from an existing SBM) was tried first and rejected -- it shrinks the
# *active*-subgraph average degree as isolation increases (neighbors of
# isolated nodes also lose degree), confounding fragmentation with degree.
# Instead we use a two-stage generator: (1) pick isolation_fraction * n
# nodes up front to be isolated (degree 0, no edges at all); (2) generate
# the 2-community SBM using ONLY the remaining "active" nodes, calibrated
# with the same closed-form a = 2hd, b = 2(1-h)d approach as the homophily
# sweep, to hit the target homophily and average degree among those active
# nodes specifically; (3) isolated nodes are added to the final graph as
# zero-degree nodes, still carrying a (randomly assigned) community label
# for downstream classification. This was verified in a sandbox (numpy +
# networkx only) to hold homophily and active-subgraph average degree
# fixed in tight bands (homophily 0.589-0.610, active_avg_degree
# 5.95-6.18) across isolation_fraction 0.0-0.30 -- a clean, non-confounded
# single-axis control.

N_NODES = 1000
AVG_DEGREE = 6.0
HOMOPHILY = 0.73  # ABOVE h* ~= 0.7041, matching real Citeseer's homophily (0.7355) -- Phase 11b test
FEAT_DIM = 50
FEATURE_SEPARATION = 0.15  # unchanged from synthetic_homophily_sweep_v2.py -- do NOT re-tune this.
                           # It was already validated there to land noisy-LogReg accuracy in a
                           # realistic 0.63-0.69 range. Isolated nodes don't interact with feature
                           # generation (features are label-conditional, not structure-conditional),
                           # so the same constant applies unchanged here.
FEATURE_SEED = 123         # feature generation seed, fixed so features are comparable across
                           # isolation levels and graph seeds

ISOLATION_FRACTIONS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30]
GRAPH_SEEDS = [42, 43, 44]     # three independent graph realizations, for replication
NOISE_SEEDS = list(range(5))   # noise draws per graph realization
NOISE_FRACTION = 0.5  # fixed at "medium" noise, matching the homophily sweep's choice


def generate_fragmented_sbm(n_nodes, avg_degree, homophily_target, isolation_fraction, seed):
    """Two-community SBM among 'active' nodes, plus a set of isolated (degree-0) nodes.

    Returns (G, labels, isolated_nodes) where G has exactly n_nodes nodes
    (ids 0..n_nodes-1), labels is a length-n_nodes array of community labels
    for every node (including isolated ones), and isolated_nodes is the set
    of node ids with no edges.
    """
    rng = np.random.default_rng(seed)
    n_isolated = int(round(isolation_fraction * n_nodes))
    n_active = n_nodes - n_isolated

    perm = rng.permutation(n_nodes)
    isolated_nodes = set(int(x) for x in perm[:n_isolated])
    active_nodes = perm[n_isolated:]  # original node ids that will carry the SBM structure

    sizes = [n_active // 2, n_active - n_active // 2]
    s = sizes[0]
    intra_deg = homophily_target * avg_degree
    inter_deg = avg_degree - intra_deg
    p_intra = np.clip(intra_deg / max(s - 1, 1), 0, 1)
    p_inter = np.clip(inter_deg / max(s, 1), 0, 1)
    probs = [[p_intra, p_inter], [p_inter, p_intra]]

    sbm_seed = int(rng.integers(0, 2**31 - 1))
    G_active_local = nx.stochastic_block_model(sizes, probs, seed=sbm_seed)
    active_labels_local = np.array([G_active_local.nodes[i]["block"] for i in range(n_active)])

    # Relabel the active SBM's local node ids (0..n_active-1) into the
    # original node-id space (the ids in `active_nodes`).
    mapping = {i: int(active_nodes[i]) for i in range(n_active)}
    G_active = nx.relabel_nodes(G_active_local, mapping)

    G = nx.Graph()
    G.add_nodes_from(range(n_nodes))  # ensures isolated nodes are present, and node order is 0..n-1
    G.add_edges_from(G_active.edges())

    labels = np.zeros(n_nodes, dtype=int)
    for i, orig_id in enumerate(active_nodes):
        labels[int(orig_id)] = active_labels_local[i]

    if n_isolated > 0:
        iso_list = sorted(isolated_nodes)
        iso_labels = rng.integers(0, 2, size=len(iso_list))
        for j, node in enumerate(iso_list):
            labels[node] = iso_labels[j]

    return G, labels, isolated_nodes


def measure_fragmentation(G, labels, isolated_nodes, n_nodes):
    """Diagnostics confirming isolation_fraction is a clean, single-axis control."""
    n_components = nx.number_connected_components(G)
    isolated_actual = len(isolated_nodes)

    active_degrees = [d for node, d in G.degree() if node not in isolated_nodes]
    active_avg_degree = float(np.mean(active_degrees)) if active_degrees else 0.0

    same = sum(1 for u, v in G.edges() if labels[u] == labels[v])
    total = G.number_of_edges()
    actual_homophily = same / total if total > 0 else float("nan")

    largest_cc = max(nx.connected_components(G), key=len)
    largest_cc_frac = len(largest_cc) / n_nodes

    return {
        "isolated_actual": isolated_actual,
        "n_components": n_components,
        "active_avg_degree": active_avg_degree,
        "actual_homophily": actual_homophily,
        "largest_cc_frac": largest_cc_frac,
    }


def make_features(labels, n_nodes, feat_dim, sep, seed):
    """Label-conditional Gaussian features, independent of graph structure.

    Identical to synthetic_homophily_sweep_v2.py's make_features -- carried
    over unchanged. Isolated nodes still have a label, so they get features
    the same way as active nodes; this is intentional (it is what makes the
    "isolated nodes get zero denoising benefit" comparison meaningful -- the
    feature signal is there, only GraphStoch's ability to exploit graph
    structure to recover it differs for isolated nodes).
    """
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


# Models (identical to the homophily sweep / cora / citeseer / pubmed scripts).

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


# Main sweep: outer loop over graph seed (replication), then isolation_fraction,
# then noise seed. Homophily is fixed at HOMOPHILY throughout.

results = {gs: {f: {"logreg_noisy": [], "logreg_denoised": [], "gcn": [], "graphsage": [], "gat": []}
                for f in ISOLATION_FRACTIONS}
           for gs in GRAPH_SEEDS}

property_log = {gs: {} for gs in GRAPH_SEEDS}

for graph_seed in GRAPH_SEEDS:
    print(f"\nGraph realization seed {graph_seed}")
    for frac in ISOLATION_FRACTIONS:
        G, labels, isolated_nodes = generate_fragmented_sbm(N_NODES, AVG_DEGREE, HOMOPHILY, frac, graph_seed)
        frag_stats = measure_fragmentation(G, labels, isolated_nodes, N_NODES)
        X_clean = make_features(labels, N_NODES, FEAT_DIM, FEATURE_SEPARATION, FEATURE_SEED)
        A = nx.to_numpy_array(G)
        edge_index = graph_to_edge_index(G, N_NODES)
        train_mask, val_mask, test_mask = make_masks(N_NODES, labels, seed=0)

        giant = max(nx.connected_components(G), key=len)
        Gg = G.subgraph(giant).copy()
        lam2 = np.linalg.eigvalsh(nx.laplacian_matrix(Gg).astype(float).toarray())[1] if Gg.number_of_nodes() > 1 else float("nan")
        property_log[graph_seed][frac] = {**frag_stats, "lambda2_giant": lam2}

        feature_std = X_clean.std()
        noise_std = NOISE_FRACTION * feature_std

        gs_model = GraphSDE(A, noise_level=0.5, dt=0.01)
        safe_dt = gs_model.stable_dt() * 0.5
        gs_model = GraphSDE(A, noise_level=0.5, dt=safe_dt)

        print(f"  isolation_fraction {frac:.2f} (isolated {frag_stats['isolated_actual']}, "
              f"n_components {frag_stats['n_components']}), active_avg_degree "
              f"{frag_stats['active_avg_degree']:.4f}, actual_homophily "
              f"{frag_stats['actual_homophily']:.4f}, largest_cc_frac "
              f"{frag_stats['largest_cc_frac']:.4f}, lambda2 {lam2:.5f}, using dt={safe_dt:.6f}")

        for seed in NOISE_SEEDS:
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
            acc_sage = train_gnn_and_eval(GraphSAGE, x_noisy_tensor, edge_index, labels, train_mask,
                                           val_mask, test_mask, FEAT_DIM, 2)
            torch.manual_seed(seed)
            acc_gat = train_gnn_and_eval(GAT, x_noisy_tensor, edge_index, labels, train_mask, val_mask,
                                          test_mask, FEAT_DIM, 2)

            results[graph_seed][frac]["logreg_noisy"].append(acc_noisy)
            results[graph_seed][frac]["logreg_denoised"].append(acc_denoised)
            results[graph_seed][frac]["gcn"].append(acc_gcn)
            results[graph_seed][frac]["graphsage"].append(acc_sage)
            results[graph_seed][frac]["gat"].append(acc_gat)

        print(f"  isolation_fraction {frac:.2f} done.")

print("\nSUMMARY per graph realization (medium noise, 0.5x feature std, homophily fixed at "
      f"{HOMOPHILY:.2f})")
for graph_seed in GRAPH_SEEDS:
    print(f"\nGraph seed {graph_seed}")
    for frac in ISOLATION_FRACTIONS:
        p = property_log[graph_seed][frac]
        print(f"  isolation_fraction {frac:.2f} (isolated {p['isolated_actual']}, "
              f"n_components {p['n_components']}, actual_homophily {p['actual_homophily']:.4f})")
        for method, accs in results[graph_seed][frac].items():
            arr = np.array(accs)
            print(f"    {method:16s}: {arr.mean():.4f} +/- {arr.std():.4f}")

print("\nPAIRED T-TEST per graph realization: GraphStoch(denoised)+LogReg vs each GNN baseline")
crossover_summary = {}
for graph_seed in GRAPH_SEEDS:
    print(f"\nGraph seed {graph_seed}")
    crossover_summary[graph_seed] = {}
    for frac in ISOLATION_FRACTIONS:
        denoised = np.array(results[graph_seed][frac]["logreg_denoised"])
        print(f"  isolation_fraction {frac:.2f}")
        gnn_wins = 0
        for method in ["gcn", "graphsage", "gat"]:
            other = np.array(results[graph_seed][frac][method])
            t_stat, p_val = stats.ttest_rel(denoised, other)
            mean_diff = denoised.mean() - other.mean()
            direction = "GraphStoch+LogReg wins" if mean_diff > 0 else f"{method} wins"
            if mean_diff <= 0:
                gnn_wins += 1
            sig = "significant p<0.05" if p_val < 0.05 else "not significant"
            print(f"    vs {method:10s}: mean_diff={mean_diff:+.4f} t={t_stat:.3f} p={p_val:.4f} "
                  f"{sig} {direction}")
        crossover_summary[graph_seed][frac] = gnn_wins  # 0..3, how many GNNs beat GraphStoch here

print("\nCROSSOVER SUMMARY (how many of the 3 GNNs beat GraphStoch+LogReg at each isolation_fraction, per graph seed)")
print(f"{'isolation':>10}" + "".join(f"  seed{gs:>3}" for gs in GRAPH_SEEDS))
for frac in ISOLATION_FRACTIONS:
    row = f"{frac:>10.2f}"
    for graph_seed in GRAPH_SEEDS:
        row += f"  {crossover_summary[graph_seed][frac]:>7d}"
    print(row)
print("(3 = GNNs win all three comparisons, 0 = GraphStoch+LogReg wins all three)")

with open("synthetic_fragmentation_sweep_results_h068.json", "w") as f:
    json.dump({
        "homophily_fixed": HOMOPHILY,
        "properties": {str(gs): {str(frac): property_log[gs][frac] for frac in ISOLATION_FRACTIONS}
                        for gs in GRAPH_SEEDS},
        "results": {str(gs): {str(frac): {k: list(v) for k, v in results[gs][frac].items()}
                                for frac in ISOLATION_FRACTIONS}
                    for gs in GRAPH_SEEDS},
        "crossover_summary": {str(gs): {str(frac): crossover_summary[gs][frac] for frac in ISOLATION_FRACTIONS}
                              for gs in GRAPH_SEEDS}
    }, f, indent=2)
print("\nSaved raw results to synthetic_fragmentation_sweep_results_h068.json")
