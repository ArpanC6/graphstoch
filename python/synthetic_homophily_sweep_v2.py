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

# Synthetic graph generation (Stochastic Block Model).
#
# Goal: hold node count, average degree, and feature signal strength
# fixed, and vary only edge homophily, to test whether homophily alone
# drives the GraphStoch-vs-GNN flip seen across Cora / Citeseer / PubMed.
#
# v2 change: the first version of this sweep used a single fixed graph
# realization (seed=42) and found a clean crossover between homophily
# 0.7 and 0.8. Before trusting that as a real finding rather than a
# lucky single draw, this version repeats the whole sweep across three
# independent graph realizations (seeds 42, 43, 44) to check whether the
# crossover location is stable.
#
# v2b change: adds a bistable (Allen-Cahn-type) reaction-diffusion
# denoising variant alongside plain Laplacian diffusion, tested on this
# classification pipeline (rather than continuous MSE denoising, where
# a prior sandbox test showed the fixed +/-1 reaction poles are
# mismatched with an arbitrary-range continuous signal). Classification
# is the setting where bistable reaction terms are expected to help,
# per GREAD (Choi et al., 2023) and ACMP (Wang et al., 2023).

N_NODES = 1000
AVG_DEGREE = 6.0
FEAT_DIM = 50
FEATURE_SEPARATION = 0.15  # tuned so noisy-LogReg accuracy lands around 0.63-0.69, a realistic
                           # non-trivial range similar to the real Cora/Citeseer/PubMed experiments,
                           # leaving room for denoising and GNNs to differentiate instead of
                           # saturating near 1.0
FEATURE_SEED = 123         # feature generation seed, fixed so features are comparable across
                           # homophily levels and graph seeds

HOMOPHILY_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]
GRAPH_SEEDS = [42, 43, 44]   # three independent graph realizations, for replication
NOISE_SEEDS = list(range(5))  # noise draws per graph realization
NOISE_FRACTION = 0.5  # fixed at "medium" noise for all runs, to keep total runtime tractable
                       # while still isolating the one axis of interest (homophily)
BETA_REACTION = 0.5  # bistable reaction strength to test, chosen from prior sandbox exploration


def make_sbm_graph(n_nodes, avg_degree, homophily_target, seed):
    """Two-community SBM, calibrated so avg_degree and homophily hit their targets directly."""
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


# Models (identical to the cora/citeseer/pubmed scripts).

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


# Main sweep: outer loop over graph seed (replication), then homophily level,
# then noise seed.

results = {gs: {h: {"logreg_noisy": [], "logreg_denoised": [], "logreg_denoised_reaction": [],
                     "gcn": [], "graphsage": [], "gat": []}
                for h in HOMOPHILY_LEVELS}
           for gs in GRAPH_SEEDS}

property_log = {gs: {} for gs in GRAPH_SEEDS}

for graph_seed in GRAPH_SEEDS:
    print(f"\nGraph realization seed {graph_seed}")
    for h in HOMOPHILY_LEVELS:
        G, labels = make_sbm_graph(N_NODES, AVG_DEGREE, h, graph_seed)
        X_clean = make_features(labels, N_NODES, FEAT_DIM, FEATURE_SEPARATION, FEATURE_SEED)
        A = nx.to_numpy_array(G)
        edge_index = graph_to_edge_index(G, N_NODES)
        train_mask, val_mask, test_mask = make_masks(N_NODES, labels, seed=0)

        same = sum(1 for u, v in G.edges() if labels[u] == labels[v])
        total = G.number_of_edges()
        actual_h = same / total
        avg_deg = np.mean([d for _, d in G.degree()])
        giant = max(nx.connected_components(G), key=len)
        Gg = G.subgraph(giant).copy()
        lam2 = np.linalg.eigvalsh(nx.laplacian_matrix(Gg).astype(float).toarray())[1]
        property_log[graph_seed][h] = {"actual_homophily": actual_h, "avg_degree": avg_deg,
                                        "lambda2_giant": lam2, "giant_frac": len(giant) / N_NODES}

        feature_std = X_clean.std()
        noise_std = NOISE_FRACTION * feature_std

        gs_model = GraphSDE(A, noise_level=0.5, dt=0.01)
        safe_dt = gs_model.stable_dt() * 0.5
        gs_model = GraphSDE(A, noise_level=0.5, dt=safe_dt)

        print(f"  Homophily target {h:.2f} (actual {actual_h:.4f}), avg_deg {avg_deg:.4f}, "
              f"lambda2 {lam2:.5f}, using dt={safe_dt:.6f}")

        for seed in NOISE_SEEDS:
            rng = np.random.default_rng(seed)
            X_noisy = X_clean + rng.normal(0, noise_std, size=X_clean.shape).astype(np.float32)
            X_denoised = np.array(gs_model.denoise(X_noisy, time_steps=50))
            X_denoised_reaction = gs_model.denoise_with_reaction(X_noisy, beta=BETA_REACTION, time_steps=50)

            acc_noisy = eval_logreg(X_noisy, labels, train_mask, test_mask)
            acc_denoised = eval_logreg(X_denoised, labels, train_mask, test_mask)
            acc_denoised_reaction = eval_logreg(X_denoised_reaction, labels, train_mask, test_mask)

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

            results[graph_seed][h]["logreg_noisy"].append(acc_noisy)
            results[graph_seed][h]["logreg_denoised"].append(acc_denoised)
            results[graph_seed][h]["logreg_denoised_reaction"].append(acc_denoised_reaction)
            results[graph_seed][h]["gcn"].append(acc_gcn)
            results[graph_seed][h]["graphsage"].append(acc_sage)
            results[graph_seed][h]["gat"].append(acc_gat)

        print(f"  homophily {h:.2f} done.")

print("\nSUMMARY per graph realization (medium noise, 0.5x feature std)")
for graph_seed in GRAPH_SEEDS:
    print(f"\nGraph seed {graph_seed}")
    for h in HOMOPHILY_LEVELS:
        print(f"  Homophily {h:.2f} (actual {property_log[graph_seed][h]['actual_homophily']:.4f})")
        for method, accs in results[graph_seed][h].items():
            arr = np.array(accs)
            print(f"    {method:24s}: {arr.mean():.4f} +/- {arr.std():.4f}")

print("\nPAIRED T-TEST per graph realization: GraphStoch(denoised)+LogReg vs each GNN baseline")
crossover_summary = {}
for graph_seed in GRAPH_SEEDS:
    print(f"\nGraph seed {graph_seed}")
    crossover_summary[graph_seed] = {}
    for h in HOMOPHILY_LEVELS:
        denoised = np.array(results[graph_seed][h]["logreg_denoised"])
        denoised_reaction = np.array(results[graph_seed][h]["logreg_denoised_reaction"])
        print(f"  Homophily {h:.2f}")
        gnn_wins = 0
        for method in ["gcn", "graphsage", "gat"]:
            other = np.array(results[graph_seed][h][method])
            t_stat, p_val = stats.ttest_rel(denoised, other)
            mean_diff = denoised.mean() - other.mean()
            direction = "GraphStoch+LogReg wins" if mean_diff > 0 else f"{method} wins"
            if mean_diff <= 0:
                gnn_wins += 1
            sig = "significant p<0.05" if p_val < 0.05 else "not significant"
            print(f"    vs {method:10s}: mean_diff={mean_diff:+.4f} t={t_stat:.3f} p={p_val:.4f} "
                  f"{sig} {direction}")
        t_stat_r, p_val_r = stats.ttest_rel(denoised_reaction, denoised)
        mean_diff_r = denoised_reaction.mean() - denoised.mean()
        sig_r = "significant p<0.05" if p_val_r < 0.05 else "not significant"
        print(f"    [reaction vs plain diffusion]: mean_diff={mean_diff_r:+.4f} t={t_stat_r:.3f} "
              f"p={p_val_r:.4f} {sig_r} ({'reaction wins' if mean_diff_r > 0 else 'plain wins'})")
        crossover_summary[graph_seed][h] = gnn_wins  # 0..3, how many GNNs beat GraphStoch here

print("\nCROSSOVER SUMMARY (how many of the 3 GNNs beat GraphStoch+LogReg at each homophily level, per graph seed)")
print(f"{'homophily':>10}" + "".join(f"  seed{gs:>3}" for gs in GRAPH_SEEDS))
for h in HOMOPHILY_LEVELS:
    row = f"{h:>10.2f}"
    for graph_seed in GRAPH_SEEDS:
        row += f"  {crossover_summary[graph_seed][h]:>7d}"
    print(row)
print("(3 = GNNs win all three comparisons, 0 = GraphStoch+LogReg wins all three)")

with open("synthetic_homophily_sweep_results_v2b_reaction.json", "w") as f:
    json.dump({
        "properties": {str(gs): {str(h): property_log[gs][h] for h in HOMOPHILY_LEVELS} for gs in GRAPH_SEEDS},
        "results": {str(gs): {str(h): {k: list(v) for k, v in results[gs][h].items()} for h in HOMOPHILY_LEVELS}
                    for gs in GRAPH_SEEDS},
        "crossover_summary": {str(gs): {str(h): crossover_summary[gs][h] for h in HOMOPHILY_LEVELS}
                              for gs in GRAPH_SEEDS}
    }, f, indent=2)
print("\nSaved raw results to synthetic_homophily_sweep_results_v2b_reaction.json")