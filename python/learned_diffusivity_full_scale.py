"""
Full-scale replication: Learned Attention-Diffusivity SDE for GraphStoch

Same model/method as the prototype (learned_diffusivity_sde.py), but run
at the repo's established methodology scale:
  N = 1000 nodes (not 500)
  homophily levels = [0.5, 0.6, 0.7, 0.8, 0.9] (not just 0.5/0.6/0.9)
  3 graph seeds x 5 noise seeds = 15 trials/level (not 3x3=9)

Purpose: (1) check whether GCN's own baseline numbers here match the
already-validated BENCHMARKS.md figures at this scale; (2) see whether
the h=0.60 improvement found in the small prototype replicates at full
scale, and what happens at the untested h=0.70 and h=0.80 levels.
"""

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from scipy import stats
import json
import time
import os

torch.manual_seed(0)

N_NODES = 1000
AVG_DEGREE = 6.0
FEAT_DIM = 50
FEATURE_SEPARATION = 0.15
FEATURE_SEED = 123
NOISE_FRACTION = 0.5
HOMOPHILY_LEVELS = [0.5, 0.6, 0.7, 0.8, 0.9]
GRAPH_SEEDS = [42, 43, 44]
NOISE_SEEDS = [0, 1, 2, 3, 4]


def make_sbm_graph(n_nodes, avg_degree, homophily_target, seed):
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
    rng = np.random.default_rng(seed)
    n_classes = len(np.unique(labels))
    means = rng.normal(0, 1, size=(n_classes, feat_dim)) * sep
    X = means[labels] + rng.normal(0, 1, size=(n_nodes, feat_dim))
    return X.astype(np.float32)


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


def graph_laplacian(A):
    D = np.diag(A.sum(axis=1))
    return D - A


def plain_diffuse(L, X0, dt, steps):
    X = X0.copy()
    for _ in range(steps):
        X = X + dt * (-L @ X)
    return X


class LearnedDiffusivitySDE(nn.Module):
    def __init__(self, feat_dim, hidden_dim=32, n_classes=2, n_steps=8, tau=0.5, sigma=0.05):
        super().__init__()
        self.n_steps = n_steps
        self.tau = tau
        self.sigma = sigma
        self.query = nn.Linear(feat_dim, hidden_dim)
        self.key = nn.Linear(feat_dim, hidden_dim)
        self.scale = hidden_dim ** 0.5
        self.readout = nn.Linear(feat_dim, n_classes)

    def attention_matrix(self, X, adj_mask):
        Q = self.query(X)
        K = self.key(X)
        scores = (Q @ K.T) / self.scale
        scores = scores.masked_fill(adj_mask == 0, float("-inf"))
        A_att = F.softmax(scores, dim=1)
        return A_att

    def forward(self, X0, adj_mask, training_noise=True):
        X = X0
        for _ in range(self.n_steps):
            A_att = self.attention_matrix(X, adj_mask)
            drift = (A_att @ X - X)
            X = X + self.tau * drift
            if training_noise and self.sigma > 0:
                X = X + self.sigma * (self.tau ** 0.5) * torch.randn_like(X)
        return self.readout(X)


def train_learned_sde(X_noisy, adj_mask, labels, train_mask, val_mask, test_mask,
                       n_classes, epochs=150, lr=0.01, patience=30):
    model = LearnedDiffusivitySDE(X_noisy.shape[1], n_classes=n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    X_t = torch.tensor(X_noisy, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.long)
    adj_t = torch.tensor(adj_mask, dtype=torch.float32)
    train_t = torch.tensor(train_mask)
    val_t = torch.tensor(val_mask)
    test_t = torch.tensor(test_mask)

    best_val = 0
    best_state = None
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X_t, adj_t, training_noise=True)
        loss = F.cross_entropy(out[train_t], y_t[train_t])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(X_t, adj_t, training_noise=False)
            val_acc = (out_eval[val_t].argmax(1) == y_t[val_t]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(X_t, adj_t, training_noise=False)
        test_acc = (out_eval[test_t].argmax(1) == y_t[test_t]).float().mean().item()
    return test_acc


class SimpleGCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super().__init__()
        self.w1 = nn.Linear(in_dim, hidden_dim)
        self.w2 = nn.Linear(hidden_dim, n_classes)

    def forward(self, X, A_norm):
        H = F.relu(A_norm @ self.w1(X))
        H = F.dropout(H, p=0.5, training=self.training)
        return A_norm @ self.w2(H)


def normalize_adj(A):
    A_hat = A + np.eye(A.shape[0])
    D_hat = np.diag(1.0 / np.sqrt(A_hat.sum(axis=1)))
    return D_hat @ A_hat @ D_hat


def train_gcn(X_noisy, A_norm, labels, train_mask, val_mask, test_mask, n_classes,
              epochs=200, lr=0.01, patience=30):
    model = SimpleGCN(X_noisy.shape[1], 16, n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    X_t = torch.tensor(X_noisy, dtype=torch.float32)
    A_t = torch.tensor(A_norm, dtype=torch.float32)
    y_t = torch.tensor(labels, dtype=torch.long)
    train_t = torch.tensor(train_mask)
    val_t = torch.tensor(val_mask)
    test_t = torch.tensor(test_mask)

    best_val = 0
    best_state = None
    no_improve = 0
    for epoch in range(epochs):
        model.train()
        opt.zero_grad()
        out = model(X_t, A_t)
        loss = F.cross_entropy(out[train_t], y_t[train_t])
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            out_eval = model(X_t, A_t)
            val_acc = (out_eval[val_t].argmax(1) == y_t[val_t]).float().mean().item()
        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(X_t, A_t)
        test_acc = (out_eval[test_t].argmax(1) == y_t[test_t]).float().mean().item()
    return test_acc


def eval_logreg(X_features, y, train_mask, test_mask):
    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_features[train_mask], y[train_mask])
    preds = clf.predict(X_features[test_mask])
    return (preds == y[test_mask]).mean()


results = {h: {"logreg_noisy": [], "logreg_denoised_plain": [],
               "learned_sde": [], "gcn": []} for h in HOMOPHILY_LEVELS}

t_start = time.time()
for h in HOMOPHILY_LEVELS:
    print(f"\nHomophily {h:.2f}", flush=True)
    for graph_seed in GRAPH_SEEDS:
        G, labels = make_sbm_graph(N_NODES, AVG_DEGREE, h, graph_seed)
        X_clean = make_features(labels, N_NODES, FEAT_DIM, FEATURE_SEPARATION, FEATURE_SEED)
        A = nx.to_numpy_array(G)
        L = graph_laplacian(A)
        A_norm = normalize_adj(A)
        adj_mask = (A + np.eye(N_NODES) > 0).astype(np.float32)
        train_mask, val_mask, test_mask = make_masks(N_NODES, seed=0)

        feature_std = X_clean.std()
        noise_std = NOISE_FRACTION * feature_std

        lambda_max = np.linalg.eigvalsh(L)[-1]
        safe_dt = (2.0 / lambda_max) * 0.5

        for noise_seed in NOISE_SEEDS:
            rng = np.random.default_rng(noise_seed)
            X_noisy = X_clean + rng.normal(0, noise_std, size=X_clean.shape).astype(np.float32)

            X_denoised_plain = plain_diffuse(L, X_noisy, safe_dt, 50)

            acc_noisy = eval_logreg(X_noisy, labels, train_mask, test_mask)
            acc_denoised_plain = eval_logreg(X_denoised_plain, labels, train_mask, test_mask)

            torch.manual_seed(noise_seed)
            acc_learned_sde = train_learned_sde(X_noisy, adj_mask, labels, train_mask,
                                                 val_mask, test_mask, n_classes=2)

            torch.manual_seed(noise_seed)
            acc_gcn = train_gcn(X_noisy, A_norm, labels, train_mask, val_mask, test_mask,
                                 n_classes=2)

            results[h]["logreg_noisy"].append(acc_noisy)
            results[h]["logreg_denoised_plain"].append(acc_denoised_plain)
            results[h]["learned_sde"].append(acc_learned_sde)
            results[h]["gcn"].append(acc_gcn)

        print(f"  graph seed {graph_seed} done ({time.time()-t_start:.1f}s elapsed)", flush=True)

print("\n" + "=" * 70)
print("SUMMARY (N=1000, 3 graph seeds x 5 noise seeds = 15 trials/level)")
print("=" * 70)
for h in HOMOPHILY_LEVELS:
    print(f"\nHomophily {h:.2f} (n={len(results[h]['gcn'])} trials)")
    for method, accs in results[h].items():
        arr = np.array(accs)
        print(f"  {method:24s}: {arr.mean():.4f} +/- {arr.std():.4f}")

print("\n" + "=" * 70)
print("PAIRED T-TESTS: learned_sde vs each baseline")
print("=" * 70)
for h in HOMOPHILY_LEVELS:
    print(f"\nHomophily {h:.2f}")
    learned = np.array(results[h]["learned_sde"])
    for method in ["logreg_denoised_plain", "gcn"]:
        other = np.array(results[h][method])
        t, p = stats.ttest_rel(learned, other)
        diff = learned.mean() - other.mean()
        sig = "SIGNIFICANT" if p < 0.05 else "not sig"
        direction = "learned_sde wins" if diff > 0 else f"{method} wins"
        print(f"  vs {method:24s}: diff={diff:+.4f} t={t:.3f} p={p:.4f} {sig} ({direction})")

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_OUT_PATH = os.path.join(_THIS_DIR, "learned_diffusivity_results_full_scale.json")
with open(_OUT_PATH, "w") as f:
    json.dump({str(h): {k: list(v) for k, v in results[h].items()} for h in HOMOPHILY_LEVELS}, f, indent=2)
print(f"\nTotal time: {time.time()-t_start:.1f}s")
print(f"Saved to {_OUT_PATH}")