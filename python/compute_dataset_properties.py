"""
compute_dataset_properties.py

Computes the 6 candidate graph/dataset properties from the hypothesis list
(connectivity, algebraic connectivity, feature-dim/node ratio, homophily,
average degree/density, feature sparsity) for Cora and Citeseer, to look for
structural signals that might explain why GraphStoch-denoised-features beat
GNNs on Citeseer but lost on Cora.

This is an OBSERVATIONAL step only -- with n=2 datasets we cannot claim any
statistical relationship, only report the raw numbers and note which
direction(s) are consistent with each hypothesis. Confirming any hypothesis
requires a third data point (PubMed) and/or a controlled synthetic-graph
sweep (varying one property at a time) -- both are the next steps after
this script.

Run from repo root, with the same venv used for the Cora/Citeseer GNN
experiments (needs torch_geometric, networkx, numpy, scipy).
"""

import numpy as np
import networkx as nx
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx


def load_dataset(name):
    dataset = Planetoid(root=f"./data/{name}", name=name,
                         transform=T.NormalizeFeatures())
    data = dataset[0]
    return data, dataset


def compute_properties(name):
    data, dataset = load_dataset(name)
    n_nodes = data.num_nodes
    n_features = data.num_features
    n_classes = dataset.num_classes

    # Build an undirected networkx graph (no self loops, no duplicate edges)
    # for connectivity / algebraic connectivity / degree computations.
    G = to_networkx(data, to_undirected=True)
    G.remove_edges_from(nx.selfloop_edges(G))

    n_edges = G.number_of_edges()

    # --- Property 1: connectivity / disconnected components ---
    components = list(nx.connected_components(G))
    n_components = len(components)
    largest_cc_size = max(len(c) for c in components)
    largest_cc_frac = largest_cc_size / n_nodes
    isolated_nodes = sum(1 for c in components if len(c) == 1)

    # --- Property 2: algebraic connectivity (lambda_2) ---
    # On the FULL graph, lambda_2 = 0 if disconnected (multiplicity = number
    # of components), so we also report lambda_2 restricted to the largest
    # connected component, which is the more informative number for a
    # graph with many small disconnected pieces.
    L_full = nx.laplacian_matrix(G).astype(float).toarray()
    eigvals_full = np.linalg.eigvalsh(L_full)
    lambda2_full = eigvals_full[1] if len(eigvals_full) > 1 else float("nan")

    largest_cc_nodes = max(components, key=len)
    G_giant = G.subgraph(largest_cc_nodes).copy()
    L_giant = nx.laplacian_matrix(G_giant).astype(float).toarray()
    eigvals_giant = np.linalg.eigvalsh(L_giant)
    lambda2_giant = eigvals_giant[1] if len(eigvals_giant) > 1 else float("nan")
    lambda_max_giant = eigvals_giant[-1]

    # --- Property 3: feature dimensionality vs node count ratio ---
    feature_to_node_ratio = n_features / n_nodes

    # --- Property 4: homophily ratio (edge homophily) ---
    # Fraction of edges connecting two nodes with the same label.
    y = data.y.numpy()
    same_label_edges = 0
    for u, v in G.edges():
        if y[u] == y[v]:
            same_label_edges += 1
    edge_homophily = same_label_edges / n_edges if n_edges > 0 else float("nan")

    # --- Property 5: average degree / density ---
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)
    density = nx.density(G)

    # --- Property 6: feature sparsity ---
    X = data.x.numpy()
    nonzero_frac = np.count_nonzero(X) / X.size

    return {
        "name": name,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_components": n_components,
        "largest_cc_frac": largest_cc_frac,
        "isolated_nodes": isolated_nodes,
        "lambda2_full_graph": lambda2_full,
        "lambda2_giant_component": lambda2_giant,
        "lambda_max_giant_component": lambda_max_giant,
        "feature_to_node_ratio": feature_to_node_ratio,
        "edge_homophily": edge_homophily,
        "avg_degree": avg_degree,
        "density": density,
        "feature_nonzero_frac": nonzero_frac,
    }


def print_comparison(props_list):
    names = [p["name"] for p in props_list]
    print("\n" + "=" * 85)
    header = f"{'Property':<32}" + "".join(f"{n:>17}" for n in names)
    print(header)
    print("=" * 85)

    rows = [
        ("Nodes", "n_nodes", "{:.0f}"),
        ("Edges", "n_edges", "{:.0f}"),
        ("Feature dim", "n_features", "{:.0f}"),
        ("Classes", "n_classes", "{:.0f}"),
        ("# connected components", "n_components", "{:.0f}"),
        ("Largest component (frac of nodes)", "largest_cc_frac", "{:.4f}"),
        ("Isolated nodes (count)", "isolated_nodes", "{:.0f}"),
        ("lambda_2, full graph", "lambda2_full_graph", "{:.6f}"),
        ("lambda_2, giant component only", "lambda2_giant_component", "{:.6f}"),
        ("lambda_max, giant component", "lambda_max_giant_component", "{:.4f}"),
        ("Feature-dim / node-count ratio", "feature_to_node_ratio", "{:.4f}"),
        ("Edge homophily", "edge_homophily", "{:.4f}"),
        ("Average degree", "avg_degree", "{:.4f}"),
        ("Graph density", "density", "{:.6f}"),
        ("Feature nonzero fraction (sparsity)", "feature_nonzero_frac", "{:.4f}"),
    ]

    for label, key, fmt in rows:
        vals = "".join(fmt.format(p[key]).rjust(17) for p in props_list)
        print(f"{label:<32}{vals}")

    print("\nKnown result direction (from README): GNNs win on Cora (9/9);")
    print("GraphStoch+LogReg wins on Citeseer (8/9). PubMed classification")
    print("comparison against GNNs has NOT yet been run (see next step) --")
    print("this table is purely about the graph/feature STRUCTURE, to see")
    print("whether PubMed's property combination breaks the Cora/Citeseer")
    print("bundle (all 6 properties moved together for those two) or follows")
    print("the same bundle, which would tell us whether these properties are")
    print("truly independent signals or just co-vary across citation graphs.")


def main():
    datasets = ["Cora", "Citeseer", "PubMed"]
    props_list = []
    for name in datasets:
        print(f"Loading and computing properties for {name}...")
        props_list.append(compute_properties(name))

    print_comparison(props_list)


if __name__ == "__main__":
    main()
