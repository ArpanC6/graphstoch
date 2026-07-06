"""
verify_convergence_order_v3.py

Re-run of the matched-Brownian-path strong-order check (see
verify_convergence_order_v2.py), addressing the open caveat in
THEORY.md Section 4b: the estimated strong order came out around
1.4-1.9 instead of the theoretical 1.0, and one suspected cause was
that the reference solution (dt_min) was not much finer than the
finest tested dt, so its own discretization error was not negligible.

Two changes from v2:

1. The reference resolution is now decoupled from the finest tested
   dt. In v2, dt_min was only 2x finer than the smallest tested dt.
   Here, dt_min is REF_EXTRA_DOUBLINGS doublings finer than the
   smallest tested dt (set well below the smallest tested dt), so the
   reference's own error is far smaller than any tested step's error.

2. More coarse step-doublings are tested going upward, letting the
   existing stability-bound filter decide how many are usable for
   each graph. Zachary's Karate Club has a tight stability bound
   (dt < ~0.0384) and can only ever support a couple of doublings no
   matter what we do here. The synthetic Erdos-Renyi graph has a
   looser bound and can make use of the extra levels.

Also fixed: in v2, the finest level's step count and dt_min were
defined independently, so every level actually only integrated up to
T/8, not the stated T. This version integrates to the stated T at
every level, consistent with what is printed.

Method is otherwise unchanged: matched (common-random-number)
Brownian increments across resolutions, finest resolution as the
quasi-exact reference, average pathwise error at each dt, strong
order estimated from the log2 ratio of average errors between
successive doublings.
"""

import numpy as np
import networkx as nx


def graph_laplacian(A):
    D = np.diag(A.sum(axis=1))
    return D - A


def euler_maruyama_matched(L, X0, sigma, dt, steps, dW_blocks):
    X = X0.copy()
    for k in range(steps):
        X = X + dt * (-L @ X) + sigma * dW_blocks[k]
    return X


def generate_matched_paths(n, dt_min, total_levels, rng):
    """
    total_levels: number of doubling levels above the finest (so there
    are total_levels + 1 levels total, indices 0..total_levels).
    Level 0 = finest (dt_min). Level k has dt = dt_min * 2^k and is
    built by pairwise-summing level (k-1)'s increments.
    """
    finest_steps = 2 ** total_levels
    fine_dW = rng.normal(0.0, np.sqrt(dt_min), size=(finest_steps, n))

    levels = {0: fine_dW}
    current = fine_dW
    for lvl in range(1, total_levels + 1):
        paired = current.reshape(-1, 2, n).sum(axis=1)
        levels[lvl] = paired
        current = paired
    return levels


def run_pathwise_convergence_check(A, name, T=1.0, sigma=0.5,
                                    n_trials=300,
                                    smallest_tested_dt=0.0078125,
                                    n_test_levels=8,
                                    ref_extra_doublings=7,
                                    seed_base=0):
    n = A.shape[0]
    L = graph_laplacian(A)

    X0 = np.zeros(n)
    X0[0] = 10.0

    eigvals = np.linalg.eigvalsh(L)
    lambda_max = eigvals[-1]
    max_stable_dt = 2.0 / lambda_max if lambda_max > 0 else np.inf

    dt_min = smallest_tested_dt / (2 ** ref_extra_doublings)
    total_levels = ref_extra_doublings + n_test_levels

    print(f"\n{name} (n={n} nodes)")
    print(f"lambda_max = {lambda_max:.4f}, stability requires dt < {max_stable_dt:.5f}")
    print(f"reference dt_min = {dt_min:.8f}  "
          f"({2**ref_extra_doublings}x finer than the smallest tested dt)")

    errors_by_level = {lvl: [] for lvl in range(ref_extra_doublings, total_levels + 1)}

    for trial in range(n_trials):
        rng = np.random.default_rng(seed_base + trial)
        levels = generate_matched_paths(n, dt_min, total_levels, rng)

        ref_steps = levels[0].shape[0]
        X_ref = euler_maruyama_matched(L, X0, sigma, dt_min, ref_steps, levels[0])

        for lvl in range(ref_extra_doublings, total_levels + 1):
            dt_lvl = dt_min * (2 ** lvl)
            if dt_lvl >= max_stable_dt:
                continue
            steps_lvl = levels[lvl].shape[0]
            X_lvl = euler_maruyama_matched(L, X0, sigma, dt_lvl, steps_lvl, levels[lvl])
            err = np.linalg.norm(X_lvl - X_ref)
            errors_by_level[lvl].append(err)

    dt_list = []
    avg_errors = []
    for lvl in range(ref_extra_doublings, total_levels + 1):
        if errors_by_level[lvl]:
            dt_lvl = dt_min * (2 ** lvl)
            avg_err = np.mean(errors_by_level[lvl])
            dt_list.append(dt_lvl)
            avg_errors.append(avg_err)
            print(f"  dt={dt_lvl:.6f}  avg||error|| over {len(errors_by_level[lvl])} "
                  f"trials = {avg_err:.6f}")

    print(f"\n  Empirical strong-order estimate (log2 ratio of consecutive avg errors):")
    orders = []
    for i in range(len(avg_errors) - 1):
        if avg_errors[i] > 0:
            order = np.log2(avg_errors[i + 1] / avg_errors[i])
            orders.append(order)
            print(f"    dt={dt_list[i]:.6f} -> dt={dt_list[i+1]:.6f}:  order ~ {order:.3f}")
    if orders:
        print(f"  Mean estimated strong order: {np.mean(orders):.3f}  "
              f"(order 1.0 = THEORY.md claim, order 0.5 = general SDE default)")

    return dt_list, avg_errors, orders


def main():
    results = {}

    np.random.seed(42)
    G_synth = nx.erdos_renyi_graph(10, 0.4, seed=42)
    tries = 0
    while not nx.is_connected(G_synth) and tries < 20:
        tries += 1
        G_synth = nx.erdos_renyi_graph(10, 0.4, seed=42 + tries)
    A_synth = nx.to_numpy_array(G_synth)
    results["synthetic_10node"] = run_pathwise_convergence_check(
        A_synth, "Synthetic 10-node Erdos-Renyi",
        T=1.0, sigma=0.5, n_trials=300,
        smallest_tested_dt=0.0078125, n_test_levels=9, ref_extra_doublings=7,
        seed_base=1000)

    G_karate = nx.karate_club_graph()
    A_karate = nx.to_numpy_array(G_karate)
    results["karate_club"] = run_pathwise_convergence_check(
        A_karate, "Zachary's Karate Club",
        T=1.0, sigma=0.5, n_trials=300,
        smallest_tested_dt=0.0009765625, n_test_levels=9, ref_extra_doublings=7,
        seed_base=2000)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, (dt_list, errors, orders) in results.items():
        if orders:
            print(f"{name}: mean empirical strong order = {np.mean(orders):.3f} "
                  f"({len(orders)} doubling ratios from {len(dt_list)} tested dts)")


if __name__ == "__main__":
    main()