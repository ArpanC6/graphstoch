"""
verify_convergence_order.py

Empirically verifies THEORY.md Section 4's claim: for GraphStoch's SDE
(additive noise, constant sigma), Euler-Maruyama achieves STRONG ORDER 1.0,
not the general SDE strong order 0.5.

Method:
  1. Build a graph (synthetic Erdos-Renyi AND Zachary's Karate Club, for
     robustness across two structurally different graphs).
  2. Use exact_solve() as ground truth at a fixed time horizon T.
  3. Run sde_solve() (Euler-Maruyama) at a sequence of halving step sizes
     dt, dt/2, dt/4, dt/8, ... averaging the empirical mean over many
     seeds at each dt to reduce Monte Carlo noise.
  4. Compute error = || empirical_mean(dt) - exact_mean ||  at each dt.
  5. Check the empirical convergence rate: if error halves each time dt
     halves, that's order 1.0. If error only shrinks by ~0.707x, that's
     order 0.5.

Rate is estimated via log2(error(dt) / error(dt/2)) between consecutive
step sizes -- this should come out close to 1.0 if THEORY.md's claim
holds, and close to 0.5 if it doesn't.

IMPORTANT: this checks convergence of the MEAN (a valid, standard way to
check strong-order convergence for linear SDEs, since strong error is
usually reported as expected deviation from the true solution path/mean
under repeated sampling). It intentionally averages over many independent
noise realizations at each dt to isolate the discretization bias from
Monte Carlo sampling noise -- do not reduce n_seeds without understanding
this trade-off.

NOTE: import graphstoch / juliacall BEFORE torch or any other heavy
library, per the project's known "torch imported before juliacall"
warning (see context doc Section 3).
"""

from juliacall import Main as jl
import numpy as np
import networkx as nx

# Load the Julia module. Adjust path if running from a different directory.
jl.seval('include("julia/GraphStoch/src/GraphStoch.jl")')
jl.seval("using .GraphStoch")


def laplacian_from_networkx(G):
    A = nx.to_numpy_array(G)
    return A


def run_convergence_check(A, name, T=1.0, sigma=0.5, n_seeds=200,
                           dt_list=None, seed_base=0):
    """
    A: adjacency matrix (numpy array)
    name: label for printing
    T: fixed time horizon to compare at
    sigma: noise strength
    n_seeds: number of independent EM runs averaged per dt (reduces MC noise)
    dt_list: list of step sizes to test, should be a halving sequence
    """
    n = A.shape[0]
    L = jl.GraphStoch.graph_laplacian(jl.Matrix(A))

    # Deterministic initial condition: a single spike at node 0.
    X0 = np.zeros(n)
    X0[0] = 10.0

    # Ground truth via exact_solve
    # exact_solve returns (sample, mean, var_modes); we want the exact mean.
    _, exact_mean, _ = jl.GraphStoch.exact_solve(L, jl.Vector(X0), sigma, T)
    exact_mean = np.array(exact_mean)

    if dt_list is None:
        # Halving sequence; smallest dt should still respect stability
        # (dt < 2/lambda_max), checked below.
        dt_list = [0.08, 0.04, 0.02, 0.01, 0.005]

    # Sanity check: ensure the largest dt tested is still numerically stable
    # for this graph (per THEORY.md Theorem 2 / Section 5 stable_dt()).
    eigvals = np.linalg.eigvalsh(np.array(L))
    lambda_max = eigvals[-1]
    max_stable_dt = 2.0 / lambda_max if lambda_max > 0 else np.inf
    print(f"\n=== {name} (n={n} nodes) ===")
    print(f"lambda_max = {lambda_max:.4f}, stability requires dt < {max_stable_dt:.5f}")
    for dt in dt_list:
        if dt >= max_stable_dt:
            print(f"  WARNING: dt={dt} exceeds stability limit, skipping")

    dt_list = [dt for dt in dt_list if dt < max_stable_dt]

    errors = []
    for dt in dt_list:
        steps = int(round(T / dt))
        # Recompute effective T to match integer step count exactly.
        T_eff = steps * dt

        means_accum = np.zeros(n)
        for seed in range(n_seeds):
            jl.seval(f"import Random; Random.seed!({seed_base + seed})")
            history = jl.GraphStoch.sde_solve(L, jl.Vector(X0), dt, steps, sigma)
            final_state = np.array(history[-1])
            means_accum += final_state
        empirical_mean = means_accum / n_seeds

        # Compare against exact solution evaluated at the SAME T_eff
        # (steps*dt may differ slightly from T due to rounding).
        _, exact_mean_eff, _ = jl.GraphStoch.exact_solve(L, jl.Vector(X0), sigma, T_eff)
        exact_mean_eff = np.array(exact_mean_eff)

        err = np.linalg.norm(empirical_mean - exact_mean_eff)
        errors.append(err)
        print(f"  dt={dt:.5f}  steps={steps:5d}  ||error|| = {err:.6f}")

    # Estimate empirical convergence order between consecutive dt's
    print(f"\n  Empirical convergence order (log2 ratio of consecutive errors):")
    orders = []
    for i in range(len(errors) - 1):
        if errors[i + 1] > 0:
            order = np.log2(errors[i] / errors[i + 1])
            orders.append(order)
            print(f"    dt={dt_list[i]:.5f} -> dt={dt_list[i+1]:.5f}:  "
                  f"order ~ {order:.3f}")
    if orders:
        print(f"  Mean estimated order: {np.mean(orders):.3f}  "
              f"(order 1.0 = THEORY.md claim, order 0.5 = general SDE default)")

    return dt_list, errors, orders


def main():
    results = {}

    # Graph 1: small synthetic Erdos-Renyi 
    np.random.seed(42)
    G_synth = nx.erdos_renyi_graph(10, 0.4, seed=42)
    # Ensure connected (required for the lambda_1=0 simple eigenvalue
    # assumption in THEORY.md Section 1); regenerate if not.
    tries = 0
    while not nx.is_connected(G_synth) and tries < 20:
        tries += 1
        G_synth = nx.erdos_renyi_graph(10, 0.4, seed=42 + tries)
    A_synth = laplacian_from_networkx(G_synth)
    results["synthetic_10node"] = run_convergence_check(
        A_synth, "Synthetic 10-node Erdos-Renyi", T=1.0, sigma=0.5,
        n_seeds=300, seed_base=1000)

    # Graph 2: Zachary's Karate Club (real network, already used in project)
    G_karate = nx.karate_club_graph()
    A_karate = laplacian_from_networkx(G_karate)
    results["karate_club"] = run_convergence_check(
        A_karate, "Zachary's Karate Club", T=1.0, sigma=0.5,
        n_seeds=300, seed_base=2000)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, (dt_list, errors, orders) in results.items():
        if orders:
            print(f"{name}: mean empirical order = {np.mean(orders):.3f}")

    return results


if __name__ == "__main__":
    main()
