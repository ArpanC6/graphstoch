"""
verify_convergence_order_v2.py

FIXED VERSION. The first attempt (verify_convergence_order.py) estimated the
mean via Monte Carlo averaging at each dt and compared to the exact mean.
That failed because the true discretization bias was smaller than the
Monte Carlo sampling noise at n_seeds=300 -- the "signal" (dt-dependent
bias) was buried under the "noise" (random sampling error), producing
non-monotonic, sometimes negative estimated orders. That result is NOT
evidence against THEORY.md Section 4's claim -- it just means the test
itself wasn't sensitive enough.

This version uses the STANDARD, correct method for empirically verifying
strong convergence order (see Kloeden & Platen 1992, Ch. 10): pathwise
comparison using MATCHED (coupled) Brownian paths across all step sizes,
often called "common random numbers."

Method:
  1. For each independent trial, generate ONE fine-grained sequence of
     Brownian increments at the smallest step size dt_min.
  2. For each coarser dt = dt_min * 2^k, build the correct Brownian
     increment for each coarse step by SUMMING the corresponding block of
     fine increments (valid because Brownian increments over adjacent
     sub-intervals sum exactly to the increment over the combined
     interval -- this is a defining property of Brownian motion, not an
     approximation).
  3. Run Euler-Maruyama at each dt using these path-matched increments
     (implemented directly in Python/numpy for full control over the
     noise -- the existing Julia sde_solve() samples its own internal
     noise and can't be given externally matched increments).
  4. Use the FINEST-resolution EM path (dt_min) as the reference/"quasi-
     exact" solution -- this is standard practice: since dt_min is far
     smaller than the coarser dt's under test, its own discretization
     error is negligible by comparison.
  5. Compute pathwise error |X_coarse(T) - X_reference(T)| for each of
     many independent trials (each trial = one fresh Brownian path), then
     average the ABSOLUTE error across trials at each dt. This average
     absolute error IS the standard strong-error metric -- unlike
     comparing means, it does not require the bias to be large relative
     to per-path noise, because we're measuring deviation, not trying to
     recover a small bias from a noisy average.
  6. Check the log2 ratio of average absolute errors between successive
     (doubling) dt -- this should be close to 1.0 for a strong-order-1.0
     method, and close to 0.5 for a strong-order-0.5 method.

This directly targets THEORY.md Section 4's claim: for additive-noise
SDEs the Milstein correction term vanishes, so Euler-Maruyama achieves
strong order 1.0, one full order better than the general SDE case.
"""

import numpy as np
import networkx as nx


def graph_laplacian(A):
    D = np.diag(A.sum(axis=1))
    return D - A


def euler_maruyama_matched(L, X0, sigma, dt, steps, dW_blocks):
    """
    Run Euler-Maruyama using PRE-GENERATED Brownian increments (dW_blocks),
    one increment per step, each of shape (n,). dW_blocks must already be
    scaled correctly (i.e. dW_blocks[k] ~ N(0, dt * I), matching the true
    step size dt for this resolution level).
    """
    n = len(X0)
    X = X0.copy()
    for k in range(steps):
        X = X + dt * (-L @ X) + sigma * dW_blocks[k]
    return X


def generate_matched_paths(n, dt_min, n_levels, rng):
    """
    Generate one fine Brownian path at resolution dt_min, then build
    coarser matched increments for each coarser level by summing blocks
    of the fine increments.

    Returns a dict: {level_index: array of shape (steps_at_that_level, n)}
    where level 0 = finest (dt_min), and each subsequent level doubles dt
    (halves the number of steps) by summing pairs of increments from the
    previous level.
    """
    # Finest level increments: dW ~ N(0, dt_min) per component.
    finest_steps = 2 ** (n_levels - 1)  # ensures clean halving all the way up
    fine_dW = rng.normal(0.0, np.sqrt(dt_min), size=(finest_steps, n))

    levels = {0: fine_dW}
    current = fine_dW
    for lvl in range(1, n_levels):
        # Combine adjacent pairs: increment over [t, t+2*dt] = sum of the
        # two consecutive fine increments over [t, t+dt] and [t+dt, t+2*dt].
        paired = current.reshape(-1, 2, n).sum(axis=1)
        levels[lvl] = paired
        current = paired
    return levels


def run_pathwise_convergence_check(A, name, T=1.0, sigma=0.5,
                                    n_trials=100, n_levels=6, seed_base=0):
    n = A.shape[0]
    L = graph_laplacian(A)

    X0 = np.zeros(n)
    X0[0] = 10.0

    eigvals = np.linalg.eigvalsh(L)
    lambda_max = eigvals[-1]
    max_stable_dt = 2.0 / lambda_max if lambda_max > 0 else np.inf

    # Finest level uses dt_min; coarsest tested level should still respect
    # the stability bound with margin.
    dt_min = T / (2 ** (n_levels - 1) * 8)  # extra factor of 8 for a safe reference
    # Build list of dt's actually tested (excluding the finest reference level).
    print(f"\n=== {name} (n={n} nodes) ===")
    print(f"lambda_max = {lambda_max:.4f}, stability requires dt < {max_stable_dt:.5f}")
    print(f"reference dt_min = {dt_min:.6f}")

    # Errors per tested level (levels 1..n_levels-1 relative to reference
    # built at level 0 == dt_min, but we only report levels whose dt is
    # still comfortably below the stability limit).
    errors_by_level = {lvl: [] for lvl in range(1, n_levels)}

    for trial in range(n_trials):
        rng = np.random.default_rng(seed_base + trial)
        levels = generate_matched_paths(n, dt_min, n_levels, rng)

        # Reference solution: run EM at the finest level (dt_min).
        ref_steps = levels[0].shape[0]
        X_ref = euler_maruyama_matched(L, X0, sigma, dt_min, ref_steps, levels[0])

        for lvl in range(1, n_levels):
            dt_lvl = dt_min * (2 ** lvl)
            if dt_lvl >= max_stable_dt:
                continue
            steps_lvl = levels[lvl].shape[0]
            X_lvl = euler_maruyama_matched(L, X0, sigma, dt_lvl, steps_lvl, levels[lvl])
            err = np.linalg.norm(X_lvl - X_ref)
            errors_by_level[lvl].append(err)

    dt_list = []
    avg_errors = []
    for lvl in range(1, n_levels):
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
            # dt_list is increasing, so avg_errors[i+1] (bigger dt) should be
            # bigger too; order = log2(error_ratio) for a doubling of dt.
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
        T=1.0, sigma=0.5, n_trials=200, n_levels=6, seed_base=1000)

    G_karate = nx.karate_club_graph()
    A_karate = nx.to_numpy_array(G_karate)
    results["karate_club"] = run_pathwise_convergence_check(
        A_karate, "Zachary's Karate Club",
        T=1.0, sigma=0.5, n_trials=200, n_levels=6, seed_base=2000)

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for name, (dt_list, errors, orders) in results.items():
        if orders:
            print(f"{name}: mean empirical strong order = {np.mean(orders):.3f}")


if __name__ == "__main__":
    main()
