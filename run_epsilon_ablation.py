"""
Ablation over the curvature-classification threshold epsilon (Eq. 4).
Uses the same controlled synthetic coupling protocol as Section 8:
each edge carries a systematic geodesic-deviation bias K L^2/6, and the
curvature-aware method corrects it with the factor 1/(1 + K L^2/6),
but ONLY on edges whose terminal curvature exceeds the threshold epsilon
(otherwise the edge is treated as flat / identity transport).

We report, for a range of epsilon, the fraction of edges classified as
curved vs flat and the resulting L2 continuity error, to show the result
is insensitive to epsilon across several orders of magnitude.
"""
import sys, os, json
import numpy as np

sys.path.insert(0, 'curvature_aware_deeponet')
from src.manifold_graph import create_manifold_graph

os.makedirs('results', exist_ok=True)

eps_values = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 3e-1]
n_runs = 200
K_scale = 1.5

# Build a fixed synthetic graph consistent with the paper's protocol:
# planar chain-like topology with synthetic curvature K = K_scale*sin(2 pi i / |V|).
n_v = 25
graph = create_manifold_graph('wrinkle', n_vertices=n_v, eps=0.01, seed=42)
# Override with the synthetic curvature assignment used in Section 8.
synthetic_K = K_scale * np.sin(2 * np.pi * np.arange(graph.n_v) / graph.n_v)

results = {}
print("=" * 70)
print("EPSILON THRESHOLD ABLATION (synthetic protocol, K_scale=1.5)")
print("=" * 70)
print(f"{'epsilon':>10} {'frac_curved':>12} {'flat_L2':>12} {'curv_L2':>12} {'improv%':>9}")

for eps in eps_values:
    rng = np.random.RandomState(321)
    flat_err, curv_err = [], []
    n_curved_edges = 0
    n_total_edges = 0
    for v in graph.innerVertices:
        inc = [(e, vi, vj) for e, (vi, vj) in enumerate(graph.E) if vi == v or vj == v]
        if len(inc) < 2:
            continue
        K = synthetic_K[v]
        for _ in range(n_runs):
            rho_true = rng.uniform(0.2, 0.8)
            obs, corr = [], []
            for e_idx, vi, vj in inc:
                L = np.linalg.norm(graph.positions_3d[vj] - graph.positions_3d[vi])
                bias = 1.0 + K * L * L / 6.0
                rho_obs = rho_true / bias + rng.randn() * 0.01
                obs.append(rho_obs)
                # Only correct if |K| exceeds threshold, else identity (flat)
                if abs(K) > eps:
                    corr.append(bias)
                else:
                    corr.append(1.0)
            obs = np.array(obs); corr = np.array(corr)
            flat_err.append((np.mean(obs) - rho_true) ** 2)
            curv_err.append((np.mean(obs * corr) - rho_true) ** 2)
        # count classification
        for e_idx, vi, vj in inc:
            n_total_edges += 1
            if abs(K) > eps:
                n_curved_edges += 1

    flat_L2 = float(np.sqrt(np.mean(flat_err)))
    curv_L2 = float(np.sqrt(np.mean(curv_err)))
    frac = n_curved_edges / max(n_total_edges, 1)
    improv = (flat_L2 - curv_L2) / flat_L2 * 100
    results[str(eps)] = {
        'epsilon': eps,
        'frac_curved': frac,
        'flat_L2': flat_L2,
        'curv_L2': curv_L2,
        'improvement_pct': improv,
    }
    print(f"{eps:>10.0e} {frac:>12.3f} {flat_L2:>12.4e} {curv_L2:>12.4e} {improv:>9.1f}")

with open('results/epsilon_ablation.json', 'w') as f:
    json.dump(results, f, indent=2)
print("Saved to results/epsilon_ablation.json")
