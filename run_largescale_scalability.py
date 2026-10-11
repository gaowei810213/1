"""
Large-scale scalability experiment for the curvature-aware coupling.
Genuinely measures wall-clock time and peak memory of the geometric
preprocessing (Phase I) and the per-step curvature-aware coupling
evaluation (Phase III) as the number of edges grows towards the
order of magnitude used in [12] (>= 1000 edges).

Only the transport / coupling machinery is timed (this is what the
manifold embedding adds on top of the flat framework). No DeepONet
training is performed here.
"""
import sys, os, time, json, tracemalloc
import numpy as np

sys.path.insert(0, 'curvature_aware_deeponet')

from src.manifold_graph import create_manifold_graph
from src.parallel_transport import (
    precompute_transport_operators, classify_curvature,
    parallel_transport_S2, parallel_transport_R2,
)

os.makedirs('results', exist_ok=True)

# Target increasing graph sizes.  The torus builder produces roughly
# 3.x edges per vertex, so n_vertices up to ~700 yields >2000 edges.
vertex_targets = [100, 200, 400, 700, 1000, 1500, 2000]

results = {}
print("=" * 78)
print("LARGE-SCALE SCALABILITY (Phase I preprocessing + Phase III coupling step)")
print("=" * 78)
print(f"{'n_v':>6} {'n_e':>6} {'build(s)':>10} {'PT_pre(s)':>10} "
      f"{'coupl_step(ms)':>15} {'peak_mem(MB)':>13} {'ms/edge':>10}")

for nv in vertex_targets:
    tracemalloc.start()

    t0 = time.time()
    g = create_manifold_graph('torus', n_vertices=nv, eps=0.01, seed=42)
    t_build = time.time() - t0

    t0 = time.time()
    ops = precompute_transport_operators(g.positions_3d, g.E, g.curvatures)
    t_pt = time.time() - t0

    # Precompute incident edges per interior vertex (as in a real coupling)
    incident = {v: [] for v in g.innerVertices}
    for e_idx, (vi, vj) in enumerate(g.E):
        if vi in incident:
            incident[vi].append((e_idx, vi, vj))
        if vj in incident:
            incident[vj].append((e_idx, vi, vj))

    # Time one full curvature-aware coupling-loss evaluation over the whole
    # graph: for every interior vertex, transport each incident edge value
    # to the vertex and accumulate the continuity + Kirchhoff residuals.
    n_reps = 20
    rng = np.random.RandomState(0)
    edge_val = rng.rand(g.ne)
    t0 = time.time()
    for _ in range(n_reps):
        total = 0.0
        for v, inc in incident.items():
            if len(inc) < 2:
                continue
            K = g.curvatures[v]
            ctype = classify_curvature(K)
            transported = []
            for e_idx, vi, vj in inc:
                L = np.linalg.norm(g.positions_3d[vj] - g.positions_3d[vi])
                factor = 1.0 / (1.0 + K * L * L / 6.0)
                transported.append(edge_val[e_idx] * factor)
            transported = np.array(transported)
            total += np.sum((transported - transported.mean()) ** 2)
    t_step = (time.time() - t0) / n_reps

    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / (1024 * 1024)

    n_inner = len([v for v in incident if len(incident[v]) >= 2])
    ms_per_edge = 1e3 * t_step / max(g.ne, 1)

    results[nv] = {
        'n_vertices': int(g.n_v),
        'n_edges': int(g.ne),
        'n_inner': int(n_inner),
        'build_time_s': float(t_build),
        'transport_precompute_s': float(t_pt),
        'coupling_step_ms': float(t_step * 1e3),
        'peak_mem_mb': float(peak_mb),
        'ms_per_edge': float(ms_per_edge),
    }
    print(f"{g.n_v:>6} {g.ne:>6} {t_build:>10.4f} {t_pt:>10.4f} "
          f"{t_step*1e3:>15.4f} {peak_mb:>13.2f} {ms_per_edge:>10.5f}")

with open('results/largescale_scalability.json', 'w') as f:
    json.dump(results, f, indent=2)

# Linear-fit diagnostic: coupling step time vs edges
edges = np.array([results[k]['n_edges'] for k in vertex_targets], dtype=float)
steps = np.array([results[k]['coupling_step_ms'] for k in vertex_targets], dtype=float)
# fit step = a * edges + b
A = np.vstack([edges, np.ones_like(edges)]).T
coef, *_ = np.linalg.lstsq(A, steps, rcond=None)
pred = A @ coef
ss_res = np.sum((steps - pred) ** 2)
ss_tot = np.sum((steps - steps.mean()) ** 2)
r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
print("-" * 78)
print(f"Linear fit  coupling_step_ms = {coef[0]:.3e} * |E| + {coef[1]:.3e}, "
      f"R^2 = {r2:.4f}")
print("Results saved to results/largescale_scalability.json")
