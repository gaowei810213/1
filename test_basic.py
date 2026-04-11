"""Basic validation test for the curvature-aware deeponet modules."""
import sys
import os
sys.path.insert(0, 'curvature_aware_deeponet')

print("=== Test 1: manifolds.py ===")
from src.manifolds import Sphere, Torus, EllipticParaboloid, HyperbolicParaboloid, Wrinkle
from src.manifolds import estimate_gaussian_curvature
import numpy as np

s = Sphere(R=1.0)
p = s.position(np.pi/4, np.pi/3)
n = s.normal(np.pi/4, np.pi/3)
K = s.gaussian_curvature_analytic(np.pi/4, np.pi/3)
print(f"  Sphere pos={p}, normal={n}, K={K}")
assert abs(K - 1.0) < 1e-10, "Sphere curvature should be 1.0"
assert abs(np.linalg.norm(p) - 1.0) < 1e-10, "Point should be on unit sphere"
assert abs(np.dot(p, n) - 1.0) < 1e-10, "Normal should equal position on unit sphere"

t = Torus(R=3.0, r=1.0)
p2 = t.position(0, 0)
K2 = t.gaussian_curvature_analytic(0, 0)
print(f"  Torus pos={p2}, K={K2:.6f}")
assert abs(p2[0] - 4.0) < 1e-10, "Torus (0,0) should be at x=R+r=4"
assert K2 > 0, "Outer torus should have positive curvature"

ep = EllipticParaboloid(c=1.0)
p3 = ep.position(0, 0)
print(f"  EllipticParaboloid origin={p3}")
assert np.allclose(p3, [0, 0, 0]), "Origin of paraboloid should be (0,0,0)"

hp = HyperbolicParaboloid(c=1.0)
p4 = hp.position(1, 1)
print(f"  HyperbolicParaboloid (1,1)={p4}")
assert abs(p4[2] - 0.0) < 1e-10, "z = (1-1)/2 = 0"

w = Wrinkle(A=0.5, omega_x=2.0, omega_y=2.0)
p5 = w.position(0, 0)
print(f"  Wrinkle origin={p5}")
assert abs(p5[2]) < 1e-10, "sin(0)*sin(0)=0"

print("  [PASS] All manifold tests passed.\n")

print("=== Test 2: parallel_transport.py ===")
from src.parallel_transport import (
    classify_curvature, parallel_transport_R2, parallel_transport_S2,
    parallel_transport_H2, precompute_transport_operators
)

assert classify_curvature(0.5) == 'S2'
assert classify_curvature(0.0) == 'R2'
assert classify_curvature(-0.5) == 'H2'
assert classify_curvature(1e-4) == 'R2'
print("  Curvature classification: OK")

h = np.array([1.0, 2.0, 3.0])
h_out = parallel_transport_R2(h)
assert np.allclose(h, h_out), "R2 transport should be identity"
print("  R2 transport (identity): OK")

# S2 transport: transport along equator
u_pos = np.array([1.0, 0.0, 0.0])
v_pos = np.array([0.0, 1.0, 0.0])
h_v = np.array([0.0, 0.0, 1.0])  # tangent to sphere at v, pointing up
h_u = parallel_transport_S2(u_pos, v_pos, h_v)
print(f"  S2 transport: h_v={h_v} -> h_u={h_u}")
# The transported vector should still be tangent to sphere at u
assert abs(np.dot(h_u, u_pos)) < 1e-10, "Transported vector should be tangent to sphere at u"
# Parallel transport preserves norm
assert abs(np.linalg.norm(h_u) - np.linalg.norm(h_v)) < 1e-10, "Transport should preserve norm"
print("  S2 transport preserves tangency and norm: OK")

# S2 transport: identity when u=v
h_same = parallel_transport_S2(u_pos, u_pos, h_v)
assert np.allclose(h_same, h_v), "Transport to same point should be identity"
print("  S2 self-transport: OK")

# H2 transport: segment parallel to y-axis
u_h2 = np.array([0.0, 2.0])
v_h2 = np.array([0.0, 1.0])
h_v_h2 = np.array([1.0, 0.0])
h_u_h2 = parallel_transport_H2(u_h2, v_h2, h_v_h2)
expected = np.array([2.0, 0.0])  # scale by y_u/y_v = 2
assert np.allclose(h_u_h2, expected), f"H2 vertical transport: expected {expected}, got {h_u_h2}"
print(f"  H2 vertical transport: {h_v_h2} -> {h_u_h2}: OK")

# Precompute transport operators
positions = np.array([[1,0,0],[0,1,0],[0,0,1]], dtype=float)
edges = [[0,1],[1,2]]
curvatures = np.array([1.0, 0.5, -0.5])
ops = precompute_transport_operators(positions, edges, curvatures)
assert len(ops) == 2
assert ops[0]['type_origin'] == 'S2'
assert ops[1]['type_terminal'] == 'H2'
print("  Precompute transport operators: OK")

print("  [PASS] All parallel transport tests passed.\n")

print("=== Test 3: curvature_attention.py ===")
from src.curvature_attention import (
    tensor_field_11, tensor_field_02, subtree_partition
)

metric = np.eye(3)
w1 = np.array([1.0, 0.0, 0.0])
w2 = np.array([0.0, 1.0, 0.0])
h_test = np.array([0.0, 3.0, 0.0])
result_11 = tensor_field_11(w1, w2, h_test, metric)
assert np.allclose(result_11, [3.0, 0.0, 0.0]), f"(1,1)-tensor: expected [3,0,0], got {result_11}"
print(f"  (1,1)-tensor field: OK, result={result_11}")

a1 = np.array([1.0, 0.0, 0.0])
a2 = np.array([0.0, 1.0, 0.0])
h1 = np.array([2.0, 0.0, 0.0])
h2 = np.array([0.0, 5.0, 0.0])
result_02 = tensor_field_02(a1, a2, h1, h2, metric)
assert abs(result_02 - 10.0) < 1e-10, f"(0,2)-tensor: expected 10.0, got {result_02}"
print(f"  (0,2)-tensor field: OK, result={result_02}")

# Subtree partition
edges_sub = [[0,1],[1,2],[2,3],[3,4],[0,2],[1,3]]
subtrees, mapping = subtree_partition(edges_sub, 5, max_depth=2, seed=0)
assert len(mapping) == 5, "All 5 nodes should be assigned"
print(f"  Subtree partition: {len(subtrees)} subtrees, mapping={mapping}")
print("  [PASS] All attention tests passed.\n")

print("=== Test 4: manifold_graph.py ===")
from src.manifold_graph import create_manifold_graph

for mtype in ['sphere', 'torus', 'elliptic', 'hyperbolic', 'wrinkle']:
    g = create_manifold_graph(mtype, n_vertices=16, eps=0.01, seed=42)
    print(f"  {mtype}: {g.n_v} vertices, {g.ne} edges, "
          f"K in [{g.curvatures.min():.4f}, {g.curvatures.max():.4f}]")
    assert g.ne > 0, f"Graph on {mtype} should have edges"
    assert len(g.curvatures) == g.n_v
    assert len(g.transport_ops) == g.ne

print("  [PASS] All manifold graph tests passed.\n")

print("=== Test 5: curvature_aware_coupling.py ===")
from src.curvature_aware_coupling import rbf_kernel, make_coupling_params

import jax.numpy as jnp
t = jnp.linspace(0, 1, 10)
centers = jnp.linspace(0, 1, 5)
K_rbf = rbf_kernel(t, centers, length_scale=0.2)
assert K_rbf.shape == (10, 5), f"RBF kernel shape: expected (10,5), got {K_rbf.shape}"
print(f"  RBF kernel shape: {K_rbf.shape}: OK")

import jax
beta, t_c = make_coupling_params(3, n_beta=5, key=jax.random.PRNGKey(0))
assert len(beta) == 3
assert t_c.shape == (5,)
print(f"  Coupling params: {len(beta)} edges, {t_c.shape[0]} centers: OK")
print("  [PASS] All coupling tests passed.\n")

print("=" * 40)
print("ALL TESTS PASSED!")
print("=" * 40)
