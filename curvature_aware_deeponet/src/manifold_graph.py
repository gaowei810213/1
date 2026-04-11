"""
Metric graph embedded in a discrete manifold.
Extends the original Graph class with manifold geometry:
  - vertex positions on the manifold
  - Gaussian curvature at each vertex
  - geodesic curvature along each edge
  - parallel transport operators
"""

import numpy as np
import jax.numpy as jnp
import networkx as nx
import sys
import os

# Add original src to path for reuse
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'physics-informed-operator-networks-for-pdes-on-metric-graphs-main', 'src'))
from graph import Graph

from .manifolds import (
    Sphere, Torus, EllipticParaboloid, HyperbolicParaboloid, Wrinkle,
    estimate_gaussian_curvature
)
from .parallel_transport import (
    precompute_transport_operators, classify_curvature,
    parallel_transport_edge, CURVATURE_THRESHOLD
)


class ManifoldGraph(Graph):
    """Metric graph embedded in a 2D Riemannian manifold.

    Extends the base Graph class with:
      - 3D vertex positions on the manifold surface
      - Gaussian curvature at each vertex
      - Precomputed parallel transport operators for each edge
      - Curvature-augmented sensor measurements
    """

    def __init__(self, manifold, adjacency, positions_3d, eps=1e-2, v=None):
        """
        Args:
            manifold: a Manifold instance (Sphere, Torus, etc.)
            adjacency: (n_v, n_v) adjacency matrix (weighted by edge lengths)
            positions_3d: (n_v, 3) array of vertex positions on the manifold
            eps: diffusion coefficient
            v: edge velocities, array of length n_edges or scalar
        """
        super().__init__()

        self.manifold = manifold
        self.A = adjacency
        self.eps = eps
        self.positions_3d = np.array(positions_3d)

        # Use 2D projection for graph layout
        self.pos = self.positions_3d[:, :2]

        # Build the base graph structure
        self.buildGraph()

        # Set edge velocities
        if v is None:
            self.v = np.ones(self.ne)
        elif np.isscalar(v):
            self.v = v * np.ones(self.ne)
        else:
            self.v = np.array(v)

        # Bounding box for (t, x)
        self.lb = np.array([0.0, 0.0])
        self.ub = np.array([1.0, 1.0])

        # Compute geometry
        self._compute_geometry()

        self.title = f"ManifoldGraph ({self.ne} edges, {self.n_v} vertices)"

    def _compute_geometry(self):
        """Compute all geometric quantities needed for curvature-aware coupling."""

        # 1. Compute vertex normals
        self.normals = np.zeros((self.n_v, 3))
        for i in range(self.n_v):
            # Estimate normal from neighboring faces
            neighbors = self._get_neighbors(i)
            if len(neighbors) >= 2:
                # Use cross product of edges to estimate normal
                vi = self.positions_3d[i]
                normals_list = []
                for k in range(len(neighbors) - 1):
                    e1 = self.positions_3d[neighbors[k]] - vi
                    e2 = self.positions_3d[neighbors[k + 1]] - vi
                    n = np.cross(e1, e2)
                    norm = np.linalg.norm(n)
                    if norm > 1e-10:
                        normals_list.append(n / norm)
                if normals_list:
                    avg_normal = np.mean(normals_list, axis=0)
                    norm = np.linalg.norm(avg_normal)
                    if norm > 1e-10:
                        self.normals[i] = avg_normal / norm
                    else:
                        self.normals[i] = np.array([0, 0, 1])
                else:
                    self.normals[i] = np.array([0, 0, 1])
            else:
                self.normals[i] = np.array([0, 0, 1])

        # 2. Compute Gaussian curvature at each vertex
        neighbors_list = [self._get_neighbors(i) for i in range(self.n_v)]
        self.curvatures = estimate_gaussian_curvature(
            self.positions_3d, neighbors_list, self.normals
        )

        # 3. Compute geodesic curvature for each edge
        self.geodesic_curvatures = np.zeros(self.ne)
        for idx, (vi, vj) in enumerate(self.E):
            # Approximate geodesic curvature as the deviation from a geodesic
            edge_vec = self.positions_3d[vj] - self.positions_3d[vi]
            edge_len = np.linalg.norm(edge_vec)
            if edge_len > 1e-10:
                # Project edge onto tangent plane at midpoint
                mid_normal = (self.normals[vi] + self.normals[vj]) / 2
                mid_normal = mid_normal / (np.linalg.norm(mid_normal) + 1e-15)
                tangent_comp = edge_vec - np.dot(edge_vec, mid_normal) * mid_normal
                self.geodesic_curvatures[idx] = (
                    np.linalg.norm(edge_vec - tangent_comp) / (edge_len + 1e-15)
                )

        # 4. Precompute parallel transport operators
        self.transport_ops = precompute_transport_operators(
            self.positions_3d, self.E, self.curvatures
        )

        # 5. Classify curvature type at each vertex
        self.curvature_types = [
            classify_curvature(self.curvatures[i]) for i in range(self.n_v)
        ]

    def _get_neighbors(self, vertex_idx):
        """Get list of neighbor vertex indices."""
        neighbors = []
        for vi, vj in self.E:
            if vi == vertex_idx:
                neighbors.append(vj)
            elif vj == vertex_idx:
                neighbors.append(vi)
        return list(set(neighbors))

    def get_curvature_sensor(self, edge_idx):
        """Get curvature information for sensor augmentation.

        Returns:
            (K_origin, K_terminal, kappa_e) tuple of curvatures.
        """
        vi, vj = self.E[edge_idx]
        return (
            self.curvatures[vi],
            self.curvatures[vj],
            self.geodesic_curvatures[edge_idx]
        )

    def transport_to_vertex(self, value, edge_idx, to_origin=True):
        """Parallel transport a value along an edge to a vertex.

        Args:
            value: scalar or vector to transport.
            edge_idx: index of the edge.
            to_origin: if True, transport to origin vertex; else to terminal.

        Returns:
            Transported value.
        """
        vi, vj = self.E[edge_idx]
        ops = self.transport_ops[edge_idx]

        if to_origin:
            target_pos = ops['origin_pos']
            source_pos = ops['terminal_pos']
            K = ops['K_origin']
        else:
            target_pos = ops['terminal_pos']
            source_pos = ops['origin_pos']
            K = ops['K_terminal']

        if np.isscalar(value) or (isinstance(value, np.ndarray) and value.ndim == 0):
            # Scalar: parallel transport is trivial
            return value

        return parallel_transport_edge(target_pos, source_pos, value, K)


def create_manifold_graph(manifold_type, n_vertices=20, eps=1e-2,
                          n_inflow=2, n_outflow=2, seed=42):
    """Factory function to create a ManifoldGraph on a given manifold.

    Args:
        manifold_type: str, one of 'sphere', 'torus', 'elliptic', 'hyperbolic', 'wrinkle'
        n_vertices: approximate number of vertices
        eps: diffusion coefficient
        n_inflow: number of inflow boundary vertices
        n_outflow: number of outflow boundary vertices
        seed: random seed

    Returns:
        ManifoldGraph instance.
    """
    rng = np.random.RandomState(seed)

    manifold_map = {
        'sphere': Sphere(R=1.0),
        'torus': Torus(R=3.0, r=1.0),
        'elliptic': EllipticParaboloid(c=1.0),
        'hyperbolic': HyperbolicParaboloid(c=1.0),
        'wrinkle': Wrinkle(A=0.5, omega_x=2.0, omega_y=2.0),
    }

    if manifold_type not in manifold_map:
        raise ValueError(f"Unknown manifold type: {manifold_type}. "
                         f"Choose from {list(manifold_map.keys())}")

    manifold = manifold_map[manifold_type]

    # Generate vertex positions on the manifold
    if manifold_type == 'sphere':
        # Fibonacci sphere sampling
        golden_ratio = (1 + np.sqrt(5)) / 2
        indices = np.arange(n_vertices)
        theta = np.arccos(1 - 2 * (indices + 0.5) / n_vertices)
        phi = 2 * np.pi * indices / golden_ratio
        positions = np.array([manifold.position(t, p) for t, p in zip(theta, phi)])

    elif manifold_type == 'torus':
        n_u = int(np.sqrt(n_vertices * manifold.R / manifold.r))
        n_v = max(n_vertices // n_u, 3)
        u_vals = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
        v_vals = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
        uu, vv = np.meshgrid(u_vals, v_vals)
        uu, vv = uu.ravel(), vv.ravel()
        positions = np.array([manifold.position(u, v) for u, v in zip(uu, vv)])

    else:  # paraboloids and wrinkle
        side = int(np.sqrt(n_vertices))
        u_vals = np.linspace(-2, 2, side)
        v_vals = np.linspace(-2, 2, side)
        uu, vv = np.meshgrid(u_vals, v_vals)
        uu, vv = uu.ravel(), vv.ravel()
        positions = np.array([manifold.position(u, v) for u, v in zip(uu, vv)])

    n_actual = len(positions)

    # Build adjacency via k-nearest neighbors
    from scipy.spatial import KDTree
    tree = KDTree(positions)
    k = min(6, n_actual - 1)
    _, indices = tree.query(positions, k=k + 1)  # +1 for self

    # Build directed adjacency matrix
    A = np.zeros((n_actual, n_actual))
    for i in range(n_actual):
        for j_idx in range(1, k + 1):  # skip self
            j = indices[i, j_idx]
            dist = np.linalg.norm(positions[i] - positions[j])
            # Make directed: only add edge if i < j (then reverse some for flow)
            if i < j:
                A[i, j] = dist

    # Assign random velocities
    n_edges = int(np.sum(A > 0))
    velocities = rng.uniform(0.5, 2.0, size=n_edges)

    # Set inflow/outflow boundary conditions
    dirichlet_alpha = np.zeros(n_actual)
    dirichlet_beta = np.zeros(n_actual)

    # Pick boundary nodes (nodes with fewest connections)
    out_degrees = np.sum(A > 0, axis=1)
    in_degrees = np.sum(A > 0, axis=0)
    boundary_candidates = np.where((out_degrees > 0) & (in_degrees == 0))[0]
    if len(boundary_candidates) < n_inflow:
        boundary_candidates = np.argsort(in_degrees)[:n_inflow + n_outflow]

    inflow_nodes = boundary_candidates[:n_inflow]
    outflow_candidates = np.where((in_degrees > 0) & (out_degrees == 0))[0]
    if len(outflow_candidates) < n_outflow:
        outflow_candidates = np.argsort(out_degrees)[:n_outflow]
    outflow_nodes = outflow_candidates[:n_outflow]

    for i in inflow_nodes:
        dirichlet_alpha[i] = 1.0
    for i in outflow_nodes:
        dirichlet_beta[i] = 1.0

    graph = ManifoldGraph(
        manifold=manifold,
        adjacency=A,
        positions_3d=positions,
        eps=eps,
        v=velocities
    )
    graph.dirichletAlpha = dirichlet_alpha
    graph.dirichletBeta = dirichlet_beta

    return graph
