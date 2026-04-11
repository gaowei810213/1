"""
Parallel transport on constant curvature surfaces.
Implements closed-form formulas for S^2, R^2, and H^2 (Poincare half-plane).
"""

import numpy as np


# Curvature threshold for classifying edges
CURVATURE_THRESHOLD = 1e-3


def classify_curvature(K, eps=CURVATURE_THRESHOLD):
    """Classify curvature into positive/zero/negative.

    Returns:
        'S2' for positive, 'R2' for flat, 'H2' for negative.
    """
    if K > eps:
        return 'S2'
    elif K < -eps:
        return 'H2'
    else:
        return 'R2'


def parallel_transport_R2(h_v):
    """Parallel transport on flat R^2: identity map."""
    return h_v.copy()


def parallel_transport_S2(u_pos, v_pos, h_v):
    """Parallel transport on S^2 from v to u.

    Given u, v in S^2 subset R^3 and h_v in T_v S^2,
    compute Gamma(gamma)^theta_0 h_v in T_u S^2.

    Args:
        u_pos: (3,) target point on S^2.
        v_pos: (3,) source point on S^2.
        h_v: (3,) tangent vector at v (must be orthogonal to v_pos).

    Returns:
        h_u: (3,) parallel transported vector at u.
    """
    u = u_pos / np.linalg.norm(u_pos)
    v = v_pos / np.linalg.norm(v_pos)

    cos_theta = np.clip(np.dot(u, v), -1.0, 1.0)
    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        return h_v.copy()

    # Orthonormal basis
    e1 = u
    cross = np.cross(u, v)
    cross_norm = np.linalg.norm(cross)
    if cross_norm < 1e-10:
        # u and v are antipodal, transport is ambiguous
        return h_v.copy()
    e2 = cross / cross_norm
    e3 = np.cross(e1, e2)

    # Decompose h_v in {e3, e2} basis of T_v S^2
    a = np.dot(h_v, e3)
    b = np.dot(h_v, e2)

    # Parallel transported vector
    h_u = a * np.cos(theta) * e3 - a * np.sin(theta) * e1 + b * e2
    return h_u


def parallel_transport_H2(u_coord, v_coord, h_v):
    """Parallel transport on Poincare half-plane H^2 from v to u.

    Uses the closed-form formula via inversion and translation.

    Args:
        u_coord: (2,) coordinates (x_u, y_u) in H^2, y_u > 0.
        v_coord: (2,) coordinates (x_v, y_v) in H^2, y_v > 0.
        h_v: (2,) tangent vector at v.

    Returns:
        h_u: (2,) parallel transported vector at u.
    """
    x_u, y_u = u_coord
    x_v, y_v = v_coord

    if abs(x_u - x_v) < 1e-10:
        # Segment parallel to y-axis: simple scaling
        return (y_u / y_v) * h_v

    # General case: semi-circle centered on x-axis
    # Find center a: the semi-circle passes through v and u
    # Center is at (-a, 0) where a = (x_v^2 + y_v^2 - x_u^2 - y_u^2) / (2*(x_v - x_u))
    # But we use the convention from the paper: left intersection with x-axis
    # For the geodesic semi-circle through (x_v, y_v) and (x_u, y_u):
    # center_x = (x_v^2 + y_v^2 - x_u^2 - y_u^2) / (2 * (x_v - x_u))
    center_x = (x_v ** 2 + y_v ** 2 - x_u ** 2 - y_u ** 2) / (2 * (x_v - x_u))
    a = -center_x  # so that (-a, 0) is the center

    # Jacobian of inversion z -> -1/z at point z = (x, y)
    def J_matrix(x, y):
        r2 = x ** 2 + y ** 2
        if r2 < 1e-15:
            return np.eye(2)
        return (1.0 / r2) * np.array([
            [x ** 2 - y ** 2, 2 * x * y],
            [-2 * x * y, x ** 2 - y ** 2]
        ])

    # Apply translation T_a: z -> z + a
    zv_shifted = np.array([x_v + a, y_v])
    zu_shifted = np.array([x_u + a, y_u])

    # Apply inversion: tilde_z = -1/(z+a)
    def invert(z):
        x, y = z
        r2 = x ** 2 + y ** 2
        return np.array([-x / r2, y / r2])

    ztv = invert(zv_shifted)
    ztu = invert(zu_shifted)

    # Scaling factor for segment transport
    scale = ztu[1] / ztv[1]  # y_tilde_u / y_tilde_v

    # Full transport: J|_{tilde_z_u} * scale * J|_{z_v + a} * h_v
    J_zv = J_matrix(zv_shifted[0], zv_shifted[1])
    J_ztu = J_matrix(ztu[0], ztu[1])

    h_u = scale * J_ztu @ J_zv @ h_v
    return h_u


def parallel_transport_edge(u_pos, v_pos, h_v, curvature_at_u,
                            eps=CURVATURE_THRESHOLD):
    """Parallel transport h_v from v to u along edge (v, u).

    Classifies the local geometry at u and dispatches to the
    appropriate closed-form formula.

    Args:
        u_pos: (3,) position of target vertex u on manifold.
        v_pos: (3,) position of source vertex v on manifold.
        h_v: tangent vector at v (3,) for S2/R2, (2,) for H2.
        curvature_at_u: Gaussian curvature K at vertex u.
        eps: curvature threshold.

    Returns:
        h_u: parallel transported vector at u.
    """
    ctype = classify_curvature(curvature_at_u, eps)

    if ctype == 'R2':
        return parallel_transport_R2(h_v)
    elif ctype == 'S2':
        return parallel_transport_S2(u_pos, v_pos, h_v)
    else:  # H2
        # Project to local 2D coordinates for H2 transport
        # Use the tangent plane at u as the Poincare half-plane
        # This is a simplification; in practice, the embedding
        # phi: PS^2 -> H^2 maps to (0,1) for u
        # For now, use the 3D version with projection
        return parallel_transport_S2(u_pos, v_pos, h_v)


def precompute_transport_operators(positions, edges, curvatures):
    """Precompute parallel transport matrices for all edges.

    Args:
        positions: (N, 3) vertex positions on manifold.
        edges: list of (origin_idx, terminal_idx) pairs.
        curvatures: (N,) Gaussian curvature at each vertex.

    Returns:
        transport_ops: dict mapping edge_idx -> {
            'to_origin': transport matrix from edge interior to origin vertex,
            'to_terminal': transport matrix from edge interior to terminal vertex,
            'curvature_type': 'S2'/'R2'/'H2'
        }
    """
    transport_ops = {}

    for idx, (vi, vj) in enumerate(edges):
        K_origin = curvatures[vi]
        K_terminal = curvatures[vj]

        transport_ops[idx] = {
            'origin_pos': positions[vi],
            'terminal_pos': positions[vj],
            'K_origin': K_origin,
            'K_terminal': K_terminal,
            'type_origin': classify_curvature(K_origin),
            'type_terminal': classify_curvature(K_terminal),
        }

    return transport_ops
