"""
Manifold definitions and geometric computations.
Provides parametric surfaces (sphere, torus, elliptic/hyperbolic paraboloid, wrinkle)
with Gaussian curvature estimation and metric tensor computation.
"""

import numpy as np
from abc import ABC, abstractmethod


class Manifold(ABC):
    """Base class for 2D manifolds embedded in R^3."""

    @abstractmethod
    def position(self, u, v):
        """Map parametric coordinates (u,v) to R^3."""
        pass

    @abstractmethod
    def normal(self, u, v):
        """Unit normal vector at parametric coordinates (u,v)."""
        pass

    def first_fundamental_form(self, u, v, du=1e-6):
        """Compute metric tensor g_ij via finite differences."""
        p = self.position(u, v)
        pu = (self.position(u + du, v) - self.position(u - du, v)) / (2 * du)
        pv = (self.position(u, v + du) - self.position(u, v - du)) / (2 * du)
        E = np.dot(pu, pu)
        F = np.dot(pu, pv)
        G = np.dot(pv, pv)
        return np.array([[E, F], [F, G]])

    def gaussian_curvature_analytic(self, u, v):
        """Override in subclasses for analytic Gaussian curvature."""
        raise NotImplementedError


class Sphere(Manifold):
    """Unit sphere S^2 with radius R."""

    def __init__(self, R=1.0):
        self.R = R

    def position(self, theta, phi):
        R = self.R
        x = R * np.sin(theta) * np.cos(phi)
        y = R * np.sin(theta) * np.sin(phi)
        z = R * np.cos(theta)
        return np.array([x, y, z])

    def normal(self, theta, phi):
        return self.position(theta, phi) / self.R

    def gaussian_curvature_analytic(self, theta, phi):
        return 1.0 / (self.R ** 2)


class Torus(Manifold):
    """Torus with major radius R and minor radius r."""

    def __init__(self, R=3.0, r=1.0):
        self.R = R
        self.r = r

    def position(self, u, v):
        R, r = self.R, self.r
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        return np.array([x, y, z])

    def normal(self, u, v):
        x = np.cos(v) * np.cos(u)
        y = np.cos(v) * np.sin(u)
        z = np.sin(v)
        return np.array([x, y, z])

    def gaussian_curvature_analytic(self, u, v):
        R, r = self.R, self.r
        return np.cos(v) / (r * (R + r * np.cos(v)))


class EllipticParaboloid(Manifold):
    """Elliptic paraboloid z = (x^2 + y^2) / (2c)."""

    def __init__(self, c=1.0):
        self.c = c

    def position(self, u, v):
        x = u
        y = v
        z = (u ** 2 + v ** 2) / (2 * self.c)
        return np.array([x, y, z])

    def normal(self, u, v):
        c = self.c
        n = np.array([-u / c, -v / c, 1.0])
        return n / np.linalg.norm(n)


class HyperbolicParaboloid(Manifold):
    """Hyperbolic paraboloid z = (x^2 - y^2) / (2c)."""

    def __init__(self, c=1.0):
        self.c = c

    def position(self, u, v):
        x = u
        y = v
        z = (u ** 2 - v ** 2) / (2 * self.c)
        return np.array([x, y, z])

    def normal(self, u, v):
        c = self.c
        n = np.array([-u / c, v / c, 1.0])
        return n / np.linalg.norm(n)


class Wrinkle(Manifold):
    """Wrinkle surface with mixed positive/zero/negative curvature.
    z = A * sin(omega_x * x) * sin(omega_y * y)
    """

    def __init__(self, A=0.5, omega_x=2.0, omega_y=2.0):
        self.A = A
        self.omega_x = omega_x
        self.omega_y = omega_y

    def position(self, u, v):
        x = u
        y = v
        z = self.A * np.sin(self.omega_x * u) * np.sin(self.omega_y * v)
        return np.array([x, y, z])

    def normal(self, u, v):
        A, wx, wy = self.A, self.omega_x, self.omega_y
        dz_du = A * wx * np.cos(wx * u) * np.sin(wy * v)
        dz_dv = A * wy * np.sin(wx * u) * np.cos(wy * v)
        n = np.array([-dz_du, -dz_dv, 1.0])
        return n / np.linalg.norm(n)


def estimate_gaussian_curvature(positions, neighbors, normals):
    """Estimate Gaussian curvature at each vertex using normal bundle theory.

    K|_{v_i} = (1/2) sum_{v_j in N(v_i)} (v_j - v_i)/||v_i - v_j||^2 * phi_ij . n

    Args:
        positions: (N, 3) array of vertex positions.
        neighbors: list of lists, neighbors[i] = list of neighbor indices of vertex i.
        normals: (N, 3) array of unit normals at each vertex.

    Returns:
        curvatures: (N,) array of Gaussian curvature estimates.
    """
    N = len(positions)
    curvatures = np.zeros(N)

    for i in range(N):
        vi = positions[i]
        ni = normals[i]
        K_sum = np.zeros(3)
        for j in neighbors[i]:
            vj = positions[j]
            diff = vj - vi
            dist_sq = np.dot(diff, diff)
            if dist_sq < 1e-15:
                continue
            # Dihedral angle approximation: use angle between normals
            phi_ij = np.arccos(np.clip(np.dot(normals[i], normals[j]), -1, 1))
            K_sum += (diff / dist_sq) * phi_ij
        curvatures[i] = 0.5 * np.dot(K_sum, ni)

    return curvatures


def compute_metric_tensor_at_vertex(position, curvature_R1212):
    """Approximate metric tensor via Jacobi field expansion.
    g_ij(x) = delta_ij - (1/3) R_{iklj} x^k x^l + O(|x|^3)
    For 2D surface: only R_1212 is independent.
    """
    # In normal coordinates centered at vertex, g_ij = delta_ij at the vertex itself
    # The correction is second-order in distance from vertex
    return np.eye(2)  # At the vertex, metric is identity in normal coords
