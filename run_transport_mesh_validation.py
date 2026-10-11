"""Validate the corrected transport operators on actual surface meshes.

This experiment replaces the former scalar-rescaling studies.  It uses the
surface generators already used by run_all_experiments.py and evaluates only
quantities that parallel transport is required to preserve: tangency, norm,
and reverse-edge consistency.  It also times the same operator on increasingly
dense torus meshes.  No DeepONet inference or accuracy claim is made here.
"""
import json
import os
import time
import tracemalloc

import numpy as np

from curvature_aware_deeponet.src.manifolds import (
    EllipticParaboloid,
    Sphere,
    Torus,
    Wrinkle,
)
from curvature_aware_deeponet.src.parallel_transport import (
    parallel_transport_S2,
    parallel_transport_normals,
)


def structured_surface(name, nu, nv):
    if name == "sphere":
        surface = Sphere(1.0)
        us = np.linspace(0.15, np.pi - 0.15, nu)
        vs = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
        periodic_v = True
    elif name == "torus":
        surface = Torus(3.0, 1.0)
        us = np.linspace(0.0, 2.0 * np.pi, nu, endpoint=False)
        vs = np.linspace(0.0, 2.0 * np.pi, nv, endpoint=False)
        periodic_v = True
    elif name == "paraboloid":
        surface = EllipticParaboloid(1.0)
        us = np.linspace(-1.5, 1.5, nu)
        vs = np.linspace(-1.5, 1.5, nv)
        periodic_v = False
    elif name == "wrinkle":
        surface = Wrinkle(0.5, 2.0, 2.0)
        us = np.linspace(-1.5, 1.5, nu)
        vs = np.linspace(-1.5, 1.5, nv)
        periodic_v = False
    else:
        raise ValueError(name)

    params = [(u, v) for u in us for v in vs]
    positions = np.array([surface.position(u, v) for u, v in params])
    normals = np.array([surface.normal(u, v) for u, v in params])
    edges = []
    for i in range(nu):
        for j in range(nv):
            p = i * nv + j
            if i + 1 < nu:
                edges.append((p, (i + 1) * nv + j))
            elif name == "torus":
                edges.append((p, j))
            if j + 1 < nv:
                edges.append((p, i * nv + j + 1))
            elif periodic_v:
                edges.append((p, i * nv))
    return positions, normals, edges


def tangent_vector(normal, rng):
    h = rng.normal(size=3)
    h -= np.dot(h, normal) * normal
    return h / np.linalg.norm(h)


def validate_surface(name, nu=24, nv=24, seed=2026):
    positions, normals, edges = structured_surface(name, nu, nv)
    rng = np.random.default_rng(seed)
    tangency = []
    norm_error = []
    reverse_error = []
    sphere_closed_form_difference = []
    for source, target in edges:
        h0 = tangent_vector(normals[source], rng)
        h1 = parallel_transport_normals(normals[source], normals[target], h0)
        h_back = parallel_transport_normals(normals[target], normals[source], h1)
        tangency.append(abs(np.dot(h1, normals[target])))
        norm_error.append(abs(np.linalg.norm(h1) - np.linalg.norm(h0)))
        reverse_error.append(np.linalg.norm(h_back - h0))
        if name == "sphere":
            exact = parallel_transport_S2(
                positions[target], positions[source], h0
            )
            sphere_closed_form_difference.append(np.linalg.norm(h1 - exact))
    result = {
        "vertices": len(positions),
        "edges": len(edges),
        "tangency_mean": float(np.mean(tangency)),
        "tangency_max": float(np.max(tangency)),
        "norm_error_mean": float(np.mean(norm_error)),
        "norm_error_max": float(np.max(norm_error)),
        "reverse_error_mean": float(np.mean(reverse_error)),
        "reverse_error_max": float(np.max(reverse_error)),
    }
    if sphere_closed_form_difference:
        result["sphere_closed_form_difference_mean"] = float(
            np.mean(sphere_closed_form_difference)
        )
        result["sphere_closed_form_difference_max"] = float(
            np.max(sphere_closed_form_difference)
        )
    return result


def scalability():
    rows = []
    for side in [16, 24, 32, 48, 64, 80]:
        positions, normals, edges = structured_surface("torus", side, side)
        rng = np.random.default_rng(77)
        vectors = np.array([tangent_vector(normals[i], rng) for i, _ in edges])
        elapsed_samples = []
        peak_samples = []
        transported = None
        for _ in range(5):
            tracemalloc.start()
            start = time.perf_counter()
            transported = [
                parallel_transport_normals(normals[i], normals[j], h)
                for (i, j), h in zip(edges, vectors)
            ]
            elapsed_samples.append(time.perf_counter() - start)
            _, peak = tracemalloc.get_traced_memory()
            peak_samples.append(peak)
            tracemalloc.stop()
        elapsed = float(np.median(elapsed_samples))
        peak = float(np.max(peak_samples))
        # Consume the result so the timing cannot be optimized away.
        checksum = float(np.sum(transported))
        rows.append(
            {
                "vertices": len(positions),
                "edges": len(edges),
                "transport_time_ms": elapsed * 1e3,
                "microseconds_per_edge": elapsed * 1e6 / len(edges),
                "peak_memory_mb": peak / (1024.0 * 1024.0),
                "timing_repetitions": 5,
                "checksum": checksum,
            }
        )
    edge_count = np.array([row["edges"] for row in rows], dtype=float)
    timings = np.array([row["transport_time_ms"] for row in rows])
    design = np.column_stack([edge_count, np.ones_like(edge_count)])
    slope, intercept = np.linalg.lstsq(design, timings, rcond=None)[0]
    fitted = design @ np.array([slope, intercept])
    r2 = 1.0 - np.sum((timings - fitted) ** 2) / np.sum(
        (timings - timings.mean()) ** 2
    )
    return rows, {
        "slope_ms_per_edge": float(slope),
        "intercept_ms": float(intercept),
        "r_squared": float(r2),
    }


def main():
    os.makedirs("results", exist_ok=True)
    surfaces = {
        name: validate_surface(name)
        for name in ["sphere", "torus", "paraboloid", "wrinkle"]
    }
    rows, fit = scalability()
    results = {"surface_validation": surfaces, "scalability": rows, "fit": fit}
    with open("results/transport_mesh_validation.json", "w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)
    for name, values in surfaces.items():
        print(
            f"{name:12s} |E|={values['edges']:5d} "
            f"tan_max={values['tangency_max']:.3e} "
            f"norm_max={values['norm_error_max']:.3e} "
            f"reverse_max={values['reverse_error_max']:.3e}"
        )
    print(
        f"scalability: slope={fit['slope_ms_per_edge']:.3e} ms/edge, "
        f"R^2={fit['r_squared']:.4f}"
    )


if __name__ == "__main__":
    main()
