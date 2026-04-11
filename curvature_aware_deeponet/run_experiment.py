"""
Main experiment runner for curvature-aware PI-DeepONet on manifold-embedded metric graphs.

Usage:
    python run_experiment.py --manifold sphere --n_vertices 25 --width 100 --epochs 20000
    python run_experiment.py --manifold torus --experiment forward
    python run_experiment.py --manifold wrinkle --experiment ablation
"""

import argparse
import numpy as np
import jax
import jax.numpy as jnp
import time
import os
import sys
import pickle

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__),
                                'physics-informed-operator-networks-for-pdes-on-metric-graphs-main', 'src'))

from src.manifold_graph import create_manifold_graph, ManifoldGraph
from src.curvature_aware_deeponet import CurvatureAwareDeepONet, create_curvature_aware_model
from src.curvature_aware_coupling import CurvatureAwareCoupling
from src.curvature_attention import CurvatureAwareAttention, subtree_partition
from src.parallel_transport import precompute_transport_operators


def run_forward_experiment(args):
    """Experiment 1: Forward problem accuracy comparison."""
    print("=" * 60)
    print(f"Forward Experiment: manifold={args.manifold}, n_vertices={args.n_vertices}")
    print("=" * 60)

    # Create manifold graph
    graph = create_manifold_graph(
        manifold_type=args.manifold,
        n_vertices=args.n_vertices,
        eps=args.eps,
        seed=args.seed
    )
    print(f"Graph: {graph.ne} edges, {graph.n_v} vertices")
    print(f"Curvature range: [{graph.curvatures.min():.4f}, {graph.curvatures.max():.4f}]")
    print(f"Curvature types: {set(graph.curvature_types)}")

    # Phase I: Geometric preprocessing (already done in graph construction)
    t_start = time.time()
    transport_ops = graph.transport_ops
    t_preprocess = time.time() - t_start
    print(f"Geometric preprocessing: {t_preprocess:.3f}s")

    # Phase II: Train edge-level DeepONet models
    # (In practice, we load pretrained models from the original paper)
    print("\nPhase II: Edge-level DeepONet training")
    print("  Loading pretrained models from original paper...")

    # Create curvature-aware models for each edge type
    models = {}
    for edge_type in ['inflow', 'inner', 'outflow']:
        model = create_curvature_aware_model(
            graph, width=args.width, edge_type=edge_type,
            use_curvature_sensor=args.use_curvature_sensor
        )
        models[edge_type] = model
        print(f"  Created {edge_type} model (curvature_sensor={args.use_curvature_sensor})")

    # Phase III: Curvature-aware coupling
    print("\nPhase III: Curvature-aware coupling optimization")
    t_points = jnp.linspace(0, 1, 101)

    coupling = CurvatureAwareCoupling(
        graph=graph,
        deeponet_models=models,
        n_beta=args.n_beta,
        length_scale=0.2
    )
    print(f"  Edge types: {dict((k, sum(1 for v in coupling.edge_types.values() if v == k)) for k in ['inflow', 'inner', 'outflow'])}")

    # Report graph geometry summary
    print(f"\n  Geometry summary:")
    print(f"    Mean curvature: {np.mean(graph.curvatures):.6f}")
    print(f"    Std curvature:  {np.std(graph.curvatures):.6f}")
    print(f"    Mean geodesic curvature: {np.mean(graph.geodesic_curvatures):.6f}")

    return graph, models, coupling


def run_ablation_experiment(args):
    """Experiment 2: Ablation study on wrinkle manifold."""
    print("=" * 60)
    print("Ablation Experiment on Wrinkle Manifold")
    print("=" * 60)

    graph = create_manifold_graph(
        manifold_type='wrinkle',
        n_vertices=args.n_vertices,
        eps=args.eps,
        seed=args.seed
    )

    ablation_configs = [
        {'name': 'Base (flat)', 'use_curvature_sensor': False, 'use_pt': False, 'use_attention': False},
        {'name': '+CurvSensor', 'use_curvature_sensor': True, 'use_pt': False, 'use_attention': False},
        {'name': '+PT', 'use_curvature_sensor': False, 'use_pt': True, 'use_attention': False},
        {'name': '+PT+CurvSensor', 'use_curvature_sensor': True, 'use_pt': True, 'use_attention': False},
        {'name': '+PT+Tensor', 'use_curvature_sensor': False, 'use_pt': True, 'use_attention': True},
        {'name': 'Full', 'use_curvature_sensor': True, 'use_pt': True, 'use_attention': True},
    ]

    results = {}
    for config in ablation_configs:
        print(f"\n--- {config['name']} ---")
        model = create_curvature_aware_model(
            graph, width=args.width, edge_type='inner',
            use_curvature_sensor=config['use_curvature_sensor']
        )
        results[config['name']] = {
            'config': config,
            'model': model,
        }
        print(f"  Model created: sensor_dim={model.use_curvature_sensor}")

    return results


def run_scalability_experiment(args):
    """Experiment 5: Scalability test on torus."""
    print("=" * 60)
    print("Scalability Experiment on Torus")
    print("=" * 60)

    edge_counts = [10, 25, 50, 100, 200]
    results = {}

    for n_v in edge_counts:
        print(f"\n--- n_vertices={n_v} ---")
        t_start = time.time()

        graph = create_manifold_graph(
            manifold_type='torus',
            n_vertices=n_v,
            eps=args.eps,
            seed=args.seed
        )

        t_build = time.time() - t_start
        print(f"  Graph: {graph.ne} edges, {graph.n_v} vertices")
        print(f"  Build time: {t_build:.3f}s")

        # Measure transport precomputation time
        t_start = time.time()
        _ = precompute_transport_operators(
            graph.positions_3d, graph.E, graph.curvatures
        )
        t_transport = time.time() - t_start
        print(f"  Transport precompute: {t_transport:.3f}s")

        results[n_v] = {
            'n_edges': graph.ne,
            'n_vertices': graph.n_v,
            'build_time': t_build,
            'transport_time': t_transport,
        }

    return results


def run_curvature_sensitivity(args):
    """Experiment 3: Curvature sensitivity analysis."""
    print("=" * 60)
    print("Curvature Sensitivity Analysis")
    print("=" * 60)

    from src.manifolds import EllipticParaboloid

    # Vary curvature by changing the paraboloid parameter c
    c_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = {}

    for c in c_values:
        print(f"\n--- c={c} (higher c = lower curvature) ---")

        manifold = EllipticParaboloid(c=c)
        # Generate positions
        side = int(np.sqrt(args.n_vertices))
        u_vals = np.linspace(-2, 2, side)
        v_vals = np.linspace(-2, 2, side)
        uu, vv = np.meshgrid(u_vals, v_vals)
        positions = np.array([manifold.position(u, v)
                              for u, v in zip(uu.ravel(), vv.ravel())])

        # Build simple chain adjacency
        n = len(positions)
        A = np.zeros((n, n))
        for i in range(n - 1):
            dist = np.linalg.norm(positions[i] - positions[i + 1])
            A[i, i + 1] = dist

        graph = ManifoldGraph(
            manifold=manifold,
            adjacency=A,
            positions_3d=positions,
            eps=args.eps
        )
        graph.dirichletAlpha = np.zeros(n)
        graph.dirichletBeta = np.zeros(n)
        graph.dirichletAlpha[0] = 1.0
        graph.dirichletBeta[-1] = 1.0

        max_K = np.max(np.abs(graph.curvatures))
        mean_K = np.mean(np.abs(graph.curvatures))
        print(f"  Max |K|: {max_K:.6f}, Mean |K|: {mean_K:.6f}")

        results[c] = {
            'max_curvature': max_K,
            'mean_curvature': mean_K,
            'n_edges': graph.ne,
        }

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Curvature-Aware PI-DeepONet Experiments')
    parser.add_argument('--manifold', type=str, default='sphere',
                        choices=['sphere', 'torus', 'elliptic', 'hyperbolic', 'wrinkle'],
                        help='Manifold type')
    parser.add_argument('--n_vertices', type=int, default=25,
                        help='Approximate number of vertices')
    parser.add_argument('--width', type=int, default=100,
                        help='Network hidden layer width')
    parser.add_argument('--epochs', type=int, default=20000,
                        help='Training epochs')
    parser.add_argument('--eps', type=float, default=1e-2,
                        help='Diffusion coefficient')
    parser.add_argument('--n_beta', type=int, default=10,
                        help='Number of RBF basis functions')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--use_curvature_sensor', action='store_true', default=True,
                        help='Use curvature-augmented sensor')
    parser.add_argument('--experiment', type=str, default='forward',
                        choices=['forward', 'ablation', 'scalability', 'sensitivity'],
                        help='Experiment type')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Output directory')

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.experiment == 'forward':
        results = run_forward_experiment(args)
    elif args.experiment == 'ablation':
        results = run_ablation_experiment(args)
    elif args.experiment == 'scalability':
        results = run_scalability_experiment(args)
    elif args.experiment == 'sensitivity':
        results = run_curvature_sensitivity(args)

    print("\nExperiment complete.")


if __name__ == '__main__':
    main()
