# Curvature-Aware PI-DeepONet for Drift-Diffusion on Manifold-Embedded Metric Graphs

## Project Structure

```
curvature_aware_deeponet/
├── src/
│   ├── __init__.py                    # Package init
│   ├── manifolds.py                   # Manifold definitions (S2, Torus, paraboloids, wrinkle)
│   ├── parallel_transport.py          # Closed-form parallel transport on S2, R2, H2
│   ├── manifold_graph.py              # ManifoldGraph: metric graph embedded in manifold
│   ├── curvature_aware_deeponet.py    # Curvature-aware PI-DeepONet model
│   ├── curvature_aware_coupling.py    # Curvature-aware coupling loss optimizer
│   ├── curvature_attention.py         # Curvature-aware graph attention mechanism
│   └── GPs.py                         # Gaussian process sampling
├── run_experiment.py                  # Main experiment runner
└── README.md
```

## Dependencies

- JAX + JAXlib
- NumPy, SciPy
- optax (JAX optimizer library)
- networkx
- matplotlib

## Usage

```bash
# Forward problem on sphere
python run_experiment.py --manifold sphere --experiment forward

# Ablation study on wrinkle manifold
python run_experiment.py --manifold wrinkle --experiment ablation

# Scalability test on torus
python run_experiment.py --experiment scalability

# Curvature sensitivity analysis
python run_experiment.py --experiment sensitivity
```
