# Drift–diffusion on manifold-embedded metric graphs

Python code and recorded numerical results accompanying the current second-revision manuscript, `manifold_drift_diffusion_algorithm.tex` in the source workspace. 中文整理说明见 [docs/PACKAGING_NOTES.zh-CN.md](docs/PACKAGING_NOTES.zh-CN.md).

## Scope

The paper reports controlled FVM/calibration experiments and independent tangent-vector transport checks. The scalar factor `1 + K L^2 / 6` is a **prescribed bias/calibration law**, not Levi–Civita transport of density or scalar flux. The recorded results do not demonstrate a trained, end-to-end DeepONet. Neural-network modules are retained as architecture/prototype code and need additional dependencies.

## Layout

```text
.
├── run_epsilon_ablation.py
├── run_largescale_scalability.py
├── run_inverse_poc.py
├── run_transport_mesh_validation.py
├── curvature_aware_deeponet/
│   ├── src/                         # Geometry, transport and optional model modules
│   ├── experiments/                 # Current FVM/calibration/transport scripts
│   ├── results/                     # Current recorded FVM and comparison results
│   └── run_experiment.py            # Optional architecture prototype
├── physics-informed-operator-networks-for-pdes-on-metric-graphs-main/
│   ├── src/                         # Required reference graph/FVM code; model support
│   ├── LICENSE                      # Original MIT notice, retained verbatim
│   └── README.md                    # Original upstream README, retained verbatim
├── results/                         # Current supplementary experiment results
├── archive/legacy/                  # Earlier scripts, results and console logs
├── docs/                            # Experiment mapping, provenance and verification
├── manifest.csv                     # Original/copied paths, sizes and SHA-256
├── requirements.txt                 # Dependencies for the reported experiments
├── requirements-optional.txt        # Additional dependencies for JAX prototypes
└── test_basic.py                    # Original optional module checks (requires JAX)
```

Keep both project directories at these locations: the scripts resolve the reference solver using their relative paths.

## Environment

Packaging verification used Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1, NetworkX 3.3 and Matplotlib 3.9.2. These are the environment verified during packaging, not a recovered lockfile from the original runs. JAX and Optax were not installed in this verification environment. See [verification details](docs/VALIDATION.md).

From this folder:

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Reproduce the reported experiments

Run every command from this folder. Each script writes to the result paths listed below and **overwrites the corresponding recorded JSON**. Use a separate copy of the repository to retain the published result snapshot.

| Paper experiment | Command | Recorded data / JSON section |
| --- | --- | --- |
| 1: controlled forward test | `python curvature_aware_deeponet/experiments/run_proof_of_concept.py` | `curvature_aware_deeponet/results/poc_results.json`: `runs`, `summary` |
| 2, 3, 4: calibration ablation, curvature sweep, topology timing | `python curvature_aware_deeponet/experiments/run_remaining_v2.py` | `curvature_aware_deeponet/results/remaining_v2.json`: `experiment2`, `experiment3`, `experiment4` |
| 4b: large-scale controlled coupling | `python run_largescale_scalability.py` | `results/largescale_scalability.json` |
| 4c: threshold classification sweep | `python run_epsilon_ablation.py` | `results/epsilon_ablation.json` |
| 5: multiple topologies | `python curvature_aware_deeponet/experiments/run_exp5_exp6.py` | `curvature_aware_deeponet/results/exp5_exp6.json`: **`experiment5` only** |
| 6: controlled algebraic baseline comparison | `python curvature_aware_deeponet/experiments/run_comparison.py` | `curvature_aware_deeponet/results/comparison.json`: `partA` |
| 7: sphere transport comparison | Same `run_comparison.py` command | Same `comparison.json`: `partB` |
| 7b, 7c: surface transport checks and kernel scaling | `python run_transport_mesh_validation.py` | `results/transport_mesh_validation.json`: `surface_validation`, `scalability`, `fit` |
| 8: controlled velocity recovery | `python run_inverse_poc.py` | `results/inverse_poc.json` |

Machine-readable mapping: [docs/experiment_index.csv](docs/experiment_index.csv). Detailed caveats and data fields: [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

The numerical inputs are generated in the scripts from their fixed seeds, graph definitions and surface parameters. The JSON files store the recorded outputs; the folder does not contain a separate complete raw FVM/training dataset. Timing and floating-point residuals can differ between machines. No original reference-project weights or Fourier-feature `.npy` files are required for the eight reported experiment commands.

## Optional prototype modules

`curvature_aware_deeponet/run_experiment.py`, the JAX model/attention/coupling modules and `test_basic.py` are supplied for inspection and further development; they are not the reproduction entry points for the paper's tables.

```bash
python -m pip install -r requirements-optional.txt
```

These additional dependencies and the neural prototypes were not verified during packaging. Installation alone does not establish a complete training or inference pipeline.

## Provenance and third-party code

All copied Python and data files are byte-for-byte copies of the source workspace; their SHA-256 hashes are listed in `manifest.csv`. Earlier experiment files are separated under `archive/legacy/` and do not replace current paper results. The workspace did not contain two complete historical source snapshots; this package preserves the versions actually present.

The reference project's original README and MIT license are included. Its `GPs.py` is already a locally modified NumPy sampler, and `graph.py` already supports a NumPy fallback. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No new license has been assigned to the author's own code or data.

This folder is prepared for upload. Upload its contents as the repository root to keep these links and commands valid.
