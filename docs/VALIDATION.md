# Packaging verification

Verification date: 2026-10-11.

## File integrity and organization

- All 44 copied source/data/notice files match their original SHA-256 hashes and byte sizes; originals remain unchanged.
- All 26 Python files parse successfully, including archived and optional code.
- All 11 supplied experiment JSON files parse successfully.
- All 12 experiment-index rows point to existing scripts, data files and JSON sections.
- Both project directories retain their required relative positions.
- The package excludes Python bytecode, manuscript build artifacts and upstream pretrained assets.

## Full current experiment execution

All eight current experiment commands completed successfully in an independent copy of this package. Rerunning did not change the result snapshots supplied for upload. The verification environment was Python 3.12.7, NumPy 1.26.4, SciPy 1.13.1, NetworkX 3.3 and Matplotlib 3.9.2; JAX and Optax were absent. BLAS/OpenMP thread limits were set to 1 and Matplotlib used the Agg backend.

Durations below are total script execution durations from this verification, not the benchmark times reported in the manuscript.

| Script | Execution | Duration (s) | Comparison with packaged snapshot, excluding runtime/allocation fields |
| --- | --- | --- | --- |
| `curvature_aware_deeponet/experiments/run_proof_of_concept.py` | Pass | 3.74 | Max non-percentage difference 5.392e-07; max percentage-point difference 3.980e-06 |
| `curvature_aware_deeponet/experiments/run_remaining_v2.py` | Pass | 33.30 | Max non-percentage difference 3.517e-07; max percentage-point difference 1.907e-05 |
| `curvature_aware_deeponet/experiments/run_comparison.py` | Pass | 42.45 | Max non-percentage difference 5.533e-08; max percentage-point difference 0.000e+00 |
| `curvature_aware_deeponet/experiments/run_exp5_exp6.py` | Pass | 34.23 | Max non-percentage difference 4.437e-08; max percentage-point difference 7.745e-06 |
| `run_largescale_scalability.py` | Pass | 32.88 | Graph counts identical; timing/allocation fields excluded |
| `run_epsilon_ablation.py` | Pass | 1.99 | Numerical fields identical |
| `run_inverse_poc.py` | Pass | 7.95 | Numerical fields identical |
| `run_transport_mesh_validation.py` | Pass | 28.61 | Numerical fields identical |

The largest absolute difference among compared non-percentage numerical fields was 5.392e-07; the largest improvement-percentage difference was 1.907e-05 percentage points. This establishes close numerical agreement, not bit-for-bit equality across environments. The threshold, inverse and mesh-transport numerical fields were identical in this verification. Timing and allocation fields, including preprocessing durations, per-edge times and timing-fit coefficients, were excluded from these equality comparisons because they depend on the environment. Rerun JSON key sets and array lengths were also checked.

## Limits

Optional JAX prototype code, `test_basic.py`, and archived experiment scripts were syntax-checked but not executed. No training pipeline, missing raw dataset or historical environment was reconstructed. No GitHub upload, Git commit, or repository publication was performed.

Full local execution logs and rerun outputs remain outside the upload package in the source workspace's `tmp/github_upload_verification_20261011/` directory. The published snapshots remain the original copies identified by `manifest.csv`.
