# Earlier experiments and outputs

These files preserve earlier material found in the workspace. They do not supply the current second-revision paper tables.

- `run_all_experiments.py` with `results/all_experiments.json`: earlier synthetic experiment suite.
- `run_experiments_v2.py` with `results/experiments_v2.json`: earlier revised synthetic suite; distinct from the restored FVM `experiments/run_remaining_v2.py`.
- `curvature_aware_deeponet/experiments/run_remaining_experiments.py` with `curvature_aware_deeponet/results/remaining_experiments.json`: earlier remaining-experiments implementation.
- `curvature_aware_deeponet/results/poc_results.pkl`: old prototype output from April 2026; not a binary copy of the current September JSON.
- `results/poc_output.txt`, `results/comparison_output.txt`: earlier console logs, retained for provenance. They may contain local paths and values different from current results.

The directory structure beneath this archive records original relative paths. These scripts were moved here only for organization, with their bytes unchanged. Their imports still expect the original layout. To run one, restore it to the corresponding path in a **separate copy** of the repository root, keeping the shared project/source directories. Additional JAX/Optax dependencies can be required, and an exact historical environment/source snapshot is unavailable.

Some earlier descriptions and conclusions were superseded in the current manuscript. Use the root README and `docs/EXPERIMENTS.md` to interpret current evidence.
