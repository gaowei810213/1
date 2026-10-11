# Third-party code

`physics-informed-operator-networks-for-pdes-on-metric-graphs-main/src/` comes from the local reference-project checkout accompanying **Physics-Informed DeepONets for drift-diffusion on metric graphs: simulation and parameter identification**, by Jan Blechschmidt, Tom-Christian Riemer, Max Winkler, Martin Stoll and Jan-F. Pietschmann (2025).

The source checkout's README and full MIT license are retained verbatim next to `src/`. Its license names Copyright (c) 2025 Jan Blechschmidt. The original README contains the publication citation and acknowledges code based on Physics-informed-DeepONets.

Local modifications already present before packaging include the NumPy Gaussian-process sampler in `src/GPs.py` and the optional NumPy fallback for `jax.numpy` in `src/graph.py`. Packaging makes no further changes to these files. Their provenance is recorded in `manifest.csv`.

Only Python source, README and LICENSE from that reference checkout are included. Its notebooks, pretrained weights and Fourier-feature `.npy` files belong to the reference workflow and are not consumed by the current paper's eight experiment commands. In particular, the upstream README describes a full upstream repository; those assets are not included in this subset.

The reference MIT license applies to that reference code. This package does not assign a new license to the author's own code or experimental results.
