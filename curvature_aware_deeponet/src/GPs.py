"""
Gaussian process sampling for generating random initial/boundary conditions.
Reuses the original implementation with minor extensions.
"""

import jax.numpy as jnp
from jax import config, random


def RBF(x1, x2, params):
    """Radial basis function kernel."""
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - \
            jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs ** 2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)


def get_sample_fns(length_scale=0.2, N=512):
    """Get GP sampling functions for initial/boundary conditions.

    Returns:
        (sample_u_in_fn, sample_u_out_fn, sample_u_init_fn)
    """
    config.update("jax_enable_x64", True)

    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(0, 1, N)
    K = RBF(X.reshape((N, 1)), X.reshape((N, 1)), gp_params)
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(N))
    config.update("jax_enable_x64", False)

    def sample_GP_values(key):
        return jnp.dot(L, random.normal(key, (N,)))

    def _make_sample_fn():
        def sample_fn(key, shift=0.0):
            gp_sample = sample_GP_values(key)
            gp_scale = 1.0 / (gp_sample.max() - gp_sample.min() + shift)
            gp_sample = gp_scale * (gp_sample - gp_sample.min() + shift)
            return lambda t: jnp.interp(t, X, gp_sample)
        return sample_fn

    return _make_sample_fn(), _make_sample_fn(), _make_sample_fn()
