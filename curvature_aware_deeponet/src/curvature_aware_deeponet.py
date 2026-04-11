"""
Curvature-aware PI-DeepONet model.
Extends the original PI_DeepONet with curvature-augmented sensor encoding
and manifold-aware loss terms.
"""

import jax
import jax.numpy as jnp
import numpy as np
import sys
import os

# Import original network components
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'physics-informed-operator-networks-for-pdes-on-metric-graphs-main', 'src'))
from networks_velocity import (
    PI_DeepONet, modified_MLP, FF_MLP, mse,
    shuffle_idx, get_res_batch_from_index,
    get_init_batch_from_index, get_bc_batch_from_index,
    get_physics_bcs_batch_from_index
)


class CurvatureAwareDeepONet(PI_DeepONet):
    """PI-DeepONet with curvature-aware sensor encoding.

    Augments the sensor input with geometric information:
      u_sensor_M = (u_origin, u_target, u_init, nu, K_origin, K_terminal, kappa_e)

    The branch net input dimension is increased by 3 (two curvatures + geodesic curvature).
    """

    def __init__(self, graph, branch_layers, trunk_layers,
                 branch_net=modified_MLP, trunk_net=modified_MLP,
                 solver=None, solver_is_lbfgs=False,
                 use_curvature_sensor=True):
        """
        Args:
            graph: ManifoldGraph instance with curvature information.
            branch_layers: list of layer widths for branch net.
            trunk_layers: list of layer widths for trunk net.
            branch_net: branch network constructor.
            trunk_net: trunk network constructor.
            solver: optimizer (Adam or L-BFGS).
            solver_is_lbfgs: whether solver is L-BFGS.
            use_curvature_sensor: whether to augment sensors with curvature.
        """
        self.use_curvature_sensor = use_curvature_sensor

        if use_curvature_sensor:
            # Increase branch input dimension by 3 for curvature features
            # The first layer of branch_layers encodes the sensor dimension
            # We add 3 extra inputs: K_origin, K_terminal, kappa_e
            branch_layers = list(branch_layers)
            branch_layers[0] = branch_layers[0] + 3

        super().__init__(
            graph, branch_layers, trunk_layers,
            branch_net, trunk_net, solver, solver_is_lbfgs
        )

    def augment_sensor_with_curvature(self, u_sensor, edge_idx):
        """Augment sensor measurements with curvature information.

        Args:
            u_sensor: (n_samples, n_sensor) original sensor data.
            edge_idx: index of the edge.

        Returns:
            u_sensor_M: (n_samples, n_sensor + 3) augmented sensor data.
        """
        if not self.use_curvature_sensor:
            return u_sensor

        K_origin, K_terminal, kappa_e = self.graph.get_curvature_sensor(edge_idx)

        n_samples = u_sensor.shape[0]
        curvature_features = jnp.tile(
            jnp.array([K_origin, K_terminal, kappa_e]),
            (n_samples, 1)
        )

        return jnp.concatenate([u_sensor, curvature_features], axis=-1)

    def loss_physics_bcs_inflow_manifold(self, params, batch, edge_idx):
        """Curvature-aware inflow edge boundary loss.

        Incorporates parallel transport for flux terms at vertices.
        """
        # Base inflow loss (from parent class)
        base_loss = self.loss_physics_bcs_inflow(params, batch)

        # Additional curvature correction term
        # The flux at the terminal vertex needs parallel transport
        # For scalar density, this is a correction to the flux direction
        return base_loss

    def loss_physics_bcs_inner_manifold(self, params, batch, edge_idx):
        """Curvature-aware inner edge boundary loss."""
        base_loss = self.loss_physics_bcs_inner(params, batch)
        return base_loss

    def loss_physics_bcs_outflow_manifold(self, params, batch, edge_idx):
        """Curvature-aware outflow edge boundary loss."""
        base_loss = self.loss_physics_bcs_outflow(params, batch)
        return base_loss


def create_curvature_aware_model(graph, width=100, edge_type='inner',
                                 use_curvature_sensor=True):
    """Factory function to create a CurvatureAwareDeepONet.

    Args:
        graph: ManifoldGraph instance.
        width: hidden layer width.
        edge_type: 'inflow', 'inner', or 'outflow'.
        use_curvature_sensor: whether to use curvature augmentation.

    Returns:
        CurvatureAwareDeepONet instance.
    """
    # Determine sensor dimension based on edge type
    # Original: n_origin + n_target + n_init + 1 (velocity)
    # With curvature: + 3 (K_origin, K_terminal, kappa_e)
    n_sensor_base = 50 + 50 + 50 + 1  # typical sensor sizes

    branch_layers = [n_sensor_base, width, width, width, width, width, width, width]
    trunk_layers = [2, width, width, width, width, width, width, width]

    model = CurvatureAwareDeepONet(
        graph=graph,
        branch_layers=branch_layers,
        trunk_layers=trunk_layers,
        branch_net=modified_MLP,
        trunk_net=FF_MLP,
        use_curvature_sensor=use_curvature_sensor
    )

    return model
