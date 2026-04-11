"""
Curvature-aware graph attention mechanism for enhanced vertex coupling.
Implements tensor field operations and multi-head attention on manifolds.
"""

import jax
import jax.numpy as jnp
import numpy as np
from .parallel_transport import (
    parallel_transport_S2, parallel_transport_H2, parallel_transport_R2,
    classify_curvature, CURVATURE_THRESHOLD
)


def tensor_field_11(w1, w2, h, metric):
    """Apply (1,1)-tensor w1 ⊗ w2* to tangent vector h.

    (w1 ⊗ w2*)(h) = w1 * <w2, h>_{g|_v}

    Args:
        w1: (d,) parameter vector.
        w2: (d,) parameter vector.
        h: (d,) tangent vector.
        metric: (d, d) metric tensor at the vertex.

    Returns:
        (d,) transformed tangent vector.
    """
    inner = jnp.dot(w2, metric @ h)
    return w1 * inner


def tensor_field_02(a1, a2, h, h_prime, metric):
    """Apply (0,2)-tensor a1* ⊗ a2* to two tangent vectors.

    (a1* ⊗ a2*)(h, h') = <a1, h>_{g|_v} * <a2, h'>_{g|_v}

    Args:
        a1: (d,) parameter vector.
        a2: (d,) parameter vector.
        h: (d,) first tangent vector.
        h_prime: (d,) second tangent vector.
        metric: (d, d) metric tensor.

    Returns:
        scalar attention score.
    """
    return jnp.dot(a1, metric @ h) * jnp.dot(a2, metric @ h_prime)


def parallel_transport_jax(h_v, source_pos, target_pos, curvature_type):
    """JAX-compatible parallel transport dispatcher.

    Args:
        h_v: tangent vector at source.
        source_pos: (3,) source position.
        target_pos: (3,) target position.
        curvature_type: 'S2', 'R2', or 'H2'.

    Returns:
        Transported tangent vector at target.
    """
    if curvature_type == 'R2':
        return h_v
    elif curvature_type == 'S2':
        # Sphere transport
        u = target_pos / jnp.linalg.norm(target_pos)
        v = source_pos / jnp.linalg.norm(source_pos)
        cos_theta = jnp.clip(jnp.dot(u, v), -1.0, 1.0)
        theta = jnp.arccos(cos_theta)

        e1 = u
        cross = jnp.cross(u, v)
        cross_norm = jnp.linalg.norm(cross)
        e2 = cross / (cross_norm + 1e-10)
        e3 = jnp.cross(e1, e2)

        a = jnp.dot(h_v, e3)
        b = jnp.dot(h_v, e2)

        return a * jnp.cos(theta) * e3 - a * jnp.sin(theta) * e1 + b * e2
    else:
        # H2: use sphere approximation for simplicity in JAX
        return h_v  # placeholder


class CurvatureAwareAttention:
    """Multi-head curvature-aware attention for vertex coupling.

    Implements:
      m'_v = sum_{e in E_v} alpha_ve * (w1 ⊗ w2*)(Gamma_e^v[h_e])
      alpha_ve = softmax_e((a1* ⊗ a2*)(h_v, Gamma_e^v[h_e]))
    """

    def __init__(self, n_heads=3, feature_dim=3, subtree_depth=4):
        """
        Args:
            n_heads: number of attention heads.
            feature_dim: dimension of tangent vectors (3 for surfaces in R^3).
            subtree_depth: maximum depth for subtree partition.
        """
        self.n_heads = n_heads
        self.feature_dim = feature_dim
        self.subtree_depth = subtree_depth

    def init_params(self, key):
        """Initialize learnable parameters for all heads.

        Returns:
            params: dict with keys 'w1', 'w2', 'w1_tilde', 'w2_tilde',
                    'a1', 'a2' for each head.
        """
        params = {}
        for i in range(self.n_heads):
            key, *subkeys = jax.random.split(key, 7)
            params[f'w1_{i}'] = jax.random.normal(subkeys[0], (self.feature_dim,)) * 0.1
            params[f'w2_{i}'] = jax.random.normal(subkeys[1], (self.feature_dim,)) * 0.1
            params[f'w1t_{i}'] = jax.random.normal(subkeys[2], (self.feature_dim,)) * 0.1
            params[f'w2t_{i}'] = jax.random.normal(subkeys[3], (self.feature_dim,)) * 0.1
            params[f'a1_{i}'] = jax.random.normal(subkeys[4], (self.feature_dim,)) * 0.1
            params[f'a2_{i}'] = jax.random.normal(subkeys[5], (self.feature_dim,)) * 0.1
        return params

    def attention_score(self, params, h_u, h_v_transported, head_idx, metric):
        """Compute curvature-aware attention score for one head.

        alpha = (a1* ⊗ a2*)(h_u, Gamma[h_v])
        """
        a1 = params[f'a1_{head_idx}']
        a2 = params[f'a2_{head_idx}']
        return tensor_field_02(a1, a2, h_u, h_v_transported, metric)

    def message(self, params, h_v_transported, head_idx, metric):
        """Compute message from one neighbor for one head.

        m = (w1 ⊗ w2*)(Gamma[h_v])
        """
        w1 = params[f'w1_{head_idx}']
        w2 = params[f'w2_{head_idx}']
        return tensor_field_11(w1, w2, h_v_transported, metric)

    def forward(self, params, h_u, neighbor_features, transport_ops, metric_u):
        """Compute multi-head curvature-aware attention at vertex u.

        Args:
            params: attention parameters.
            h_u: (feature_dim,) feature at vertex u.
            neighbor_features: list of (h_v, source_pos, target_pos, curv_type) tuples.
            transport_ops: parallel transport info for each neighbor.
            metric_u: (d, d) metric tensor at u.

        Returns:
            h_u_new: (n_heads * feature_dim,) updated feature.
        """
        head_outputs = []

        for head_idx in range(self.n_heads):
            # Compute attention scores and messages
            scores = []
            messages = []

            for h_v, src_pos, tgt_pos, ctype in neighbor_features:
                # Parallel transport h_v to T_u M
                h_v_transported = parallel_transport_jax(h_v, src_pos, tgt_pos, ctype)

                # Attention score
                score = self.attention_score(params, h_u, h_v_transported,
                                             head_idx, metric_u)
                scores.append(score)

                # Message
                msg = self.message(params, h_v_transported, head_idx, metric_u)
                messages.append(msg)

            # Softmax over scores
            scores = jnp.array(scores)
            alpha = jax.nn.softmax(scores)

            # Weighted sum of messages
            aggregated = jnp.zeros(self.feature_dim)
            for k, msg in enumerate(messages):
                aggregated = aggregated + alpha[k] * msg

            # Self-update
            w1t = params[f'w1t_{head_idx}']
            w2t = params[f'w2t_{head_idx}']
            self_update = tensor_field_11(w1t, w2t, h_u, metric_u)

            head_outputs.append(jax.nn.tanh(self_update + aggregated))

        # Concatenate heads
        return jnp.concatenate(head_outputs)


def subtree_partition(edges, n_vertices, max_depth=4, seed=42):
    """Partition graph into subtrees for parameter sharing.

    Args:
        edges: list of (vi, vj) pairs.
        n_vertices: total number of vertices.
        max_depth: maximum BFS depth per subtree.
        seed: random seed.

    Returns:
        subtrees: list of (source_node, node_set) tuples.
        node_to_subtree: dict mapping node -> subtree index.
    """
    rng = np.random.RandomState(seed)
    visited = set()
    subtrees = []
    node_to_subtree = {}

    # Build adjacency list
    adj = {i: [] for i in range(n_vertices)}
    for vi, vj in edges:
        adj[vi].append(vj)
        adj[vj].append(vi)

    # Randomly select source nodes and BFS
    unvisited = list(range(n_vertices))
    rng.shuffle(unvisited)

    subtree_idx = 0
    for source in unvisited:
        if source in visited:
            continue

        # BFS from source with max depth
        queue = [(source, 0)]
        tree_nodes = set()
        while queue:
            node, depth = queue.pop(0)
            if node in visited or depth > max_depth:
                continue
            visited.add(node)
            tree_nodes.add(node)
            node_to_subtree[node] = subtree_idx

            if depth < max_depth:
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        queue.append((neighbor, depth + 1))

        subtrees.append((source, tree_nodes))
        subtree_idx += 1

    return subtrees, node_to_subtree
