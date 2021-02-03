import numpy as np


def k_nearest_nodes(q_nodes, q_sample, k=1):
    """Returns the k nearest nodes to q_sample
    from a list of q_nodes.
    """
    q_nodes = np.block(q_nodes)
    node_delta = q_nodes - q_sample

    max_vals = min(node_delta.shape[1], k)

    min_idx = np.argpartition(np.linalg.norm(node_delta, axis=0), max_vals - 1)

    return q_nodes[:, min_idx[:k]].reshape(3, -1)
