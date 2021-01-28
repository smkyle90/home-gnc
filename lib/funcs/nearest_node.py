import numpy as np


def nearest_node(q_nodes, q_sample):

    q_nodes = np.block(q_nodes)
    node_delta = q_nodes - q_sample

    min_idx = np.argmin(np.linalg.norm(node_delta, axis=0))

    return q_nodes[:, min_idx].reshape(-1, 1)
