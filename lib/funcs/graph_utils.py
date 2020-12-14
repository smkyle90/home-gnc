#!/usr/bin/env python3
import graph_tool.all as gt
import numpy as np


def build_graph(g, nodes_info):
    for node_data in nodes_info.values():
        new_node = g.vertex(node_data["NODE_ID"], add_missing=True)
        g.vp["name"][new_node] = node_data["NODE_ID"]
        g.vp["pos"][new_node] = (node_data["COORD"][0], node_data["COORD"][1])

    for _node, node_data in nodes_info.items():
        src = node_data["NODE_ID"]
        for tgt in node_data["CONNECTS_TO"]:

            e = g.add_edge(src, tgt)

            q_src = np.array(g.vp["pos"][src])
            q_tgt = np.array(g.vp["pos"][tgt])
            q_dst = q_src - q_tgt
            q_dst = np.linalg.norm(q_dst)

            g.ep["dist"][e] = q_dst

    return g


def reconstruct_path(src, tgt, preds):
    """Reconstruct a path back form the target
    to the source using the predecessor array that
    is returned from the dijkstra or astar searches.

    Args:

    Returns:

    """

    vertices = []
    while int(tgt) != int(src):
        vertices.append(tgt)
        tgt = preds.a[tgt]

    vertices.reverse()

    return vertices


def get_route(g, src, task_list):
    src_node = g.vertex(src)
    dists = g.ep["dist"]

    task_order = [src]
    node_order = [src]

    if src in task_list:
        task_list.remove(src)

    while task_list:
        dist_map, pred_map = gt.astar_search(g, source=src_node, weight=dists)
        min_dist = np.inf
        next_node = src_node
        for t in task_list:
            if dist_map[t] < min_dist:
                next_node = t

        node_order.extend(reconstruct_path(src_node, next_node, pred_map))

        src_node = g.vertex(next_node)
        task_list.remove(next_node)
        task_order.append(next_node)

    return task_order, node_order
