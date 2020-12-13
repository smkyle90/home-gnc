#!/usr/bin/env python3
# https://developer.gnome.org/gnome-devel-demos/stable/widget_drawing.py.html.en
import argparse
import json
import time

import graph_tool.all as gt
import numpy as np
import yaml
import zmq


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


def main(config):

    push_addr = config["TASK"]["ADDR"]
    push_port = int(config["TASK"]["PORT"])

    context = zmq.Context()
    socket = context.socket(zmq.PUB)

    # Binds the socket to a predefined port on localhost
    socket.bind("tcp://{}:{}".format(push_addr, push_port))
    time.sleep(1)

    g = gt.Graph()

    # Add some graph properties
    g.vp["name"] = g.new_vertex_property("string")
    g.vp["pos"] = g.new_vertex_property("vector<float>")
    g.ep["dist"] = g.new_edge_property("double")

    g = build_graph(g, config["NODES"])

    while True:
        task_list = []
        while True:
            try:
                src = input("Input the start node: ")

                if str(src) in g.vp["name"]:
                    break
            except Exception as e:
                print(e)

        while True:
            try:
                num = input("Add a location to the task list, or press r to run: ")
                if (num == "r") or (num == "R"):
                    break
                task_list.append(int(num))
            except Exception as e:
                print(e)

        task_list = list(set(task_list))
        _task_order, node_order = get_route(g, src, task_list)

        print("Starting at node {}".format(src))
        for n in node_order:
            print("Visit Node: {} at ({}, {})".format(g.vp["name"][n], *g.vp["pos"][n]))

        coord_order = [g.vp["pos"][v].get_array().tolist() for v in node_order]
        print("Logger: Publishing task to controller.")
        json_msg = json.dumps(coord_order)
        socket.send_json(json_msg)
        print("Logger: Task published")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="State Estimate.")
    parser.add_argument(
        "path",
        metavar="path/to/config/file",
        type=str,
        help="Path to configuration file.",
    )

    args = parser.parse_args()
    with open(args.path, "r") as f:
        config = yaml.safe_load(f)

    main(config["GUIDANCE"])
