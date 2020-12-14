#!/usr/bin/env python3
import argparse
import json
import time

import graph_tool.all as gt
import numpy as np
import yaml
import zmq

from lib.funcs import scan_networks
from lib.objects import Person, Router


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
    push_addr = config["STATE"]["ADDR"]
    push_port = int(config["STATE"]["PORT"])

    context = zmq.Context()
    socket = context.socket(zmq.PUB)

    # Binds the socket to a predefined port on localhost
    socket.bind("tcp://{}:{}".format(push_addr, push_port))
    time.sleep(1)

    interface = config["INTERFACE"]
    sudo = config["SUDO"]
    router_loc = {rtr[0]: Router(*rtr[:-1]) for rtr in config["ROUTER_LOC"]}
    process_covar = config["PROC_COVAR"] ** 2 * np.eye(2)

    est_state = np.mean(
        np.block([rtr.loc() for rtr in router_loc.values()]), axis=1
    ).reshape(-1, 1)
    est_covar = np.diag([10.0 ** 2, 10.0 ** 2])

    scott = Person(est_state[0, 0], est_state[1, 0], est_covar)

    n_wait = 0
    while True:
        scott.predict(process_covar)
        print("Logger: Predict Step Complete.")

        n_meas = 0
        if n_wait % 5 == 0:
            print("Logger: Scanning for network data.")
            net_data = scan_networks(interface, sudo)

            print("Logger: Network Data received.")

            n_meas = scott.update(net_data, router_loc)
            print("Logger: Update Step Complete using {} measurements.".format(n_meas))

        if n_meas < 3:
            print("Logger: System is unobservable. Number of measurements is too low.")

        # Publish the data
        print("Logger: Publishing data.")
        json_msg = json.dumps(
            {"loc": scott.loc().tolist(), "cov": scott.cov.tolist(), "n_meas": n_meas}
        )
        socket.send_json(json_msg)
        print("Logger: Data published")
        n_wait += 1
        time.sleep(1)


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

    main(config["NAVIGATION"])
