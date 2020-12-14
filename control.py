#!/usr/bin/env python3
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import yaml
import zmq
from scipy import signal

from lib.objects import Controller


def main(config):
    N = 8
    fig = plt.figure()
    grid = plt.GridSpec(nrows=N, ncols=N, hspace=0.2, wspace=0.2)

    ax_main = fig.add_subplot(grid[1 : (N - 1), : (N - 1)])
    ax_vx = fig.add_subplot(grid[:1, : (N - 1)])
    ax_vy = fig.add_subplot(grid[1 : (N - 1), (N - 1) :])

    plt.ion()

    state_sub_addr = config["NAVIGATION"]["STATE"]["ADDR"]
    state_sub_port = int(config["NAVIGATION"]["STATE"]["PORT"])
    context = zmq.Context()
    state_socket = context.socket(zmq.SUB)
    state_socket.connect("tcp://{}:{}".format(state_sub_addr, state_sub_port))
    state_socket.subscribe("")
    print(
        "Logger: Connected to state data at tcp://{}:{}".format(
            state_sub_addr, state_sub_port
        )
    )

    task_sub_addr = config["GUIDANCE"]["TASK"]["ADDR"]
    task_sub_port = int(config["GUIDANCE"]["TASK"]["PORT"])
    context = zmq.Context()
    task_socket = context.socket(zmq.SUB)
    task_socket.connect("tcp://{}:{}".format(task_sub_addr, task_sub_port))
    task_socket.subscribe("")
    print(
        "Logger: Connected to task data at tcp://{}:{}".format(
            task_sub_addr, task_sub_port
        )
    )

    rtr_locs = config["NAVIGATION"]["ROUTER_LOC"]
    vtx_locs = config["GUIDANCE"]["NODES"]

    rtr_locs = {rtr[-1]: rtr[1:3] for rtr in rtr_locs}
    vtx_locs = {
        node_data["NODE_ID"]: node_data["COORD"] for node_data in vtx_locs.values()
    }

    controller = Controller(
        ax_main, ax_vx, ax_vy, state_socket, task_socket, rtr_locs, vtx_locs
    )

    A = np.array([[1, 1], [0, 1]])
    B = np.array([[1, 0], [0, 1]])
    poles = np.array([-1, -2])

    K = signal.place_poles(A, B, poles)
    K = K.gain_matrix
    print("Logger: Controller configured. Starting main loop.")

    while True:

        q = controller.get_curr_state()

        controller.get_task(config["CONTROL"]["ORIGIN"])

        if controller.task is not None:
            print("Logger: Checking Task Status.")
            controller.check_task_status(q["loc"])

            qd = controller.get_ref_state()
            print("Logger: Reference state received.")

            print("Logger: Calculating controls.")
            u = -K.dot(q["loc"] - qd)
        else:
            print("Logger: Awaiting Guidance input.")
            qd = []
            u = []

        controller.render(q, qd, u)

        plt.show()
        plt.pause(1.0)


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

    main(config)
