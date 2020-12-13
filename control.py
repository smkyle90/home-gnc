#!/usr/bin/env python3
import argparse
import json

import matplotlib.pyplot as plt
import numpy as np
import yaml
import zmq
from scipy import signal


class Controller:
    def __init__(
        self,
        ax,
        state_socket,
        task_socket,
        routers,
        nodes,
        poles=[-0.25, -0.5],
        task_delta=2,
    ):
        self.ax = ax
        self.state_socket = state_socket
        self.task_socket = task_socket
        self.poles = poles

        self.task = []
        self.original_task = []
        self.task_delta = task_delta

        self.routers = routers
        self.nodes = nodes
        # self.visited=[False for i in self.task]

    def get_curr_state(self):
        json_data = self.state_socket.recv_json()

        dict_data = json.loads(json_data)
        return {k: np.array(v) for k, v in dict_data.items()}

    def get_ref_state(self):
        if self.task:
            return np.array(self.task[0]).reshape(-1, 1)
        else:
            return []

    def get_task(self):
        json_data = self.task_socket.recv_json()
        self.task = json.loads(json_data)
        self.original_task = json.loads(json_data)

    def check_task_status(self, curr_loc):
        if not self.task:
            return

        task_loc = np.array(self.task[0]).reshape(-1, 1)
        if np.linalg.norm(task_loc - curr_loc) < self.task_delta:
            self.task = self.task[1:]

    def render(self, q, qd, u):
        self.ax.clear()
        self.ax.plot(q["loc"][0, 0], q["loc"][1, 0], "g+")
        self.ax.plot(qd[0, 0], qd[1, 0], "gx")

        for rtr_name, rtr in self.routers.items():
            self.ax.plot(*rtr, "b+")
            self.ax.text(*rtr, rtr_name)

        for node_id, node in self.nodes.items():
            if node in self.task:
                self.ax.plot(*node, "bo")
            elif node in self.original_task:
                self.ax.plot(*node, "go")
            else:
                self.ax.plot(*node, "ko")

            self.ax.text(*node, node_id)

        self.ax.set_xlim([-30, 30])
        self.ax.set_ylim([-30, 30])


def main(config):

    fig = plt.figure()
    ax = fig.add_subplot(111)
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

    rtr_locs = {rtr[0]: rtr[1:3] for rtr in rtr_locs}
    vtx_locs = {
        node_data["NODE_ID"]: node_data["COORD"] for node_data in vtx_locs.values()
    }

    controller = Controller(ax, state_socket, task_socket, rtr_locs, vtx_locs)

    print("Logger: Getting task from guidance.")
    controller.get_task()
    print("Logger: Task received.")

    A = np.array([[1, 1], [0, 1]])
    B = np.array([[1, 0], [0, 1]])
    poles = np.array([-1, -2])

    K = signal.place_poles(A, B, poles)
    K = K.gain_matrix
    print("Logger: Controller configured. Starting main loop.")

    while True:

        q = controller.get_curr_state()
        controller.check_task_status(q["loc"])
        qd = controller.get_ref_state()

        u = -K.dot(q["loc"] - qd)

        controller.render(q, qd, u)

        plt.show()
        plt.pause(0.5)


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
