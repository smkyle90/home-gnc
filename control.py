#!/usr/bin/env python3
import argparse
import json

import numpy as np
import yaml
import zmq
from scipy import signal


class Controller:
    def __init__(self, state_socket, task_socket, poles=[-0.25, -0.5], task_delta=2):
        self.state_socket = state_socket
        self.task_socket = task_socket
        self.poles = poles

        self.task = []
        self.original_task = []
        self.task_delta = task_delta
        # self.visited=[False for i in self.task]

    def get_curr_state(self):
        json_data = self.state_socket.recv_json()
        return np.array(json.loads(json_data))

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


def main(config):

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

    controller = Controller(state_socket, task_socket)

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

        x = controller.get_curr_state()
        controller.check_task_status(x)
        xd = controller.get_ref_state()

        print("State Estimate ", x)
        print("Next node ", xd)

        u = -K.dot(x - xd)

        print("Control ", u)


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
