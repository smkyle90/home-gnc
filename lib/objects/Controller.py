#!/usr/bin/env python3
import json

import numpy as np
import zmq
from matplotlib.patches import Circle
from numpy.linalg import eig, norm

USER = "#90C550"
NODE = "#E53D00"
RTR = "#031111"


class Controller:
    def __init__(
        self,
        ax,
        ax_vx,
        ax_vy,
        state_socket,
        task_socket,
        routers,
        nodes,
        poles=[-0.25, -0.5],
        task_delta=2,
    ):
        self.ax = ax
        self.ax_vx = ax_vx
        self.ax_vy = ax_vy

        self.state_socket = state_socket
        self.task_socket = task_socket
        self.poles = poles

        self.task = None
        self.original_task = None
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

    def get_task(self, origin):

        if origin:
            task = [[0, 0]]
        else:
            try:
                json_data = self.task_socket.recv_json(zmq.NOBLOCK)
                task = json.loads(json_data)
            except zmq.error.Again:
                task = None

        if task is not None:
            self.task = task
            self.original_task = task

    def check_task_status(self, curr_loc):
        if not self.task:
            return

        task_loc = np.array(self.task[0]).reshape(-1, 1)
        if norm(task_loc - curr_loc) < self.task_delta:
            self.task = self.task[1:]

        if not self.task:
            self.task = None

    def render(self, q, qd, u):
        self.ax.clear()
        self.ax_vx.clear()
        self.ax_vy.clear()

        if len(q):
            e_val, __ = eig(q["cov"])
            r = np.sqrt(np.sum(e_val))
            self.ax.plot(
                q["loc"][0, 0], q["loc"][1, 0], marker="o", color=USER, markersize=6
            )

            c = Circle(
                (q["loc"][0, 0], q["loc"][1, 0]),
                3 * r,
                fill=True,
                color=USER,
                alpha=0.2,
            )
            self.ax.add_patch(c)

            self.ax.text(10, -4, "{} Routers Used".format(q["n_meas"]), color=RTR)

        if len(qd):
            self.ax.plot(qd[0, 0], qd[1, 0], color=USER, marker="o", markersize=10)
            self.ax.plot(qd[0, 0], qd[1, 0], color="w", marker="o", markersize=8)

        if self.original_task is not None:
            for i, e in enumerate(self.original_task):
                if i:
                    self.ax.plot(
                        [e[0], self.original_task[i - 1][0]],
                        [e[1], self.original_task[i - 1][1]],
                        color=NODE,
                        linewidth=1,
                        alpha=0.2,
                    )

        for rtr_name, rtr in self.routers.items():
            self.ax.plot(*rtr, marker="+", color=RTR)
            self.ax.text(rtr[0] + 0.2, rtr[1], rtr_name, color=RTR)

        for node_id, node in self.nodes.items():
            self.ax.plot(*node, marker="o", color=NODE)
            self.ax.text(node[0] - 0.6, node[1], node_id, color=NODE)

        self.ax.set_xlim([-10, 25])
        self.ax.set_ylim([-10, 25])
        self.ax.grid()

        try:
            u = np.where(u < -5, -5, u)
            u = np.where(u > 5, 5, u)

            self.ax_vx.plot(u[0, 0], 0, "ko", markersize=15)
            self.ax_vy.plot(0, u[1, 0], "ko", markersize=15)
        except Exception:
            self.ax_vx.plot(0, 0, "ko", markersize=15)
            self.ax_vy.plot(0, 0, "ko", markersize=15)

        self.ax_vx.set_xlim([-6, 6])
        self.ax_vx.set_ylim([-1, 1])
        self.ax_vx.set_xticks([-5, 5])
        self.ax_vx.set_xticklabels(["Left", "Right"])
        self.ax_vx.set_yticks([])
        self.ax_vx.xaxis.set_ticks_position("top")

        self.ax_vy.set_xlim([-1, 1])
        self.ax_vy.set_ylim([-6, 6])
        self.ax_vy.set_xticks([])
        self.ax_vy.set_yticks([-5, 5])
        self.ax_vy.set_yticklabels(["Reverse", "Forward"])
        self.ax_vy.yaxis.set_ticks_position("right")