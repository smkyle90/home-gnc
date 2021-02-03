import random

import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np

from lib.funcs import (
    check_line_of_sight,
    feedback_linearisation_controller,
    k_nearest_nodes,
)
from lib.objects import Arena, Environment, Obstacle


def get_node_name(q):
    return "({}, {}, {})".format(*q.reshape(-1,))


def manhattan_distance(q1, q2):
    return np.abs(q1 - q2).sum()


# Arena configuration
x_min, x_max = 0, 20
y_min, y_max = 0, 20
dx, dy = 0.5, 0.5

# Obstacle configuration
n_stat = 0
n_dyn = 0
o_min = 1
o_max = 2
obs_val = 100
epsilon = 1

plot_data = True

# Generate obstacles
static_obs = [
    Obstacle(
        random.randint(x_min, x_max),
        random.randint(x_min, x_max),
        random.randint(o_min, o_max),
    )
    for i in range(n_stat)
]

dynamic_obs = [
    Obstacle(
        random.randint(x_min, x_max),
        random.randint(x_min, x_max),
        random.randint(1, o_max),
    )
    for i in range(n_dyn)
]

arena = Arena(x_min, x_max, dx, y_min, y_max, dy)

env = Environment(arena, static_obs, dynamic_obs)

q_init = np.array([[1.0], [1.0], [0]])

# q_nodes = [q_init]

i = 0

q_free = env.reachable_coordinates()

q_dest = np.array([[15.0], [15.0], [np.pi / 4]])
# q_dest = np.append(random.choice(q_free), np.random.uniform(0, 2 * np.pi)).reshape(
#     -1, 1
# )
n_samples = 1

# def rrt_plan(env, src, tgt):
#     """RRT plan with manhattan heuristic.
#     """
#     q_init = src
#     q_dest = tgt

path_exists = False
dest_checked = False

# q_nodes = {get_node_name(q_init): {"pos": q_init, "cost": 0, "prev": None, "path": []}}
Q = []


# g.vp["cost"] = g.new_vertex_property("float")


# Get the reachable coordinates in the environment
q_free = env.reachable_coordinates()
node_coords = [q_init]


class ConstrainedGraph:
    def __init__(self):
        g = gt.Graph()

        g.vp["name"] = g.new_vertex_property("string")
        g.vp["pos"] = g.new_vertex_property("vector<float>")
        g.vp["cost"] = g.new_vertex_property("double")
        g.ep["cost"] = g.new_edge_property("double")

        self.node_map = {}
        self.g = g
        self.node_coords = []

    def add_vertex(self, q, cost):
        node_name = get_node_name(q)
        new_node = self.g.add_vertex(1)

        self.node_map[node_name] = new_node
        self.node_coords.append(q.reshape(-1, 1))
        self.g.vp["name"][new_node] = node_name
        self.g.vp["pos"][new_node] = q.reshape(-1,).tolist()
        self.g.vp["cost"][new_node] = cost

    def add_edge(self, src_coord, tgt_coord, cost):
        src = self.get_node(src_coord)
        tgt = self.get_node(tgt_coord)

        e = self.g.add_edge(src, tgt)
        self.g.ep["cost"][e] = cost

    def get_node(self, q):
        return self.node_map[get_node_name(q)]


cg = ConstrainedGraph()

cg.add_vertex(q_init, 0)

while not path_exists:
    min_cost = np.inf
    samples = 0

    q_sample = np.append(
        random.choice(q_free), np.random.uniform(0, 2 * np.pi)
    ).reshape(-1, 1)

    q_nearest = k_nearest_nodes(cg.node_coords, q_sample, 3)

    min_cost = np.inf
    min_move_cost = np.inf
    min_coord = None
    parent_coord = None

    for nearest in q_nearest.T:
        # Distance from source to target
        dn = np.linalg.norm(nearest[:2].reshape(-1, 1) - q_sample[:2])
        u_ff, q_path = feedback_linearisation_controller(
            nearest.reshape(-1, 1), q_sample, T_max=2 * dn
        )

        # if a path exists
        if not u_ff:
            continue

        Q.append(q_path)
        q_path = np.array(q_path)

        if env.check_valid_path(q_path[:, 0], q_path[:, 1]):
            # This is the point we actually reach
            q_reach = np.array(q_path[-1]).reshape(-1, 1)

            # Add the node to the set of eligible nodes
            nearest_node = cg.get_node(nearest)
            prev_cost = cg.g.vp["cost"][nearest_node]
            move_cost = manhattan_distance(
                nearest.reshape(-1, 1), q_reach
            )  # + np.sqrt(np.trace(np.array(u_ff).T.dot(np.array(u_ff))))

            if prev_cost + move_cost < min_cost:
                min_cost = prev_cost + move_cost
                min_move_cost = move_cost
                min_coord = q_reach
                parent_coord = nearest

            # Check the distance to the goal and angle
            dq = np.linalg.norm(q_reach[:2] - q_dest[:2])
            dt = np.abs(q_reach[2, 0] - q_dest[2, 0])

            print("Distance to goal: {}".format(round(dq, 2)))
            print("Angle to goal: {}".format(round(dt, 2)))

            if (dq < 2) and (dt < np.pi / 8):
                path_exists = True
                q_final = nearest.reshape(-1, 1)

    if min_coord is not None:
        cg.add_vertex(min_coord, min_cost)
        cg.add_edge(parent_coord, min_coord, min_move_cost)

fig = plt.figure()
ax = fig.add_subplot(111)
ax = env.plot_environment(ax, [])

src = cg.get_node(q_init)
tgt = cg.get_node(q_final)

dist, pred = gt.astar_search(cg.g, src, weight=cg.g.ep["cost"])

prev_node = tgt
while prev_node != src:
    q0 = cg.g.vp["pos"][prev_node][:2]
    theta = cg.g.vp["pos"][prev_node][2]
    r0 = 2 * np.array([np.cos(theta), np.sin(theta)])
    ax.plot(*q0, "k.")
    ax.plot([q0[0], q0[0] + r0[0]], [q0[1], q0[1] + r0[1]], "k-")
    prev_node = cg.g.vertex(pred[prev_node])

ax.plot(np.block(cg.node_coords)[0, :], np.block(cg.node_coords)[1, :], "bo")
ax.plot(*cg.g.vp["pos"][src][:2], "ko")
ax.plot(*cg.g.vp["pos"][tgt][:2], "kx")
plt.show()

#     # return Q

#     # print("Iteration: {}".format(i))
# global_path = []
# prev_node = get_node_name(q_reach)
# while prev_node is not None:
#     print(prev_node)
#     global_path.extend(q_nodes[prev_node]["path"][::-1])
#     prev_node = q_nodes[prev_node]["prev"]

# global_path.reverse()
# global_path = np.array(global_path)

# node_coords = [v["pos"] for v in q_nodes.values()]
# for q_path in Q:
#     q_path = np.array(q_path)
#     ax.plot(q_path[:, 0], q_path[:, 1], "k--", alpha=0.1)

# ax.plot(global_path[:, 0], global_path[:, 1], 'g--', alpha=0.3)
# ax.plot(q_init[0, 0], q_init[1, 0], "ro")
# ax.plot(q_dest[0, 0], q_dest[1, 0], "rx")
# plt.show()
