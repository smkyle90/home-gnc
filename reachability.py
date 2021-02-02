import random

import matplotlib.pyplot as plt
import numpy as np

from lib.funcs import (
    check_line_of_sight,
    feedback_linearisation_controller,
    nearest_node,
)
from lib.objects import Arena, Environment, Obstacle


def get_node_name(q):
    return "({}, {}, {})".format(*q.reshape(-1,))


def manhattan_distance(q1, q2):
    return np.abs(q1 - q2).sum()


# Arena configuration
x_min, x_max = 0, 30
y_min, y_max = 0, 30
dx, dy = 1, 1

# Obstacle configuration
n_stat = 0
n_dyn = 1
o_min = 1
o_max = 3
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
        10,  # random.randint(x_min, x_max),
        10,  # random.randint(x_min, x_max),
        random.randint(1, o_max),
    )
    for i in range(n_dyn)
]

arena = Arena(x_min, x_max, dx, y_min, y_max, dy)

env = Environment(arena, static_obs, dynamic_obs)

q_init = np.array([[1.0], [1.0], [np.pi / 2]])

# q_nodes = [q_init]

i = 0

q_free = env.reachable_coordinates()

q_dest = np.array([[1.0], [20.0], [3 * np.pi / 2]])
# q_dest = np.append(random.choice(q_free), np.random.uniform(0, 2 * np.pi)).reshape(
#     -1, 1
# )
n_samples = 4

fig = plt.figure()
ax = fig.add_subplot(111)
ax = env.plot_environment(ax, [])


# def rrt_plan(env, src, tgt):
#     """RRT plan with manhattan heuristic.
#     """
#     q_init = src
#     q_dest = tgt

path_exists = False
dest_checked = False

q_nodes = {get_node_name(q_init): {"pos": q_init, "cost": 0, "prev": None}}
node_coords = [q_init]
Q = []

# Get the reachable coordinates in the environment
q_free = env.reachable_coordinates()

while not path_exists:
    if False:
        # if not dest_checked:
        if env.check_line_of_sight(*q_init[:2, 0], *q_dest[:2, 0]):
            q_source = q_init
            q_target = q_dest

        dest_checked = True
    else:
        min_cost = np.inf
        samples = 0
        while samples < n_samples:
            q_sample = np.append(
                random.choice(q_free), np.random.uniform(0, 2 * np.pi)
            ).reshape(-1, 1)

            # q_sample = np.array([
            #     [np.random.uniform(x_min, x_max)],
            #     [np.random.uniform(y_min, y_max)],
            #     [np.random.uniform(0, 2 * np.pi)],
            #     ])

            q_nearest = nearest_node(node_coords, q_sample)

            # Check if the change is angle is feasible
            if np.abs(q_nearest[2, 0] - q_sample[2, 0]) > np.pi / 2:
                continue
            # Check if there are obstacles in the way
            elif not env.check_line_of_sight(*q_nearest[:2, 0], *q_sample[:2, 0]):
                continue

            proj_cost = q_nodes[get_node_name(q_nearest)]["cost"] + manhattan_distance(
                q_sample, q_dest
            )

            if proj_cost < min_cost:
                min_cost = proj_cost
                q_target = q_sample
                q_source = q_nearest

            samples += 1

    # Distance from source to target
    dn = np.linalg.norm(q_source[:2] - q_target[:2])
    u_ff, q_path = feedback_linearisation_controller(q_source, q_target, T_max=2 * dn)

    # if a path exists
    if u_ff:
        q_reach = np.array(q_path[-1]).reshape(-1, 1)

        # Check the distance to the goal and angle
        dq = np.linalg.norm(q_reach[:2] - q_dest[:2])
        dt = np.abs(q_reach[2, 0] - q_dest[2, 0])

        print("Distance to goal: {}".format(round(dq, 2)))
        print("Angle to goal: {}".format(round(dt, 2)))

        if (dq < 2) and (dt < np.pi / 4):
            path_exists = True

        # Add the node to the set of eligible nodes
        prev_cost = q_nodes[get_node_name(q_source)]["cost"]
        move_cost = manhattan_distance(
            q_source, q_reach
        )  # + np.sqrt(np.trace(np.array(u_ff).T.dot(np.array(u_ff))))

        q_nodes[get_node_name(q_reach)] = {
            "pos": q_reach,
            "cost": prev_cost + move_cost,
            "prev": get_node_name(q_source),
        }
        node_coords.append(q_reach)
        Q.append(q_path)

    # return Q

    # print("Iteration: {}".format(i))
prev_node = get_node_name(q_reach)
while prev_node is not None:
    print(prev_node)
    prev_node = q_nodes[prev_node]["prev"]

node_coords = [v["pos"] for v in q_nodes.values()]
ax.plot(np.block(node_coords)[0, :], np.block(node_coords)[1, :], "bo")
for q_path in Q:
    q_path = np.array(q_path)
    ax.plot(q_path[:, 0], q_path[:, 1], "k--", alpha=0.1)

ax.plot(q_init[0, 0], q_init[1, 0], "ro")
ax.plot(q_dest[0, 0], q_dest[1, 0], "rx")
plt.show()
