import matplotlib.pyplot as plt
import numpy as np

from lib.funcs import feedback_linearisation_controller, nearest_node

q_init = np.array([[0], [0], [0]])

q_min = q_init

q_dest = np.array([[20], [20], [np.pi / 3],])

# u, q = feedback_linearisation_controller(q, q_dest)


path_exists = False
dest_checked = False
q_nodes = [q_init]
closest = []
Q = []
i = 0

while not path_exists:
    # if not dest_checked:
    if False:
        q_sample = q_dest
        dest_checked = True
    else:
        q_sample = q_dest + np.array(
            [
                [np.random.normal(0, 10)],
                [np.random.normal(0, 10)],
                [np.random.uniform(0, 2 * np.pi)],
            ]
        )

        q_sample[2, :] = q_sample[2, :] % (2 * np.pi)

    q_nearest = nearest_node(q_nodes, q_sample)

    # Check if the change is angle is feasible
    if np.abs(q_nearest[2, 0] - q_sample[2, 0]) > np.pi / 2:
        continue
    # Check if there are obstacles in the way
    elif i < 5:
        i += 1
        continue

    u_ff, q_path = feedback_linearisation_controller(q_nearest, q_sample)

    # if a path exists
    if u_ff:

        q_reach = np.array(q_path[-1]).reshape(-1, 1)
        # Check the distance to the goal
        dq = np.linalg.norm(q_reach - q_dest)
        print("Distance to goal: {}".format(round(dq, 2)))

        if dq < 2:
            path_exists = True

        # Add the node to the set of eligible nodes
        q_nodes.append(q_reach)
        Q.append(q_path)

        q_min = q_nearest

    print("Iteration: {}".format(i))

    i += 1

    # if a control exists, the path exists
    # if u:
plt.plot(np.block(q_nodes)[0, :], np.block(q_nodes)[1, :], "bo")
for q_path in Q:
    q_path = np.array(q_path)
    plt.plot(q_path[:, 0], q_path[:, 1], "k--", alpha=0.1)

plt.plot(q_dest[0, 0], q_dest[1, 0], "rx")
plt.show()
