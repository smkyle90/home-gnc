import copy
import random
import time

import matplotlib.pyplot as plt
import numpy as np

from lib.objects import Arena, Environment, Obstacle

# def main():
if True:
    # Arena configuration
    x_min, x_max = 0, 30
    y_min, y_max = 0, 30
    dx, dy = 1, 1

    # Obstacle configuration
    n_stat = 2
    n_dyn = 4
    o_min = 1
    o_max = 2
    obs_val = 100
    epsilon = 0

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

    # Define an arena
    arena = Arena(x_min, x_max, dx, y_min, y_max, dy)

    # Define an environment
    env = Environment(arena, static_obs, dynamic_obs)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()

    # Define a task

    while True:
        for dyn in dynamic_obs:
            dyn.x += random.gauss(0, 1)
            dyn.y += random.gauss(0, 1)

        t1 = time.perf_counter()
        env.dynamic_obs = dynamic_obs
        t2 = time.perf_counter()
        print("Update dynamic obstacles: {} ms".format(round(1000 * (t2 - t1), 2)))

        q_reach = env.reachable_coordinates()
        src_pos = random.choice(q_reach)
        tgt_pos = random.choice(q_reach)
        src = env.get_node_by_coord(*src_pos)
        tgt = env.get_node_by_coord(*tgt_pos)

        t1 = time.perf_counter()
        nodes_visited = env.get_route(src, tgt)
        t2 = time.perf_counter()
        print("Calculate route: {} ms".format(round(1000 * (t2 - t1), 2)))
        print("Line of sight: {}".format(env.check_line_of_sight(*src_pos, *tgt_pos)))

        min_path = env.get_coordinates(nodes_visited)

        if plot_data:
            ax = env.plot_environment(ax, nodes_visited)
            plt.show()
            plt.pause(1)


# if __name__ == "__main__":
#     main()
