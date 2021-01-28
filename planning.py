import copy
import random
import time

import graph_tool.all as gt
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from lib.objects import Arena, Environment, Obstacle


def main():
    # Arena configuration
    x_min, x_max = 0, 30
    y_min, y_max = 0, 30
    dx, dy = 1, 1

    # Obstacle configuration
    n_stat = 2
    n_dyn = 2
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
            random.randint(x_min, x_max),
            random.randint(x_min, x_max),
            random.randint(1, o_max),
        )
        for i in range(n_dyn)
    ]

    arena = Arena(x_min, x_max, dx, y_min, y_max, dy)

    t1 = time.perf_counter()
    env = Environment(arena, static_obs, dynamic_obs)
    t2 = time.perf_counter()
    print("Instantiate environment: {} ms".format(round(1000 * (t2 - t1), 2)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.ion()

    while True:
        dynamic_obs = [
            Obstacle(
                random.randint(x_min, x_max),
                random.randint(x_min, x_max),
                random.randint(1, o_max),
            )
            for i in range(n_dyn)
        ]

        t1 = time.perf_counter()
        env.DynamicObstacles = dynamic_obs
        t2 = time.perf_counter()
        print("Update dynamic obstacles: {} ms".format(round(1000 * (t2 - t1), 2)))
        q_reach = env.reachable_coordinates()

        src_pos = random.choice(q_reach)
        tgt_pos = random.choice(q_reach)

        src_name = "({}, {})".format(*src_pos)
        tgt_name = "({}, {})".format(*tgt_pos)

        src = env.node_map[src_name]
        tgt = env.node_map[tgt_name]
        t1 = time.perf_counter()
        nodes_visited = env.get_route(src, tgt)
        t2 = time.perf_counter()
        print("Calculate route: {} ms".format(round(1000 * (t2 - t1), 2)))

        if plot_data:
            ax = env.plot_environment(ax, nodes_visited)
            plt.show()
            plt.pause(1)


if __name__ == "__main__":
    main()
