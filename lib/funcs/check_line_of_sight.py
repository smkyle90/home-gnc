import random

import numpy as np


def check_line_of_sight(q0, qd, obs_list, v_speed=1, epsilon=1):
    """Check the line of sight between a point q0 and qd, i.e.,
    the agent can move from q0 to qd without the presence of an
    object.

    Args:
        q0 (np.ndarray): current location of agent
        qd (np.ndarray): desired location of agent
        obs_list (np.ndarray): list of obstacles and associated radii
        v_speed (float): constant velocity of vehicle
        epsilon (float): addition radius buffer for obstacle

    Returns:
        free_path (bool): if a path exists between q0 and qd.
    """

    q0 = q0.reshape(-1, 1)
    qd = qd.reshape(-1, 1)
    obs_list = obs_list.reshape(-1, 3)

    dq = qd - q0
    d_crow = np.linalg.norm(dq)

    v_norm = dq / d_crow

    # Get directional velocities
    v_agent = v_speed * v_norm

    # Keep the to account for moving obstacles
    v_obs = 0 * np.array([[np.cos(0)], [np.sin(0)]])

    # Quadratic equation terms
    a_x = obs_list[:, 0] - q0[0, 0]
    a_y = obs_list[:, 1] - q0[1, 0]

    b_x = v_obs[0, :] - v_agent[0, :]
    b_y = v_obs[1, :] - v_agent[1, :]

    a0 = b_x ** 2 + b_y ** 2
    b0 = 2.0 * a_x * b_x + 2.0 * a_y * b_y
    c0 = a_x ** 2 + a_y ** 2 - (obs_list[:, 2] + epsilon) ** 2

    discriminant = b0 ** 2 - 4.0 * a0 * c0

    # The trajectories intersect if the discriminant is positive
    discriminant = np.where(discriminant > 0.0, discriminant, 0.0)

    t_star_1 = (-b0 - np.sqrt(discriminant)) / (2.0 * a0)
    t_star_2 = (-b0 + np.sqrt(discriminant)) / (2.0 * a0)

    # There are three distinct scenarios:
    # 1. Both positive -- collision is in front of us
    collision_flag = np.logical_and(
        np.logical_and(t_star_1 > 0.0, t_star_2 > 0.0), t_star_1 != t_star_2
    )

    # 2. One positive, one negative -- we are within the bubble
    in_sigma_flag = np.sign(t_star_1) != np.sign(t_star_2)

    # 3. Both negative -- collision is behind us
    no_collision = np.logical_or(
        np.logical_and(t_star_1 < 0.0, t_star_2 < 0.0), np.isclose(t_star_1, t_star_2),
    )

    free_path = True
    # Check if the collision if before or after  we reach out desintation
    for idx, collision in enumerate(collision_flag):
        if collision:
            t_star = np.maximum(0, np.minimum(t_star_1[idx], t_star_2[idx]))
            d_star = v_speed * t_star

            if d_crow > d_star:
                free_path = False
                return free_path

    return free_path
