import numpy as np
from scipy import signal

from .lateral_distance import lateral_distance


def feedback_linearisation_controller(
    q0, qd, poles=[-0.25, -0.5], dt=0.1, vd=1.0, T_max=10, epsilon=0.1
):
    """A feedback linearisation controller for a unicycle.

    Change unicycle in the linearised feedback form

        z = A_hat * z + B_hat * nu

    where z is the characterised by the lateral error, e_l, and heading error, e_h, given by
    z = [e_l, vd * sin(e_h)].

    Note this relies on a constant velocity model, i.e., vd is fixed.

    Note that A_hat = [[0, 1], [0, 0]] and B_hat = [[0], [1]], and the feedback law
    is given by nu = -K * z, where K can be found using the pole placement method.

    The control law nu = vd * omegad * cos(e_h) ==> omegad = nu / (vd * cos(e_h)).

    Therefore, u = [[vd], [omegad]] for the original unicycle system.

    Args:
        q0 (np.ndarray): initial configuration of the system
        qf (np.ndarray): desired configuration of the system
        poles (list): location of closed loop poles for feedback system
        dt (float): timestep for calculating trajectory and controls
        vd (float): constant velocity of agent
        T_max (float): max look ahead time before no controls are returned
        epsilon (float): stop condition for || q-qd || < epsilon

    Returns:
        u (list): a list of control inputs for unicycle that steer system from q0 to qd.
        An empty list implies the path is not followable or reachable in the time allotment.
        q (list): the list of configurations
    """
    q0 = q0.reshape(3, 1)
    qd = qd.reshape(3, 1)

    ## Feedback linearisation matrices.
    A_hat = np.array([[0, 1], [0, 0],])

    B_hat = np.array([[0], [1],])

    # Closed loop control gains.
    K = signal.place_poles(A_hat, B_hat, poles)
    K = K.gain_matrix

    # Initialise controls and state
    u = []
    q = []

    # Check d_theta is in range (-pi/2, pi/2), if not, a valid path does not exist
    d_theta = q0[2, 0] - qd[2, 0]

    if np.abs(d_theta) > np.pi / 2:
        return u, q

    # Initialise the time
    T = 0
    while np.linalg.norm(q0 - qd) > epsilon:
        e_h = q0[2, 0] - qd[2, 0]
        e_l = lateral_distance(*q0[:2, 0], *qd[:, 0])

        z = np.array([[e_l], [vd * np.sin(e_h)],])
        nu = -K.dot(z)

        omegad = nu[0, 0] / (vd * np.cos(e_h))

        u.append((vd, omegad))

        G = np.array([[np.cos(q0[2, 0]), 0], [np.sin(q0[2, 0]), 0], [0, 1],])
        u_in = np.array([[vd], [omegad],])

        q0 = q0 + G.dot(u_in) * dt
        q.append(q0[:, 0].tolist())
        T += dt

        if T > T_max:
            return [], []

    return u, q
