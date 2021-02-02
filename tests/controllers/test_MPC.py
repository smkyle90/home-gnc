"""Tests for MPC object
"""


import numpy as np
import pytest
from controlib.controllers import MPC

# @pytest

# flake8: noqa: W0212
def test_init(one_d_cart):
    # Test all variations of the __init__ function

    y_set = np.array([[1], [0]])

    __, nu = one_d_cart.B.shape
    nm, __ = one_d_cart.C.shape

    Q = np.eye(nm)
    R = np.eye(nu)

    t_sample = 1.0
    n_predict = 10

    controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict)

    assert np.allclose(controller.sys.A, one_d_cart.disc(t_sample).A, atol=1e-3)
    assert np.allclose(controller.setpoint, y_set, atol=1e-3)
    assert np.allclose(controller.Q, Q, atol=1e-3)
    assert np.allclose(controller.R, R, atol=1e-3)
    assert np.allclose(controller.t_sample, t_sample, atol=1e-3)
    assert np.allclose(controller.n_predict, n_predict, atol=1e-3)

    # Check the min and max are +- infinity
    assert controller.ymin.min() == -np.inf
    assert controller.ymax.max() == np.inf

    # Set output bounds
    y_min = [-1, -1]
    y_max = [1, 1]

    Y_min = np.array(y_min * n_predict).reshape(-1, 1)
    Y_max = np.array(y_max * n_predict).reshape(-1, 1)

    controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict, y_min, y_max)

    assert np.allclose(controller.ymin, np.array(y_min).reshape(-1, 1), atol=1e-3)
    assert np.allclose(controller.ymax, np.array(y_max).reshape(-1, 1), atol=1e-3)
    assert np.allclose(controller._MPC__Y_min, Y_min, atol=1e-3)
    assert np.allclose(controller._MPC__Y_max, Y_max, atol=1e-3)

    # Set input bounds
    u_min = [-1]
    u_max = [1]

    U_min = np.array(u_min * n_predict).reshape(-1, 1)
    U_max = np.array(u_max * n_predict).reshape(-1, 1)

    controller = MPC(
        one_d_cart, y_set, Q, R, t_sample, n_predict, y_min, y_max, u_min, u_max
    )
    assert np.allclose(controller.umin, np.array(u_min).reshape(-1, 1), atol=1e-3)
    assert np.allclose(controller.umax, np.array(u_max).reshape(-1, 1), atol=1e-3)
    assert np.allclose(controller._MPC__U_min, U_min, atol=1e-3)
    assert np.allclose(controller._MPC__U_max, U_max, atol=1e-3)

    # Check the hidden attributes. Make this simple.
    t_sample = 1.0
    n_predict = 5

    controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict)

    Q_expected = np.eye(n_predict * nm)
    R_expected = np.eye(n_predict * nu)

    A_expected = np.array(
        [
            [1, 1],
            [0, 1],
            [1, 2],
            [0, 1],
            [1, 3],
            [0, 1],
            [1, 4],
            [0, 1],
            [1, 5],
            [0, 1],
        ]
    )

    B_expected = np.array(
        [
            [0.5, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1.5, 0.5, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [2.5, 1.5, 0.5, 0, 0],
            [1, 1, 1, 0, 0],
            [3.5, 2.5, 1.5, 0.5, 0],
            [1, 1, 1, 1, 0],
            [4.5, 3.5, 2.5, 1.5, 0.5],
            [1, 1, 1, 1, 1],
        ]
    )

    assert np.allclose(controller._MPC__A_t, A_expected, atol=1e-3)
    assert np.allclose(controller._MPC__B_t, B_expected, atol=1e-3)
    assert np.allclose(controller._MPC__Q_mpc, Q_expected, atol=1e-3)
    assert np.allclose(controller._MPC__R_mpc, R_expected, atol=1e-3)

    # Test updating the parameters
    Q = 10 * np.eye(nm)
    controller.Q = Q
    controller.initialise_controller()
    Q_expected = 10 * np.eye(n_predict * nm)
    assert np.allclose(controller._MPC__Q_mpc, Q_expected, atol=1e-3)

    R = 10 * np.eye(nu)
    controller.R = R
    R_expected = np.eye(n_predict * nu)
    assert np.allclose(controller._MPC__R_mpc, R_expected, atol=1e-3)


# flake8: noqa: W0212
def test_update_setpoint(one_d_cart):
    y_set = np.array([[1], [0]])

    __, nu = one_d_cart.B.shape
    nm, __ = one_d_cart.C.shape

    Q = np.eye(nm)
    R = np.eye(nu)

    t_sample = 1.0
    n_predict = 10

    controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict)

    y_new = np.array([[10], [-10]])
    Y_new = np.block([[y_new] for i in range(n_predict)])

    controller.update_setpoint(y_new)

    assert np.allclose(controller.setpoint, y_new, atol=1e-3)
    assert np.allclose(controller._MPC__Y_d, Y_new, atol=1e-3)


def test_calculate_control(one_d_cart, ip_system):

    # One-Dimensional Cart
    y_set = np.array([[1], [0]])

    ns, nu = one_d_cart.B.shape
    nm, __ = one_d_cart.C.shape

    Q = np.eye(ns)
    R = np.eye(nu)

    t_sample = 0.2
    n_predict = 25

    # Unconstrained case, should be the same as LQR
    mpc = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict)

    u = mpc.calculate_control()

    assert len(u) == nu

    # Add some constraints
    x_max = 5
    v_max = 1
    u_max = 10
    u_min = -10

    # Compare control outputs
    U_max = -np.inf
    U_min = np.inf
    X_max = -np.inf
    V_max = -np.inf

    # Define MPC object
    controller = MPC(
        one_d_cart,
        y_set,
        Q,
        R,
        t_sample,
        n_predict,
        [-x_max, -v_max],
        [x_max, v_max],
        [u_min],
        [u_max],
    )
    T = 0

    while T < 20:

        # Update the initial condition
        controller.sys.x0 = one_d_cart.x0

        # Calcualte the controller
        u = controller.calculate_control()

        # Apply the controls
        one_d_cart.apply_control(u, t_sample)

        U_max = max(U_max, u.max())
        U_min = min(U_min, u.min())
        X_max = max(X_max, one_d_cart.x0[0, 0])
        V_max = max(V_max, one_d_cart.x0[1, 0])

        T += t_sample

    assert np.allclose(one_d_cart.C.dot(one_d_cart.x0), y_set, atol=1e-6)
    assert U_max <= u_max
    assert U_min >= u_min
    assert X_max <= x_max
    assert V_max <= v_max

    # Inverted Pendulum
    y_set = np.array([[0.2], [0]])

    ns, nu = ip_system.B.shape
    nm, __ = ip_system.C.shape

    Q = np.eye(nm)
    R = 0.01 * np.eye(nu)

    t_sample = 0.02
    n_predict = 50

    # Add some constraints
    ymin = [-1, -2 * np.pi]
    ymax = [1, 2 * np.inf]

    umin = [-1]
    umax = [1]

    # Compare control inputs
    U_max = -np.inf
    U_min = np.inf

    # Define MPC object
    controller = MPC(
        ip_system, y_set, Q, R, t_sample, n_predict, ymin, ymax, umin, umax,
    )

    T = 0
    while T < 20:
        # Update the initial condition
        controller.sys.x0 = ip_system.x0

        # Calcualte the controller
        u = controller.calculate_control()

        # Apply the controls
        ip_system.apply_control(u, t_sample)

        U_max = max(U_max, u.max())
        U_min = min(U_min, u.min())

        T += t_sample

    assert np.allclose(ip_system.C.dot(ip_system.x0), y_set, atol=1e-6)
    assert U_max <= u_max
    assert U_min >= u_min


def test_improper_types(one_d_cart):

    y_set = np.array([[1], [0]])

    ns, nu = one_d_cart.B.shape

    Q = np.eye(ns)
    R = np.eye(nu)

    t_sample = 1.0
    n_predict = 10

    bad_system = [
        5,
        5.0,
        "system",
        {"system": one_d_cart},
    ]
    for bad_sys in bad_system:
        with pytest.raises(TypeError):
            controller = MPC(bad_sys, y_set, Q, R, t_sample, n_predict)

    bad_setpoint = [[], [1, 2, 3], np.array([]), np.array([[3, 2, 1]])]
    for bad_set in bad_setpoint:
        with pytest.raises(ValueError):
            controller = MPC(one_d_cart, bad_set, Q, R, t_sample, n_predict)

    bad_setpoint = ["bad_setpoint"]
    for bad_set in bad_setpoint:
        with pytest.raises(TypeError):
            controller = MPC(one_d_cart, bad_set, Q, R, t_sample, n_predict)

    bad_Q = [1, [1]]
    for bad_q in bad_Q:
        with pytest.raises(TypeError):
            controller = MPC(one_d_cart, y_set, bad_q, R, t_sample, n_predict)

    bad_Q = [np.eye(1), np.eye(3)]
    for bad_q in bad_Q:
        with pytest.raises(ValueError):
            controller = MPC(one_d_cart, y_set, bad_q, R, t_sample, n_predict)

    bad_R = [1, [1]]
    for bad_r in bad_R:
        with pytest.raises(TypeError):
            controller = MPC(one_d_cart, y_set, Q, bad_r, t_sample, n_predict)

    bad_R = [np.array([]), np.eye(2)]
    for bad_r in bad_R:
        with pytest.raises(ValueError):
            controller = MPC(one_d_cart, y_set, Q, bad_r, t_sample, n_predict)

    bad_time = "bad_time"
    with pytest.raises(TypeError):
        controller = MPC(one_d_cart, y_set, Q, R, bad_time, n_predict)

    bad_time = -1.0
    with pytest.raises(ValueError):
        controller = MPC(one_d_cart, y_set, Q, R, bad_time, n_predict)

    bad_predict = ["prediction", [1]]
    for bad_p in bad_predict:
        with pytest.raises(TypeError):
            controller = MPC(one_d_cart, y_set, Q, R, t_sample, bad_p)

    bad_predict = [
        -1,
    ]
    for bad_p in bad_predict:
        with pytest.raises(ValueError):
            controller = MPC(one_d_cart, y_set, Q, R, t_sample, bad_p)

    bad_ymin = [[1], [1, 2, 3], np.array([1]), np.array([1, 2, 3])]
    for ymin in bad_ymin:
        with pytest.raises(ValueError):
            controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict, ymin=ymin)

    bad_ymin = [1, "bad_ymin"]
    for ymin in bad_ymin:
        with pytest.raises(TypeError):
            controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict, ymin=ymin)

    bad_ymax = [[1], [1, 2, 3], np.array([1]), np.array([1, 2, 3])]
    for ymax in bad_ymax:
        with pytest.raises(ValueError):
            controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict, ymax=ymax)

    bad_ymax = [1, "bad_ymax"]
    for ymax in bad_ymax:
        with pytest.raises(TypeError):
            controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict, ymax=ymax)

    bad_umin = [[1, 2], [], np.array([1, 2]), np.array([])]
    for umin in bad_umin:
        with pytest.raises(ValueError):
            controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict, umin=umin)

    bad_umin = [1, "bad_umin"]
    for umin in bad_umin:
        with pytest.raises(TypeError):
            controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict, umin=umin)

    bad_umax = [[1, 2], [], np.array([1, 2]), np.array([])]
    for umax in bad_umax:
        with pytest.raises(ValueError):
            controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict, umax=umax)

    bad_umax = [1, "bad_umax"]
    for umax in bad_umax:
        with pytest.raises(TypeError):
            controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict, umax=umax)


def test_repr(one_d_cart):
    y_set = np.array([[1], [0]])

    ns, nu = one_d_cart.B.shape
    nm, __ = one_d_cart.C.shape

    Q = np.eye(ns)
    R = np.eye(nu)

    t_sample = 1.0
    n_predict = 10

    controller = MPC(one_d_cart, y_set, Q, R, t_sample, n_predict)

    print(controller)
