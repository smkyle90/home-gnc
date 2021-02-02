"""Tests for KalmanFilter object
"""

import copy
import random

import matplotlib.pyplot as plt
import numpy as np
import pytest
from controlib.filters import KalmanFilter
from controlib.systems import LinearSystem


def test_init(one_d_cart,):
    # Run all variations of the __init__

    kf = KalmanFilter(one_d_cart)

    assert np.allclose(kf.sys.A, one_d_cart.A, atol=1e-3)
    assert np.allclose(kf.sys.B, one_d_cart.B, atol=1e-3)
    assert np.allclose(kf.sys.C, one_d_cart.C, atol=1e-3)
    assert np.allclose(kf.sys.D, one_d_cart.D, atol=1e-3)
    assert np.allclose(kf.sys.x0, one_d_cart.x0, atol=1e-3)

    est_covar = 10 * np.eye(2)
    kf = KalmanFilter(one_d_cart, est_covar)
    assert np.allclose(kf.est_covar, est_covar, atol=1e-3)

    proc_covar = 10 * np.eye(1)
    kf = KalmanFilter(one_d_cart, est_covar, proc_covar)
    assert np.allclose(kf.proc_covar, proc_covar, atol=1e-3)

    meas_covar = 10 * np.eye(2)
    kf = KalmanFilter(one_d_cart, est_covar, proc_covar, meas_covar)
    assert np.allclose(kf.meas_covar, meas_covar, atol=1e-3)

    # Test __repr__
    print(kf)


def test_predict_correct():

    # Define a simple one dimensional system
    A = np.array([[0]])
    B = np.array([[1]])
    C = np.array([[1]])

    sys = LinearSystem(A, B, C)

    # Discretize the system
    dt = 1.0
    sys_d = sys.disc(dt)

    # Define some covariances
    sigma_x = 1.0
    sigma_u = 1.0
    sigma_y = 1.0

    est_covar = np.diag([sigma_x ** 2])
    proc_covar = np.diag([sigma_u ** 2])
    meas_covar = np.diag([sigma_y ** 2])

    kf = KalmanFilter(sys_d, est_covar, proc_covar, meas_covar)

    # Define a control
    u = 1

    # Apply the control
    kf.predict(u)
    assert np.allclose(kf.sys.x0, np.array([[1]]), atol=1e-3)
    assert np.allclose(kf.est_covar, np.array([[2.0]]), atol=1e-3)

    # Send a measurement
    y = np.array([[1]])
    kf.correct(y)
    assert np.allclose(kf.sys.x0, np.array([[1]]), atol=1e-3)
    assert np.allclose(kf.est_covar, np.array([[2 / 3]]), atol=1e-3)


def test_filter(one_d_cart):

    # Discrete Case
    dt = 0.1
    sys_d = one_d_cart.disc(dt)

    # Make only one state measurable
    sys_d.C = np.array([[1, 0]])
    sys_d.D = 0

    # Initialise the true system
    true_sys = copy.deepcopy(sys_d)

    # Initialise some covariances
    sigma_x = 1e-15
    sigma_v = 1e-15
    sigma_y = 1e-15
    sigma_u = 1e-15

    est_covar = np.diag([sigma_x ** 2, sigma_v ** 2])
    proc_covar = np.diag([sigma_u ** 2])
    meas_covar = np.diag([sigma_y ** 2])

    kf = KalmanFilter(sys_d, est_covar, proc_covar, meas_covar)

    T = 0
    while T < 20:

        # Specify the input
        u = 1

        # update true system
        true_sys.apply_control(u)

        # Get the true output
        y = true_sys.C.dot(true_sys.x0)

        # Measured responses
        u_meas = u + random.gauss(0, sigma_u)
        y_meas = y + random.gauss(0, sigma_y)

        # Run KF
        kf.predict(u_meas)
        kf.correct(y_meas)

        T += dt

    assert np.allclose(kf.sys.x0, true_sys.x0, atol=1e-3)

    # Continuous Case
    dt = 0.1
    sys_c = one_d_cart

    print(sys_c.x0)
    # Make only one state measurable
    sys_c.C = np.array([[1, 0]])
    sys_c.D = 0

    # Initialise the true system
    true_sys = copy.deepcopy(sys_c)

    # Initialise some covariances
    sigma_x = 1e-15
    sigma_v = 1e-15
    sigma_y = 1e-15
    sigma_u = 1e-15

    est_covar = np.diag([sigma_x ** 2, sigma_v ** 2])
    proc_covar = np.diag([sigma_u ** 2])
    meas_covar = np.diag([sigma_y ** 2])

    kf = KalmanFilter(sys_c, est_covar, proc_covar, meas_covar)

    T = 0
    while T < 20:
        # Specify the input
        u = 1

        # update true system
        true_sys.apply_control(u, dt)

        # Get the true output
        y = true_sys.C.dot(true_sys.x0)

        # Measured responses
        u_meas = u + random.gauss(0, sigma_u)
        y_meas = y + random.gauss(0, sigma_y)

        # Run KF
        kf.predict(u_meas, dt)
        kf.correct(y_meas)

        T += dt

    assert np.allclose(kf.sys.x0, true_sys.x0, atol=2)


def test_improper_types(one_d_cart):

    with pytest.raises(TypeError):
        kf = KalmanFilter("bad system")
        kf = KalmanFilter(1)
        kf = KalmanFilter(np.eye(10))

    # Instantiate a proper system
    kf = KalmanFilter(one_d_cart)
    with pytest.raises(TypeError):
        kf.est_covar = 1
        kf.est_covar = [1, 2, 3]
        kf.est_covar = "covar"

    with pytest.raises(ValueError):
        kf.est_covar = np.eye(4)
        kf.est_covar = np.array([1, 2, 3, 4])
        kf.est_covar = np.array([[1], [2], [3], [4]])

    with pytest.raises(TypeError):
        kf.meas_covar = 1
        kf.meas_covar = [1, 2, 3]
        kf.meas_covar = "covar"

    with pytest.raises(ValueError):
        kf.meas_covar = np.eye(4)
        kf.meas_covar = np.array([1, 2, 3, 4])
        kf.meas_covar = np.array([[1], [2], [3], [4]])

    with pytest.raises(TypeError):
        kf.proc_covar = 1
        kf.proc_covar = [1, 2, 3]
        kf.proc_covar = "covar"

    with pytest.raises(ValueError):
        kf.proc_covar = np.eye(4)
        kf.proc_covar = np.array([1, 2, 3, 4])
        kf.proc_covar = np.array([[1], [2], [3], [4]])

    # Bad timestep
    with pytest.raises(ValueError):
        kf.predict(1, -1)

    # Bad measurement vector
    with pytest.raises(ValueError):
        kf.correct(np.array([1, 2]))
        kf.correct(np.array([[1]]))
