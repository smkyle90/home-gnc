"""Tests for MPC object
"""

import math

import numpy as np
import pytest
from controlib.controllers import LQR
from controlib.systems import LinearSystem


def test_init(one_d_cart):
    # Test all variations of the __init__ function

    ns, nu = one_d_cart.B.shape

    Q = np.eye(ns)
    R = np.eye(nu)

    controller = LQR(one_d_cart, Q, R)

    assert np.allclose(controller.sys.A, one_d_cart.A, atol=1e-3)
    assert np.allclose(controller.Q, Q, atol=1e-3)
    assert np.allclose(controller.R, R, atol=1e-3)


def test_calculate_control(ip_system):
    # Use the inverted pendulum system

    # Continuous Case
    Q = np.diag([1, 0, 1, 0])
    R = np.diag([1])
    lqr = LQR(ip_system, Q, R)
    K, _S, _E = lqr.calculate_control()
    K_expected = np.array([[-1.0000, -1.6567, 18.6854, 3.4594]])
    assert np.allclose(K, K_expected, atol=0.2)

    Q = np.diag([5000, 0, 100, 0])
    R = np.diag([1])
    lqr = LQR(ip_system, Q, R)
    K, _S, _E = lqr.calculate_control()
    K_expected = np.array([[-70.7107, -37.8345, 105.5298, 20.9238]])
    assert np.allclose(K, K_expected, atol=0.2)

    # Discrete Case
    T_s = 1 / 100
    ip_system_d = ip_system.disc(T_s)

    Q = np.diag([1, 0, 1, 0])
    R = np.diag([1])
    lqr = LQR(ip_system_d, Q, R)
    K, _S, _E = lqr.calculate_control()
    K_expected = np.array([[-0.9384, -1.5656, 18.0351, 3.3368]])
    assert np.allclose(K, K_expected, atol=0.2)

    Q = np.diag([5000, 0, 100, 0])
    R = np.diag([1])
    lqr = LQR(ip_system_d, Q, R)
    K, _S, _E = lqr.calculate_control()
    K_expected = np.array([[-61.9933, -33.5040, 95.0597, 18.8300]])
    assert np.allclose(K, K_expected, atol=0.2)


def test_improper_types(one_d_cart):

    y_set = np.array([[1], [0]])

    ns, nu = one_d_cart.B.shape

    Q = np.eye(ns)
    R = np.eye(nu)

    bad_system = [
        5,
        5.0,
        "system",
        {"system": one_d_cart},
    ]
    for bad_sys in bad_system:
        with pytest.raises(TypeError):
            controller = LQR(bad_sys, Q, R)

    bad_Q = [1, [1]]
    for bad_q in bad_Q:
        with pytest.raises(TypeError):
            controller = LQR(one_d_cart, bad_q, R)

    bad_Q = [np.eye(1), np.eye(3)]
    for bad_q in bad_Q:
        with pytest.raises(ValueError):
            controller = LQR(one_d_cart, bad_q, R)

    bad_R = [1, [1]]
    for bad_r in bad_R:
        with pytest.raises(TypeError):
            controller = LQR(one_d_cart, Q, bad_r)

    bad_R = [np.array([]), np.eye(2)]
    for bad_r in bad_R:
        with pytest.raises(ValueError):
            controller = LQR(one_d_cart, Q, bad_r)
