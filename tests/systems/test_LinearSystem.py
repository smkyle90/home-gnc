"""Tests for LinearSystems object
"""

import copy

import numpy as np
import pytest
from controlib.systems import LinearSystem


def test_init_LinearSystem():
    # Define number of states, controls and measurements
    ns = 2
    nc = 1
    nm = 1

    # Test all version of the init function
    A = np.eye(ns)
    B = np.ones((ns, nc))

    sys = LinearSystem(A, B)

    assert np.array_equal(A, sys.A)
    assert np.array_equal(B, sys.B)
    assert np.array_equal(np.eye(ns), sys.C)
    assert np.array_equal(np.zeros((ns, 1)), sys.x0)

    C = np.ones((nm, ns))
    sys = LinearSystem(A, B, C)

    assert np.array_equal(C, sys.C)

    D = np.ones((nm, nc))
    sys = LinearSystem(A, B, C, D)

    assert np.array_equal(D, sys.D)

    continuous = False
    sys = LinearSystem(A, B, C, D, continuous)

    assert not sys.continuous

    x0 = np.array([[10], [3]])
    sys = LinearSystem(A, B, C, D, continuous, x0)

    assert np.array_equal(x0, sys.x0)

    sys = LinearSystem(A, B, C, 1, continuous, x0)
    sys = LinearSystem(A, B, C, [1], continuous, x0)


def test_repr(smd_system):
    # Continuous case
    print(smd_system)

    # Discontinuous case
    smd_system.continuous = False
    print(smd_system)


def test_disc(dc_motor_system):
    # https://www.mathworks.com/help/control/getstart/converting-between-continuous-and-discrete-time-systems.html
    delta_t = 0.01
    sys_d = dc_motor_system.disc(delta_t)

    A_d = np.array([[0.96079, -0.00027976], [0.006994, 0.90484],])

    B_d = np.array([[0.019605], [7.1595e-005],])

    assert np.allclose(A_d, sys_d.A, atol=0.01)
    assert np.allclose(B_d, sys_d.B, atol=0.01)

    # Try and run the discretizer on an already discretized system
    sys_d.disc(delta_t)


def test_ctrb(smd_system, ip_system):
    c_smd = np.array([[0, 1], [1, -1],])

    C_g, ctrb = smd_system.ctrb()
    assert np.allclose(c_smd, C_g, atol=0.01)
    assert ctrb

    c_ip = np.array(
        [
            [0.0, 1.8182, -0.33054876, 12.19415176],
            [1.8182, -0.33054876, 12.19415176, -4.42554097],
            [0.0, 4.54, -0.8263719, 141.71560641],
            [4.54, -0.8263719, 141.71560641, -31.31000529],
        ]
    )
    C_g, ctrb = ip_system.ctrb()
    assert np.allclose(c_ip, C_g, atol=0.01)
    assert ctrb


def test_obsv(smd_system, ip_system):
    o_smd = np.array([[1.0, 0.0], [0.0, 1.0],])

    O_g, obsv = smd_system.obsv()
    assert np.allclose(o_smd, O_g, atol=0.01)
    assert obsv

    o_ip = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, -0.1818, 2.6727, 0.0],
            [0.0, -0.4545, 31.1818, 0.0],
            [0.0, 0.03305124, -0.48589686, 2.6727],
            [0.0, 0.0826281, -1.21474215, 31.1818],
        ]
    )

    O_g, obsv = ip_system.obsv()
    assert np.allclose(o_ip, O_g, atol=0.01)
    assert obsv


def test_poles(smd_system, ip_system):

    p_smd = np.array([-0.5 + 0.8660254j, -0.5 - 0.8660254j])
    assert np.allclose(p_smd, smd_system.poles(), atol=0.1)

    p_ip = np.array([0.0, -0.14281773, -5.60409025, 5.56510798])
    assert np.allclose(p_ip, ip_system.poles(), atol=0.1)


def test_stable(smd_system, ip_system):
    # Continuous Case
    assert smd_system.stable()
    assert not ip_system.stable()

    # Discrete case
    T_s = 0.1
    assert smd_system.disc(T_s).stable()
    assert not ip_system.disc(T_s).stable()


def test_apply_control(one_d_cart):

    # Specify some different "looking" controls
    sys = copy.deepcopy(one_d_cart)
    dt = 1.0
    sys.apply_control(1, dt)
    sys.apply_control([1], dt)
    sys.apply_control(np.array([1]), dt)

    del sys

    # Re-init one_d_cart
    sys = copy.deepcopy(one_d_cart)

    # Specify a control and timestep
    u = np.array([[1]])
    dt = 1.0

    # apply the control once
    sys.apply_control(u, dt)
    x_expected = np.array([[0.5], [1]])
    assert np.allclose(sys.x0, x_expected, atol=1e-3)

    # apply the control twice
    sys.apply_control(u, dt)
    x_expected = np.array([[2], [2]])
    assert np.allclose(sys.x0, x_expected, atol=1e-3)

    # apply the control three times
    sys.apply_control(u, dt)
    x_expected = np.array([[4.5], [3]])
    assert np.allclose(sys.x0, x_expected, atol=1e-3)

    del sys

    # Discretise system and reset init condits
    sys_d = one_d_cart.disc(dt)

    # apply the control once
    sys_d.apply_control(u)
    x_expected = np.array([[0.5], [1]])
    assert np.allclose(sys_d.x0, x_expected, atol=1e-3)

    # apply the control twice
    sys_d.apply_control(u)
    x_expected = np.array([[2], [2]])
    assert np.allclose(sys_d.x0, x_expected, atol=1e-3)

    # apply the control three times
    sys_d.apply_control(u)
    x_expected = np.array([[4.5], [3]])
    assert np.allclose(sys_d.x0, x_expected, atol=1e-3)


def test_improper_types(one_d_cart):

    # Non-square A matrix
    with pytest.raises(ValueError):
        one_d_cart.A = np.array([[1, 2]])
        one_d_cart.A = np.array([[1], [2]])

    # Wrong Dimension B matrix
    with pytest.raises(ValueError):
        one_d_cart.B = np.array([[1]])
        one_d_cart.B = np.array([[1], [2], [3]])

    # Wrong Dimension C matrix
    with pytest.raises(ValueError):
        one_d_cart.C = np.array([[1]])
        one_d_cart.C = np.array([[1], [2], [3]])

    # Wrong Dimension of initial conditions
    with pytest.raises(ValueError):
        one_d_cart.x0 = np.array([[1]])
        one_d_cart.x0 = np.array([[1], [2], [3]])

    # System timestep
    with pytest.raises(ValueError):
        one_d_cart.disc(-1)

    # System timestep
    with pytest.raises(TypeError):
        one_d_cart.disc([1])
        one_d_cart.disc(np.array([1]))
        one_d_cart.disc({"data": 1})

    # Wrong Dimension of control
    with pytest.raises(ValueError):
        one_d_cart.apply_control(np.array([[1, 2]]), 1)
        one_d_cart.apply_control(np.array([[1], [2]]), 1)
        one_d_cart.apply_control(np.array([]), 1)

    # Negative, or no timestep for a continuous system
    with pytest.raises(ValueError):
        one_d_cart.apply_control(1, -1)
        one_d_cart.apply_control(1)
