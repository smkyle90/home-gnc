import numpy as np
import pytest
from controlib.filters import KalmanFilter
from controlib.systems import LinearSystem


@pytest.fixture
def one_d_cart():
    """One dimension cart system.
    """
    # define some parameters
    m = 1

    # Define the system matrices
    A = np.array([[0, 1], [0, 0],])

    B = np.array([[0], [1 / m],])

    C = np.eye(2)

    sys = LinearSystem(A, B, C)

    return sys


@pytest.fixture
def smd_system():
    """Spring-mass-damper system
    """
    # define some parameters
    m = 1
    k = 1
    b = 1

    # Define the system matrices
    A = np.array([[0, 1], [-k / m, -b / m],])

    B = np.array([[0], [1 / m],])

    C = np.array([[1, 0]])

    sys = LinearSystem(A, B, C)

    return sys


@pytest.fixture
def dc_motor_system():

    A = np.array([[-4, -0.03], [0.75, -10],])

    B = np.array([[2], [0],])

    C = np.array([[0, 1]])

    # D = 0

    sys = LinearSystem(A, B, C)

    return sys


@pytest.fixture
def ip_system():
    """Inverted pendulum on a cart (continuous)
    """
    # http://ctms.engin.umich.edu/CTMS/index.php?example=InvertedPendulum&section=ControlStateSpace

    # Define the system matrices
    A = np.array(
        [[0, 1, 0, 0], [0, -0.1818, 2.6727, 0], [0, 0, 0, 1], [0, -0.4545, 31.1818, 0],]
    )

    B = np.array([[0], [1.8182], [0], [4.54],])

    C = np.array([[1, 0, 0, 0], [0, 0, 1, 0],])

    sys = LinearSystem(A, B, C)

    return sys


@pytest.fixture
def kf_system():
    """
    Ref:
    https://www.mathworks.com/help/control/ug/kalman-filter-design.html;jsessionid=ccdae7bd4b40d3b0fef5b54d18ed

    """
    # Define the system matrices
    A = np.array([[1.1269, -0.4940, 0.1129], [1.0000, 0, 0], [0, 1.0000, 0],])

    B = np.array([[-0.3832], [0.5919], [0.5191],])

    C = np.array([[1, 0, 0]])

    sys = LinearSystem(A, B, C, continuous=False)

    # Define process and measurement covariances

    proc_covar = np.array([[2.3]])
    meas_covar = np.array([[1.0]])
    est_covar = np.eye(3)

    kf_system = KalmanFilter(
        sys, est_covar=est_covar, proc_covar=proc_covar, meas_covar=meas_covar
    )

    return kf_system
