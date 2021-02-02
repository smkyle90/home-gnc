"""Tests for LinearSystems object
"""

import numpy as np
import pytest
from controlib.systems import LaguerreNetwork


@pytest.mark.skip
def test_laguerre_projection():
    L = LaguerreNetwork(6, 0.5)

    y = np.array([[i for i in range(10)],])
    t = [i for i in range(10)]

    L.laguerre_projection(y, t)


@pytest.mark.skip
def test_construct_A():
    _L = LaguerreNetwork(6, 0.5)
    # L._construct_A()


@pytest.mark.skip
def test_construct_B():
    _L = LaguerreNetwork(6, 0.5)
    # L._construct_B()
