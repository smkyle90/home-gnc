"""LQR solver
"""

import numpy as np
import scipy

from ..systems import LinearSystem


class LQR(object):
    """Solve the continuous time Linear Quadratic Regulator (LQR) controller.

        dx/dt = A x + B u

        cost = integral x.T*Q*x + u.T*R*u

        ref Bertsekas, p.151

    Args:
        sys (LinearSystem): an controlib LinearSystem type
        Q (np.ndarray): an n x n state cost matrix
        R (np.ndarray): an m x m input cost matrix

    Returns:
        K (np.ndarray): optimal control gains
        X (np.ndarray): solution to Ricatti equation
        e_val (np.ndarray): corresponding closed-loop system eigenvalues
    """

    def __init__(self, sys, Q, R):
        self.sys = sys
        self.Q = Q
        self.R = R

    @property
    def sys(self):
        """sys matrix."""
        return self._sys

    @sys.setter
    def sys(self, value):
        """Setter for sys.

        Args:
            value (LinearSystem): LinearSystem

        Raises:
            TypeError: Type of value must be of type LinearSystem.
        """
        if not isinstance(value, LinearSystem):
            raise TypeError("System must be a LinearSystem type.")

        self._sys = value

    @property
    def Q(self):
        """Q matrix."""
        return self._Q

    @Q.setter
    def Q(self, value):
        """Setter for Q.

        Args:
            value (np.ndarray): output weighting matrix

        Raises:
            TypeError: Type of value must be a np.ndarray of dimension.
        """

        ns, __ = self.sys.B.shape

        if not isinstance(value, np.ndarray):
            raise TypeError("Q must be a square nd.nparray of dimension {}".format(ns))

        if value.shape != (ns, ns):
            raise ValueError(
                "The dimension of Q needs to be a square matrix of dimension {}.".format(
                    ns
                )
            )

        self._Q = value

    @property
    def R(self):
        """R matrix."""
        return self._R

    @R.setter
    def R(self, value):
        """Setter for R.

        Args:
            value (np.ndarray): output weighting matrix

        Raises:
            TypeError: Type of value must be a np.ndarray of dimension.
        """

        __, nc = self.sys.B.shape

        if not isinstance(value, np.ndarray):
            raise TypeError("R must be a square nd.nparray of dimension {}".format(nc))

        if value.shape != (nc, nc):
            raise ValueError(
                "The dimension of R needs to be a square matrix of dimenson {}.".format(
                    nc
                )
            )

        self._R = value

    def calculate_control(self):
        """Optimal optimal control strategy

        Args:
            self

        Returns:
            K (np.ndarray): optimal gain matrix
            X (np.ndarray): solution to Algebraic Ricatti equation
            e_val (np.ndarray): closed-loop eigenvalues
        """

        if self.sys.continuous:
            # first, try to solve the ricatti equation
            X = scipy.linalg.solve_continuous_are(
                self.sys.A, self.sys.B, self.Q, self.R
            )

            # compute the LQR gain
            K = np.linalg.inv(self.R).dot(self.sys.B.T.dot(X))
        else:
            # first, try to solve the ricatti equation
            X = scipy.linalg.solve_discrete_are(self.sys.A, self.sys.B, self.Q, self.R)

            # compute the LQR gain
            K = np.linalg.inv(self.sys.B.T.dot(X).dot(self.sys.B) + self.R).dot(
                self.sys.B.T.dot(X).dot(self.sys.A)
            )

        e_val, _e_vec = np.linalg.eig(self.sys.A - self.sys.B.dot(K))

        return K, X, e_val
