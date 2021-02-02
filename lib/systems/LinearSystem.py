"""Class for a Continuous, Linear system, of the form

    dx/dt = Ax + Bu
    y = Cx + Du

    This is a general class, where the user specifies the system matrices,
    and there are methods available to:
        1. Discretise the system,
        2. Perform stability, controllability, observability analyses of the system
        3. Calculate closed-loop control gains for the system using LQR, Pole placement etc.

    This module does not inheret data from the python-controls module, as there are
    dependencies on slycot, which is released up to Python 3.5. As such,
    cusstom methods will be built to extend the functionality inhibited by
    this constraint.

    Note: tried to build a Python 3.5 docker image in install Slycot
"""

import numpy as np
import scipy

import control


class LinearSystem(object):
    def __init__(self, A, B, C=0, D=0, continuous=True, x0=None):
        """Default constructor for the linear system

        Args:
            A (np.array): an n x n system state transition matrix
            B (np.array): an n x m system input matrix
            C (np.array): an n x p system output matrix
            D (np.array): an n x m system feedthrough matrix

        Returns:
            self (LinearSystem)

        """

        ns, _nc = B.shape

        self.A = A
        self.B = B

        if hasattr(C, "shape"):
            self.C = C
        else:
            self.C = np.eye(ns)

        self.D = D
        self.continuous = continuous

        if x0 is None:
            x0 = np.zeros((ns, 1))

        self.x0 = x0

    def __repr__(self):
        if self.continuous:
            sys_type = "continuous"
            sys_des = "dx/dt = Ax + Bu"
            sys_out = "y = Cx + Du"

        else:
            sys_type = "discrete"
            sys_des = "x[(k+1)*dt] = Ax[k*dt] + Bu[k*dt]"
            sys_out = "y[k*dt] = Cx[k*dt] + Du[k*dt]"

        return "The {} system of the form:\n\t{}\n\t{}\nis given by:\nA\n===\n{}\nB\n===\n{}\nC\n===\n{}\nD\n===\n{}.".format(
            sys_type, sys_des, sys_out, self.A, self.B, self.C, self.D
        )

    @property
    def A(self):
        """A matrix."""
        return self._A

    @A.setter
    def A(self, value):
        """Setter for A.

        Args:
            value (np.array): system A matrix.

        Raises:
            TypeError: Type of value must be of type float.
        """
        nr, nc = value.shape

        if nr != nc:
            raise ValueError("A must be square matrix.")

        self._A = value

    @property
    def B(self):
        """B matrix."""
        return self._B

    @B.setter
    def B(self, value):
        """Setter for B.

        Args:
            value (np.array): system B matrix.

        Raises:
            TypeError: Type of value must be of type float.
        """
        ns, _nc = value.shape

        if ns != self.A.shape[0]:
            raise ValueError("B must have the same number of states as A.")

        self._B = value

    @property
    def C(self):
        """C matrix."""
        return self._C

    @C.setter
    def C(self, value):
        """Setter for C.

        Args:
            value (np.array): system C matrix.

        Raises:
            TypeError: Type of value must be of type float.
        """

        _nr, nc = value.shape

        if nc != self.A.shape[0]:
            raise ValueError(
                "C must not have the same number of columns as the number of states."
            )

        self._C = value

    @property
    def D(self):
        """D matrix."""
        return self._D

    @D.setter
    def D(self, value):
        """Setter for D.

        Args:
            value (np.array): system D matrix.

        Raises:
            TypeError: Type of value must be of type float.
        """

        _ns, nc = self.B.shape
        nm, __ = self.C.shape

        if isinstance(value, int) or isinstance(value, float):
            if value:
                value = np.array([value])
            else:
                value = np.zeros((nm, nc))
        if isinstance(value, list):
            value = np.array(value)
        try:
            value = np.reshape(value, (nm, nc))
        except Exception:
            value = np.zeros((nm, nc))

        self._D = value

    @property
    def x0(self):
        """x0 matrix."""
        return self._x0

    @x0.setter
    def x0(self, value):
        """Setter for x0.

        Args:
            value (np.array): system state x0 .

        Raises:
            TypeError: Type of value must be of type float.
        """

        ns, __ = self.B.shape

        # Reshape the state vector
        value = np.reshape(value, (-1, 1))

        if value.shape != (ns, 1):
            raise ValueError("x0 must be a 1-d vector of states.")

        self._x0 = value

    def disc(self, dt):
        """Discretise system

        Args:
            dt (float): discretisation time
        """

        if not self.continuous:
            return self

        if isinstance(dt, float) or isinstance(dt, int):
            dt = float(dt)
            if dt < 0:
                raise ValueError("Discretisation time must be greater than zero.")
        else:
            raise TypeError("The timestep must be a float or integer.")

        ns, nc = self.B.shape

        blk = np.block([[self.A, self.B], [np.zeros((nc, ns)), np.zeros((nc, nc))]])

        matrix_exp = scipy.linalg.expm(blk * dt)

        A_d = matrix_exp[:ns, :ns]
        B_d = matrix_exp[:ns, ns:]

        return LinearSystem(A_d, B_d, self.C, self.D, False)

    def ctrb(self):
        """Return the controllability grammian for the system.
        """

        C_g = np.asarray(control.ctrb(self.A, self.B))
        r = np.linalg.matrix_rank(C_g)
        return C_g, r == self.A.shape[0]

    def obsv(self):
        """Return the controllability grammian for the system.
        """
        O_g = np.asarray(control.obsv(self.A, self.C))
        r = np.linalg.matrix_rank(O_g)

        return O_g, r == self.A.shape[0]

    def poles(self):
        """Return the poles of the system
        """
        e_val, _e_vec = np.linalg.eig(self.A)

        return e_val

    def stable(self):
        """Checks the stability of a system for a continuous
        or discrete system.

        """
        e_val = self.poles()

        if self.continuous:
            return np.all(e_val <= 0).min()
        else:
            return np.all(np.absolute(e_val) <= 1).min()

    def apply_control(self, u, delta_t=0):
        """Apply a control to the system.

        Args:
            u (np.array): an m x 1 array of control actions.
        """

        ns, nc = self.B.shape

        if isinstance(u, int):
            u = np.array([u])
        elif isinstance(u, list):
            u = np.array(u)

        # Reshape the control vector
        u = np.reshape(u, (-1, 1))

        if u.shape != (nc, 1):
            raise ValueError("u must be of dimension {}x{}".format(nc, 1))

        if self.continuous:
            if (not delta_t) or (delta_t < 0):
                raise ValueError("Must provide a suitable timestep.")

            sys_d = self.disc(delta_t)

            self.x0 = sys_d.A.dot(self.x0) + sys_d.B.dot(u)
        else:
            self.x0 = self.A.dot(self.x0) + self.B.dot(u)
