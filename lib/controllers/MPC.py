"""LQR solver
"""

import numpy as np
import quadprog as qp

from ..systems import LinearSystem


# flake8: noqa: C901
class MPC(object):
    """Solve the continuous time Model Predictive Controller (MPC) for a linear system of the
    form:

        dx/dt = Ax + Bu
        y = Cx

    for a specific sampling time, t_sample, and prediction horizon, n_predict.

    As per the quadprog repo, the system is converted into a Quadratic Programming
    Problem of the form:

        min x.T*G*x - a.T*x
        s.t. C.T*u >= b

    for some matrices G, a, C, b. In this application, x represents the
    sequence of control inputs.

    Note that the control inequalities, output constraints and output limits
    are also contained in the G, a, C, and b matrices appropriately.

    This is solved using the quadprog library. References:

        Matlab example: https://www.mathworks.com/help/optim/ug/quadprog.html

    Convex optimizers: https://nbviewer.jupyter.org/github/cvxgrp/cvx_short_course/blob/master/intro/control.ipynb

    Quadprog repo: https://github.com/rmcgibbo/quadprog

    Stanford resource: https://web.stanford.edu/class/archive/ee/ee392m/ee392m.1056/Lecture14_MPC.pdf

    Args:
        sys (LinearSystem): an controlib LinearSystem type
        t_sample (float): sampling time of the Model Preditive controller
        n_predict (int): number of prediction steps
        setpoint (np.ndarray): an n x 1 output setpoint (desired trajectory) OR an n x n_predict output setpoint
        Q (np.ndarray): an n x n output cost matrix
        R (np.ndarray): an m x m input cost matrix
        ymin (np.ndarray): an n x 1 vector of minimum allowed states for the system
        ymax (np.ndarray): an n x 1 vector of maximum allowed states for the system
        umin (np.ndarray): an m x 1 vector of minimum allowed control inputs for the system
        umax (np.ndarray): an m x 1 vector of maximum allowed control inputs for the system

    Returns:
        u (np.array): optimal control input

    """

    def __init__(
        self,
        sys,
        setpoint,
        Q,
        R,
        t_sample,
        n_predict,
        ymin=None,
        ymax=None,
        umin=None,
        umax=None,
    ):

        self.t_sample = t_sample
        self.n_predict = n_predict
        self.sys = sys
        self.setpoint = setpoint
        self.Q = Q
        self.R = R

        __, nc = self.sys.B.shape
        nm, __ = self.sys.C.shape

        if ymin is None:
            ymin = [-np.inf for i in range(nm)]
        if ymax is None:
            ymax = [np.inf for i in range(nm)]
        if umin is None:
            umin = [-np.inf for i in range(nc)]
        if umax is None:
            umax = [np.inf for i in range(nc)]

        self.ymin = ymin
        self.ymax = ymax
        self.umin = umin
        self.umax = umax

        # Set hidden attributes
        self.initialise_controller()

        self.update_setpoint(self.setpoint)

        self.__extrapolate_constraints()

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

        # get the system in a useful format
        if value.continuous:
            # Discretize the system
            value = value.disc(self.t_sample)

        self._sys = value

    @property
    def setpoint(self):
        """setpoint matrix."""
        return self._setpoint

    @setpoint.setter
    def setpoint(self, value):
        """Setter for setpoint.

        Args:
            value (np.ndarray): array of setpoints

        Raises:
            TypeError: Type of value must be a np.ndarray of dimension {}.
        """
        # Check if it is a list, if so, convert to an array
        if isinstance(value, list):
            value = np.array([value])

        if isinstance(value, np.ndarray):
            # Get setpoint shape
            value = value.reshape(-1, 1)

            np_d, __ = value.shape
            nm, __ = self.sys.C.shape

            # Calcualte the desired trajectory
            if np_d != nm:
                raise ValueError(
                    "Setpoint must be for a sequence of points of dimension (nm x 1)."
                )
        else:
            raise TypeError("Setpoint should be a list or np.ndarray")

        self._setpoint = value

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

        nm, __ = self.sys.C.shape

        if not isinstance(value, np.ndarray):
            raise TypeError("Q must be a square nd.nparray of dimension {}".format(nm))

        if value.shape != (nm, nm):
            raise ValueError(
                "The dimension of Q needs to be a square matrix of dimension {}.".format(
                    nm
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

    @property
    def t_sample(self):
        """t_sample matrix."""
        return self._t_sample

    @t_sample.setter
    def t_sample(self, value):
        """Setter for t_sample.

        Args:
            value (np.ndarray): output weighting matrix

        Raises:
            TypeError: Type of value must be an int or float, or greater than zero.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("The timestep must be an integer or float.")

        if value < 0:
            raise ValueError("The timestep must be greater than zero.")

        self._t_sample = value

    @property
    def n_predict(self):
        """n_predict matrix."""
        return self._n_predict

    @n_predict.setter
    def n_predict(self, value):
        """Setter for n_predict.

        Args:
            value (int): prediction horizon

        Raises:
            TypeError: Type of value must be an int or float, or greater than zero.
        """
        if not isinstance(value, (int, float)):
            raise TypeError("The timestep must be an integer or float.")

        if value < 0:
            raise ValueError("The timestep must be greater than zero.")

        self._n_predict = int(value)

    @property
    def ymin(self):
        """ymin matrix."""
        return self._ymin

    @ymin.setter
    def ymin(self, value):
        """Setter for ymin.

        Args:
            value (np.ndarray): minimum output value allowed

        Raises:
            ValueError: Must be of correct dimension.
            TypeError: Must be a list of np.ndarray.
        """

        if isinstance(value, list):
            value = np.array(value)

        nm, __ = self.sys.C.shape

        if isinstance(value, np.ndarray):
            value = value.reshape(-1, 1)
            if value.shape != (nm, 1):
                raise ValueError(
                    "Minimum output state dimensions should be {} x {}".format(nm, 1)
                )
        else:
            raise TypeError("Minimum output should be a list or np.ndarray")

        self._ymin = value

    @property
    def ymax(self):
        """ymax matrix."""
        return self._ymax

    @ymax.setter
    def ymax(self, value):
        """Setter for ymax.

        Args:
            value (np.ndarray): minimum output value allowed

        Raises:
            ValueError: Must be of correct dimension.
            TypeError: Must be a list of np.ndarray.
        """

        if isinstance(value, list):
            value = np.array(value)

        nm, __ = self.sys.C.shape

        if isinstance(value, np.ndarray):
            value = value.reshape(-1, 1)
            if value.shape != (nm, 1):
                raise ValueError(
                    "Minimum output state dimensions should be {} x {}".format(nm, 1)
                )
        else:
            raise TypeError("Maximum output should be a list or np.ndarray")

        self._ymax = value

    @property
    def umin(self):
        """umin matrix."""
        return self._umin

    @umin.setter
    def umin(self, value):
        """Setter for umin.

        Args:
            value (np.ndarray): minimum input value allowed

        Raises:
            ValueError: Must be of correct dimension.
            TypeError: Must be a list of np.ndarray.
        """

        __, nc = self.sys.B.shape

        if isinstance(value, list):
            value = np.array(value)
        if isinstance(value, np.ndarray):
            value = value.reshape(-1, 1)

            if value.shape != (nc, 1):
                raise ValueError(
                    "Minimum control input dimensions should be {} x {}".format(nc, 1)
                )
        else:
            raise TypeError(
                "Minimum control output should be a list or np.ndarray with {} values.".format(
                    nc
                )
            )

        self._umin = value

    @property
    def umax(self):
        """umax matrix."""
        return self._umax

    @umax.setter
    def umax(self, value):
        """Setter for umax.

        Args:
            value (np.ndarray): maximum input value allowed

        Raises:
            ValueError: Must be of correct dimension.
            TypeError: Must be a list of np.ndarray.
        """

        if isinstance(value, list):
            value = np.array(value)

        __, nc = self.sys.B.shape

        if isinstance(value, np.ndarray):
            value = value.reshape(-1, 1)
            if value.shape != (nc, 1):
                raise ValueError(
                    "Minimum control input dimensions should be {} x {}".format(nc, 1)
                )
        else:
            raise TypeError(
                "Minimum control output should be a list or np.ndarray with {} values.".format(
                    nc
                )
            )

        self._umax = value

    def __extrapolate_constraints(self):
        """Extrapolate input and output constraint data
        """

        # Extrapolate output min and max
        self.__Y_min = np.block([[self.ymin] for i in range(self.n_predict)])
        self.__Y_max = np.block([[self.ymax] for i in range(self.n_predict)])

        # Extrapolate control min and max
        self.__U_min = np.block([[self.umin] for i in range(self.n_predict)])
        self.__U_max = np.block([[self.umax] for i in range(self.n_predict)])

    # Define private functions
    def __set_Q_mpc(self):
        # Define the MPC Q matrix
        return np.kron(np.eye(self.n_predict), self.Q)

    def __set_R_mpc(self):
        # Define the MPC R matrix
        return np.kron(np.eye(self.n_predict), self.R)

    def __set_A_t(self):

        # Calculate the predicted state and output
        A_t = np.block(
            [
                [self.sys.C.dot(np.linalg.matrix_power(self.sys.A, j + 1))]
                for j in range(self.n_predict)
            ]
        )

        return A_t

    def __set_B_t(self):
        # Define the lower diagonal value for the predicted states
        B0 = 0 * self.sys.C.dot(self.sys.A).dot(self.sys.B)
        B_t = np.block(
            [
                [
                    self.sys.C.dot(np.linalg.matrix_power(self.sys.A, j - k)).dot(
                        self.sys.B
                    )
                    if k <= j
                    else B0
                    for k in range(self.n_predict)
                ]
                for j in range(self.n_predict)
            ]
        )
        return B_t

    def __set_G(self):
        # Calculate QP matrices
        return self.__B_t.T.dot(self.__Q_mpc).dot(self.__B_t) + self.__R_mpc

    def initialise_controller(self):
        """Update the MPC matrices if user makes an update.
        """
        self.__Q_mpc = self.__set_Q_mpc()
        self.__R_mpc = self.__set_R_mpc()
        self.__A_t = self.__set_A_t()
        self.__B_t = self.__set_B_t()
        self.__G = self.__set_G()

    def update_setpoint(self, setpoint):
        """Update the controller setpoint
        """
        self.setpoint = setpoint
        # Extrapolate the setpoint
        self.__Y_d = np.block([[self.setpoint] for i in range(self.n_predict)])

    # Define control calculation
    def calculate_control(self):
        """Calcualte the control input for the MPC object for the LinearSystem.

        Args:
            self

        Returns:
            u (np.ndarray): a nu x 1 control input vector.

        """

        __, nc = self.sys.B.shape

        # Predicted system value -- use state estimate?
        Y_meas = self.__A_t.dot(self.sys.x0)

        # Predicted system error
        Y_error = Y_meas - self.__Y_d

        # f
        f = self.__B_t.T.dot(Y_error)

        A_constraint = np.block(
            [
                [self.__B_t],
                [-self.__B_t],
                [np.eye(nc * self.n_predict)],
                [-np.eye(nc * self.n_predict)],
            ]
        )

        b_constraint = np.block(
            [
                [self.__Y_max - self.__A_t.dot(self.sys.x0)],
                [-(self.__Y_min - self.__A_t.dot(self.sys.x0))],
                [self.__U_max],
                [-self.__U_min],
            ]
        )

        # Solve the opimization problem
        result = qp.solve_qp(
            self.__G, -f.reshape(-1,), -A_constraint.T, -b_constraint.reshape(-1,)
        )

        # get controls
        U_c = result[0]

        # Return optimal controls for number of inputs
        return U_c[:nc].reshape(-1, 1)

    def __repr__(self):
        # return "MPC Object for sampling time {} and prediction horizon {}".format(
        #     self.t_sample, self.n_predict
        # )
        return "MPC"
