"""Class for a Laguerre Network, of the form

    dx/dt = Ax + Bu
    y = Cx

    This is a general class, where the user specifies the system dimension,
    and associated pole location.

    This method returns a inherits a LinearSystem.
"""

import numpy as np

from . import LinearSystem


class LaguerreNetwork(object):
    def __init__(self, dim, alpha, inputs=1, outputs=1, continuous=True, x0=None):
        """Default constructor for the LaguerreNetwork

        Args:
            dim (int): the dimension of the Laguerre Network
            alpha (float): pole location of Laguerre Network

        Returns:
            self (LinearSystem)

        """

        self.dim = dim
        self.alpha = alpha
        self.inputs = inputs
        self.outputs = outputs
        self.continuous = continuous

        if x0 is None:
            x0 = self._construct_B()

        self.x0 = x0

        # Implicit function parameters
        self.A = self._construct_A()
        self.B = self._construct_B()
        self.C = np.zeros((self.outputs, self.dim))

    def _contruct_polynomials(self, time):
        """Construct N Laguerre Polynomials for network
        at current time.

        Args:
            time (float): current system time

        Returns:
            L (np.array): value of Laguerre polynomials 0 - N for time.

        """
        # First two Laugerre Polynomials

        L = [1.0, 1.0 - time]

        # Recursive addition
        for k in range(1, self.dim):
            Lk = ((2 * k + 1 - time) * L[-1] - k * L[-2]) / (k + 1)
            L.append(Lk)

        return L

    def laguerre_projection(self, measured_output, time_series):
        """Estimate Laguerre coefficients based on some measured outputs
        for a series of time.

        Args:
            measured_output (np.array): array of measured outputs
            time_series (list or np.array): time series related to measured data
        """

        if hasattr(measured_output, "shape"):
            ns, __ = measured_output.shape
        else:
            raise TypeError(
                "Measured Output Vector must be a numpy array. Each row must be a state value."
            )

        if ns != self.outputs:
            raise ValueError(
                "Number of measured states by equal number of Laguerre outputs."
            )

        if isinstance(time_series, np.ndarray):
            time_series = np.reshape(time_series, (-1,)).tolist()

        if not isinstance(time_series, list):
            raise TypeError("Time series must be a list of np.ndarray.")

        # Get Laguerre parameters for each function for each timestep
        Lp = np.array([self._contruct_polynomials(t) for t in time_series])

        # Initialise output laguerre coefficients
        C_lag = np.array([])

        # loop through the states. Get the laguerre coefficients for each
        for state in measured_output:
            y_meas = np.reshape(np.array(state), (-1, 1))

            # Solve for laguerre coefficients
            c_lag = np.linalg.inv(Lp.T.dot(Lp)).dot(Lp.T).dot(y_meas)

            # Add the data to coefficients to the system matrix
            C_lag = np.hstack((C_lag, np.reshape(c_lag, (-1,))))

        C_lag = np.reshape(C_lag, (ns, -1))

        # Update the Laguerre System
        self.C = C_lag

        # Return Lp so we can reconstruct the state
        return Lp

    def online_projection(self, u, dt, P):
        """Get Laguerre projection in real-time
        """
        # Define the laguerre system as a linear system
        pass

    def _construct_A(self):
        # Construct A matrix
        # Calcualte initial Laguerre vector
        beta = 1 - self.alpha ** 2
        # Laguerre dynamics
        Al = np.zeros((self.dim, self.dim))
        for i in range(self.dim):
            if not i:
                Al += self.alpha * np.eye(self.dim)
            else:
                Al[i:, :-i] += (
                    (-1) ** (i - 1)
                    * self.alpha ** (i - 1)
                    * beta
                    * np.eye(self.dim - i)
                )

        return Al

    def _construct_B(self):
        # Construct B matrix
        beta = 1 - self.alpha ** 2
        B = np.sqrt(beta) * np.array(
            [
                [(-1) ** (i - 1) * self.alpha ** (i - 1) for j in range(self.inputs)]
                for i in range(1, self.dim + 1)
            ]
        )
        return B

    def __repr__(self):
        return "Laguerre Network of dimension {}, with pole at {}.".format(
            self.dim, self.alpha
        )

    @property
    def dim(self):
        """dim matrix."""
        return self._dim

    @dim.setter
    def dim(self, value):
        """Setter for dim.

        dimrgs:
            value (int or flaot): system dimension.

        Raises:
            TypeError: Type of value must be of type integer (or float).
        """

        if isinstance(value, float):
            value = int(value)

        if not isinstance(value, int):
            raise TypeError("Dimension must be an integer value.")

        self._dim = value

    @property
    def alpha(self):
        """alpha matrix."""
        return self._alpha

    @alpha.setter
    def alpha(self, value):
        """Setter for alpha.

        Args:
            value (float): Laguerre Network pole location.

        Raises:
            ValueError: Value must be between 0 and 1 inclusive.
        """

        if 0 <= value <= 1:
            self._alpha = value
        else:
            raise ValueError("The value of alpha must be between 0 and 1.")

    @property
    def inputs(self):
        """inputs matrix."""
        return self._inputs

    @inputs.setter
    def inputs(self, value):
        """Setter for inputs.

        Args:
            value (int): Number of system inputs

        Raises:
            ValueError: Value must be greater than 1
        """

        if value < 1:
            raise ValueError("The value of inputs must be greater than 1")
        else:
            self._inputs = int(value)

    @property
    def outputs(self):
        """outputs matrix."""
        return self._outputs

    @outputs.setter
    def outputs(self, value):
        """Setter for outputs.

        Args:
            value (int): Number of system outputs

        Raises:
            ValueError: Value must be greater than 1
        """

        if value < 1:
            raise ValueError("The value of outputs must be greater than 1")
        else:
            self._outputs = int(value)

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

        # Reshape the state vector
        value = np.reshape(value, (self.dim, -1))

        if value.shape != (self.dim, 1):
            raise ValueError("x0 must be a 1-d vector of states.")

        self._x0 = value
