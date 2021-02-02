"""A base class for a  Kalman Filter for a Linear System.
"""

import numpy as np

from ..systems import LinearSystem


class KalmanFilter(object):
    def __init__(self, sys, est_covar=None, proc_covar=None, meas_covar=None):
        """Contructor to KalmanFilter.

        Args:
        sys (LinearSystem): controlib LinearSystem object
        est_covar (np.array): n x n state estimate covariance matrix
        proc_covar (np.array): m x m input covariance matrix
        meas_covar (np.array): p x p measurement covariance matrix

        """
        self.sys = sys

        ns, nc = sys.B.shape
        nm, __ = sys.C.shape

        if est_covar is None:
            est_covar = np.eye(ns)

        if proc_covar is None:
            proc_covar = np.eye(nc)

        if meas_covar is None:
            meas_covar = np.eye(nm)

        self.est_covar = est_covar
        self.proc_covar = proc_covar
        self.meas_covar = meas_covar

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
    def est_covar(self):
        """est_covar matrix."""
        return self._est_covar

    @est_covar.setter
    def est_covar(self, value):
        """Setter for est_covar.

        Args:
            value (np.array): system est_covar matrix.

        Raises:
            TypeError: Type of value must be of type float.
        """
        if not isinstance(value, np.ndarray):
            raise TypeError("The estimate covariance must be an np.ndarray.")

        if value.shape != self.sys.A.shape:
            raise ValueError(
                "est_covar must be a square matrix of dimension {}.".format(
                    self.sys.A.shape[0]
                )
            )

        self._est_covar = value

    @property
    def proc_covar(self):
        """proc_covar matrix."""
        return self._proc_covar

    @proc_covar.setter
    def proc_covar(self, value):
        """Setter for proc_covar.

        Args:
            value (np.array): system proc_covar matrix.

        Raises:
            TypeError: Type of value must be of type float.
        """

        if not isinstance(value, np.ndarray):
            raise TypeError("The process covariance must be an np.ndarray.")

        __, nc = self.sys.B.shape

        if value.shape != (nc, nc):
            raise ValueError(
                "proc_covar must be a square matrix of dimension {}.".format(nc)
            )

        self._proc_covar = value

    @property
    def meas_covar(self):
        """meas_covar matrix."""
        return self._meas_covar

    @meas_covar.setter
    def meas_covar(self, value):
        """Setter for meas_covar.

        Args:
            value (np.array): system meas_covar matrix.

        Raises:
            TypeError: Type of value must be of type float.
        """

        if not isinstance(value, np.ndarray):
            raise TypeError("The measurement covariance must be an np.ndarray.")

        nm, __ = self.sys.C.shape

        if value.shape != (nm, nm):
            raise ValueError(
                "meas_covar must be a square matrix of dimension {}.".format(nm)
            )

        self._meas_covar = value

    # Predictive Step
    def predict(self, u, dt=0):
        """Preditive step

        Args:
            u (np.ndarray): an nu x 1 vector of control (process) measurements
            dt (float): time delta since last update
        """

        # Discretise the system for this amount of time
        if self.sys.continuous:
            if dt > 0:
                sys_d = self.sys.disc(dt)

                # Update the state
                sys_d.x0 = self.sys.x0
            else:
                raise ValueError("Timestep dt must be greater than zero.")
        else:
            sys_d = self.sys

        sys_d.apply_control(u)

        # Update the state
        self.sys.x0 = sys_d.x0

        # Update estimate covar using discrete system
        self.est_covar = sys_d.A.dot(self.est_covar).dot(sys_d.A.T) + sys_d.B.dot(
            self.proc_covar
        ).dot(sys_d.B.T)

    def correct(self, y_meas):
        """Corrective step

        Args:
            y_meas (np.ndarray): a nm x 1 array of measured values

        """

        # Check dimensions of measured and expected values
        if y_meas.shape != self.sys.C.dot(self.sys.x0).shape:
            raise ValueError("Measurement vector must equal the dimension of C*x.")

        ns, __ = self.est_covar.shape

        # Calculate innovation
        meas_inn = y_meas - self.sys.C.dot(self.sys.x0)

        # Calculate innovation covariance
        covar_inn = self.sys.C.dot(self.est_covar).dot(self.sys.C.T) + self.meas_covar

        # Calculate Kalmain Gain
        kal_gain = self.est_covar.dot(self.sys.C.T).dot(np.linalg.inv(covar_inn))

        # Update estimate and covar
        self.sys.x0 = self.sys.x0 + kal_gain.dot(meas_inn)
        self.est_covar = (np.eye(ns) - kal_gain.dot(self.sys.C)).dot(self.est_covar)

    def __repr__(self):
        return "{}, {}, {}".format(self.est_covar, self.proc_covar, self.meas_covar)
