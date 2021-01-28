import numpy as np


class Unicycle:
    def __init__(self, q=None, u=None):
        """
        """

        if q is None:
            q = np.zeros((3, 1))
        else:
            q = q.reshape(3, 1)

        if u is None:
            u = np.zeros((2, 1))
        else:
            u = u.reshape(2, 1)

        self.q = q
        self.u = u

        self.__update()

    def __update(self):
        """
        """
        self.G = np.array(
            [[np.cos(self.q[2, 0]), 0], [np.sin(self.q[2, 0]), 0], [0, 1],]
        )

    def apply_control(self, u, dt):
        """
        """
        self.__update()
        self.q = self.q + self.G.dot(self.u) * dt

    def linear_matrices(self, u):
        """
        """
        A = np.array(
            [
                [0, 0, -self.u[0, 0] * np.sin(self.q[2, 0])],
                [0, 0, self.u[0, 0] * np.cos(self.q[2, 0])],
                [0, 0, 0],
            ]
        )

        B = np.array([[np.cos(self.q[2, 0]), 0], [np.sin(self.q[2, 0]), 0], [0, 1],])

        return A, B
