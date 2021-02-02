import numpy as np


class Unicycle:
    def __init__(self, x, y, theta, v, omega):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.omega = omega

        self.__nq = 3
        self.__nu = 2

    def as_vec(self):
        return np.array([self.x, self.y, self.theta]).reshape(self.__nq, 1)

    def G(self):
        return np.array([[np.cos(self.q[2, 0]), 0], [np.sin(self.q[2, 0]), 0], [0, 1],])

    def apply_control(self, u, dt):
        u = u.reshape(self.__nu, 1)

        q = self.as_vec()
        q = q + self.G().dot(u) * dt

        self.x = q[0, 0]
        self.y = q[1, 0]
        self.theta = q[2, 0]
        self.v = u[0, 0]
        self.omega = u[1, 0]

    def linear_matrices(self, q, u):
        q = q.reshape(self.__nq, 1)
        u = u.reshape(self.__nu, 1)

        A = np.array(
            [
                [0, 0, -u[0, 0] * np.sin(q[2, 0])],
                [0, 0, u[0, 0] * np.cos(q[2, 0])],
                [0, 0, 0],
            ]
        )

        B = np.array([[np.cos(q[2, 0]), 0], [np.sin(q[2, 0]), 0], [0, 1],])

        return A, B
