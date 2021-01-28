import numpy as np


class Agent:
    def __init__(self, x, y, theta, v, omega):
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        self.omega = omega

    def as_vec(self):
        return np.array([self.x, self.y, self.theta]).reshape(-1, 1)

    def controls(self):
        return np.array([self.v, self.omega]).reshape(-1, 1)

    def update(self, dt):
        q = self.as_vec()
        u = self.controls()
        q = q + q.dot(u) * dt

        self.x = q[0, 0]
        self.y = q[1, 0]
        self.theta = q[2, 0]
